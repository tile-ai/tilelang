/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_schedule.cc
 * \brief AutoSchedule pass for TileLang
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "./common/attr.h"
#include "./common/collector.h"
#include "auto_schedule.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

// Forward declaration
Stmt ApplyWarpgroupPartitionToIRStructure(IRStructure *root, IterVar thread_var,
                                          BarrierManager &barrier_manager,
                                          const bool enable_epi,
                                          PrimExpr thread_count[2]);

// Helper function to collect all prefix tasks (consecutive tasks without
// register region at the beginning of sequences)
void CollectPrefixTasks(IRStructure *node,
                        std::unordered_set<TaskNode *> &prefix_tasks,
                        bool &prefix_valid) {
  if (!node)
    return;

  if (!prefix_valid)
    return;

  if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (auto &child : seq->children) {
      CollectPrefixTasks(child.get(), prefix_tasks, prefix_valid);
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);
    prefix_valid = false;
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    CollectPrefixTasks(wrapper->child.get(), prefix_tasks, prefix_valid);
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    CollectPrefixTasks(unit->child.get(), prefix_tasks, prefix_valid);
  } else if (node->IsTask()) {
    if (CountRegisterRegions(node) == 0) {
      auto task = static_cast<TaskNode *>(node);
      prefix_tasks.insert(task);
    } else {
      prefix_valid = false;
    }
  } else {
    LOG(FATAL);
  }
}

// Global warpgroup id assignment - should be called from the top level
// Tasks that use the same register region must have the same warpgroup id
// Goal: balance weighted latency between two warpgroups (0 and 1)
// Weighted latency = latency * tripcount (tripcount = 100 for non-constant loop
// extent)
bool AssignWarpgroupIdsGlobal(IRStructure *root) {
  if (!root) {
    LOG(FATAL) << "Empty root";
  }
  // PrintIRStructure(root);

  // Step 1: Collect all TaskNodes with context information
  std::vector<TaskNodeWithContext> all_tasks;
  CollectAllTaskNodesWithContext(root, all_tasks);

  if (all_tasks.empty()) {
    LOG(FATAL) << "No task";
  }

  int n = all_tasks.size();

  // Collect all prefix tasks (consecutive tasks without register region at the
  // beginning of sequences)
  std::unordered_set<TaskNode *> prefix_tasks;
  bool prefix_valid = true;
  CollectPrefixTasks(root, prefix_tasks, prefix_valid);

  // Step 2: Initialize all warpgroup ids to -1 (unassigned)
  for (auto &task_ctx : all_tasks) {
    task_ctx.task->SetWarpgroupId(-1);
  }

  // Step 3: Build union-find based on shared register regions
  TaskUnionFind uf(n);
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (UseSameRegisterRegion(all_tasks[i].task, all_tasks[j].task)) {
        uf.unite(i, j);
      }
    }
  }

  // Step 4: Group tasks by connected component
  std::unordered_map<int, std::vector<int>> components; // root -> indices
  for (int i = 0; i < n; i++) {
    int root_idx = uf.find(i);
    components[root_idx].push_back(i);
  }

  // Step 5: Calculate weighted latency for each component that has register
  // regions
  std::vector<ComponentInfo> component_infos;
  for (const auto &kv : components) {
    int root = kv.first;
    const std::vector<int> &indices = kv.second;
    // Check if this component has any task with register regions
    bool has_register_region = false;
    for (int idx : indices) {
      if (CountRegisterRegions(all_tasks[idx].task) > 0) {
        has_register_region = true;
        break;
      }
    }
    int64_t total_weighted_latency = 0;
    bool has_task = false;
    bool has_tma_core = false;
    bool has_tensor_core = false;
    for (int idx : indices) {
      TaskNode *task = all_tasks[idx].task;
      if (prefix_tasks.find(task) != prefix_tasks.end()) {
        // This is a prefix task, skip it (it won't participate in warpgroup
        // specialize)
        continue;
      }
      if (task->ContainsLoopBreak()) {
        // Skip tasks with loop_break, they won't participate in warpgroup
        // specialize and keep warpgroup_id = -1
        continue;
      }
      has_task = true;
      int64_t latency = task->GetLatency();
      int64_t tripcount = all_tasks[idx].tripcount;
      total_weighted_latency += latency * tripcount;
      has_tma_core |= task->UsesTMACore();
      has_tensor_core |= task->UsesTensorCore();
    }
    if (has_task) {
      component_infos.push_back({root, total_weighted_latency, indices,
                                 has_tma_core, has_tensor_core});
    }
  }

  // Step 6: Sort components by weighted latency (descending)
  // Greedy assignment: assign larger components to the warpgroup with less
  // current usage
  std::sort(component_infos.begin(), component_infos.end(),
            [](const ComponentInfo &a, const ComponentInfo &b) {
              return a.weighted_latency > b.weighted_latency;
            });

  // Step 7: Greedy assignment to balance weighted latency
  int64_t warpgroup0_latency = 0;
  int64_t warpgroup1_latency = 0;

  for (const auto &comp : component_infos) {
    int assigned_warpgroup = 0;
    if (warpgroup0_latency <= warpgroup1_latency) {
      assigned_warpgroup = 0;
      warpgroup0_latency += comp.weighted_latency;
    } else {
      assigned_warpgroup = 1;
      warpgroup1_latency += comp.weighted_latency;
    }
  }

  int64_t max_latency = std::max(warpgroup0_latency, warpgroup1_latency);
  int64_t min_latency = std::min(warpgroup0_latency, warpgroup1_latency);
  if ((double)max_latency / min_latency < 1.1) {
    int64_t warpgroup0_latency = 0;
    int64_t warpgroup1_latency = 0;

    for (const auto &comp : component_infos) {
      int assigned_warpgroup = 0;
      if (warpgroup0_latency <= warpgroup1_latency) {
        assigned_warpgroup = 0;
        warpgroup0_latency += comp.weighted_latency;
      } else {
        assigned_warpgroup = 1;
        warpgroup1_latency += comp.weighted_latency;
      }

      // Assign warpgroup id to all tasks in this component
      // Skip tasks that contain loop_break (keep warpgroup_id = -1)
      for (int idx : comp.task_indices) {
        TaskNode *task = all_tasks[idx].task;
        if (!task->ContainsLoopBreak()) {
          task->SetWarpgroupId(assigned_warpgroup);
        }
        // Tasks with loop_break keep warpgroup_id = -1
      }
    }
    return true;
  } else {
    int64_t warpgroup0_latency = 0;
    int64_t warpgroup1_latency = 0;
    for (const auto &comp : component_infos) {
      int assigned_warpgroup = 0;
      if (comp.uses_tensor_core_ && !comp.uses_tma_core_) {
        assigned_warpgroup = 0;
        warpgroup0_latency += comp.weighted_latency;
      } else if (!comp.uses_tensor_core_ && comp.uses_tma_core_) {
        assigned_warpgroup = 1;
        warpgroup1_latency += comp.weighted_latency;
      } else if (warpgroup0_latency <= warpgroup1_latency) {
        assigned_warpgroup = 0;
        warpgroup0_latency += comp.weighted_latency;
      } else {
        assigned_warpgroup = 1;
        warpgroup1_latency += comp.weighted_latency;
      }

      // Assign warpgroup id to all tasks in this component
      // Skip tasks that contain loop_break (keep warpgroup_id = -1)
      for (int idx : comp.task_indices) {
        TaskNode *task = all_tasks[idx].task;
        if (!task->ContainsLoopBreak()) {
          task->SetWarpgroupId(assigned_warpgroup);
        }
        // Tasks with loop_break keep warpgroup_id = -1
      }
    }
    return false;
  }
}

// Builder that collects ScheduleUnits from IRStructure
class ScheduleUnitBuilder {
public:
  bool Build(IRStructure *root) {
    ScheduleRecursive(root);

    // Global warpgroup id assignment from the top level
    return AssignWarpgroupIdsGlobal(root);
  }

  // New recursive scheduling function that replaces Collect method
  // Directly schedules the entire IRStructure tree recursively in place
  void ScheduleRecursive(IRStructure *node);

  // Z3-based scheduler that calls Python implementation via FFI
  std::vector<IRStructure *>
  Z3SchedulePython(const std::vector<IRStructure *> &nodes) {
    size_t n = nodes.size();
    if (n <= 1) {
      if (n == 1) {
        // For TaskNode, set start time
        if (nodes[0]->IsTask()) {
          auto task = static_cast<TaskNode *>(nodes[0]);
          task->SetStartTime(0);
        }
      }
      return nodes;
    }

    try {
      // Get the Python-registered function using ffi::Function::GetGlobal
      static std::optional<ffi::Function> z3_schedule_func =
          ffi::Function::GetGlobal("tl.transform.z3_schedule_python");
      if (!z3_schedule_func.has_value()) {
        LOG(WARNING) << "Python Z3 scheduler not registered, falling back to "
                        "topological sort";
        return nodes;
      }

      // Prepare input data
      std::vector<int64_t> latencies;
      std::vector<int64_t> iis;
      std::vector<int64_t> resource_flags;
      std::vector<std::pair<int64_t, int64_t>> data_deps;
      std::vector<std::pair<int64_t, int64_t>> resource_deps;

      latencies.reserve(n);
      iis.reserve(n);
      resource_flags.reserve(n);

      for (size_t i = 0; i < n; ++i) {
        const IRStructure *node = nodes[i];
        latencies.push_back(node->GetLatency());
        iis.push_back(node->GetII());

        // Encode resource flags as bitmask
        int64_t flags = 0;
        if (node->UsesCUDACore())
          flags |= 1;
        if (node->UsesTMACore())
          flags |= 2;
        if (node->UsesTensorCore())
          flags |= 4;
        resource_flags.push_back(flags);
      }

      // Collect data dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasDependency(nodes[i], nodes[j])) {
            data_deps.emplace_back(i, j);
          }
        }
      }

      // Collect resource dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasResourceDependency(nodes[i], nodes[j])) {
            resource_deps.emplace_back(i, j);
          }
        }
      }

      // Convert vectors to TVM containers
      ffi::Array<int64_t> tvm_latencies;
      ffi::Array<int64_t> tvm_iis;
      ffi::Array<int64_t> tvm_resource_flags;
      ffi::Array<ffi::Array<int64_t>> tvm_data_deps;
      ffi::Array<ffi::Array<int64_t>> tvm_resource_deps;

      for (auto val : latencies) {
        tvm_latencies.push_back(val);
      }
      for (auto val : iis) {
        tvm_iis.push_back(val);
      }
      for (auto val : resource_flags) {
        tvm_resource_flags.push_back(val);
      }
      for (const auto &dep : data_deps) {
        ffi::Array<int64_t> pair;
        pair.push_back(dep.first);
        pair.push_back(dep.second);
        tvm_data_deps.push_back(pair);
      }
      for (const auto &dep : resource_deps) {
        ffi::Array<int64_t> pair;
        pair.push_back(dep.first);
        pair.push_back(dep.second);
        tvm_resource_deps.push_back(pair);
      }

      // Extract results
      // Python function returns only start_times, C++ side will sort by
      // start_time
      auto start_times =
          z3_schedule_func
              .value()(tvm_latencies, tvm_iis, tvm_resource_flags,
                       tvm_data_deps, tvm_resource_deps)
              .cast<ffi::Array<int64_t>>();

      if (start_times.size() != n) {
        LOG(WARNING) << "Python Z3 scheduler returned invalid results (size "
                        "mismatch), falling back to topological sort";
        return nodes;
      }

      // Apply start times to nodes
      for (size_t i = 0; i < n; ++i) {
        // Only TaskNode has SetStartTime method
        if (nodes[i]->IsTask()) {
          auto task = static_cast<TaskNode *>(nodes[i]);
          task->SetStartTime(start_times[i]);
        }
      }

      // Create sorted task list based on start_time (and original index as
      // tie-breaker)
      std::vector<std::pair<int64_t, size_t>> start_time_with_idx;
      start_time_with_idx.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        start_time_with_idx.emplace_back(start_times[i], i);
      }

      // Sort by start_time, then by original index
      std::sort(start_time_with_idx.begin(), start_time_with_idx.end(),
                [](const std::pair<int64_t, size_t> &a,
                   const std::pair<int64_t, size_t> &b) {
                  if (a.first != b.first)
                    return a.first < b.first;
                  return a.second < b.second;
                });

      // Create sorted node list
      std::vector<IRStructure *> sorted_nodes;
      sorted_nodes.reserve(n);
      for (const auto &p : start_time_with_idx) {
        sorted_nodes.push_back(nodes[p.second]);
      }

      return sorted_nodes;

    } catch (const std::exception &e) {
      LOG(WARNING) << "Python Z3 scheduler failed with exception: " << e.what()
                   << ", falling back to topological sort";
      return nodes;
    } catch (...) {
      LOG(WARNING) << "Python Z3 scheduler failed with unknown exception, "
                      "falling back to topological sort";
      return nodes;
    }
  }

  // Z3-based scheduler for loops that calls Python implementation via FFI
  // with distance-aware dependencies
  void Z3SchedulePythonLoop(ControlNode *ctrl) {
    auto seq_body = static_cast<SequenceNode *>(ctrl->child.get());
    std::vector<IRStructure *> nodes;
    nodes.reserve(seq_body->children.size());
    for (const auto &child : seq_body->children) {
      nodes.push_back(child.get());
    }

    size_t n = nodes.size();
    int64_t ii = 1; // default II

    auto num_stages = 1;
    auto num_stages_val = ctrl->control.get()->annotations.Get("num_stages");
    if (num_stages_val.has_value()) {
      num_stages = num_stages_val.value().cast<IntImm>()->value;
    }

    static std::optional<ffi::Function> z3_schedule_loop_func =
        ffi::Function::GetGlobal("tl.transform.z3_schedule_loop_python");
    if (!z3_schedule_loop_func.has_value()) {
      LOG(FATAL) << "Python Z3 loop scheduler not registered, falling back "
                    "to topological sort";
    }

    // Prepare input data
    std::vector<int64_t> latencies;
    std::vector<int64_t> iis;
    std::vector<int64_t> resource_flags;
    std::vector<std::tuple<int64_t, int64_t, int64_t>>
        data_deps; // (i, j, distance)
    std::vector<std::pair<int64_t, int64_t>> resource_deps;

    latencies.reserve(n);
    iis.reserve(n);
    resource_flags.reserve(n);

    for (size_t i = 0; i < n; ++i) {
      const IRStructure *node = nodes[i];
      latencies.push_back(node->GetLatency());
      iis.push_back(node->GetII());

      // Encode resource flags as bitmask
      int64_t flags = 0;
      if (node->UsesCUDACore())
        flags |= 1;
      if (node->UsesTMACore())
        flags |= 2;
      if (node->UsesTensorCore())
        flags |= 4;
      resource_flags.push_back(flags);
    }

    // Collect data dependencies with distance
    // distance = 0 if i < j (same iteration), distance = 1 if i > j (next
    // iteration)
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (i == j)
          continue; // Skip self-dependency
        if (HasDependency(nodes[i], nodes[j])) {
          if (i < j) {
            data_deps.emplace_back(i, j, 0);
          } else {
            int64_t distance =
                HasRegisterDependency(nodes[i], nodes[j]) ? 1 : num_stages;
            data_deps.emplace_back(i, j, distance);
          }
        }
      }
    }

    // Collect resource dependencies
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        if (HasResourceDependency(nodes[i], nodes[j])) {
          resource_deps.emplace_back(i, j);
        }
      }
    }

    // Convert vectors to TVM containers
    ffi::Array<int64_t> tvm_latencies;
    ffi::Array<int64_t> tvm_iis;
    ffi::Array<int64_t> tvm_resource_flags;
    ffi::Array<ffi::Array<int64_t>>
        tvm_data_deps; // each element is [i, j, distance]
    ffi::Array<ffi::Array<int64_t>> tvm_resource_deps;

    for (auto val : latencies) {
      tvm_latencies.push_back(val);
    }
    for (auto val : iis) {
      tvm_iis.push_back(val);
    }
    for (auto val : resource_flags) {
      tvm_resource_flags.push_back(val);
    }
    for (const auto &dep : data_deps) {
      ffi::Array<int64_t> triple;
      triple.push_back(std::get<0>(dep));
      triple.push_back(std::get<1>(dep));
      triple.push_back(std::get<2>(dep));
      tvm_data_deps.push_back(triple);
    }
    for (const auto &dep : resource_deps) {
      ffi::Array<int64_t> pair;
      pair.push_back(dep.first);
      pair.push_back(dep.second);
      tvm_resource_deps.push_back(pair);
    }

    // Extract results
    // Python function returns (start_times, stages) as a tuple of two
    // arrays
    auto return_val =
        z3_schedule_loop_func
            .value()(num_stages, tvm_latencies, tvm_iis, tvm_resource_flags,
                     tvm_data_deps, tvm_resource_deps)
            .cast<ffi::Tuple<ffi::Array<int64_t>, ffi::Array<int>, int64_t>>();

    ffi::Array<int64_t> start_times = return_val.get<0>();
    ffi::Array<int> stages = return_val.get<1>();
    ii = return_val.get<2>();

    // Apply start times and promote flags to nodes
    std::map<IRStructure *, int> stage_map;
    size_t num_promoted = 0;
    int min_stage = ii * 2;
    for (size_t i = 0; i < n; ++i) {
      nodes[i]->SetStartTime(start_times[i]);
      min_stage = std::min(min_stage, stages[i]);
    }
    // Force all stages to start from 0
    for (size_t i = 0; i < n; ++i) {
      stage_map[nodes[i]] = stages[i] - min_stage;
      if (stages[i] != min_stage) {
        num_promoted++;
      }
    }
    std::vector<std::pair<int64_t, size_t>> start_time_with_idx;
    start_time_with_idx.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      start_time_with_idx.emplace_back(
          start_times[i] + stages[i] * return_val.get<2>(), i);
    }

    // Sort by start_time, then by original index
    std::sort(start_time_with_idx.begin(), start_time_with_idx.end(),
              [](const std::pair<int64_t, size_t> &a,
                 const std::pair<int64_t, size_t> &b) {
                if (a.first != b.first)
                  return a.first < b.first;
                return a.second < b.second;
              });

    // Create sorted node list
    std::vector<IRStructure *> sorted_nodes;
    sorted_nodes.reserve(n);
    for (const auto &p : start_time_with_idx) {
      sorted_nodes.push_back(nodes[p.second]);
    }

    // Reorder children if order changed
    bool order_changed = false;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
      if (sorted_nodes[i] != nodes[i]) {
        order_changed = true;
        break;
      }
    }

    if (order_changed) {
      std::unordered_map<IRStructure *, size_t> node_to_index;
      for (size_t i = 0; i < nodes.size(); ++i) {
        node_to_index[nodes[i]] = i;
      }

      std::vector<std::unique_ptr<IRStructure>> reordered_children;
      reordered_children.reserve(sorted_nodes.size());
      for (IRStructure *sorted_node : sorted_nodes) {
        auto it = node_to_index.find(sorted_node);
        if (it == node_to_index.end()) {
          LOG(FATAL) << "[ScheduleRecursive] IRStructure not found in "
                        "children mapping";
        }
        reordered_children.push_back(std::move(seq_body->children[it->second]));
      }
      seq_body->children = std::move(reordered_children);
    }

    for (auto &node : seq_body->children) {
      auto unit = std::make_unique<ScheduleUnit>();
      unit->stage = stage_map[node.get()];
      unit->child = std::move(node);
      node = std::move(unit);
    }

    if (num_promoted > 0) {
      ctrl->SetPromote(true);
    }

    // Estimate overall latency: II * tripcount
    // Get tripcount from For loop extent and step
    int64_t tripcount = 100; // default if not constant
    const ForNode *for_node = ctrl->control.get();
    PrimExpr loop_extent = for_node->extent;
    PrimExpr loop_step = for_node->step.has_value()
                             ? for_node->step.value()
                             : IntImm(DataType::Int(32), 1);

    if (const auto *extent_int = loop_extent.as<IntImmNode>()) {
      if (const auto *step_int = loop_step.as<IntImmNode>()) {
        // Calculate ceil(extent / step)
        int64_t extent = extent_int->value;
        int64_t step = step_int->value;
        if (step > 0) {
          // ceil(extent / step) = (extent + step - 1) / step
          tripcount = (extent + step - 1) / step;
        } else {
          // Invalid step, use extent as fallback
          tripcount = extent;
        }
      } else {
        // Step is not constant, use 100 as default
        tripcount = 100;
      }
    } else {
      // Extent is not constant, use 100 as default
      tripcount = 100;
    }
    int64_t overall_latency = ii * tripcount;

    // Set II and latency on the ControlNode (which delegates to child)
    ctrl->SetII(overall_latency);
    ctrl->SetLatency(overall_latency);
  }

  // Set thread index variable for warpgroup partition
  void SetThreadVar(IterVar thread_var) { thread_var_ = thread_var; }

private:
  IterVar thread_var_; // Thread index variable for warpgroup partition

  // Check if two regions refer to the same buffer
  bool SameBuffer(const BufferRegion &a, const BufferRegion &b) const {
    return a->buffer.same_as(b->buffer);
  }

  // Check if two IRStructures have data dependency (excluding read-after-read)
  bool HasDependency(const IRStructure *a, const IRStructure *b) const {
    // Check if either node contains loop_break (if it's a TaskNode)
    // Tasks with loop_break have control dependencies with all other tasks
    // because loop_break can change control flow and affect execution order
    if (a->IsTask()) {
      const TaskNode *task_a = static_cast<const TaskNode *>(a);
      if (task_a->ContainsLoopBreak()) {
        // If task_a contains loop_break, it has dependency with b
        // because loop_break affects control flow and execution order
        return true;
      }
    }
    if (b->IsTask()) {
      const TaskNode *task_b = static_cast<const TaskNode *>(b);
      if (task_b->ContainsLoopBreak()) {
        // If task_b contains loop_break, it has dependency with a
        // because loop_break affects control flow and execution order
        return true;
      }
    }

    // Check all combinations of accesses
    // a writes, b reads (RAW)
    // a reads, b writes (WAR)
    // a writes, b writes (WAW)
    // a reads, b reads (RAR) - no dependency

    // For simplicity, we check if they access the same buffer
    // and at least one of them writes to that buffer
    for (const auto &write_region_a : a->GetWriteRegions()) {
      for (const auto &read_region_b : b->GetReadRegions()) {
        if (SameBuffer(write_region_a, read_region_b))
          return true;
      }
      for (const auto &write_region_b : b->GetWriteRegions()) {
        if (SameBuffer(write_region_a, write_region_b))
          return true;
      }
    }
    for (const auto &read_region_a : a->GetReadRegions()) {
      for (const auto &write_region_b : b->GetWriteRegions()) {
        if (SameBuffer(read_region_a, write_region_b))
          return true;
      }
    }
    return false;
  }

  // Check if two IRStructures have data dependency (excluding read-after-read)
  bool HasRegisterDependency(const IRStructure *a, const IRStructure *b) const {
    // Check if either node contains loop_break (if it's a TaskNode)
    // Tasks with loop_break have control dependencies with all other tasks
    // because loop_break can change control flow and affect execution order
    if (a->IsTask()) {
      const TaskNode *task_a = static_cast<const TaskNode *>(a);
      if (task_a->ContainsLoopBreak()) {
        // If task_a contains loop_break, it has dependency with b
        // because loop_break affects control flow and execution order
        return true;
      }
    }
    if (b->IsTask()) {
      const TaskNode *task_b = static_cast<const TaskNode *>(b);
      if (task_b->ContainsLoopBreak()) {
        // If task_b contains loop_break, it has dependency with a
        // because loop_break affects control flow and execution order
        return true;
      }
    }

    // Check all combinations of accesses
    // a writes, b reads (RAW)
    // a reads, b writes (WAR)
    // a writes, b writes (WAW)
    // a reads, b reads (RAR) - no dependency

    // For simplicity, we check if they access the same buffer
    // and at least one of them writes to that buffer
    for (const auto &write_region_a : a->GetWriteRegions()) {
      if (IsSharedBuffer(write_region_a.get()->buffer))
        continue;
      for (const auto &read_region_b : b->GetReadRegions()) {
        if (SameBuffer(write_region_a, read_region_b))
          return true;
      }
      for (const auto &write_region_b : b->GetWriteRegions()) {
        if (SameBuffer(write_region_a, write_region_b))
          return true;
      }
    }
    for (const auto &read_region_a : a->GetReadRegions()) {
      if (IsSharedBuffer(read_region_a.get()->buffer))
        continue;
      for (const auto &write_region_b : b->GetWriteRegions()) {
        if (SameBuffer(read_region_a, write_region_b))
          return true;
      }
    }
    return false;
  }

  // Check if an IRStructure has any register region
  bool HasRegisterRegion(const IRStructure *node) const {
    return CountRegisterRegions(node) > 0;
  }

  // Check if two IRStructures have resource dependency (use same hardware
  // resource)
  bool HasResourceDependency(const IRStructure *a, const IRStructure *b) const {
    // Resource dependencies occur when two tasks use the same hardware resource
    // that cannot be used simultaneously (or has limited throughput)

    // Check TMA core dependency
    if (a->UsesTMACore() && b->UsesTMACore()) {
      return true; // Both use TMA core, cannot execute simultaneously
    }

    // Check Tensor core dependency
    if (a->UsesTensorCore() && b->UsesTensorCore()) {
      return true; // Both use Tensor core, cannot execute simultaneously
    }

    // Check CUDA core dependency (more nuanced - CUDA cores can often be
    // pipelined) For now, we treat CUDA core as a shared resource with limited
    // throughput Could be refined based on actual hardware constraints
    if (a->UsesCUDACore() && b->UsesCUDACore()) {
      // CUDA cores are more plentiful, but we still mark dependency for now
      // This could be refined to allow some level of parallelism
      return true;
    }

    // TODO: Add more specific resource dependency checks:
    // - Memory bandwidth constraints (shared between TMA and other operations)
    // - Shared memory bank conflicts
    // - Register file limitations

    return false;
  }
};

// Implementation of ScheduleRecursive function
void ScheduleUnitBuilder::ScheduleRecursive(IRStructure *node) {
  if (!node)
    return;

  if (node->IsTask()) {
    // TaskNode: no further scheduling needed
    return;
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);

    // First, recursively schedule all children
    for (size_t i = 0; i < seq->children.size(); ++i) {
      ScheduleRecursive(seq->children[i].get());
    }

    // Now collect child nodes for potential scheduling
    std::vector<IRStructure *> child_nodes;
    child_nodes.reserve(seq->children.size());
    for (const auto &child : seq->children) {
      child_nodes.push_back(child.get());
    }

    std::vector<IRStructure *> sorted_nodes;
    sorted_nodes = Z3SchedulePython(child_nodes);

    // Check if order changed
    bool order_changed = false;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
      if (sorted_nodes[i] != child_nodes[i]) {
        order_changed = true;
        break;
      }
    }

    if (order_changed) {
      // Create mapping from IRStructure pointer to child index
      std::unordered_map<IRStructure *, size_t> node_to_index;
      for (size_t i = 0; i < child_nodes.size(); ++i) {
        node_to_index[child_nodes[i]] = i;
      }

      // Reorder children according to sorted_nodes
      std::vector<std::unique_ptr<IRStructure>> reordered_children;
      reordered_children.reserve(sorted_nodes.size());

      for (IRStructure *sorted_node : sorted_nodes) {
        auto it = node_to_index.find(sorted_node);
        if (it == node_to_index.end()) {
          LOG(FATAL) << "[ScheduleRecursive] IRStructure not found in "
                        "children mapping";
        }
        size_t old_idx = it->second;
        reordered_children.push_back(std::move(seq->children[old_idx]));
      }

      // Move reordered children back
      seq->children = std::move(reordered_children);
    }
    for (auto &node : seq->children) {
      auto unit = std::make_unique<ScheduleUnit>();
      unit->stage = -1;
      unit->child = std::move(node);
      node = std::move(unit);
    }
    return;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);

    // Now schedule the ControlNode's internal tasks (if any) as a unit
    // The body should now be a SequenceNode containing the tasks
    if (ctrl->child && ctrl->child->IsSequence()) {
      auto seq_body = static_cast<SequenceNode *>(ctrl->child.get());
      for (const auto &child : seq_body->children) {
        ScheduleRecursive(child.get());
      }
      Z3SchedulePythonLoop(ctrl);
    } else {
      ScheduleRecursive(ctrl->child.get());
    }
    return;
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    if (wrapper->child) {
      ScheduleRecursive(wrapper->child.get());
      wrapper->SetII(wrapper->child->GetII());
      wrapper->SetLatency(wrapper->child->GetLatency());
    }
    return;
  }

  LOG(FATAL) << "[ScheduleRecursive] Unknown IRStructure type" << node;
}

// Mutator to update thread extent in AttrStmt nodes
// Used after warpgroup partition to double thread extent
class ThreadExtentUpdater : public StmtExprMutator {
public:
  explicit ThreadExtentUpdater(PrimExpr updated_extent)
      : updated_thread_extent_(updated_extent) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      auto thread_iv_ = Downcast<IterVar>(op->node);
      if (thread_iv_->thread_tag == "threadIdx.x") {
        // Visit the body first (to update any references)
        AttrStmt attr_stmt =
            Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));

        // Update the thread extent

        // Create new IterVar with updated domain
        Range new_dom =
            Range::FromMinExtent(thread_iv_->dom->min, updated_thread_extent_);

        // Update the AttrStmt with new IterVar and value
        thread_iv_.CopyOnWrite()->dom = new_dom;
        attr_stmt.CopyOnWrite()->node = thread_iv_;
        attr_stmt.CopyOnWrite()->value = updated_thread_extent_;

        // Clear the saved reference
        thread_iv_ = {};

        return attr_stmt;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  PrimExpr updated_thread_extent_;
  IterVar thread_iv_;
};

// Visitor to extract the body of tilelang_root block
class TilelangRootBodyExtractor : public StmtVisitor {
public:
  Stmt body;

  void VisitStmt_(const BlockNode *op) override {
    if (op->name_hint == "tilelang_root") {
      body = op->body;
      return; // Don't visit children
    }
    StmtVisitor::VisitStmt_(op);
  }
};

// Mutator to replace the body of tilelang_root block
class TilelangRootBodyReplacer : public StmtMutator {
public:
  explicit TilelangRootBodyReplacer(Stmt new_body) : new_body_(new_body) {}

  Stmt VisitStmt_(const BlockNode *op) override {
    auto block = GetRef<Block>(op);
    if (op->name_hint == "tilelang_root") {
      // Keep all block attributes but replace the body
      return Block(op->iter_vars, op->reads, op->writes, op->name_hint,
                   new_body_, op->init, op->alloc_buffers, op->match_buffers,
                   op->annotations);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  Stmt new_body_;
};

// Visitor to build IRStructure from TIR statements
class IRStructureBuilder : public StmtVisitor {
public:
  std::unique_ptr<IRStructure> Build(const Stmt &stmt) {
    VisitStmt(stmt);
    if (!root_) {
      LOG(WARNING)
          << "IRStructureBuilder: root_ is null after visiting statement. "
          << "This may indicate an unhandled statement type.";
      // Return an empty TaskNode as fallback
      auto task_node = std::make_unique<TaskNode>();
      task_node->stmts.push_back(stmt);
      return task_node;
    }
    return std::move(root_);
  }

protected:
  void VisitStmt_(const SeqStmtNode *op) override {
    auto seq_node = std::make_unique<SequenceNode>();

    for (size_t i = 0; i < op->seq.size(); i++) {
      VisitStmt(op->seq[i]);
      if (root_) {
        seq_node->children.push_back(std::move(root_));
      }
    }
    root_ = std::move(seq_node);
  }

  void VisitStmt_(const ForNode *op) override {
    // Determine if this is a sequential or parallel for
    if (op->kind == ForKind::kSerial) {
      // Sequential For -> ControlNode
      auto control_node = std::make_unique<ControlNode>();
      control_node->control = GetRef<For>(op);

      // Process the loop body
      VisitStmt(op->body);
      if (root_) {
        control_node->child = std::move(root_);
      } else {
      }

      root_ = std::move(control_node);
    } else {
      // Parallel For -> TaskNode
      auto task_node = std::make_unique<TaskNode>();
      task_node->stmts.push_back(GetRef<Stmt>(op));

      // Analyze the loop body for resource usage
      AnalyzeResourceUsage(op->body, task_node.get());

      root_ = std::move(task_node);
    }
  }

  void VisitStmt_(const EvaluateNode *op) override {
    // Evaluate statement (usually a Call) -> TaskNode
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze the expression for resource usage
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    // If statement -> treat as TaskNode for now (could be refined later)
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze both branches for resource usage
    AnalyzeResourceUsage(op->then_case, task_node.get());
    if (op->else_case) {
      AnalyzeResourceUsage(op->else_case.value(), task_node.get());
    }

    root_ = std::move(task_node);
  }

  void VisitStmt_(const LetStmtNode *op) override {
    // Wrapper statement -> WrapperNode
    auto wrapper_node = std::make_unique<WrapperNode>();
    wrapper_node->wrapper = GetRef<Stmt>(op);

    // Process the wrapperbody
    VisitStmt(op->body);
    if (root_) {
      wrapper_node->child = std::move(root_);
    }

    root_ = std::move(wrapper_node);
  }

  void VisitStmt_(const AttrStmtNode *op) override {
    // Wrapper statement -> WrapperNode
    auto wrapper_node = std::make_unique<WrapperNode>();
    wrapper_node->wrapper = GetRef<Stmt>(op);

    // Process the wrapperbody
    VisitStmt(op->body);
    if (root_) {
      wrapper_node->child = std::move(root_);
    }

    root_ = std::move(wrapper_node);
  }

  void VisitStmt_(const WhileNode *op) override {
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze condition and body for resource usage
    AnalyzeResourceUsage(Evaluate(op->condition), task_node.get());
    AnalyzeResourceUsage(op->body, task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const BlockNode *op) override {
    // All blocks are treated as TaskNode
    // Note: tilelang_root block should have been extracted by
    // TilelangRootBodyExtractor If we encounter it here, it means we're
    // processing the entire function body (not extracted), which should only
    // happen when there's no tilelang_root block
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());
    root_ = std::move(task_node);
  }

private:
  std::unique_ptr<IRStructure> root_;

  void AnalyzeResourceUsage(const Stmt &stmt, TaskNode *task_node) {
    // Recursively analyze statements to determine resource usage
    struct ResourceAnalyzer : public StmtExprVisitor {
      TaskNode *task_node;
      bool found_tma{false};
      bool found_tensor{false};
      bool found_cuda{false};

      // Tensor Core shape information (multiple shapes possible)
      struct TensorCoreShape {
        int64_t m;
        int64_t n;
        int64_t k;
        TensorCoreShape(int64_t m = 0, int64_t n = 0, int64_t k = 0)
            : m(m), n(n), k(k) {}
      };
      std::vector<TensorCoreShape> tensor_core_shapes;

      ResourceAnalyzer(TaskNode *node) : task_node(node) {}

      void VisitExpr_(const CallNode *op) override {
        // Check for specific TileLang operations
        static const auto copy_op = Op::Get("tl.tileop.copy");
        static const auto gemm_py_op = Op::Get("tl.tileop.gemm_py");
        static const auto gemm_op = Op::Get("tl.tileop.gemm");
        static const auto reduce_op = Op::Get("tl.tileop.reduce");
        static const auto fill_op = Op::Get("tl.tileop.fill");
        static const auto region_op = Op::Get("tl.tileop.region");

        // Try to get operation name for logging
        std::string op_name = "unknown";
        if (const auto *op_ptr = op->op.as<OpNode>()) {
          op_name = op_ptr->name;
        }

        // Check if this is a TMA copy operation
        if (op->op.same_as(copy_op)) {
          bool found_global = false;
          for (unsigned idx = 0; idx != 2; ++idx) {
            auto region = Downcast<Call>(op->args[0]);
            if (const auto *buffer_load =
                    region->args[0].as<BufferLoadNode>()) {
              Buffer buffer = buffer_load->buffer;
              String scope = buffer.scope();
              MemoryType mem_type = GetMemoryTypeFromScope(scope);
              if (mem_type == MemoryType::kGlobal) {
                found_global = true;
              }
            }
          }
          found_tma |= found_global;
        } else if (op->op.same_as(gemm_py_op) || op->op.same_as(gemm_op)) {
          found_tensor = true;

          // Try to extract Tensor Core shape information from gemm arguments
          // gemm arguments: A, B, C, transpose_A, transpose_B, policy,
          // clear_accum, k_pack, wg_wait, mbar We need to extract shape from A,
          // B, C buffers
          if (op->args.size() >= 3) {
            // Extract buffer regions from arguments
            auto try_extract_shape = [&](const PrimExpr &arg, int64_t &m,
                                         int64_t &n) -> bool {
              // Check if argument is a BufferLoad
              if (const auto *buffer_load = arg.as<BufferLoadNode>()) {
                Buffer buffer = buffer_load->buffer;
                // Get buffer shape
                if (buffer->shape.size() >= 2) {
                  // Try to get constant dimensions
                  if (const auto *m_int =
                          buffer->shape[buffer->shape.size() - 2]
                              .as<IntImmNode>()) {
                    if (const auto *n_int =
                            buffer->shape[buffer->shape.size() - 1]
                                .as<IntImmNode>()) {
                      m = m_int->value;
                      n = n_int->value;
                      return true;
                    }
                  }
                }
              }
              // Check if argument is a BufferRegion
              else if (const auto *buffer_region = arg.as<BufferRegionNode>()) {
                Buffer buffer = buffer_region->buffer;
                // Get buffer shape
                if (buffer->shape.size() >= 2) {
                  // Try to get constant dimensions
                  if (const auto *m_int =
                          buffer->shape[buffer->shape.size() - 2]
                              .as<IntImmNode>()) {
                    if (const auto *n_int =
                            buffer->shape[buffer->shape.size() - 1]
                                .as<IntImmNode>()) {
                      m = m_int->value;
                      n = n_int->value;
                      return true;
                    }
                  }
                }
              }
              return false;
            };

            // Extract shapes from A, B, C arguments
            int64_t a_m = 0, a_n = 0;
            int64_t b_m = 0, b_n = 0;
            int64_t c_m = 0, c_n = 0;

            if (try_extract_shape(op->args[0], a_m, a_n) &&
                try_extract_shape(op->args[1], b_m, b_n) &&
                try_extract_shape(op->args[2], c_m, c_n)) {

              // For gemm: C[M, N] = A[M, K] * B[K, N]
              // C shape gives us M and N
              int64_t m = c_m;
              int64_t n = c_n;
              int64_t k = a_n; // A's last dimension (default: no transpose)

              // Extract transpose flags from gemm arguments
              // Arguments after C: transpose_A (arg[3]), transpose_B (arg[4])
              if (op->args.size() > 3) {
                if (const auto *transpose_a = op->args[3].as<IntImmNode>()) {
                  if (transpose_a->value != 0) {
                    // A is transposed: A[K, M]
                    k = a_m;
                  }
                }
              }
              if (op->args.size() > 4) {
                if (const auto *transpose_b = op->args[4].as<IntImmNode>()) {
                  if (transpose_b->value != 0) {
                    // B is transposed: B[N, K]
                    // Check consistency with A's K dimension
                    if (k == 0) {
                      k = b_n;
                    } else if (k != b_n) {
                      // Inconsistent K dimensions, use minimum
                      k = std::min(k, b_n);
                    }
                  }
                }
              }

              // If K is still 0, try to infer from B's first dimension
              if (k == 0) {
                k = b_m;
              }

              // Add this Tensor Core shape to the vector
              if (m > 0 && n > 0 && k > 0) {
                tensor_core_shapes.emplace_back(m, n, k);
              }
            }
          }
        } else if (op->op.same_as(reduce_op) || op->op.same_as(fill_op)) {
          // Reduce and fill operations use CUDA core
          found_cuda = true;
        } else if (op->op.same_as(region_op)) {
          // Handle tl.tileop.region call for memory access analysis
          // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2:
          // write, 3: read/write) args[2..] = extents
          if (op->args.size() >= 2) {
            // Extract access type
            if (const auto *access_int = op->args[1].as<IntImmNode>()) {
              int access_type = access_int->value;
              // For now, just mark as CUDA operation (memory access)
              found_cuda = true;
              // TODO: Extract buffer region and add to task_node->read_regions
              // or write_regions BufferLoad buffer_load =
              // Downcast<BufferLoad>(op->args[0]); Construct BufferRegion from
              // buffer_load and extents if (access_type == 1 || access_type ==
              // 3) task_node->read_regions.push_back(region); if (access_type
              // == 2 || access_type == 3)
              // task_node->write_regions.push_back(region);
            }
          }
        } else {
          // Check for other known operations that use CUDA core
          // For now, assume any other call is a basic computation
          found_cuda = true;
        }

        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const AddNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const SubNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const MulNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const DivNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }
    };

    ResourceAnalyzer analyzer(task_node);
    analyzer(stmt);

    // Set task node flags based on what was found
    if (analyzer.found_tma) {
      task_node->SetUsesTMACore(true);
    }
    if (analyzer.found_tensor) {
      task_node->SetUsesTensorCore(true);
      // Set Tensor Core shape information if available
      for (const auto &shape : analyzer.tensor_core_shapes) {
        if (shape.m > 0 && shape.n > 0 && shape.k > 0) {
          task_node->AddTensorCoreShape(shape.m, shape.n, shape.k);
        }
      }
    }
    // If neither TMA nor Tensor core was used, and CUDA operations were found,
    // set CUDA core flag
    if (!analyzer.found_tma && !analyzer.found_tensor) {
      task_node->SetUsesCUDACore(true);
    }

    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(stmt);
    std::vector<BufferRegion> read_regions = memory_detector.GetReadRegions();
    std::vector<BufferRegion> write_regions = memory_detector.GetWriteRegions();

    // Merge with existing regions (avoid duplicates)
    for (const auto &region : read_regions) {
      task_node->AddReadRegion(region);
    }

    for (const auto &region : write_regions) {
      task_node->AddWriteRegion(region);
    }

    // Estimate latency and initiation interval for this task
    LatencyEstimator latency_estimator;
    latency_estimator.Estimate(task_node);
  }
};

// The main pass function
tvm::transform::Pass AutoSchedule(const bool enable_epi) {
  using namespace tir::transform;
  auto pass_func =
      [enable_epi](PrimFunc func, const IRModule &mod,
                   const tvm::transform::PassContext &ctx) -> PrimFunc {
    // Extract the body of tilelang_root block if it exists
    TilelangRootBodyExtractor extractor;
    extractor(func->body);
    Stmt body_to_schedule;
    bool has_tilelang_root = false;
    PrimExpr updated_thread_extent; // Will be set if warpgroup partition
                                    // doubles thread extent
    IterVar thread_var; // Thread index variable for warpgroup partition

    if (extractor.body.defined()) {
      body_to_schedule = extractor.body;
      has_tilelang_root = true;
    } else {
      LOG(FATAL);
      body_to_schedule = func->body;
    }

    // Build IRStructure from the body to schedule
    IRStructureBuilder builder;
    auto ir_structure = builder.Build(body_to_schedule);

    // Print the built IRStructure with all statements
    ICHECK(ir_structure) << "IRStructure is null (empty body?)";

    // First print the summary view
    // PrintIRStructure(ir_structure.get());

    // Then print all statements
    // PrintAllStmts(ir_structure.get());

    // Build ScheduleUnits from IRStructure
    ScheduleUnitBuilder unit_builder;
    // Get thread index variable for warpgroup partition
    // First try to get from body_to_schedule, if not found, try from the entire
    // function body
    thread_var = ThreadTagChecker::GetThreadVar(body_to_schedule);
    if (!thread_var.defined()) {
      thread_var = ThreadTagChecker::GetThreadVar(func->body);
    }
    if (thread_var.defined()) {
      unit_builder.SetThreadVar(thread_var);
    } else {
      LOG(FATAL) << "Could not find thread index variable, warpgroup "
                    "partition will use default";
    }
    bool double_thread = unit_builder.Build(ir_structure.get());

    // Print the modified summary view
    // PrintIRStructure(ir_structure.get());

    // Analyze buffer dependencies and insert barriers before warpgroup
    // partition
    // Create barrier manager
    BarrierManager barrier_manager;
    // Determine thread count for barrier arrive_count calculations
    PrimExpr thread_count[2] = {thread_var->dom->extent,
                                double_thread ? thread_var->dom->extent
                                              : IntImm(DataType::Int(32), 128)};
    LoopNestingInfo loop_info;
    std::vector<MultiVersionBufferInfo> buffer_infos;
    AnalyzeAndInsertBarriers(ir_structure.get(), barrier_manager, thread_count,
                             loop_info, buffer_infos);

    // Print the modified summary view
    // PrintIRStructure(ir_structure.get());

    // Apply warpgroup partition to entire IRStructure
    Stmt new_body = ApplyWarpgroupPartitionToIRStructure(
        ir_structure.get(), thread_var, barrier_manager, enable_epi,
        thread_count);

    if (double_thread) {
      updated_thread_extent = thread_var->dom->extent * 2;
    } else {
      updated_thread_extent =
          thread_var->dom->extent + IntImm(DataType::Int(32), 128);
    }

    // If we extracted from tilelang_root block, replace the body
    Stmt final_body;
    TilelangRootBodyReplacer replacer(new_body);
    final_body = replacer(func->body);
    // Apply thread extent update if warpgroup partition was applied
    ThreadExtentUpdater extent_updater(updated_thread_extent);
    final_body = extent_updater(final_body);
    // Add create_list_of_mbarrier statement and barrier annotations
    if (barrier_manager.HasBarriers()) {
      // First, insert create_list_of_mbarrier statement at the beginning of the
      // body
      Stmt create_mbarrier_stmt = CreateListOfMBarrierStmt(barrier_manager);
      // Get barrier map for annotations
      Map<ObjectRef, ObjectRef> barrier_map =
          BarrierManagerToMap(barrier_manager);
      // Create a new mutator to add the statement and annotations
      class BarrierInserter : public StmtMutator {
      public:
        BarrierInserter(Stmt barrier_stmt) : barrier_stmt_(barrier_stmt) {}

        Stmt VisitStmt_(const BlockNode *op) override {
          auto block = GetRef<Block>(op);
          if (op->name_hint == "tilelang_root") {
            // Insert barrier statement at the beginning of the block body
            Stmt new_body = SeqStmt({barrier_stmt_, op->body});
            return Block(op->iter_vars, op->reads, op->writes, op->name_hint,
                         new_body, op->init, op->alloc_buffers,
                         op->match_buffers, op->annotations);
          }
          return StmtMutator::VisitStmt_(op);
        }

      private:
        Stmt barrier_stmt_;
      };

      BarrierInserter inserter(create_mbarrier_stmt);
      final_body = inserter(final_body);
    }

    // Apply multi-version alloc_buffer rewrite if needed
    if (!buffer_infos.empty()) {
      final_body = RewriteAllocBuffers(final_body, buffer_infos);
    }

    // Create a new PrimFunc with the updated body
    auto new_func = PrimFunc(func->params, final_body, func->ret_type,
                             func->buffer_map, func->attrs);
    return new_func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

// Helper function to clone IRStructure with warpgroup filter
std::unique_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id) {
  if (!node || !node->containWarpgroupId(warpgroup_id))
    return nullptr;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);
    return task->Clone();
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    auto new_seq = std::make_unique<SequenceNode>();
    for (const auto &child : seq->children) {
      auto new_child =
          CloneIRStructureWithWarpgroupFilter(child.get(), warpgroup_id);
      if (new_child) {
        new_seq->children.push_back(std::move(new_child));
      }
    }
    return new_seq;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);
    auto new_ctrl = std::make_unique<ControlNode>();
    new_ctrl->control = ctrl->control;
    new_ctrl->SetPromote(ctrl->hasPromote());
    new_ctrl->child =
        CloneIRStructureWithWarpgroupFilter(ctrl->child.get(), warpgroup_id);
    return new_ctrl;
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    auto new_wrapper = std::make_unique<WrapperNode>();
    new_wrapper->wrapper = wrapper->wrapper;
    new_wrapper->child =
        CloneIRStructureWithWarpgroupFilter(wrapper->child.get(), warpgroup_id);
    return new_wrapper;
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    auto new_unit = std::make_unique<ScheduleUnit>();
    new_unit->before[warpgroup_id] = unit->before[warpgroup_id];
    new_unit->after[warpgroup_id] = unit->after[warpgroup_id];
    new_unit->stage = unit->stage;
    new_unit->child =
        CloneIRStructureWithWarpgroupFilter(unit->child.get(), warpgroup_id);
    return new_unit;
  }
  LOG(FATAL);
}

// Apply warpgroup partition to entire IRStructure (top-level IfThenElse)
Stmt ApplyWarpgroupPartitionToIRStructure(IRStructure *root, IterVar thread_var,
                                          BarrierManager &barrier_manager,
                                          const bool outer_enable_epi,
                                          PrimExpr thread_count[2]) {
  if (!root)
    return Evaluate(0);

  if (root->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(root);
    Stmt body = Evaluate(0);
    if (wrapper->child) {
      body = ApplyWarpgroupPartitionToIRStructure(
          wrapper->child.get(), thread_var, barrier_manager, outer_enable_epi,
          thread_count);
    }
    if (const auto *let = wrapper->wrapper.as<LetStmtNode>()) {
      return LetStmt(let->var, let->value, body);
    } else if (const auto *attr = wrapper->wrapper.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    } else {
      LOG(FATAL);
    }
  }

  // Check if there are tasks with mixed warpgroup ids
  std::vector<TaskNodeWithContext> all_tasks;
  CollectAllTaskNodesWithContext(root, all_tasks);

  bool has_warpgroup0 = false;
  bool has_warpgroup1 = false;
  bool has_warpgroup_neutral = false;
  for (auto &task : all_tasks) {
    int wg_id = task.task->GetWarpgroupId();
    if (wg_id == 0)
      has_warpgroup0 = true;
    else if (wg_id == 1)
      has_warpgroup1 = true;
    else if (wg_id == -1)
      has_warpgroup_neutral = true;
  }

  // Convert IRStructure to Stmt for IfThenElse
  std::function<Stmt(IRStructure *)> irstructure_to_stmt;
  irstructure_to_stmt = [&irstructure_to_stmt,
                         outer_enable_epi](IRStructure *structure) -> Stmt {
    if (!structure) {
      return Evaluate(0);
    }

    if (structure->IsTask()) {
      auto task = static_cast<TaskNode *>(structure);
      if (task->stmts.empty()) {
        return Evaluate(0);
      } else if (task->stmts.size() == 1) {
        return task->stmts[0];
      } else {
        return SeqStmt(task->stmts);
      }
    } else if (structure->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(structure);
      std::vector<Stmt> stmts;
      for (const auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        for (auto &before : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        Stmt child_stmt = irstructure_to_stmt(unit->child.get());
        stmts.push_back(child_stmt);
        for (auto &after : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
      }
      auto flattened = SeqStmt::Flatten(stmts);
      return flattened;
    } else if (structure->IsControl()) {
      auto ctrl = static_cast<ControlNode *>(structure);
      Var loop_var = ctrl->control->loop_var;
      PrimExpr loop_start = ctrl->control->min;
      PrimExpr loop_extent = ctrl->control->extent;
      PrimExpr loop_step = ctrl->control->step.has_value()
                               ? ctrl->control->step.value()
                               : IntImm(DataType::Int(32), 1);
      int min_stages = 100, max_stages = -1;
      if (ctrl->child->IsSequence()) {
        auto seq = static_cast<SequenceNode *>(ctrl->child.get());
        for (auto &child : seq->children) {
          auto unit = static_cast<ScheduleUnit *>(child.get());
          min_stages = std::min(min_stages, unit->stage);
          max_stages = std::max(max_stages, unit->stage);
        }
      }
      if (!ctrl->hasPromote() || !ctrl->child->IsSequence() ||
          min_stages == max_stages) {
        std::vector<Stmt> stmts;
        if (ctrl->child->IsScheduleUnit()) {
          auto unit = static_cast<ScheduleUnit *>(ctrl->child.get());
          for (auto &before : unit->before) {
            for (auto &stmt : before) {
              stmts.push_back(stmt);
            }
          }
          stmts.push_back(irstructure_to_stmt(unit->child.get()));
          for (auto &after : unit->after) {
            for (auto &stmt : after) {
              stmts.push_back(stmt);
            }
          }
        } else if (ctrl->child->IsSequence()) {
          auto seq = static_cast<SequenceNode *>(ctrl->child.get());
          for (auto &child : seq->children) {
            ICHECK(child->IsScheduleUnit());
            auto unit = static_cast<ScheduleUnit *>(child.get());
            for (auto &before : unit->before) {
              for (auto &stmt : before) {
                stmts.push_back(stmt);
              }
            }
            stmts.push_back(irstructure_to_stmt(unit->child.get()));
            for (auto &after : unit->after) {
              for (auto &stmt : after) {
                stmts.push_back(stmt);
              }
            }
          }
        } else {
          LOG(FATAL);
        }
        Stmt body = SeqStmt::Flatten(stmts);
        // Filter out "num_stages" annotation
        Map<String, Any> filtered_annotations = ctrl->control->annotations;
        filtered_annotations.erase("num_stages");
        return For(loop_var, loop_start, loop_extent, ctrl->control->kind, body,
                   ctrl->control->thread_binding, filtered_annotations);
      }
      auto seq = static_cast<SequenceNode *>(ctrl->child.get());
      Stmt body = Evaluate(0);
      std::vector<std::vector<Stmt>> unit_stages;
      unit_stages.resize(max_stages - min_stages + 1);
      for (auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        std::vector<Stmt> stmts;
        for (auto &before : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        stmts.push_back(irstructure_to_stmt(unit->child.get()));
        for (auto &after : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
        unit_stages[unit->stage - min_stages].push_back(
            SeqStmt::Flatten(stmts));
      }
      // Check if any task in this control node contains loop_break
      // If any task contains loop_break, disable prologue
      std::function<bool(IRStructure *)> check_contains_loop_break;
      check_contains_loop_break =
          [&check_contains_loop_break](IRStructure *structure) -> bool {
        if (!structure)
          return false;

        if (structure->IsTask()) {
          auto task = static_cast<TaskNode *>(structure);
          return task->ContainsLoopBreak();
        } else if (structure->IsSequence()) {
          auto seq = static_cast<SequenceNode *>(structure);
          for (const auto &child : seq->children) {
            auto unit = static_cast<ScheduleUnit *>(child.get());
            if (check_contains_loop_break(unit->child.get())) {
              return true;
            }
          }
          return false;
        } else if (structure->IsScheduleUnit()) {
          auto unit = static_cast<ScheduleUnit *>(structure);
          return check_contains_loop_break(unit->child.get());
        } else if (structure->IsControl()) {
          auto ctrl = static_cast<ControlNode *>(structure);
          return check_contains_loop_break(ctrl->child.get());
        } else if (structure->IsWrapper()) {
          auto wrapper = static_cast<WrapperNode *>(structure);
          return check_contains_loop_break(wrapper->child.get());
        }
        return false;
      };

      // Set enable_pro to true only if no task contains loop_break
      bool enable_pro = !check_contains_loop_break(ctrl->child.get());
      bool enable_epi = outer_enable_epi && enable_pro;
      std::vector<Stmt> steady;

      for (auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        std::vector<Stmt> stmts;
        for (auto &before : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        stmts.push_back(irstructure_to_stmt(unit->child.get()));
        for (auto &after : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
        Map<Var, PrimExpr> substitution;
        PrimExpr condition =
            And(loop_var < loop_extent, loop_var >= loop_start);
        if (unit->stage == min_stages) {
          condition = loop_var >= loop_start;
        }
        if (unit->stage == max_stages) {
          condition = loop_var < loop_extent;
        }
        Stmt stmt = IfThenElse(condition, SeqStmt::Flatten(stmts));
        substitution.Set(loop_var,
                         loop_var - loop_step * (max_stages - unit->stage));
        steady.push_back(Substitute(stmt, substitution));
      }
      Stmt new_body = SeqStmt::Flatten(steady);
      auto new_var = loop_var.copy_with_suffix("");
      // Filter out "num_stages" annotation
      Map<String, Any> filtered_annotations = ctrl->control->annotations;
      filtered_annotations.erase("num_stages");
      Map<Var, PrimExpr> substitution;
      substitution.Set(loop_var, new_var);
      For for_op =
          For(new_var, loop_start,
              ctrl->control->extent + loop_step * (max_stages - min_stages),
              ctrl->control->kind, Substitute(new_body, substitution),
              ctrl->control->thread_binding, filtered_annotations);

      Stmt prologue = Evaluate(0);
      if (enable_pro) {
        Map<Var, PrimExpr> sub;
        For new_for = for_op;
        auto pro = loop_var.copy_with_suffix("_prologue");
        sub.Set(new_var, pro);
        new_for.CopyOnWrite()->loop_var = pro;
        new_for.CopyOnWrite()->kind = ForKind::kUnrolled;
        new_for.CopyOnWrite()->extent =
            min(max_stages - min_stages, for_op.get()->extent);
        for_op.CopyOnWrite()->min += loop_step * (max_stages - min_stages);
        for_op.CopyOnWrite()->extent =
            max(0, for_op.get()->extent - (max_stages - min_stages));
        prologue = Substitute(new_for, sub);
      }
      Stmt epilogue = Evaluate(0);
      if (enable_epi) {
        Map<Var, PrimExpr> sub;
        For new_for = for_op;
        auto epi = loop_var.copy_with_suffix("_epilogue");
        sub.Set(new_var, epi);
        new_for.CopyOnWrite()->loop_var = epi;
        new_for.CopyOnWrite()->kind = ForKind::kUnrolled;
        new_for.CopyOnWrite()->min =
            for_op.get()->min +
            loop_step * (for_op.get()->extent - (max_stages - min_stages));
        new_for.CopyOnWrite()->extent =
            min(max_stages - min_stages, for_op.get()->extent);
        for_op.CopyOnWrite()->extent =
            max(0, for_op.get()->extent - (max_stages - min_stages));
        epilogue = Substitute(new_for, sub);
      }
      return SeqStmt({prologue, for_op, epilogue});
    } else if (structure->IsWrapper()) {
      auto wrapper = static_cast<const WrapperNode *>(structure);
      Stmt body = Evaluate(0);
      if (wrapper->child) {
        body = irstructure_to_stmt(wrapper->child.get());
      }
      if (const auto *let = wrapper->wrapper.as<LetStmtNode>()) {
        return LetStmt(let->var, let->value, body);
      } else if (const auto *attr = wrapper->wrapper.as<AttrStmtNode>()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value, body);
      } else {
        LOG(FATAL);
      }
    }

    LOG(FATAL)
        << "Failed to convert IRStructure to Stmt, returning empty statement";
    return Evaluate(0);
  };

  // If all tasks belong to the same warpgroup, no partition needed
  if (!(has_warpgroup0 && has_warpgroup1)) {
    return irstructure_to_stmt(root);
  }

  // Helper function to clone IRStructure filtering tasks with warpgroup_id ==
  // -1 (neutral tasks)
  std::function<std::unique_ptr<IRStructure>(IRStructure *)>
      clone_neutral_filter;
  clone_neutral_filter =
      [&clone_neutral_filter](
          IRStructure *node) -> std::unique_ptr<IRStructure> {
    if (!node)
      return nullptr;

    if (node->IsTask()) {
      auto task = static_cast<TaskNode *>(node);
      if (task->GetWarpgroupId() == -1) {
        return task->Clone();
      } else {
        auto new_task = std::make_unique<TaskNode>();
        // Empty statements
        return new_task;
      }
    } else if (node->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(node);
      auto new_seq = std::make_unique<SequenceNode>();
      for (const auto &child : seq->children) {
        if (child) {
          auto node = static_cast<ScheduleUnit *>(child.get());
          auto new_node = clone_neutral_filter(node->child.get());
          if (new_node) {
            auto new_unit = std::make_unique<ScheduleUnit>();
            new_unit->child = std::move(new_node);
            new_seq->children.push_back(std::move(new_unit));
          }
        }
      }
      return new_seq;
    } else if (node->IsControl() || node->IsWrapper()) {
      return nullptr;
    }
    LOG(FATAL);
  };

  // Clone IRStructure for warpgroup neutral, 0 and 1
  auto wg_neutral_structure =
      has_warpgroup_neutral ? clone_neutral_filter(root) : nullptr;
  auto wg0_structure = CloneIRStructureWithWarpgroupFilter(root, 0);
  auto wg1_structure = CloneIRStructureWithWarpgroupFilter(root, 1);

  // Check if both clones have actual statements
  auto has_actual_statements = [](IRStructure *node) -> bool {
    std::vector<TaskNodeWithContext> tasks;
    CollectAllTaskNodesWithContext(node, tasks);
    for (auto &task : tasks) {
      if (!task.task->stmts.empty()) {
        return true;
      }
    }
    return false;
  };

  bool wg_neutral_has_stmts =
      wg_neutral_structure ? has_actual_statements(wg_neutral_structure.get())
                           : false;
  bool wg0_has_stmts = has_actual_statements(wg0_structure.get());
  bool wg1_has_stmts = has_actual_statements(wg1_structure.get());

  PrimExpr condition = thread_var->var < thread_count[0];

  Stmt neutral_body = wg_neutral_has_stmts
                          ? irstructure_to_stmt(wg_neutral_structure.get())
                          : Evaluate(0);
  Stmt then_body =
      wg0_has_stmts ? irstructure_to_stmt(wg0_structure.get()) : Evaluate(0);
  Stmt else_body =
      wg1_has_stmts ? irstructure_to_stmt(wg1_structure.get()) : Evaluate(0);

  // Create IfThenElse statement with barrier synchronization if both warpgroups
  // have statements
  Stmt if_then_else;
  if (wg0_has_stmts && wg1_has_stmts) {
    // Both warpgroups exist: insert barriers for cross-warpgroup
    // synchronization
    if_then_else = IfThenElse(condition, then_body, else_body);
  } else if (wg0_has_stmts) {
    // Only warpgroup 0 has statements, execute unconditionally
    if_then_else = then_body;
  } else if (wg1_has_stmts) {
    // Only warpgroup 1 has statements, execute unconditionally
    if_then_else = else_body;
  } else {
    // Neither warpgroup 0 nor 1 has statements
    if_then_else = Evaluate(0);
  }

  // Combine neutral tasks (warpgroup -1) with the if-then-else statement
  // Add barrier synchronization between neutral tasks and warpgroup-specific
  // work
  Stmt combined_stmt;
  if (wg_neutral_has_stmts) {
    if (!IsEvaluateZero(if_then_else) && !IsEvaluateZero(neutral_body)) {
      // Both have statements: insert barriers for neutral-to-warpgroup
      // synchronization
      combined_stmt = InsertBarriersForNeutralSync(
          neutral_body, if_then_else, barrier_manager, thread_count);
    } else if (!IsEvaluateZero(if_then_else) || !IsEvaluateZero(neutral_body)) {
      // Only one has actual statements
      std::vector<Stmt> stmts;
      if (!IsEvaluateZero(neutral_body)) {
        stmts.push_back(neutral_body);
      }
      if (!IsEvaluateZero(if_then_else)) {
        stmts.push_back(if_then_else);
      }
      if (stmts.size() == 1) {
        combined_stmt = stmts[0];
      } else {
        combined_stmt = SeqStmt(stmts);
      }
    } else {
      // Both are empty
      combined_stmt = Evaluate(0);
    }
  } else {
    combined_stmt = if_then_else;
  }

  return combined_stmt;
}

// StmtMutator to rewrite alloc_buffers in Block nodes
class AllocBufferRewriter : public StmtMutator {
public:
  AllocBufferRewriter(const std::vector<MultiVersionBufferInfo> &buffer_infos)
      : buffer_infos_(buffer_infos) {
    // Create mapping from original buffer to new buffer
    for (const auto &info : buffer_infos_) {
      buffer_remap_[info.buffer] = info.new_buffer;
    }
  }

private:
  Stmt VisitStmt_(const BlockNode *op) override {
    Stmt new_body = this->VisitStmt(op->body);

    // Check if we need to update alloc_buffers
    bool needs_update = false;
    Array<Buffer> new_alloc_buffers;

    for (auto buffer : op->alloc_buffers) {
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        new_alloc_buffers.push_back(it->second);
        needs_update = true;
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }

    auto new_block = CopyOnWrite(op);
    new_block->body = new_body;
    if (needs_update) {
      new_block->alloc_buffers = new_alloc_buffers;
    }
    return Stmt(new_block);
  }

  const std::vector<MultiVersionBufferInfo> &buffer_infos_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
};

// Main function to rewrite alloc_buffers
Stmt RewriteAllocBuffers(
    const Stmt &stmt, const std::vector<MultiVersionBufferInfo> &buffer_infos) {
  if (buffer_infos.empty()) {
    return stmt;
  }

  AllocBufferRewriter rewriter(buffer_infos);
  return rewriter(stmt);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm
