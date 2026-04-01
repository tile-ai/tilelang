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
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>
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
#include "../op/gemm_py.h"
#include "../target/utils.h"
#include "./common/attr.h"
#include "./common/collector.h"
#include "auto_schedule.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

// Extract all sequencial task nodes from the IR structure tree
void GatherTaskNodesSingle(
    const std::shared_ptr<IRStructure> &node,
    std::vector<std::shared_ptr<IRStructure>> &task_nodes);
void GatherTaskNodes(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                     std::vector<std::shared_ptr<IRStructure>> &task_nodes) {
  for (const auto &node : nodes) {
    if (node->IsTask()) {
      task_nodes.emplace_back(node);
    } else if (node->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(node.get());
      GatherTaskNodes(seq->children, task_nodes);
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<WrapperNode *>(node.get());
      if (wrapper->task)
        task_nodes.emplace_back(wrapper->task);
      if (wrapper->child)
        GatherTaskNodesSingle(wrapper->child, task_nodes);
    } else if (node->IsControl()) {
      task_nodes.emplace_back(node);
    } else {
      LOG(FATAL) << "Unknown node type in GatherTaskNodes";
    }
  }
}

void GatherTaskNodesSingle(
    const std::shared_ptr<IRStructure> &node,
    std::vector<std::shared_ptr<IRStructure>> &task_nodes) {
  return GatherTaskNodes({node}, task_nodes);
}

// Forward declaration
Stmt ApplyWarpgroupPartitionToIRStructure(
    IRStructure *root, IterVar thread_var, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, const bool enable_epi,
    PrimExpr thread_count[2], bool producer_consumer,
    const WarpSpecializeConfig &config, Buffer neutral_sync_shared_barrier);

Stmt ConvertIRStructureToStmt(IRStructure *root, const bool outer_enable_epi);

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

void CollectSuffixTaskCandidates(IRStructure *node,
                                 std::vector<TaskNode *> &suffix_candidates,
                                 bool &suffix_valid) {
  if (!node)
    return;

  if (!suffix_valid)
    return;

  if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (auto it = seq->children.rbegin(); it != seq->children.rend(); ++it) {
      CollectSuffixTaskCandidates(it->get(), suffix_candidates, suffix_valid);
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);
    suffix_valid = false;
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    CollectSuffixTaskCandidates(wrapper->child.get(), suffix_candidates,
                                suffix_valid);
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    CollectSuffixTaskCandidates(unit->child.get(), suffix_candidates,
                                suffix_valid);
  } else if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);
    suffix_candidates.push_back(task);
  } else {
    LOG(FATAL);
  }
}

void CollectSuffixTaskCandidates(IRStructure *node,
                                 std::vector<TaskNode *> &suffix_candidates) {
  bool suffix_valid = true;
  CollectSuffixTaskCandidates(node, suffix_candidates, suffix_valid);
}

void CollectSuffixTasks(IRStructure *root,
                        const std::vector<TaskNodeWithContext> &all_tasks,
                        const TaskUnionFind &uf,
                        std::unordered_set<TaskNode *> &suffix_tasks) {
  std::vector<TaskNode *> suffix_candidates;
  CollectSuffixTaskCandidates(root, suffix_candidates);

  std::unordered_set<TaskNode *> candidate_set(suffix_candidates.begin(),
                                               suffix_candidates.end());

  std::unordered_map<TaskNode *, int> task_to_index;
  task_to_index.reserve(all_tasks.size());
  for (int i = 0; i < static_cast<int>(all_tasks.size()); ++i) {
    task_to_index[all_tasks[i].task] = i;
  }

  std::unordered_map<int, std::vector<TaskNode *>> component_tasks;
  component_tasks.reserve(all_tasks.size());
  for (int i = 0; i < static_cast<int>(all_tasks.size()); ++i) {
    int root_idx = uf.find(i);
    component_tasks[root_idx].push_back(all_tasks[i].task);
  }

  for (int i = 0; i < static_cast<int>(suffix_candidates.size()); ++i) {
    TaskNode *task = suffix_candidates[i];
    if (suffix_tasks.count(task)) {
      continue; // Already added via component
    }
    if (CountRegisterRegions(task) == 0) {
      suffix_tasks.insert(task);
      continue;
    }

    auto it = task_to_index.find(task);
    if (it == task_to_index.end()) {
      continue;
    }
    int component_root = uf.find(it->second);

    auto comp_it = component_tasks.find(component_root);
    if (comp_it == component_tasks.end()) {
      continue;
    }

    bool all_in_candidates = true;
    for (TaskNode *component_task : comp_it->second) {
      if (candidate_set.find(component_task) == candidate_set.end()) {
        all_in_candidates = false;
        break;
      }
    }

    if (!all_in_candidates) {
      break;
    }

    for (TaskNode *component_task : comp_it->second) {
      suffix_tasks.insert(component_task);
    }
  }
}

// Global warpgroup id assignment - should be called from the top level
// Tasks that use the same register region must have the same warpgroup id
// Goal: balance weighted latency between two warpgroups (0 and 1)
// Weighted latency = latency * tripcount (tripcount = 100 for non-constant loop
// extent)
bool AssignWarpgroupIdsGlobal(IRStructure *root, bool enable_warp_partition) {
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

  // Collect all prefix tasks (consecutive tasks without register region at the
  // beginning of sequences)
  std::unordered_set<TaskNode *> prefix_tasks;
  bool prefix_valid = true;
  CollectPrefixTasks(root, prefix_tasks, prefix_valid);

  std::unordered_set<TaskNode *> suffix_tasks;
  if (enable_warp_partition) {
    CollectSuffixTasks(root, all_tasks, uf, suffix_tasks);
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
      if (suffix_tasks.find(task) != suffix_tasks.end()) {
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
  if ((double)max_latency < 1.1 * min_latency) {
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
    ScheduleRecursive(root, {});

    // Global warpgroup id assignment from the top level
    return AssignWarpgroupIdsGlobal(root, enable_warp_partition_);
  }

  // New recursive scheduling function that replaces Collect method
  // Directly schedules the entire IRStructure tree recursively in place
  void ScheduleRecursive(IRStructure *node,
                         const std::set<Buffer> &used_buffers);

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
  void Z3SchedulePythonLoop(ControlNode *ctrl,
                            const std::set<Buffer> &used_buffers) {
    if (ctrl->child == nullptr) {
      LOG(WARNING)
          << "Z3SchedulePythonLoop called on a control node without child";
      return;
    }
    std::vector<std::shared_ptr<IRStructure>> flat_children;
    if (!ctrl->child->IsSequence()) {
      GatherTaskNodesSingle(ctrl->child, flat_children);
    } else {
      auto seq_node = static_cast<SequenceNode *>(ctrl->child.get());
      GatherTaskNodes(seq_node->children, flat_children);
    }
    auto seq_node = std::make_shared<SequenceNode>();
    seq_node->children = flat_children;
    ctrl->child = std::move(seq_node);

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

    // Collect all shared buffers
    // The negative number means we can use multi-buffering for this buffer, so
    // we need to create a variable for the number of versions for this buffer
    // in z3 scheduler.
    std::vector<int64_t> buffer_sizes;
    std::map<Buffer, int64_t> buffer_to_num_versions;
    int64_t memory_limit = shared_memory_limit_;
    for (const auto &region_access : ctrl->GetReadWriteRegions()) {
      const auto &buffer = region_access.region->buffer;
      if (!IsSharedBuffer(buffer)) {
        continue; // Only consider shared buffers for multi-buffer
      }
      if (buffer_to_num_versions.count(buffer)) {
        continue;
      }
      if (used_buffers.count(buffer)) {
        buffer_to_num_versions[buffer] = 1;
        memory_limit -= GetBufferSize(buffer);
      } else {
        buffer_sizes.push_back(GetBufferSize(buffer));
        buffer_to_num_versions[buffer] = -(int64_t)buffer_sizes.size();
      }
    }

    // Collect data dependencies with distance
    // distance = 0 if i < j (same iteration), distance = 1 if i > j (next
    // iteration)
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (HasDependency(nodes[i], nodes[j])) {
          if (i < j) {
            data_deps.emplace_back(i, j, 0);
          } else {
            if (HasRegisterDependency(nodes[i], nodes[j])) {
              data_deps.emplace_back(i, j, 1);
            } else {
              auto deps = GetSharedDependencies(nodes[i], nodes[j]);
              for (const auto &buffer : deps) {
                data_deps.emplace_back(i, j, buffer_to_num_versions[buffer]);
              }
            }
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
    ffi::Array<int64_t> tvm_buffer_sizes;

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
    for (auto val : buffer_sizes) {
      tvm_buffer_sizes.push_back(val);
    }

    // Extract results
    // Python function returns (start_times, stages) as a tuple of two
    // arrays
    auto return_val =
        z3_schedule_loop_func
            .value()(num_stages, tvm_latencies, tvm_iis, tvm_resource_flags,
                     tvm_data_deps, tvm_resource_deps, tvm_buffer_sizes,
                     memory_limit)
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

      std::vector<std::shared_ptr<IRStructure>> reordered_children;
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

    // Reorder & Copy Let-defined variables
    auto IsVarDecl = [&](IRStructure *node) -> bool {
      if (!node)
        return false;

      if (node->IsTask()) {
        auto task = static_cast<TaskNode *>(node);
        if (task->stmts.size() == 1) {
          return task->stmts[0].as<LetStmtNode>() != nullptr;
        }
      }
      return false;
    };
    auto SolveConflictVar = [&]() -> bool {
      for (int i = 0; i < n; ++i)
        if (IsVarDecl(seq_body->children[i].get())) {
          for (int j = 0; j < n; ++j) {
            if (i == j)
              continue;

            auto node_i = seq_body->children[i].get();
            auto node_j = seq_body->children[j].get();
            int rem_stage_j = stage_map[node_j];

            if (!HasDependency(node_i, node_j))
              continue;

            // LOG(INFO) << "[ScheduleRecursive] Conflict var detection between
            // " << i << " and " << j;

            if (stage_map[node_j] == stage_map[node_i])
              continue;

            auto node_i_task = static_cast<TaskNode *>(node_i);
            auto node_i_let_stmt = node_i_task->stmts[0].as<LetStmtNode>();

            auto iter = ctrl->control->loop_var;
            auto step = ctrl->control->step.has_value()
                            ? ctrl->control->step.value()
                            : 1;
            auto cloned_value = node_i_let_stmt->value;
            auto cloned_let_stmt =
                LetStmt(node_i_let_stmt->var.copy_with_suffix(""), cloned_value,
                        Evaluate(0));
            auto cloned_task = std::make_shared<TaskNode>();
            cloned_task->stmts.push_back(cloned_let_stmt);
            stage_map[cloned_task.get()] = rem_stage_j;

            for (int k = j; k < n; ++k) {
              auto node_k = seq_body->children[k].get();
              auto task_k = static_cast<TaskNode *>(node_k);
              if (rem_stage_j != stage_map[node_k])
                continue;
              if (HasDependency(node_i, node_k)) {
                for (size_t id = 0; id < task_k->stmts.size(); ++id) {
                  task_k->stmts[id] = Substitute(
                      task_k->stmts[id],
                      {{node_i_let_stmt->var, cloned_let_stmt->var}});
                }
                task_k->SubstituteVar(node_i_let_stmt->var,
                                      cloned_let_stmt->var);
                stage_map[node_k] = rem_stage_j;
              }
            }

            seq_body->children.insert(seq_body->children.begin() + j,
                                      std::move(cloned_task));
            n += 1;
            return true; // Conflict resolved, restart the loop
          }
        }
      return false;
    };
    // Resolve conflicts until no more conflicts exist or max iterations reached
    // (to avoid infinite loop)
    int conflict_count = 0;
    while (SolveConflictVar() && ++conflict_count < 100)
      ;

    for (auto &node : seq_body->children) {
      auto unit = std::make_shared<ScheduleUnit>();
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
    ctrl->SetIIperIter(ii);
  }

  // Set thread index variable for warpgroup partition
  void SetThreadVar(IterVar thread_var) { thread_var_ = thread_var; }

  // Set enable_warp_partition flag
  void SetEnableWarpPartition(bool enable) { enable_warp_partition_ = enable; }

  // Set shared memory limit for pipeline (in bytes)
  void SetSharedMemoryLimit(int64_t bytes) { shared_memory_limit_ = bytes; }

private:
  IterVar thread_var_; // Thread index variable for warpgroup partition
  bool enable_warp_partition_ = false;
  int64_t shared_memory_limit_ = 48 * 1024;

  // Check if two regions refer to the same buffer
  bool SameBuffer(const BufferRegion &a, const BufferRegion &b) const {
    return a->buffer.same_as(b->buffer);
  }

  // Check if two variables are the same
  bool SameVar(const Var &a, const Var &b) const { return a.same_as(b); }

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
    for (const auto &write_var_a : a->GetWriteVars()) {
      for (const auto &read_var_b : b->GetReadVars()) {
        if (SameVar(write_var_a, read_var_b))
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

  // Get shared buffers two IRStructures both access (excluding read-after-read)
  std::set<Buffer> GetSharedDependencies(const IRStructure *a,
                                         const IRStructure *b) const {
    std::set<Buffer> deps;
    for (const auto &write_region_a : a->GetWriteRegions()) {
      if (!IsSharedBuffer(write_region_a->buffer))
        continue;
      for (const auto &read_region_b : b->GetReadRegions()) {
        if (SameBuffer(write_region_a, read_region_b))
          deps.insert(write_region_a->buffer);
      }
      for (const auto &write_region_b : b->GetWriteRegions()) {
        if (SameBuffer(write_region_a, write_region_b))
          deps.insert(write_region_a->buffer);
      }
    }
    for (const auto &read_region_a : a->GetReadRegions()) {
      if (!IsSharedBuffer(read_region_a->buffer))
        continue;
      for (const auto &write_region_b : b->GetWriteRegions()) {
        if (SameBuffer(read_region_a, write_region_b))
          deps.insert(read_region_a->buffer);
      }
    }
    return deps;
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
void ScheduleUnitBuilder::ScheduleRecursive(
    IRStructure *node, const std::set<Buffer> &used_buffers) {
  if (!node)
    return;

  auto ChildrenScheduleHelper =
      [&](std::vector<std::shared_ptr<IRStructure>> origin_children)
      -> std::vector<std::shared_ptr<IRStructure>> {
    // Now collect child nodes for potential scheduling
    std::vector<IRStructure *> child_nodes;
    child_nodes.reserve(origin_children.size());
    for (const auto &child : origin_children) {
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
      std::vector<std::shared_ptr<IRStructure>> reordered_children;
      reordered_children.reserve(sorted_nodes.size());

      for (IRStructure *sorted_node : sorted_nodes) {
        auto it = node_to_index.find(sorted_node);
        if (it == node_to_index.end()) {
          LOG(FATAL) << "[ScheduleRecursive] IRStructure not found in "
                        "children mapping";
        }
        size_t old_idx = it->second;
        reordered_children.emplace_back(origin_children[old_idx]);
      }

      // Move reordered children back
      origin_children = reordered_children;
    }
    for (auto &node : origin_children) {
      auto unit = std::make_shared<ScheduleUnit>();
      unit->stage = -1;
      unit->child = std::shared_ptr<IRStructure>(node);
      node = unit;
    }
    return origin_children;
  };

  if (node->IsTask()) {
    // TaskNode: no further scheduling needed
    return;
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);

    // First, recursively schedule all children
    std::vector<std::shared_ptr<IRStructure>> seq_children, origin_children;
    GatherTaskNodes(seq->children, origin_children);
    for (auto &child : origin_children) {
      auto child_used_buffers = used_buffers;
      for (auto &other_child : origin_children) {
        if (child.get() != other_child.get()) {
          for (const auto &region : other_child->GetReadRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
          for (const auto &region : other_child->GetWriteRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
        }
      }
      ScheduleRecursive(child.get(), child_used_buffers);
    }

    seq->children = ChildrenScheduleHelper(origin_children);
    return;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);

    // Now schedule the ControlNode's internal tasks (if any) as a unit
    // The body should now be a SequenceNode containing the tasks
    if (ctrl->child) {
      if (ctrl->child->IsSequence()) {
        auto seq_body = static_cast<SequenceNode *>(ctrl->child.get());
        std::vector<std::shared_ptr<IRStructure>> origin_children;
        GatherTaskNodes(seq_body->children, origin_children);
        for (auto &child : origin_children) {
          auto child_used_buffers = used_buffers;
          for (auto &other_child : origin_children) {
            if (child.get() != other_child.get()) {
              for (const auto &region : other_child->GetReadRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
              for (const auto &region : other_child->GetWriteRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
            }
          }
          ScheduleRecursive(child.get(), child_used_buffers);
        }
        Z3SchedulePythonLoop(ctrl, used_buffers);
      } else if (ctrl->child->IsWrapper()) {
        auto wrapper = static_cast<WrapperNode *>(ctrl->child.get());
        std::vector<std::shared_ptr<IRStructure>> origin_children;
        GatherTaskNodes({wrapper->child}, origin_children);
        for (auto &child : origin_children) {
          auto child_used_buffers = used_buffers;
          for (auto &other_child : origin_children) {
            if (child.get() != other_child.get()) {
              for (const auto &region : other_child->GetReadRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
              for (const auto &region : other_child->GetWriteRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
            }
          }
          ScheduleRecursive(child.get(), child_used_buffers);
        }
        Z3SchedulePythonLoop(ctrl, used_buffers);
      } else {
        ScheduleRecursive(ctrl->child.get(), used_buffers);
      }
    }
    return;
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    std::vector<std::shared_ptr<IRStructure>> origin_children;
    GatherTaskNodes({wrapper->child}, origin_children);
    for (auto &child : origin_children) {
      auto child_used_buffers = used_buffers;
      for (auto &other_child : origin_children) {
        if (child.get() != other_child.get()) {
          for (const auto &region : other_child->GetReadRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
          for (const auto &region : other_child->GetWriteRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
        }
      }
      ScheduleRecursive(child.get(), child_used_buffers);
    }
    auto seq_node = std::make_shared<SequenceNode>();
    seq_node->children = ChildrenScheduleHelper(origin_children);
    *node = *seq_node;
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
  std::shared_ptr<IRStructure> Build(const Stmt &stmt, int64_t thread_count = 1,
                                     Target target = Target()) {
    thread_count_ = thread_count;
    target_ = target;
    VisitStmt(stmt);
    if (!root_) {
      LOG(WARNING)
          << "IRStructureBuilder: root_ is null after visiting statement. "
          << "This may indicate an unhandled statement type.";
      // Return an empty TaskNode as fallback
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmts.push_back(stmt);
      return task_node;
    }
    return std::move(root_);
  }

protected:
  void VisitStmt_(const SeqStmtNode *op) override {
    auto seq_node = std::make_shared<SequenceNode>();

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
      auto control_node = std::make_shared<ControlNode>();
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
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmts.push_back(GetRef<Stmt>(op));

      // Analyze the loop body for resource usage
      AnalyzeResourceUsage(op->body, task_node.get());

      root_ = std::move(task_node);
    }
  }

  void VisitStmt_(const EvaluateNode *op) override {
    // Evaluate statement (usually a Call) -> TaskNode
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze the expression for resource usage
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    // If statement -> treat as TaskNode for now (could be refined later)
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    AnalyzeMemoryExpr(op->condition, task_node.get());
    AnalyzeResourceUsage(Evaluate(op->condition), task_node.get(), true);

    // Analyze both branches for resource usage
    AnalyzeResourceUsage(op->then_case, task_node.get());
    if (op->else_case) {
      AnalyzeResourceUsage(op->else_case.value(), task_node.get());
    }

    root_ = std::move(task_node);
  }

  void VisitStmt_(const LetStmtNode *op) override {
    // Wrapper statement -> WrapperNode
    auto wrapper_node = std::make_shared<WrapperNode>();
    wrapper_node->wrapper = GetRef<Stmt>(op);
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetLetDecl(op));
    AnalyzeResourceUsage(GetLetDecl(op), task_node.get());
    wrapper_node->task = std::move(task_node);

    // Process the wrapperbody
    VisitStmt(op->body);
    if (root_) {
      wrapper_node->child = std::move(root_);
    }

    root_ = std::move(wrapper_node);
  }

  void VisitStmt_(const AttrStmtNode *op) override {
    // Wrapper statement -> WrapperNode
    auto wrapper_node = std::make_shared<WrapperNode>();
    wrapper_node->wrapper = GetRef<Stmt>(op);
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetAttrDecl(op));
    AnalyzeResourceUsage(GetAttrDecl(op), task_node.get());
    wrapper_node->task = std::move(task_node);

    // Process the wrapperbody
    VisitStmt(op->body);
    if (root_) {
      wrapper_node->child = std::move(root_);
    }

    root_ = std::move(wrapper_node);
  }

  void VisitStmt_(const WhileNode *op) override {
    auto task_node = std::make_shared<TaskNode>();
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
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());
    root_ = std::move(task_node);
  }

private:
  std::shared_ptr<IRStructure> root_;
  int64_t thread_count_ = 1;
  Target target_;

  void AnalyzeResourceUsage(const Stmt &stmt, TaskNode *task_node,
                            bool only_variables = false) {
    // Recursively analyze statements to determine resource usage
    struct ResourceAnalyzer : public StmtExprVisitor {
      TaskNode *task_node;
      bool found_tma{false};
      bool found_tensor{false};
      bool found_cuda{false};

      bool found_tma_load{false};

      // Tensor Core shape information (multiple shapes possible)
      struct TensorCoreShape {
        int64_t m;
        int64_t n;
        int64_t k;
        TensorCoreShape(int64_t m = 0, int64_t n = 0, int64_t k = 0)
            : m(m), n(n), k(k) {}
      };
      std::vector<TensorCoreShape> tensor_core_shapes;

      // GemmInst: the resolved tensor core instruction (single, asserted
      // shared)
      GemmInst gemm_inst{GemmInst::kMMA};
      bool has_gemm_inst{false};

      // Target and block_size for GemmInst determination
      Target target;
      int64_t block_size;

      ResourceAnalyzer(TaskNode *node, Target target = Target(),
                       int64_t block_size = 128)
          : task_node(node), target(target), block_size(block_size) {}

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
          bool found_global = false, found_shared = false;
          int idx_global = -1, idx_shared = -1;
          for (unsigned idx = 0; idx != 2; ++idx) {
            auto region = Downcast<Call>(op->args[idx]);
            if (const auto *buffer_load =
                    region->args[0].as<BufferLoadNode>()) {
              Buffer buffer = buffer_load->buffer;
              String scope = buffer.scope();
              MemoryType mem_type = GetMemoryTypeFromScope(scope);
              if (mem_type == MemoryType::kGlobal) {
                found_global = true;
                idx_global = idx;
              }
              if (mem_type == MemoryType::kShared) {
                found_shared = true;
                idx_shared = idx;
              }
            }
          }
          found_tma = false;
          if (found_global && found_shared) {
            if (idx_global == 0 && idx_shared == 1)
              found_tma = true;
            if (idx_global == 1 && idx_shared == 0)
              found_tma = true;
          }
        } else if (op->op.same_as(gemm_py_op) || op->op.same_as(gemm_op)) {
          found_tensor = true;

          int64_t m = op->args[5].as<IntImmNode>()->value;
          int64_t n = op->args[6].as<IntImmNode>()->value;
          int64_t k = op->args[7].as<IntImmNode>()->value;
          tensor_core_shapes.emplace_back(m, n, k);

          // Determine the final GemmInst using GemmPyNode::getGemmInst
          if (target.defined()) {
            GemmPy gemm_py(op->args);
            GemmInst inst =
                gemm_py->getGemmInst(static_cast<int>(block_size), target);
            ICHECK(!has_gemm_inst || gemm_inst == inst)
                << "All gemm operations in a task must use the same GemmInst, "
                << "but got " << GemmInstToString(gemm_inst) << " and "
                << GemmInstToString(inst);
            gemm_inst = inst;
            has_gemm_inst = true;
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

    ResourceAnalyzer analyzer(task_node, target_, thread_count_);
    analyzer(stmt);

    if (!only_variables) {
      // Set task node flags based on what was found
      if (analyzer.found_tma) {
        task_node->SetUsesTMACore(true);
        if (analyzer.found_tma_load) {
          task_node->SetHasTMALoad(true);
        }
      }
      if (analyzer.found_tensor) {
        task_node->SetUsesTensorCore(true);
        // Set Tensor Core shape information if available
        for (const auto &shape : analyzer.tensor_core_shapes) {
          if (shape.m > 0 && shape.n > 0 && shape.k > 0) {
            task_node->AddTensorCoreShape(shape.m, shape.n, shape.k);
          }
        }
        // Set GemmInst information
        if (analyzer.has_gemm_inst) {
          task_node->SetGemmInst(analyzer.gemm_inst);
        }
      }
      // If neither TMA nor Tensor core was used, and CUDA operations were
      // found, set CUDA core flag
      if (!analyzer.found_tma && !analyzer.found_tensor) {
        task_node->SetUsesCUDACore(true);
      }
    }

    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(stmt);
    std::vector<BufferRegion> read_regions = memory_detector.GetReadRegions();
    std::vector<BufferRegion> write_regions = memory_detector.GetWriteRegions();
    std::vector<Var> read_vars = memory_detector.GetReadVars();
    std::vector<Var> write_vars = memory_detector.GetWriteVars();

    // Merge with existing regions (avoid duplicates)
    for (const auto &region : read_regions) {
      task_node->AddReadRegion(region);
    }

    for (const auto &region : write_regions) {
      task_node->AddWriteRegion(region);
    }

    for (const auto &var : read_vars) {
      task_node->AddReadVar(var);
    }

    for (const auto &var : write_vars) {
      task_node->AddWriteVar(var);
    }

    // Estimate latency and initiation interval for this task
    LatencyEstimator latency_estimator;
    latency_estimator.SetThreadCount(thread_count_);
    latency_estimator.Estimate(task_node);
  }

  void AnalyzeMemoryExpr(const PrimExpr &expr, TaskNode *task_node) {
    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(expr);
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
    latency_estimator.SetThreadCount(thread_count_);
    latency_estimator.Estimate(task_node);
  }
};

Stmt ReNestLetStmts(const Stmt &stmt);

// The main pass function
tvm::transform::Pass AutoSchedule(const bool enable_epi) {
  using namespace tir::transform;
  auto pass_func =
      [enable_epi](PrimFunc func, const IRModule &mod,
                   const tvm::transform::PassContext &ctx) -> PrimFunc {
    // Get target from PrimFunc attribute for GemmInst determination
    auto target_opt = func->GetAttr<Target>(tvm::attr::kTarget);
    Target target;
    if (target_opt.defined()) {
      target = target_opt.value();
    }
    auto config = GetWarpSpecializeConfig(target);

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

    // Get thread index variable for warpgroup partition
    // First try to get from body_to_schedule, if not found, try from the entire
    // function body
    thread_var = ThreadTagChecker::GetThreadVar(body_to_schedule);
    if (!thread_var.defined()) {
      thread_var = ThreadTagChecker::GetThreadVar(func->body);
    }

    // Calculate thread count for latency estimation
    int64_t latency_thread_count = 1;
    if (thread_var.defined() && thread_var->dom.defined()) {
      PrimExpr thread_extent = thread_var->dom->extent;
      if (const int64_t *extent_ptr = as_const_int(thread_extent)) {
        latency_thread_count = *extent_ptr;
        if (latency_thread_count < 1)
          latency_thread_count = 1;
      }
    }

    // Build IRStructure from the body to schedule
    IRStructureBuilder builder;
    auto ir_structure =
        builder.Build(body_to_schedule, latency_thread_count, target);

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
    unit_builder.SetEnableWarpPartition(config.enable_warp_partition);
    unit_builder.SetSharedMemoryLimit(config.shared_memory_limit);
    bool double_thread = unit_builder.Build(ir_structure.get());

    if (!config.enable_warpgroup_partition) {
      Stmt new_body = ConvertIRStructureToStmt(ir_structure.get(), enable_epi);

      // If we extracted from tilelang_root block, replace the body
      Stmt final_body;
      TilelangRootBodyReplacer replacer(new_body);
      final_body = replacer(func->body);

      final_body = ReNestLetStmts(final_body);

      // Create a new PrimFunc with the updated body
      auto new_func = PrimFunc(func->params, final_body, func->ret_type,
                               func->buffer_map, func->attrs);
      return new_func;
    }

    // Print the modified summary view
    // PrintIRStructure(ir_structure.get());

    // Analyze buffer dependencies and insert barriers before warpgroup
    // partition
    int next_barrier_id = 1;
    std::vector<Buffer> barrier_buffers;
    Map<ObjectRef, ObjectRef> barrier_map;
    // Determine thread count for barrier arrive_count calculations
    PrimExpr thread_count[2];
    if (!config.enable_thread_extend) {
      ICHECK(config.enable_warp_partition);
      // sm_100: use fixed warp size (32) for both partitions
      thread_count[0] = IntImm(DataType::Int(32), 32);
      thread_count[1] = IntImm(DataType::Int(32), 32);
    } else {
      // sm_90: original behavior
      thread_count[0] = thread_var->dom->extent;
      thread_count[1] = double_thread ? thread_var->dom->extent
                                      : IntImm(DataType::Int(32),
                                               config.producer_thread_count);
    }
    LoopNestingInfo loop_info;
    std::vector<MultiVersionBufferInfo> buffer_infos;
    PrimExpr barrier_count = config.enable_thread_extend
                                 ? thread_count[0] + thread_count[1]
                                 : thread_var->dom->extent;
    Buffer neutral_sync_shared_barrier =
        makeBarrierBuffer(barrier_count, "neutral_sync_shared_barrier", 1,
                          barrier_buffers, barrier_map);
    AnalyzeAndInsertBarriers(
        ir_structure.get(), next_barrier_id, barrier_buffers, barrier_map,
        thread_count, loop_info, buffer_infos, neutral_sync_shared_barrier);

    // Print the modified summary view
    // PrintIRStructure(ir_structure.get());

    // Apply warpgroup partition to entire IRStructure
    Stmt new_body = ApplyWarpgroupPartitionToIRStructure(
        ir_structure.get(), thread_var, barrier_buffers, barrier_map,
        enable_epi, thread_count, double_thread, config,
        neutral_sync_shared_barrier);

    if (config.enable_thread_extend) {
      // sm_90: may need to update thread extent
      if (double_thread) {
        updated_thread_extent = thread_var->dom->extent * 2;
      } else {
        updated_thread_extent =
            thread_var->dom->extent +
            IntImm(DataType::Int(32), config.producer_thread_count);
      }
    }

    // If we extracted from tilelang_root block, replace the body
    Stmt final_body;
    TilelangRootBodyReplacer replacer(new_body);
    final_body = replacer(func->body);
    // Apply thread extent update if warpgroup partition was applied (sm_90
    // only)
    if (config.enable_thread_extend) {
      ThreadExtentUpdater extent_updater(updated_thread_extent);
      final_body = extent_updater(final_body);
    }
    // Add barrier buffers to tilelang_root block's alloc_buffers
    if (!barrier_buffers.empty()) {
      class TilelangRootAllocBufferAdder : public StmtMutator {
      public:
        explicit TilelangRootAllocBufferAdder(
            const std::vector<Buffer> &buffers_to_add,
            Map<ObjectRef, ObjectRef> &barrier_map)
            : buffers_to_add_(buffers_to_add), barrier_map_(barrier_map) {}

        Stmt VisitStmt_(const BlockNode *op) override {
          auto block = GetRef<Block>(op);
          if (op->name_hint == "tilelang_root") {
            // Combine existing alloc_buffers with new buffers
            Array<Buffer> new_alloc_buffers = op->alloc_buffers;
            for (const auto &buffer : buffers_to_add_) {
              new_alloc_buffers.push_back(buffer);
            }
            auto new_annotations = op->annotations;
            new_annotations.Set("barrier_init", barrier_map_);
            // Create new block with updated alloc_buffers
            return Block(op->iter_vars, op->reads, op->writes, op->name_hint,
                         op->body, op->init, new_alloc_buffers,
                         op->match_buffers, new_annotations);
          }
          return StmtMutator::VisitStmt_(op);
        }

      private:
        std::vector<Buffer> buffers_to_add_;
        Map<ObjectRef, ObjectRef> &barrier_map_;
      };

      TilelangRootAllocBufferAdder adder(barrier_buffers, barrier_map);
      final_body = adder(final_body);
    }

    // Apply multi-version alloc_buffer rewrite if needed
    if (!buffer_infos.empty()) {
      final_body = RewriteAllocBuffers(final_body, buffer_infos);
    }

    // LOG(INFO) << final_body << std::endl;

    final_body = ReNestLetStmts(final_body);

    // Create a new PrimFunc with the updated body
    auto new_func = PrimFunc(func->params, final_body, func->ret_type,
                             func->buffer_map, func->attrs);
    return new_func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

// Helper: check if a TaskNode is a LetDecl (single LetStmt with empty body)
static bool IsLetDeclTask(const TaskNode *task) {
  return task->stmts.size() == 1 && task->stmts[0].as<LetStmtNode>() != nullptr;
}

// Helper: check if an IRStructure node is a LetDecl task (or a ScheduleUnit
// wrapping one)
static bool IsLetDeclNode(const IRStructure *node) {
  if (!node)
    return false;
  if (node->IsTask()) {
    return IsLetDeclTask(static_cast<const TaskNode *>(node));
  }
  if (node->IsScheduleUnit()) {
    auto unit = static_cast<const ScheduleUnit *>(node);
    return unit->child && unit->child->IsTask() &&
           IsLetDeclTask(static_cast<const TaskNode *>(unit->child.get()));
  }
  return false;
}

// Helper: check if an IRStructure subtree contains any LetDecl tasks
static bool ContainsLetDecl(const IRStructure *node) {
  if (!node)
    return false;
  if (IsLetDeclNode(node))
    return true;
  if (node->IsSequence()) {
    auto seq = static_cast<const SequenceNode *>(node);
    for (const auto &child : seq->children) {
      if (ContainsLetDecl(child.get()))
        return true;
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<const ControlNode *>(node);
    return ContainsLetDecl(ctrl->child.get());
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(node);
    return ContainsLetDecl(wrapper->child.get());
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<const ScheduleUnit *>(node);
    return ContainsLetDecl(unit->child.get());
  }
  return false;
}

// Helper function to clone IRStructure with warpgroup filter.
std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id,
                                    Map<Var, PrimExpr> &var_remap) {
  if (!node)
    return nullptr;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);

    // LetDecl tasks are always included in every warp group clone.
    // Create a fresh variable copy so the two warp groups use different names.
    if (IsLetDeclTask(task)) {
      const auto *let = task->stmts[0].as<LetStmtNode>();
      auto new_var = let->var.copy_with_suffix("");
      // Substitute previously renamed variables in the value expression.
      PrimExpr new_value =
          var_remap.empty() ? let->value : Substitute(let->value, var_remap);
      var_remap.Set(let->var, new_var);
      auto new_task = std::make_shared<TaskNode>();
      new_task->stmts.push_back(LetStmt(new_var, new_value, Evaluate(0)));
      return new_task;
    }

    // Non-LetDecl tasks: only include if warp group matches
    if (!node->containWarpgroupId(warpgroup_id))
      return nullptr;
    auto cloned = task->Clone();
    // Substitute renamed LetDecl variables in task statements
    if (!var_remap.empty()) {
      auto ct = static_cast<TaskNode *>(cloned.get());
      for (size_t i = 0; i < ct->stmts.size(); ++i) {
        ct->stmts[i] = Substitute(ct->stmts[i], var_remap);
      }
    }
    return cloned;
  } else if (node->IsSequence()) {
    // A SequenceNode is included if it contains the target warp group
    // OR if it contains LetDecl tasks (which are always needed).
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto seq = static_cast<SequenceNode *>(node);
    auto new_seq = std::make_shared<SequenceNode>();
    for (const auto &child : seq->children) {
      auto new_child = CloneIRStructureWithWarpgroupFilter(
          child.get(), warpgroup_id, var_remap);
      if (new_child) {
        new_seq->children.push_back(std::move(new_child));
      }
    }
    if (new_seq->children.empty())
      return nullptr;
    return new_seq;
  } else if (node->IsControl()) {
    // A ControlNode is included if it contains the target warp group
    // OR if it contains LetDecl tasks.
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto ctrl = static_cast<ControlNode *>(node);
    auto new_ctrl = std::make_shared<ControlNode>();
    new_ctrl->control = ctrl->control;
    new_ctrl->SetPromote(ctrl->hasPromote());
    new_ctrl->child = CloneIRStructureWithWarpgroupFilter(
        ctrl->child.get(), warpgroup_id, var_remap);
    return new_ctrl;
  } else if (node->IsWrapper()) {
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto wrapper = static_cast<WrapperNode *>(node);
    auto new_wrapper = std::make_shared<WrapperNode>();
    // Keep the wrapper statement as-is (do NOT rename LetStmt wrappers here;
    // only LetDecl TaskNodes get renamed).
    new_wrapper->wrapper = wrapper->wrapper;
    new_wrapper->child = CloneIRStructureWithWarpgroupFilter(
        wrapper->child.get(), warpgroup_id, var_remap);
    return new_wrapper;
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    bool child_is_let_decl = IsLetDeclNode(unit->child.get());

    // Include the ScheduleUnit if the child is a LetDecl or the warp group
    // matches.
    if (!child_is_let_decl && !node->containWarpgroupId(warpgroup_id))
      return nullptr;

    auto new_unit = std::make_shared<ScheduleUnit>();
    new_unit->stage = unit->stage;
    new_unit->child = CloneIRStructureWithWarpgroupFilter(
        unit->child.get(), warpgroup_id, var_remap);

    if (!child_is_let_decl) {
      // Copy before/after for the target warp group
      new_unit->before[warpgroup_id] = unit->before[warpgroup_id];
      new_unit->after[warpgroup_id] = unit->after[warpgroup_id];
      // Substitute renamed LetDecl variables in before/after stmts
      if (!var_remap.empty()) {
        for (auto &s : new_unit->before[warpgroup_id]) {
          s = Substitute(s, var_remap);
        }
        for (auto &s : new_unit->after[warpgroup_id]) {
          s = Substitute(s, var_remap);
        }
      }
    }
    return new_unit;
  }
  LOG(FATAL);
}

// Entry point overload — creates a fresh var_remap per call
std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id) {
  Map<Var, PrimExpr> var_remap;
  return CloneIRStructureWithWarpgroupFilter(node, warpgroup_id, var_remap);
}

// Simple visitor to collect all Var references from statements/expressions
class VarRefCollector : public StmtExprVisitor {
public:
  std::unordered_set<const VarNode *> vars;
  void VisitExpr_(const VarNode *op) override { vars.insert(op); }
};

// Remove LetDecl definitions whose variables are not referenced by any
// non-LetDecl task in the IR tree.  After warp-group partitioning some
// LetDecl values may access out-of-bounds indices because the consumer
// tasks that used them ended up in the other warp group.
std::shared_ptr<IRStructure>
RemoveUnusedLetDecls(std::shared_ptr<IRStructure> root) {
  if (!root)
    return nullptr;

  // Phase 1: Collect LetDecl definitions and variable references from
  // non-LetDecl nodes (task stmts and ScheduleUnit before/after).
  struct LetDeclEntry {
    const VarNode *var;
    PrimExpr value;
  };
  std::vector<LetDeclEntry> let_decls;
  std::unordered_set<const VarNode *> referenced_vars;

  std::function<void(const IRStructure *)> collect =
      [&](const IRStructure *node) {
        if (!node)
          return;
        if (node->IsTask()) {
          auto task = static_cast<const TaskNode *>(node);
          if (IsLetDeclTask(task)) {
            const auto *let = task->stmts[0].as<LetStmtNode>();
            let_decls.push_back({let->var.get(), let->value});
          } else {
            VarRefCollector collector;
            for (const auto &stmt : task->stmts) {
              collector(stmt);
            }
            referenced_vars.insert(collector.vars.begin(),
                                   collector.vars.end());
          }
        } else if (node->IsSequence()) {
          for (const auto &child :
               static_cast<const SequenceNode *>(node)->children) {
            collect(child.get());
          }
        } else if (node->IsControl()) {
          collect(static_cast<const ControlNode *>(node)->child.get());
        } else if (node->IsWrapper()) {
          collect(static_cast<const WrapperNode *>(node)->child.get());
        } else if (node->IsScheduleUnit()) {
          auto unit = static_cast<const ScheduleUnit *>(node);
          collect(unit->child.get());
          VarRefCollector collector;
          for (const auto &stmts : unit->before) {
            for (const auto &s : stmts)
              collector(s);
          }
          for (const auto &stmts : unit->after) {
            for (const auto &s : stmts)
              collector(s);
          }
          referenced_vars.insert(collector.vars.begin(), collector.vars.end());
        }
      };
  collect(root.get());

  // Phase 2: Transitive closure — if a LetDecl var is referenced,
  // all vars in its value expression are transitively referenced too.
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &entry : let_decls) {
      if (referenced_vars.count(entry.var)) {
        VarRefCollector collector;
        collector(entry.value);
        for (const auto *v : collector.vars) {
          if (!referenced_vars.count(v)) {
            referenced_vars.insert(v);
            changed = true;
          }
        }
      }
    }
  }

  // Phase 3: Filter the tree — remove LetDecl tasks for unused vars.
  std::function<std::shared_ptr<IRStructure>(
      const std::shared_ptr<IRStructure> &)>
      filter_tree = [&](const std::shared_ptr<IRStructure> &node)
      -> std::shared_ptr<IRStructure> {
    if (!node)
      return nullptr;
    if (node->IsTask()) {
      if (IsLetDeclTask(static_cast<const TaskNode *>(node.get()))) {
        const auto *let = static_cast<const TaskNode *>(node.get())
                              ->stmts[0]
                              .as<LetStmtNode>();
        if (!referenced_vars.count(let->var.get())) {
          return nullptr; // Remove unused LetDecl
        }
      }
      return node;
    } else if (node->IsSequence()) {
      auto seq = static_cast<const SequenceNode *>(node.get());
      auto new_seq = std::make_shared<SequenceNode>();
      for (const auto &child : seq->children) {
        auto filtered = filter_tree(child);
        if (filtered)
          new_seq->children.push_back(std::move(filtered));
      }
      if (new_seq->children.empty())
        return nullptr;
      return new_seq;
    } else if (node->IsControl()) {
      auto ctrl = static_cast<const ControlNode *>(node.get());
      auto new_ctrl = std::make_shared<ControlNode>();
      new_ctrl->control = ctrl->control;
      new_ctrl->SetPromote(ctrl->hasPromote());
      new_ctrl->child = filter_tree(ctrl->child);
      if (!new_ctrl->child)
        return nullptr;
      return new_ctrl;
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<const WrapperNode *>(node.get());
      auto new_wrapper = std::make_shared<WrapperNode>();
      new_wrapper->wrapper = wrapper->wrapper;
      new_wrapper->child = filter_tree(wrapper->child);
      return new_wrapper;
    } else if (node->IsScheduleUnit()) {
      auto unit = static_cast<const ScheduleUnit *>(node.get());
      auto new_unit = std::make_shared<ScheduleUnit>();
      new_unit->stage = unit->stage;
      new_unit->before = unit->before;
      new_unit->after = unit->after;
      new_unit->child = filter_tree(unit->child);
      if (!new_unit->child)
        return nullptr;
      return new_unit;
    }
    return node;
  };

  return filter_tree(root);
}

class SimtCopyDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    SimtCopyDetector detector;
    detector.VisitStmt(stmt);
    return detector.has_simt_copy_;
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    auto scope =
        runtime::StorageScope::Create(GetPtrStorageScope(op->buffer->data));
    if (scope.to_string() != "global") {
      has_simt_copy_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool has_simt_copy_{false};
};

Stmt ConvertIRStructureToStmt(IRStructure *root, const bool outer_enable_epi) {
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

      // Set enable_pro to true only if:
      // 1. No task contains loop_break
      // 2. Loop boundaries (min and extent) are constants
      bool enable_pro = !check_contains_loop_break(ctrl->child.get());

      // Check if loop boundaries are constants
      bool loop_min_is_const = tir::is_const_int(loop_start);
      bool loop_extent_is_const = tir::is_const_int(loop_extent);

      if (!loop_min_is_const || !loop_extent_is_const) {
        enable_pro = false;
      }

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
        Map<Var, PrimExpr> substitution, substitution_cond;
        substitution.Set(loop_var,
                         loop_var - loop_step * (max_stages - unit->stage));
        substitution_cond.Set(
            loop_var,
            Max(loop_start,
                Min(loop_start + loop_extent - loop_step,
                    loop_var - loop_step * (max_stages - unit->stage))));
        if (IsLetDeclNode(unit->child.get())) {
          Stmt stmt = SeqStmt::Flatten(stmts);
          steady.push_back(Substitute(stmt, substitution_cond));
        } else {
          PrimExpr condition =
              And(loop_var < loop_start + loop_extent, loop_var >= loop_start);
          if (unit->stage == min_stages) {
            condition = loop_var >= loop_start;
          }
          if (unit->stage == max_stages) {
            condition = loop_var < loop_start + loop_extent;
          }
          Stmt stmt = IfThenElse(condition, SeqStmt::Flatten(stmts));
          steady.push_back(Substitute(stmt, substitution));
        }
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

  return irstructure_to_stmt(root);
}

// Apply warpgroup partition to entire IRStructure (top-level IfThenElse)
Stmt ApplyWarpgroupPartitionToIRStructure(
    IRStructure *root, IterVar thread_var, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, const bool outer_enable_epi,
    PrimExpr thread_count[2], bool producer_consumer,
    const WarpSpecializeConfig &config, Buffer neutral_sync_shared_barrier) {
  if (!root)
    return Evaluate(0);

  if (root->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(root);
    Stmt body = Evaluate(0);
    if (wrapper->child) {
      body = ApplyWarpgroupPartitionToIRStructure(
          wrapper->child.get(), thread_var, barrier_buffers, barrier_map,
          outer_enable_epi, thread_count, producer_consumer, config,
          neutral_sync_shared_barrier);
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

      // Set enable_pro to true only if:
      // 1. No task contains loop_break
      // 2. Loop boundaries (min and extent) are constants
      bool enable_pro = !check_contains_loop_break(ctrl->child.get());

      // Check if loop boundaries are constants
      bool loop_min_is_const = tir::is_const_int(loop_start);
      bool loop_extent_is_const = tir::is_const_int(loop_extent);

      if (!loop_min_is_const || !loop_extent_is_const) {
        enable_pro = false;
      }

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
        Map<Var, PrimExpr> substitution, substitution_cond;
        substitution.Set(loop_var,
                         loop_var - loop_step * (max_stages - unit->stage));
        substitution_cond.Set(
            loop_var,
            Max(loop_start,
                Min(loop_start + loop_extent - loop_step,
                    loop_var - loop_step * (max_stages - unit->stage))));
        if (IsLetDeclNode(unit->child.get())) {
          Stmt stmt = SeqStmt::Flatten(stmts);
          steady.push_back(Substitute(stmt, substitution_cond));
        } else {
          PrimExpr condition =
              And(loop_var < loop_start + loop_extent, loop_var >= loop_start);
          if (unit->stage == min_stages) {
            condition = loop_var >= loop_start;
          }
          if (unit->stage == max_stages) {
            condition = loop_var < loop_start + loop_extent;
          }
          Stmt stmt = IfThenElse(condition, SeqStmt::Flatten(stmts));
          steady.push_back(Substitute(stmt, substitution));
        }
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
  std::function<std::shared_ptr<IRStructure>(IRStructure *)>
      clone_neutral_filter;
  clone_neutral_filter =
      [&clone_neutral_filter](
          IRStructure *node) -> std::shared_ptr<IRStructure> {
    if (!node)
      return nullptr;

    if (node->IsTask()) {
      auto task = static_cast<TaskNode *>(node);
      if (task->GetWarpgroupId() == -1) {
        return task->Clone();
      } else {
        auto new_task = std::make_shared<TaskNode>();
        // Empty statements
        return new_task;
      }
    } else if (node->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(node);
      auto new_seq = std::make_shared<SequenceNode>();
      for (const auto &child : seq->children) {
        if (child) {
          auto node = static_cast<ScheduleUnit *>(child.get());
          auto new_node = clone_neutral_filter(node->child.get());
          if (new_node) {
            auto new_unit = std::make_shared<ScheduleUnit>();
            new_unit->child = std::move(new_node);
            new_seq->children.push_back(std::move(new_unit));
          }
        }
      }
      return new_seq;
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<WrapperNode *>(node);
      auto new_wrapper = std::make_shared<WrapperNode>();
      new_wrapper->child = clone_neutral_filter(wrapper->child.get());
      if (new_wrapper->child) {
        return new_wrapper;
      }
      return nullptr;
    } else if (node->IsControl()) {
      return nullptr;
    }
    LOG(FATAL);
  };

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

  std::function<std::shared_ptr<IRStructure>(
      IRStructure *, const std::function<bool(int)> &, int)>
      clone_neutral_filter_with_top_level;
  clone_neutral_filter_with_top_level =
      [&clone_neutral_filter_with_top_level, &clone_neutral_filter](
          IRStructure *node, const std::function<bool(int)> &include_top_level,
          int top_level_index) -> std::shared_ptr<IRStructure> {
    if (!node)
      return nullptr;

    if (node->IsTask()) {
      if (include_top_level(top_level_index)) {
        return clone_neutral_filter(node);
      } else {
        auto new_task = std::make_shared<TaskNode>();
        // Empty statements
        return new_task;
      }
    } else if (node->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(node);
      auto new_seq = std::make_shared<SequenceNode>();
      int child_index = 0;
      for (const auto &child : seq->children) {
        if (child) {
          auto schedule_unit = static_cast<ScheduleUnit *>(child.get());
          int next_top_level_index =
              top_level_index == -1 ? child_index : top_level_index;
          auto new_node = clone_neutral_filter_with_top_level(
              schedule_unit->child.get(), include_top_level,
              next_top_level_index);
          if (new_node) {
            auto new_unit = std::make_shared<ScheduleUnit>();
            new_unit->child = std::move(new_node);
            new_seq->children.push_back(std::move(new_unit));
          }
        }
        child_index++;
      }
      return new_seq;
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<WrapperNode *>(node);
      auto new_wrapper = std::make_shared<WrapperNode>();
      new_wrapper->child = clone_neutral_filter_with_top_level(
          wrapper->child.get(), include_top_level, top_level_index);
      if (new_wrapper->child) {
        return new_wrapper;
      }
      return nullptr;
    } else if (node->IsControl()) {
      return nullptr;
    }
    LOG(FATAL);
  };

  int last_warpgroup_task_top_level_index = -1;
  if (root->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(root);
    for (size_t i = 0; i < seq->children.size(); ++i) {
      const auto &child = seq->children[i];
      if (!child) {
        continue;
      }
      auto unit = static_cast<ScheduleUnit *>(child.get());
      std::vector<TaskNodeWithContext> child_tasks;
      CollectAllTaskNodesWithContext(unit->child.get(), child_tasks);
      for (const auto &task : child_tasks) {
        if (task.task->GetWarpgroupId() >= 0) {
          last_warpgroup_task_top_level_index = static_cast<int>(i);
        }
      }
    }
  }

  auto is_epi_top_level_index =
      [last_warpgroup_task_top_level_index](int top_level_index) {
        return last_warpgroup_task_top_level_index >= 0 &&
               top_level_index > last_warpgroup_task_top_level_index;
      };
  auto is_pro_top_level_index = [is_epi_top_level_index](int top_level_index) {
    return !is_epi_top_level_index(top_level_index);
  };

  auto wg_pro_neutral_structure = has_warpgroup_neutral
                                      ? clone_neutral_filter_with_top_level(
                                            root, is_pro_top_level_index, -1)
                                      : nullptr;
  auto wg_epi_neutral_structure = has_warpgroup_neutral
                                      ? clone_neutral_filter_with_top_level(
                                            root, is_epi_top_level_index, -1)
                                      : nullptr;

  auto wg0_structure =
      RemoveUnusedLetDecls(CloneIRStructureWithWarpgroupFilter(root, 0));
  auto wg1_structure =
      RemoveUnusedLetDecls(CloneIRStructureWithWarpgroupFilter(root, 1));

  bool wg_pro_neutral_has_stmts =
      wg_pro_neutral_structure
          ? has_actual_statements(wg_pro_neutral_structure.get())
          : false;
  bool wg_epi_neutral_has_stmts =
      wg_epi_neutral_structure
          ? has_actual_statements(wg_epi_neutral_structure.get())
          : false;
  bool wg0_has_stmts = has_actual_statements(wg0_structure.get());
  bool wg1_has_stmts = has_actual_statements(wg1_structure.get());

  PrimExpr condition = thread_var->var < thread_count[0];
  PrimExpr wg1_condition =
      thread_var->var < (thread_count[0] + thread_count[1]);

  Stmt pro_neutral_body =
      wg_pro_neutral_has_stmts
          ? irstructure_to_stmt(wg_pro_neutral_structure.get())
          : Evaluate(0);
  Stmt epi_neutral_body =
      wg_epi_neutral_has_stmts
          ? irstructure_to_stmt(wg_epi_neutral_structure.get())
          : Evaluate(0);
  Stmt then_body =
      wg0_has_stmts ? irstructure_to_stmt(wg0_structure.get()) : Evaluate(0);
  Stmt else_body =
      wg1_has_stmts ? irstructure_to_stmt(wg1_structure.get()) : Evaluate(0);

  Stmt if_then_else;
  if (wg0_has_stmts && wg1_has_stmts) {
    bool has_simt_copy = SimtCopyDetector::Detect(else_body);
    if (has_simt_copy || !config.enable_set_max_nreg) {
      if_then_else =
          IfThenElse(condition, then_body,
                     IfThenElse(wg1_condition, else_body, Evaluate(0)));
    } else {
      std::vector<Stmt> then_body_with_nreg{
          Evaluate(Call(DataType::Handle(), tl::set_max_nreg(),
                        {config.consumer_max_nreg, 1})),
          then_body};
      std::vector<Stmt> else_body_with_nreg{
          Evaluate(Call(DataType::Handle(), tl::set_max_nreg(),
                        {config.producer_max_nreg, 0})),
          else_body};
      if_then_else = IfThenElse(
          condition, SeqStmt(then_body_with_nreg),
          IfThenElse(wg1_condition, SeqStmt(else_body_with_nreg), Evaluate(0)));
    }
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

  PrimExpr barrier_count = config.enable_thread_extend
                               ? thread_count[0] + thread_count[1]
                               : thread_var->dom->extent;

  Stmt pro_and_warpgroup_stmt;
  if (wg_pro_neutral_has_stmts) {
    if (!IsEvaluateZero(if_then_else) && !IsEvaluateZero(pro_neutral_body)) {
      // Both have statements: insert barriers for neutral-to-warpgroup
      // synchronization
      pro_and_warpgroup_stmt = InsertBarriersForNeutralSync(
          pro_neutral_body, if_then_else, barrier_buffers, barrier_map,
          barrier_count, neutral_sync_shared_barrier);
    } else if (!IsEvaluateZero(if_then_else) ||
               !IsEvaluateZero(pro_neutral_body)) {
      // Only one has actual statements
      std::vector<Stmt> stmts;
      if (!IsEvaluateZero(pro_neutral_body)) {
        stmts.push_back(pro_neutral_body);
      }
      if (!IsEvaluateZero(if_then_else)) {
        stmts.push_back(if_then_else);
      }
      if (stmts.size() == 1) {
        pro_and_warpgroup_stmt = stmts[0];
      } else {
        pro_and_warpgroup_stmt = SeqStmt(stmts);
      }
    } else {
      // Both are empty
      pro_and_warpgroup_stmt = Evaluate(0);
    }
  } else {
    pro_and_warpgroup_stmt = if_then_else;
  }

  bool need_shared_barrier_for_epi = false;
  bool need_tmem_barrier_for_epi = false;
  if (wg_epi_neutral_structure) {
    for (const auto *warpgroup_structure :
         {wg0_structure.get(), wg1_structure.get()}) {
      need_shared_barrier_for_epi =
          need_shared_barrier_for_epi ||
          HasSharedWriteReadDependency(warpgroup_structure,
                                       wg_epi_neutral_structure.get());
      need_tmem_barrier_for_epi =
          need_tmem_barrier_for_epi ||
          HasTmemWriteReadDependency(warpgroup_structure,
                                     wg_epi_neutral_structure.get());
    }
  }

  Stmt combined_stmt;
  if (!IsEvaluateZero(pro_and_warpgroup_stmt) &&
      !IsEvaluateZero(epi_neutral_body)) {
    // Both have statements: insert barriers for warpgroup-to-epi_neutral
    // synchronization
    combined_stmt = InsertBarriersForNeutralSyncWithDependency(
        pro_and_warpgroup_stmt, epi_neutral_body, barrier_buffers, barrier_map,
        barrier_count, need_shared_barrier_for_epi, need_tmem_barrier_for_epi,
        Buffer(), thread_var->var, 0, thread_count[0]);
  } else if (!IsEvaluateZero(epi_neutral_body)) {
    combined_stmt = epi_neutral_body;
  } else {
    combined_stmt = pro_and_warpgroup_stmt;
  }

  return combined_stmt;
}

// Re-write LetStmt to nest them properly
// Example transformation:
//   SeqStmt {
//     let x = 42 { Evaluate(0) }     // standalone, empty body
//     let y = x+1 { Evaluate(0) }    // standalone, empty body
//     compute(x, y)                   // actual work
//     store(result)
//   }
// becomes:
//   let x = 42 {
//     let y = x+1 {
//       SeqStmt {
//         compute(x, y)
//         store(result)
//       }
//     }
//   }
class LetStmtNester : public StmtMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) override {
    Array<Stmt> stmts;
    for (const auto &stmt : op->seq) {
      stmts.push_back(this->VisitStmt(stmt));
    }

    Array<Stmt> flat_stmts;
    for (const auto &stmt : stmts) {
      if (const auto *inner_seq = stmt.as<SeqStmtNode>()) {
        for (const auto &inner_stmt : inner_seq->seq) {
          flat_stmts.push_back(inner_stmt);
        }
      } else {
        flat_stmts.push_back(stmt);
      }
    }
    stmts = flat_stmts;

    for (int i = static_cast<int>(stmts.size()) - 2; i >= 0; --i) {
      if (const auto *let = stmts[i].as<LetStmtNode>()) {
        if (IsEmptyBody(let->body)) {
          Stmt absorbed_body = CollectRemaining(stmts, i + 1);
          stmts = TruncateAndReplace(
              stmts, i, LetStmt(let->var, let->value, absorbed_body));
        }
      } else if (const auto *attr = stmts[i].as<AttrStmtNode>()) {
        if (IsEmptyBody(attr->body)) {
          Stmt absorbed_body = CollectRemaining(stmts, i + 1);
          stmts = TruncateAndReplace(
              stmts, i,
              AttrStmt(attr->node, attr->attr_key, attr->value, absorbed_body));
        }
      }
    }

    if (stmts.empty())
      return Evaluate(0);
    if (stmts.size() == 1)
      return stmts[0];

    return SeqStmt(stmts);
  }

private:
  // Check if a statement body is Evaluate(0) — the empty placeholder
  static bool IsEmptyBody(const Stmt &stmt) {
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *imm = eval->value.as<IntImmNode>()) {
        return imm->value == 0;
      }
    }
    return false;
  }

  // Collect stmts[start .. end) into a single Stmt
  static Stmt CollectRemaining(const Array<Stmt> &stmts, int start) {
    int n = static_cast<int>(stmts.size());
    if (start >= n) {
      return Evaluate(0);
    }
    if (start == n - 1) {
      return stmts[start];
    }
    Array<Stmt> remaining;
    for (int j = start; j < n; ++j) {
      remaining.push_back(stmts[j]);
    }
    return SeqStmt(remaining);
  }

  // Keep stmts[0..index), replace stmts[index] with new_stmt,
  // discard everything after (already absorbed into new_stmt body)
  static Array<Stmt> TruncateAndReplace(const Array<Stmt> &stmts, int index,
                                        Stmt new_stmt) {
    Array<Stmt> result;
    for (int j = 0; j < index; ++j) {
      result.push_back(stmts[j]);
    }
    result.push_back(new_stmt);
    return result;
  }
};

Stmt ReNestLetStmts(const Stmt &stmt) {
  LetStmtNester nester;
  return nester(stmt);
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
