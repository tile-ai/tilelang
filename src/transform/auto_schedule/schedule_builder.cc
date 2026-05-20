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
 * \file schedule_builder.cc
 * \brief ScheduleUnitBuilder and task gathering for TileLang AutoSchedule
 */

#include "./schedule_builder.h"

#include "../auto_schedule.h"

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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../op/builtin.h"
#include "../../op/gemm_py.h"
#include "../../op/utils.h"
#include "../../target/utils.h"
#include "../common/attr.h"
#include "../common/collector.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

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

bool SameBuffer(const BufferRegion &a, const BufferRegion &b) {
  return a->buffer.same_as(b->buffer);
}

bool SameVar(const Var &a, const Var &b) { return a.same_as(b); }

bool HasDependency(const IRStructure *a, const IRStructure *b) {
  if (a->IsTask()) {
    const TaskNode *task_a = static_cast<const TaskNode *>(a);
    if (task_a->ContainsLoopBreak())
      return true;
  }
  if (b->IsTask()) {
    const TaskNode *task_b = static_cast<const TaskNode *>(b);
    if (task_b->ContainsLoopBreak())
      return true;
  }
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

bool HasRegisterDependency(const IRStructure *a, const IRStructure *b) {
  if (a->IsTask()) {
    const TaskNode *task_a = static_cast<const TaskNode *>(a);
    if (task_a->ContainsLoopBreak())
      return true;
  }
  if (b->IsTask()) {
    const TaskNode *task_b = static_cast<const TaskNode *>(b);
    if (task_b->ContainsLoopBreak())
      return true;
  }
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

std::set<Buffer> GetSharedDependencies(const IRStructure *a,
                                       const IRStructure *b) {
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

bool HasRegisterRegion(const IRStructure *node) {
  return CountRegisterRegions(node) > 0;
}

bool HasResourceDependency(const IRStructure *a, const IRStructure *b) {
  if (a->UsesTMACore() && b->UsesTMACore())
    return true;
  if (a->UsesTensorCore() && b->UsesTensorCore())
    return true;
  if (a->UsesCUDACore() && b->UsesCUDACore())
    return true;
  return false;
}

void CollectPrefixTasks(IRStructure *root,
                        std::unordered_set<TaskNode *> &prefix_tasks) {
  if (!root)
    return;

  // Collect all top-level items (TaskNodes and ControlNodes) in order
  std::vector<IRStructure *> items;
  CollectTopLevelItems(root, items);

  // Walk forward: a task is prefix iff it has no register regions AND
  // has no dependency with any previously-rejected (non-prefix) item.
  // ControlNodes are always rejected.
  std::vector<IRStructure *> rejected;
  for (auto *item : items) {
    if (item->IsControl()) {
      rejected.push_back(item);
      continue;
    }
    if (!item->IsTask()) {
      rejected.push_back(item);
      continue;
    }
    auto *task = static_cast<TaskNode *>(item);
    if (CountRegisterRegions(task) != 0) {
      rejected.push_back(task);
      continue;
    }
    bool has_dep = false;
    for (auto *rej : rejected) {
      if (HasDependency(task, rej)) {
        has_dep = true;
        break;
      }
    }
    if (has_dep) {
      rejected.push_back(task);
    } else {
      prefix_tasks.insert(task);
    }
  }
}

void CollectSuffixTasks(IRStructure *root,
                        const std::vector<TaskNodeWithContext> &all_tasks,
                        const TaskUnionFind &uf,
                        std::unordered_set<TaskNode *> &suffix_tasks) {
  if (!root)
    return;

  // Collect all top-level items in order
  std::vector<IRStructure *> items;
  CollectTopLevelItems(root, items);

  // Walk backward: a task is a suffix candidate iff it has no dependency
  // with any subsequently-rejected (non-suffix) item.
  // ControlNodes are always rejected.
  std::vector<IRStructure *> rejected;
  std::unordered_set<TaskNode *> candidate_set;
  for (int i = static_cast<int>(items.size()) - 1; i >= 0; --i) {
    auto *item = items[i];
    if (item->IsControl()) {
      rejected.push_back(item);
      continue;
    }
    if (!item->IsTask()) {
      rejected.push_back(item);
      continue;
    }
    auto *task = static_cast<TaskNode *>(item);
    if (task->ContainsLoopBreak()) {
      rejected.push_back(task);
      continue;
    }
    bool has_dep = false;
    for (auto *rej : rejected) {
      if (HasDependency(task, rej)) {
        has_dep = true;
        break;
      }
    }
    if (has_dep) {
      rejected.push_back(task);
    } else {
      candidate_set.insert(task);
    }
  }

  // Apply register region grouping via union-find
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

  for (TaskNode *task : candidate_set) {
    if (suffix_tasks.count(task))
      continue;

    if (CountRegisterRegions(task) == 0) {
      suffix_tasks.insert(task);
      continue;
    }

    auto it = task_to_index.find(task);
    if (it == task_to_index.end())
      continue;
    int component_root = uf.find(it->second);

    auto comp_it = component_tasks.find(component_root);
    if (comp_it == component_tasks.end())
      continue;

    bool all_in_candidates = true;
    for (TaskNode *component_task : comp_it->second) {
      if (candidate_set.find(component_task) == candidate_set.end()) {
        all_in_candidates = false;
        break;
      }
    }

    if (!all_in_candidates)
      continue;

    for (TaskNode *component_task : comp_it->second) {
      suffix_tasks.insert(component_task);
    }
  }
}

bool AssignWarpgroupIdsGlobal(IRStructure *root, bool enable_warp_partition) {
  if (!root) {
    LOG(FATAL) << "Empty root";
  }

  std::vector<TaskNodeWithContext> all_tasks;
  CollectAllTaskNodesWithContext(root, all_tasks);

  if (all_tasks.empty()) {
    LOG(FATAL) << "No task";
  }

  int n = all_tasks.size();

  for (auto &task_ctx : all_tasks) {
    task_ctx.task->SetWarpgroupId(-1);
  }

  TaskUnionFind uf(n);
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (UseSameRegisterRegion(all_tasks[i].task, all_tasks[j].task)) {
        uf.unite(i, j);
      }
    }
  }

  std::unordered_set<TaskNode *> prefix_tasks;
  CollectPrefixTasks(root, prefix_tasks);

  std::unordered_set<TaskNode *> suffix_tasks;
  CollectSuffixTasks(root, all_tasks, uf, suffix_tasks);

  std::unordered_map<int, std::vector<int>> components;
  for (int i = 0; i < n; i++) {
    int root_idx = uf.find(i);
    components[root_idx].push_back(i);
  }

  std::vector<ComponentInfo> component_infos;
  for (const auto &kv : components) {
    int root = kv.first;
    const std::vector<int> &indices = kv.second;
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
        continue;
      }
      if (suffix_tasks.find(task) != suffix_tasks.end()) {
        continue;
      }
      if (task->ContainsLoopBreak()) {
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

  std::sort(component_infos.begin(), component_infos.end(),
            [](const ComponentInfo &a, const ComponentInfo &b) {
              return a.weighted_latency > b.weighted_latency;
            });

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

      for (int idx : comp.task_indices) {
        TaskNode *task = all_tasks[idx].task;
        if (!task->ContainsLoopBreak()) {
          task->SetWarpgroupId(assigned_warpgroup);
        }
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

      for (int idx : comp.task_indices) {
        TaskNode *task = all_tasks[idx].task;
        if (!task->ContainsLoopBreak()) {
          task->SetWarpgroupId(assigned_warpgroup);
        }
      }
    }
    return false;
  }
}

void ScheduleUnitBuilder::ScheduleRecursive(
    std::shared_ptr<IRStructure> &node, const std::set<Buffer> &used_buffers) {
  if (!node)
    return;

  auto ChildrenScheduleHelper =
      [&](std::vector<std::shared_ptr<IRStructure>> origin_children)
      -> std::vector<std::shared_ptr<IRStructure>> {
    std::vector<IRStructure *> child_nodes;
    child_nodes.reserve(origin_children.size());
    for (const auto &child : origin_children) {
      child_nodes.push_back(child.get());
    }

    std::vector<IRStructure *> sorted_nodes;
    sorted_nodes = Z3SchedulePython(child_nodes);

    bool order_changed = false;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
      if (sorted_nodes[i] != child_nodes[i]) {
        order_changed = true;
        break;
      }
    }

    if (order_changed) {
      std::unordered_map<IRStructure *, size_t> node_to_index;
      for (size_t i = 0; i < child_nodes.size(); ++i) {
        node_to_index[child_nodes[i]] = i;
      }

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
    return;
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node.get());

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
      ScheduleRecursive(child, child_used_buffers);
    }

    seq->children = ChildrenScheduleHelper(origin_children);
    return;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node.get());

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
          ScheduleRecursive(child, child_used_buffers);
        }
        Z3SchedulePythonLoop(ctrl, used_buffers);
      } else if (ctrl->child->IsWrapper()) {
        auto wrapper = static_cast<WrapperNode *>(ctrl->child.get());
        std::vector<std::shared_ptr<IRStructure>> origin_children;
        GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
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
          ScheduleRecursive(child, child_used_buffers);
        }
        Z3SchedulePythonLoop(ctrl, used_buffers);
      } else {
        ScheduleRecursive(ctrl->child, used_buffers);
      }
    }
    return;
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children;
    GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
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
      ScheduleRecursive(child, child_used_buffers);
    }
    auto seq_node = std::make_shared<SequenceNode>();
    seq_node->children = ChildrenScheduleHelper(origin_children);
    node = seq_node;
    return;
  }

  LOG(FATAL) << "[ScheduleRecursive] Unknown IRStructure type" << node.get();
}

// --- Naive scheduling implementation ---

bool NaiveAssignWarpgroupIds(IRStructure *root) {
  if (!root)
    LOG(FATAL) << "Empty root";

  std::vector<TaskNodeWithContext> all_tasks;
  CollectAllTaskNodesWithContext(root, all_tasks);
  if (all_tasks.empty())
    LOG(FATAL) << "No task";

  // Simple producer/consumer assignment:
  // TMA tasks → wg1 (producer), compute tasks → wg0 (consumer)
  for (auto &task_ctx : all_tasks) {
    TaskNode *task = task_ctx.task;
    if (task->ContainsLoopBreak()) {
      task->SetWarpgroupId(-1);
      continue;
    }
    if (task->UsesTMACore() && !task->UsesTensorCore()) {
      task->SetWarpgroupId(1); // producer
    } else {
      task->SetWarpgroupId(0); // consumer
    }
  }

  // Collect prefix/suffix tasks and reset them to neutral
  std::unordered_set<TaskNode *> prefix_tasks;
  CollectPrefixTasks(root, prefix_tasks);
  for (auto *task : prefix_tasks) {
    task->SetWarpgroupId(-1);
  }

  int n = all_tasks.size();
  TaskUnionFind uf(n);
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (UseSameRegisterRegion(all_tasks[i].task, all_tasks[j].task)) {
        uf.unite(i, j);
      }
    }
  }
  std::unordered_set<TaskNode *> suffix_tasks;
  CollectSuffixTasks(root, all_tasks, uf, suffix_tasks);
  for (auto *task : suffix_tasks) {
    task->SetWarpgroupId(-1);
  }

  return false; // no double_thread in naive mode
}

void ScheduleUnitBuilder::NaiveScheduleLoop(ControlNode *ctrl) {
  if (!ctrl->child)
    return;

  // Flatten children
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

  // Read num_stages from loop annotation
  int num_stages = 1;
  auto num_stages_val = ctrl->control.get()->annotations.Get("num_stages");
  if (num_stages_val.has_value()) {
    num_stages = num_stages_val.value().cast<IntImm>()->value;
  }

  // Assign pipeline stages and start times:
  // - TMA load → stage 0, start_time = 0
  // - Everything else → stage (num_stages - 1), start_time = num_stages
  // - All task latencies set to 0, IIperIter = 1
  std::map<IRStructure *, int> stage_map;
  bool has_promoted = false;
  for (auto &child : seq_body->children) {
    IRStructure *node = child.get();
    bool is_tma_load =
        node->UsesTMACore() && !node->UsesTensorCore() && !node->UsesCUDACore();
    if (is_tma_load && node->IsTask()) {
      is_tma_load = static_cast<TaskNode *>(node)->HasTMALoad();
    }
    int stage = !is_tma_load ? 0 : (num_stages);
    stage_map[node] = stage;
    if (stage != num_stages) {
      has_promoted = true;
    }
    node->SetStartTime(is_tma_load ? 0 : num_stages);
    node->SetLatency(0);
    node->SetII(0);
  }

  ctrl->SetIIperIter(1);

  // Estimate overall latency
  int64_t tripcount = 100;
  const ForNode *for_node = ctrl->control.get();
  PrimExpr loop_extent = for_node->extent;
  PrimExpr loop_step = for_node->step.has_value()
                           ? for_node->step.value()
                           : IntImm(DataType::Int(32), 1);
  if (const auto *extent_int = loop_extent.as<IntImmNode>()) {
    if (const auto *step_int = loop_step.as<IntImmNode>()) {
      int64_t extent = extent_int->value;
      int64_t step = step_int->value;
      if (step > 0) {
        tripcount = (extent + step - 1) / step;
      }
    }
  }
  ctrl->SetII(tripcount);
  ctrl->SetLatency(tripcount);

  // Wrap in ScheduleUnits with assigned stages
  for (auto &node : seq_body->children) {
    auto unit = std::make_shared<ScheduleUnit>();
    unit->stage = stage_map[node.get()];
    unit->child = std::move(node);
    node = std::move(unit);
  }

  if (has_promoted) {
    ctrl->SetPromote(true);
  }
}

void ScheduleUnitBuilder::NaiveScheduleRecursive(
    std::shared_ptr<IRStructure> &node) {
  if (!node)
    return;

  // Helper to wrap children in ScheduleUnits preserving order
  auto WrapInScheduleUnits =
      [](std::vector<std::shared_ptr<IRStructure>> &children) {
        for (auto &child : children) {
          auto unit = std::make_shared<ScheduleUnit>();
          unit->stage = -1;
          unit->child = std::move(child);
          child = std::move(unit);
        }
      };

  if (node->IsTask()) {
    return;
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children;
    GatherTaskNodes(seq->children, origin_children);
    for (auto &child : origin_children) {
      NaiveScheduleRecursive(child);
    }
    WrapInScheduleUnits(origin_children);
    seq->children = origin_children;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node.get());
    if (ctrl->child) {
      if (ctrl->child->IsSequence() || ctrl->child->IsWrapper()) {
        NaiveScheduleLoop(ctrl);
      } else {
        NaiveScheduleRecursive(ctrl->child);
      }
    }
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children;
    GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
    for (auto &child : origin_children) {
      NaiveScheduleRecursive(child);
    }
    auto seq_node = std::make_shared<SequenceNode>();
    WrapInScheduleUnits(origin_children);
    seq_node->children = origin_children;
    node = seq_node;
  } else {
    LOG(FATAL) << "[NaiveScheduleRecursive] Unknown IRStructure type";
  }
}

bool ScheduleUnitBuilder::NaiveBuild(std::shared_ptr<IRStructure> &root) {
  NaiveScheduleRecursive(root);
  return NaiveAssignWarpgroupIds(root.get());
}

} // namespace tl
} // namespace tvm
