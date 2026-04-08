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
      continue;
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
  bool prefix_valid = true;
  CollectPrefixTasks(root, prefix_tasks, prefix_valid);

  std::unordered_set<TaskNode *> suffix_tasks;
  if (enable_warp_partition) {
    CollectSuffixTasks(root, all_tasks, uf, suffix_tasks);
  }

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

} // namespace tl
} // namespace tvm
