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

#include "../../op/builtin.h"
#include "../common/attr.h"
#include "../common/collector.h"
#include "./ir_structure.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

// SequenceNode member function implementations
bool SequenceNode::UsesCUDACore() const {
  for (const auto &child : children) {
    if (child && child->UsesCUDACore())
      return true;
  }
  return false;
}

bool SequenceNode::UsesTMACore() const {
  for (const auto &child : children) {
    if (child && child->UsesTMACore())
      return true;
  }
  return false;
}

bool SequenceNode::UsesTensorCore() const {
  for (const auto &child : children) {
    if (child && child->UsesTensorCore())
      return true;
  }
  return false;
}

std::vector<BufferRegion> SequenceNode::GetReadRegions() const {
  std::vector<BufferRegion> all_read_regions;
  for (const auto &child : children) {
    if (child) {
      auto child_read_regions = child->GetReadRegions();
      all_read_regions.insert(all_read_regions.end(),
                              child_read_regions.begin(),
                              child_read_regions.end());
    }
  }
  // Remove duplicates (by buffer and region equality)
  // This is a simplified deduplication - in practice might need more
  // sophisticated logic
  std::vector<BufferRegion> deduplicated;
  for (const auto &region : all_read_regions) {
    bool found = false;
    for (const auto &existing : deduplicated) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        found = true;
        break;
      }
    }
    if (!found)
      deduplicated.push_back(region);
  }
  return deduplicated;
}

std::vector<BufferRegion> SequenceNode::GetWriteRegions() const {
  std::vector<BufferRegion> all_write_regions;
  for (const auto &child : children) {
    if (child) {
      auto child_write_regions = child->GetWriteRegions();
      all_write_regions.insert(all_write_regions.end(),
                               child_write_regions.begin(),
                               child_write_regions.end());
    }
  }
  // Remove duplicates (by buffer and region equality)
  std::vector<BufferRegion> deduplicated;
  for (const auto &region : all_write_regions) {
    bool found = false;
    for (const auto &existing : deduplicated) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        found = true;
        break;
      }
    }
    if (!found)
      deduplicated.push_back(region);
  }
  return deduplicated;
}

int64_t SequenceNode::GetLatency() const { return latency_; }

int64_t SequenceNode::GetII() const { return ii_; }

void SequenceNode::SetUsesCUDACore(bool value) {
  for (auto &child : children) {
    if (child)
      child->SetUsesCUDACore(value);
  }
}

void SequenceNode::SetUsesTMACore(bool value) {
  for (auto &child : children) {
    if (child)
      child->SetUsesTMACore(value);
  }
}

void SequenceNode::SetUsesTensorCore(bool value) {
  for (auto &child : children) {
    if (child)
      child->SetUsesTensorCore(value);
  }
}

void SequenceNode::SetReadRegions(const std::vector<BufferRegion> &regions) {
  // Not clear what this means for SequenceNode - maybe set on first child?
  if (!children.empty() && children[0]) {
    children[0]->SetReadRegions(regions);
  }
}

void SequenceNode::SetWriteRegions(const std::vector<BufferRegion> &regions) {
  if (!children.empty() && children[0]) {
    children[0]->SetWriteRegions(regions);
  }
}

void SequenceNode::SetLatency(int64_t latency) { latency_ = latency; }

void SequenceNode::SetII(int64_t ii) { ii_ = ii; }

void SequenceNode::AddReadRegion(const BufferRegion &region) {
  if (!children.empty() && children[0]) {
    children[0]->AddReadRegion(region);
  }
}

void SequenceNode::AddWriteRegion(const BufferRegion &region) {
  if (!children.empty() && children[0]) {
    children[0]->AddWriteRegion(region);
  }
}

std::unique_ptr<IRStructure> SequenceNode::Clone() const {
  auto new_seq = std::make_unique<SequenceNode>();
  new_seq->children.reserve(children.size());
  for (const auto &child : children) {
    if (child) {
      new_seq->children.push_back(child->Clone());
    } else {
      new_seq->children.push_back(nullptr);
    }
  }
  // Copy latency and II
  new_seq->SetLatency(GetLatency());
  new_seq->SetII(GetII());
  return new_seq;
}

std::unique_ptr<IRStructure> TaskNode::Clone() const {
  auto new_task = std::make_unique<TaskNode>();
  // Copy statements
  new_task->stmts = stmts;
  // Copy resource usage flags
  new_task->SetUsesCUDACore(UsesCUDACore());
  new_task->SetUsesTMACore(UsesTMACore());
  new_task->SetUsesTensorCore(UsesTensorCore());
  // Copy memory access regions
  new_task->SetReadRegions(GetReadRegions());
  new_task->SetWriteRegions(GetWriteRegions());
  // Copy latency and II
  new_task->SetLatency(GetLatency());
  new_task->SetII(GetII());
  // Copy start time
  new_task->SetStartTime(GetStartTime());
  // Copy warpgroup id
  new_task->SetWarpgroupId(GetWarpgroupId());
  // Copy loop_break cache
  new_task->contains_loop_break_cache_ = contains_loop_break_cache_;
  return new_task;
}

void TaskNode::CollectRegions(
    std::vector<RegionAccessInfo> &result,
    std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const {
  int wg_id = GetWarpgroupId();
  // Collect write regions
  for (const auto &region : GetWriteRegions()) {
    auto key = std::make_pair(region->buffer, std::make_pair(true, wg_id));
    if (visited.find(key) == visited.end()) {
      visited.insert(key);
      result.emplace_back(region, true, wg_id);
    }
  }
  // Collect read regions
  for (const auto &region : GetReadRegions()) {
    auto key = std::make_pair(region->buffer, std::make_pair(false, wg_id));
    if (visited.find(key) == visited.end()) {
      visited.insert(key);
      result.emplace_back(region, false, wg_id);
    }
  }
}

bool TaskNode::ContainsLoopBreak() const {
  // Return cached result if available
  if (contains_loop_break_cache_.has_value()) {
    return contains_loop_break_cache_.value();
  }

  // Check if any statement in this task contains a loop_break call
  bool found_loop_break = false;
  for (const auto &stmt : stmts) {
    // Helper function to check if a statement contains loop_break
    auto contains_loop_break = [](const Stmt &stmt) -> bool {
      bool found = false;
      PostOrderVisit(stmt, [&found](const ObjectRef &node) {
        if (found)
          return;
        if (const auto *call = node.as<CallNode>()) {
          if (call->op.same_as(tl::loop_break())) {
            found = true;
          }
        }
      });
      return found;
    };

    if (contains_loop_break(stmt)) {
      found_loop_break = true;
      break;
    }
  }

  contains_loop_break_cache_ = found_loop_break;
  return found_loop_break;
}

std::unique_ptr<IRStructure> ControlNode::Clone() const {
  auto new_ctrl = std::make_unique<ControlNode>();
  // Copy For control (For is a TVM object with reference counting)
  new_ctrl->control = control;
  // Clone child if exists
  if (child) {
    new_ctrl->child = child->Clone();
  }
  // Copy latency and II
  new_ctrl->SetLatency(GetLatency());
  new_ctrl->SetII(GetII());
  return new_ctrl;
}

std::unique_ptr<IRStructure> WrapperNode::Clone() const {
  auto new_wrapper = std::make_unique<WrapperNode>();
  // Copy var and value (TVM objects with reference counting)
  new_wrapper->wrapper = wrapper;
  // Clone child if exists
  if (child) {
    new_wrapper->child = child->Clone();
  }
  // Copy latency and II
  new_wrapper->SetLatency(GetLatency());
  new_wrapper->SetII(GetII());
  return new_wrapper;
}

std::unique_ptr<IRStructure> ScheduleUnit::Clone() const {
  auto new_unit = std::make_unique<ScheduleUnit>();
  // Copy var and value (TVM objects with reference counting)
  new_unit->stage = stage;
  // Clone child if exists
  if (child) {
    new_unit->child = child->Clone();
  }
  // Copy latency and II
  new_unit->SetLatency(GetLatency());
  new_unit->SetII(GetII());
  return new_unit;
}

void ControlNode::CollectRegions(
    std::vector<RegionAccessInfo> &result,
    std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const {
  if (child) {
    child->CollectRegions(result, visited);
  }
}

void WrapperNode::CollectRegions(
    std::vector<RegionAccessInfo> &result,
    std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const {
  if (child) {
    child->CollectRegions(result, visited);
  }
}

void ScheduleUnit::CollectRegions(
    std::vector<RegionAccessInfo> &result,
    std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const {
  if (child) {
    child->CollectRegions(result, visited);
  }
}

void SequenceNode::CollectRegions(
    std::vector<RegionAccessInfo> &result,
    std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const {
  for (const auto &child : children) {
    if (child) {
      child->CollectRegions(result, visited);
    }
  }
}

// Helper function to collect all TaskNodes with context information
void CollectAllTaskNodesWithContext(IRStructure *node,
                                    std::vector<TaskNodeWithContext> &all_tasks,
                                    ControlNode *current_control_node) {
  if (!node)
    return;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);
    TaskNodeWithContext task_ctx;
    task_ctx.task = task;
    task_ctx.control_node = current_control_node;

    // Calculate tripcount if inside a loop
    if (current_control_node) {
      const ForNode *for_node = current_control_node->control.get();
      PrimExpr loop_extent = for_node->extent;
      // Try to convert loop_extent to int64_t
      if (const int64_t *extent_ptr = as_const_int(loop_extent)) {
        task_ctx.tripcount = *extent_ptr;
      } else {
        // If extent is not constant, use 100 as default (as requested)
        task_ctx.tripcount = 100;
      }
    } else {
      task_ctx.tripcount = 1; // Not inside a loop
    }

    all_tasks.push_back(task_ctx);
  } else if (node->IsSequence()) {
    auto seq = static_cast<const SequenceNode *>(node);
    for (const auto &child : seq->children) {
      CollectAllTaskNodesWithContext(child.get(), all_tasks,
                                     current_control_node);
    }
  } else if (node->IsControl()) {
    auto control = static_cast<const ControlNode *>(node);
    // When entering a control node, update the current control context
    CollectAllTaskNodesWithContext(control->child.get(), all_tasks,
                                   const_cast<ControlNode *>(control));
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(node);
    // Wrapper nodes don't change control context, just recurse into child
    CollectAllTaskNodesWithContext(wrapper->child.get(), all_tasks,
                                   current_control_node);
  } else if (node->IsScheduleUnit()) {
    auto promote = static_cast<const ScheduleUnit *>(node);
    // Promote nodes don't change control context, just recurse into child
    CollectAllTaskNodesWithContext(promote->child.get(), all_tasks,
                                   current_control_node);
  } else {
    LOG(FATAL);
  }
}

} // namespace tl
} // namespace tvm
