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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/array.h>


#include <utility>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <optional>

#include "./common/attr.h"
#include "./common/collector.h"
#include "../op/builtin.h"
#include "auto_schedule.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

// Forward declaration
void ApplyWarpgroupPartition(ScheduleUnit& unit, IterVar thread_var);
std::unique_ptr<IRStructure> ApplyWarpgroupPartitionToIRStructure(IRStructure* root, IterVar thread_var);

// Helper function to compare if two regions are equal
bool RegionsEqual(const Region& a, const Region& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!tir::is_one(a[i]->min - b[i]->min) || !tir::is_one(a[i]->extent - b[i]->extent)) {
      return false;
    }
  }
  return true;
}

// Helper function to check if two ranges overlap
bool RangesOverlap(const Range& a, const Range& b) {
  // Two ranges [a_min, a_min+a_extent) and [b_min, b_min+b_extent) overlap if:
  // max(a_min, b_min) < min(a_min+a_extent, b_min+b_extent)
  // Since min and extent might be symbolic, we use arithmetic simplification
  // For simplicity, assume they are constants or can be compared
  // Use tir::is_zero to check if max(a_min, b_min) - min(a_min+a_extent, b_min+b_extent) < 0
  // Actually, we can check if they are provably non-overlapping
  // If we can't prove either way, assume they might overlap (conservative)

  // Simplify expressions
  auto analyzer = tvm::arith::Analyzer();
  PrimExpr a_min = analyzer.Simplify(a->min);
  PrimExpr a_extent = analyzer.Simplify(a->extent);
  PrimExpr b_min = analyzer.Simplify(b->min);
  PrimExpr b_extent = analyzer.Simplify(b->extent);

  // Compute a_end = a_min + a_extent, b_end = b_min + b_extent
  PrimExpr a_end = analyzer.Simplify(a_min + a_extent);
  PrimExpr b_end = analyzer.Simplify(b_min + b_extent);

  // Check if ranges are definitely disjoint
  // Case 1: a_end <= b_min
  if (tir::is_one(a_end <= b_min)) {
    return false;
  }
  // Case 2: b_end <= a_min
  if (tir::is_one(b_end <= a_min)) {
    return false;
  }

  // Otherwise, they may overlap (either proven overlap or unknown)
  return true;
}

// Helper function to check if two regions overlap
bool RegionsOverlap(const Region& a, const Region& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!RangesOverlap(a[i], b[i])) {
      return false;
    }
  }
  return true;
}





// Helper function to check if a buffer region is in register memory
bool IsRegisterRegion(const BufferRegion& region) {
  const Buffer& buffer = region->buffer;
  String scope = buffer.scope();
  MemoryType mem_type = GetMemoryTypeFromScope(scope);
  return mem_type == MemoryType::kRegister;
}

// Helper function to collect all register regions from a task
std::vector<BufferRegion> CollectRegisterRegions(const TaskNode* task) {
  std::vector<BufferRegion> reg_regions;
  // Check read regions
  for (const auto& region : task->GetReadRegions()) {
    if (IsRegisterRegion(region)) {
      reg_regions.push_back(region);
    }
  }
  // Check write regions
  for (const auto& region : task->GetWriteRegions()) {
    if (IsRegisterRegion(region)) {
      reg_regions.push_back(region);
    }
  }
  return reg_regions;
}

// Check if two TaskNodes use the same register region
// Used for warpgroup specialization: different warpgroups cannot share registers
bool UseSameRegisterRegion(const TaskNode* a, const TaskNode* b) {
  if (!a || !b) return false;

  auto reg_regions_a = CollectRegisterRegions(a);
  auto reg_regions_b = CollectRegisterRegions(b);

  // For each pair of register regions, check if they refer to the same buffer
  // and their regions overlap (conservative: if same buffer, assume overlap)
  for (const auto& region_a : reg_regions_a) {
    for (const auto& region_b : reg_regions_b) {
      // Check if same buffer
      if (region_a->buffer.same_as(region_b->buffer)) {
        // If same buffer, check if regions overlap
        if (RegionsOverlap(region_a->region, region_b->region)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Helper function to count register regions in a task
int CountRegisterRegions(const TaskNode* task) {
  auto reg_regions = CollectRegisterRegions(task);
  return static_cast<int>(reg_regions.size());
}



// Helper function to collect all TaskNodes with context information
void CollectAllTaskNodesWithContext(const IRStructure* node,
                                    std::vector<TaskNodeWithContext>& all_tasks,
                                    ControlNode* current_control_node) {
  if (!node) return;

  if (node->IsTask()) {
    TaskNode* task = const_cast<TaskNode*>(static_cast<const TaskNode*>(node));
    TaskNodeWithContext task_ctx;
    task_ctx.task = task;
    task_ctx.control_node = current_control_node;

    // Calculate tripcount if inside a loop
    if (current_control_node) {
      const ForNode* for_node = current_control_node->control.get();
      PrimExpr loop_extent = for_node->extent;
      // Try to convert loop_extent to int64_t
      if (const int64_t* extent_ptr = tir::as_const_int(loop_extent)) {
        task_ctx.tripcount = *extent_ptr;
      } else {
        // If extent is not constant, use 100 as default (as requested)
        task_ctx.tripcount = 100;
      }
    } else {
      task_ctx.tripcount = 1;  // Not inside a loop
    }

    all_tasks.push_back(task_ctx);
  } else if (node->IsSequence()) {
    const SequenceNode* seq = static_cast<const SequenceNode*>(node);
    for (const auto& child : seq->children) {
      CollectAllTaskNodesWithContext(child.get(), all_tasks, current_control_node);
    }
  } else if (node->IsControl()) {
    const ControlNode* control = static_cast<const ControlNode*>(node);
    // When entering a control node, update the current control context
    CollectAllTaskNodesWithContext(control->child.get(), all_tasks,
                                   const_cast<ControlNode*>(control));
  }
}

// SequenceNode member function implementations
bool SequenceNode::UsesCUDACore() const {
  for (const auto& child : children) {
    if (child && child->UsesCUDACore()) return true;
  }
  return false;
}

bool SequenceNode::UsesTMACore() const {
  for (const auto& child : children) {
    if (child && child->UsesTMACore()) return true;
  }
  return false;
}

bool SequenceNode::UsesTensorCore() const {
  for (const auto& child : children) {
    if (child && child->UsesTensorCore()) return true;
  }
  return false;
}

std::vector<BufferRegion> SequenceNode::GetReadRegions() const {
  std::vector<BufferRegion> all_read_regions;
  for (const auto& child : children) {
    if (child) {
      auto child_read_regions = child->GetReadRegions();
      all_read_regions.insert(all_read_regions.end(), child_read_regions.begin(), child_read_regions.end());
    }
  }
  // Remove duplicates (by buffer and region equality)
  // This is a simplified deduplication - in practice might need more sophisticated logic
  std::vector<BufferRegion> deduplicated;
  for (const auto& region : all_read_regions) {
    bool found = false;
    for (const auto& existing : deduplicated) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        found = true;
        break;
      }
    }
    if (!found) deduplicated.push_back(region);
  }
  return deduplicated;
}

std::vector<BufferRegion> SequenceNode::GetWriteRegions() const {
  std::vector<BufferRegion> all_write_regions;
  for (const auto& child : children) {
    if (child) {
      auto child_write_regions = child->GetWriteRegions();
      all_write_regions.insert(all_write_regions.end(), child_write_regions.begin(), child_write_regions.end());
    }
  }
  // Remove duplicates (by buffer and region equality)
  std::vector<BufferRegion> deduplicated;
  for (const auto& region : all_write_regions) {
    bool found = false;
    for (const auto& existing : deduplicated) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        found = true;
        break;
      }
    }
    if (!found) deduplicated.push_back(region);
  }
  return deduplicated;
}

int64_t SequenceNode::GetLatency() const {
  int64_t total = 0;
  for (const auto& child : children) {
    if (child) total += child->GetLatency();
  }
  return total;
}

int64_t SequenceNode::GetII() const {
  int64_t max_ii = 0;
  for (const auto& child : children) {
    if (child) max_ii = std::max(max_ii, child->GetII());
  }
  return max_ii;
}

void SequenceNode::SetUsesCUDACore(bool value) {
  for (auto& child : children) {
    if (child) child->SetUsesCUDACore(value);
  }
}

void SequenceNode::SetUsesTMACore(bool value) {
  for (auto& child : children) {
    if (child) child->SetUsesTMACore(value);
  }
}

void SequenceNode::SetUsesTensorCore(bool value) {
  for (auto& child : children) {
    if (child) child->SetUsesTensorCore(value);
  }
}

void SequenceNode::SetReadRegions(const std::vector<BufferRegion>& regions) {
  // Not clear what this means for SequenceNode - maybe set on first child?
  if (!children.empty() && children[0]) {
    children[0]->SetReadRegions(regions);
  }
}

void SequenceNode::SetWriteRegions(const std::vector<BufferRegion>& regions) {
  if (!children.empty() && children[0]) {
    children[0]->SetWriteRegions(regions);
  }
}

void SequenceNode::SetLatency(int64_t latency) {
  // Not clear how to distribute latency across children
  if (!children.empty() && children[0]) {
    children[0]->SetLatency(latency);
  }
}

void SequenceNode::SetII(int64_t ii) {
  // Not clear how to distribute II across children
  if (!children.empty() && children[0]) {
    children[0]->SetII(ii);
  }
}

void SequenceNode::AddReadRegion(const BufferRegion& region) {
  if (!children.empty() && children[0]) {
    children[0]->AddReadRegion(region);
  }
}

void SequenceNode::AddWriteRegion(const BufferRegion& region) {
  if (!children.empty() && children[0]) {
    children[0]->AddWriteRegion(region);
  }
}

std::unique_ptr<IRStructure> SequenceNode::Clone() const {
  auto new_seq = std::make_unique<SequenceNode>();
  new_seq->children.reserve(children.size());
  for (const auto& child : children) {
    if (child) {
      new_seq->children.push_back(child->Clone());
    } else {
      new_seq->children.push_back(nullptr);
    }
  }
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
  // Copy promote flag
  new_task->SetPromote(GetPromote());
  // Copy warpgroup id
  new_task->SetWarpgroupId(GetWarpgroupId());
  return new_task;
}

std::unique_ptr<IRStructure> ControlNode::Clone() const {
  auto new_ctrl = std::make_unique<ControlNode>();
  // Copy For control (For is a TVM object with reference counting)
  new_ctrl->control = control;
  // Clone child if exists
  if (child) {
    new_ctrl->child = child->Clone();
  }
  return new_ctrl;
}

// Global warpgroup id assignment - should be called from the top level
// Tasks that use the same register region must have the same warpgroup id
// Goal: balance weighted latency between two warpgroups (0 and 1)
// Weighted latency = latency * tripcount (tripcount = 100 for non-constant loop extent)
void AssignWarpgroupIdsGlobal(IRStructure* root) {
  if (!root) return;

  // Step 1: Collect all TaskNodes with context information
  std::vector<TaskNodeWithContext> all_tasks;
  CollectAllTaskNodesWithContext(root, all_tasks);

  if (all_tasks.empty()) return;

  int n = all_tasks.size();

  // Step 2: Initialize all warpgroup ids to -1 (unassigned)
  for (auto& task_ctx : all_tasks) {
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

  // Step 5: Calculate weighted latency for each component
  std::vector<ComponentInfo> component_infos;
  for (const auto& kv : components) {
    int root = kv.first;
    const std::vector<int>& indices = kv.second;
    int64_t total_weighted_latency = 0;
    for (int idx : indices) {
      int64_t latency = all_tasks[idx].task->GetLatency();
      int64_t tripcount = all_tasks[idx].tripcount;
      total_weighted_latency += latency * tripcount;
    }
    component_infos.push_back({root, total_weighted_latency, indices});
  }

  // Step 6: Sort components by weighted latency (descending)
  // Greedy assignment: assign larger components to the warpgroup with less current usage
  std::sort(component_infos.begin(), component_infos.end(),
            [](const ComponentInfo& a, const ComponentInfo& b) {
              return a.weighted_latency > b.weighted_latency;
            });

  // Step 7: Greedy assignment to balance weighted latency
  int64_t warpgroup0_latency = 0;
  int64_t warpgroup1_latency = 0;

  for (const auto& comp : component_infos) {
    int assigned_warpgroup = 0;
    if (warpgroup0_latency <= warpgroup1_latency) {
      assigned_warpgroup = 0;
      warpgroup0_latency += comp.weighted_latency;
    } else {
      assigned_warpgroup = 1;
      warpgroup1_latency += comp.weighted_latency;
    }

    // Assign warpgroup id to all tasks in this component
    for (int idx : comp.task_indices) {
      all_tasks[idx].task->SetWarpgroupId(assigned_warpgroup);
    }
  }

  // Log the assignment
  int count_warpgroup0 = 0;
  int count_warpgroup1 = 0;
  for (const auto& task_ctx : all_tasks) {
    if (task_ctx.task->GetWarpgroupId() == 0) count_warpgroup0++;
    else if (task_ctx.task->GetWarpgroupId() == 1) count_warpgroup1++;
  }

  LOG(INFO) << "Global warpgroup id assignment: " << count_warpgroup0
            << " tasks to warpgroup 0 (" << warpgroup0_latency
            << " weighted latency), " << count_warpgroup1
            << " tasks to warpgroup 1 (" << warpgroup1_latency
            << " weighted latency)";
}

// Assign warpgroup ids to tasks within a ScheduleUnit (legacy function, kept for compatibility)
// Tasks that use the same register region must have the same warpgroup id
// Goal: balance weighted latency between two warpgroups (0 and 1)
// Weighted latency = latency * tripcount if task is inside a loop, otherwise latency


// MemoryAccessDetector: detect read/write regions in statements
// Adapted from BlockReadWriteDetector in TVM
class MemoryAccessDetector : public StmtExprVisitor {
public:
  MemoryAccessDetector() = default;

  // Analyze a statement and collect read/write regions
  void Analyze(const Stmt& stmt) {
    read_buffers_.clear();
    write_buffers_.clear();
    read_regions_.clear();
    write_regions_.clear();
    dom_map_.clear();
    hint_map_.clear();
    pending_conditions_.clear();
    let_bindings_.clear();
    operator()(stmt);
  }

  // Return collected read regions
  std::vector<BufferRegion> GetReadRegions() const {
    return CollectRegions(read_buffers_, read_regions_);
  }

  // Return collected write regions
  std::vector<BufferRegion> GetWriteRegions() const {
    return CollectRegions(write_buffers_, write_regions_);
  }

private:
  /*! \brief Iteration range for loop_vars */
  std::unordered_map<const VarNode*, arith::IntSet> dom_map_;
  /*! \brief Extra iteration range hint for free vars */
  std::unordered_map<const VarNode*, arith::IntSet> hint_map_;
  /*! \brief Unresolved conditions within current scope. */
  std::vector<PrimExpr> pending_conditions_;
  /*! \brief The buffers that the current block reads */
  std::vector<Buffer> read_buffers_;
  /*! \brief The buffers that the current block writes */
  std::vector<Buffer> write_buffers_;
  /*! \brief The read regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> read_regions_;
  /*! \brief The write regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> write_regions_;
  /*!\ brief Internal analyzer. */
  arith::Analyzer ana_;
  /*! \brief let bindings inside the block */
  std::unordered_map<const VarNode*, PrimExpr> let_bindings_;

  /*!
   * \brief Update read/write buffers and regions with provided buffer and region
   */
  void Update(std::vector<Buffer>* buffers,
              std::vector<std::vector<arith::IntSet>>* regions,
              Buffer buffer,
              std::vector<arith::IntSet> region) {
    // Check if buffer already exists
    for (size_t i = 0; i < buffers->size(); ++i) {
      if ((*buffers)[i].same_as(buffer)) {
        // Merge regions
        ICHECK_EQ((*regions)[i].size(), region.size());
        for (size_t j = 0; j < region.size(); ++j) {
          (*regions)[i][j] = arith::Union({(*regions)[i][j], region[j]});
        }
        return;
      }
    }
    // New buffer
    buffers->push_back(buffer);
    regions->push_back(region);
  }

  /*!
   * \brief Process a buffer region argument from reduce operation
   * \param arg The argument which could be BufferRegion, BufferLoad, or tl.tileop.region call
   * \param is_read Whether this is a read (true) or write (false) access
   */
  void ProcessBufferRegion(const PrimExpr& arg, bool is_read) {
    // Check if it's a BufferRegion
    if (const auto* buffer_region = arg.as<BufferRegionNode>()) {
      Buffer buffer = buffer_region->buffer;
      const Region& region = buffer_region->region;
      std::vector<arith::IntSet> int_sets;
      int_sets.reserve(region.size());
      for (const auto& range : region) {
        // Create IntSet for range [min, min + extent)
        int_sets.push_back(arith::IntSet::FromRange(
            Range::FromMinExtent(range->min, range->extent)));
      }
      if (is_read) {
        Update(&read_buffers_, &read_regions_, buffer, int_sets);
      } else {
        Update(&write_buffers_, &write_regions_, buffer, int_sets);
      }
      return;
    }

    // Check if it's a BufferLoad
    if (const auto* buffer_load = arg.as<BufferLoadNode>()) {
      Buffer buffer = buffer_load->buffer;
      std::vector<arith::IntSet> int_sets;
      int_sets.reserve(buffer_load->indices.size());
      for (PrimExpr index : buffer_load->indices) {
        // Create IntSet for single point
        int_sets.push_back(RelaxAccessIndex(index));
      }
      if (is_read) {
        Update(&read_buffers_, &read_regions_, buffer, int_sets);
      } else {
        Update(&write_buffers_, &write_regions_, buffer, int_sets);
      }
      return;
    }

    // Check if it's a tl.tileop.region call (should already be handled by VisitExpr_)
    // but we can still process it recursively
    if (const auto* call = arg.as<CallNode>()) {
      static const auto region_op = Op::Get("tl.tileop.region");
      if (call->op.same_as(region_op)) {
        // Recursively visit this call to handle it
        VisitExpr_(call);
        return;
      }
    }

    // If we reach here, the argument type is not supported
    LOG(WARNING) << "Unsupported argument type in tl.tileop.reduce: " << arg->GetTypeKey();
  }

  /*! \brief Helper function to collect access regions. */
  std::vector<BufferRegion> CollectRegions(
      const std::vector<Buffer>& buffers,
      const std::vector<std::vector<tvm::arith::IntSet>>& regions) const {
    std::vector<BufferRegion> result;
    result.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      const Buffer& buffer = buffers[i];
      const std::vector<arith::IntSet>& int_sets = regions[i];
      Region region;
      size_t ndim = buffer->shape.size();
      size_t region_ndim = int_sets.size();

      // Assert that region dimension equals buffer dimension
      ICHECK_EQ(region_ndim, ndim) << "Region dimension " << region_ndim
                                   << " must equal buffer dimension " << ndim;

      region.reserve(ndim);
      for (size_t j = 0; j < ndim; ++j) {
        const tvm::arith::IntSet& int_set = int_sets[j];
        region.push_back(int_set.CoverRange(Range::FromMinExtent(0, buffer->shape[j])));
      }

      result.push_back(BufferRegion(buffer, region));
    }
    return result;
  }

  /*! \brief Helper function to relax the buffer indices */
  arith::IntSet RelaxAccessIndex(const PrimExpr& index) {
    PrimExpr current = index;
    PrimExpr remapped = Substitute(current, let_bindings_);
    while (!remapped.same_as(current)) {
      current = remapped;
      remapped = Substitute(current, let_bindings_);
    }
    return arith::EvalSet(arith::IntSet::Vector(current), dom_map_);
  }

  void operator()(const Stmt& stmt) {
    StmtExprVisitor::operator()(stmt);
  }

  void VisitStmt_(const ForNode* op) override {
    Range range = Range::FromMinExtent(op->min, op->extent);
    dom_map_[op->loop_var.get()] = arith::IntSet::FromRange(range);
    StmtVisitor::VisitStmt_(op);
    dom_map_.erase(op->loop_var.get());
  }

  void VisitStmt_(const IfThenElseNode* op) override {
    VisitExpr(op->condition);
    {
      // Visit then branch
      // Simplified: we don't handle conditional bounds for now
      StmtExprVisitor::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      // Visit else branch
      StmtExprVisitor::VisitStmt(op->else_case.value());
    }
  }

  void VisitStmt_(const LetStmtNode* op) override {
    let_bindings_[op->var.get()] = op->value;
    StmtVisitor::VisitStmt_(op);
    let_bindings_.erase(op->var.get());
  }

  void VisitExpr_(const BufferLoadNode* op) override {
    std::vector<arith::IntSet> relaxed_region;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->buffer->shape.size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim) << "BufferLoad indices count " << num_indices
                                        << " must equal buffer dimension " << buffer_ndim;

    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
    }
    Update(&read_buffers_, &read_regions_, op->buffer, relaxed_region);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) override {
    std::vector<arith::IntSet> relaxed_region;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->buffer->shape.size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim) << "BufferStore indices count " << num_indices
                                        << " must equal buffer dimension " << buffer_ndim;

    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
    }
    Update(&write_buffers_, &write_regions_, op->buffer, relaxed_region);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) override {
    static const auto region_op = Op::Get("tl.tileop.region");
    static const auto reduce_op = Op::Get("tl.tileop.reduce");

    // Check for tl.tileop.region call
    if (op->op.same_as(region_op)) {
      // Handle tl.tileop.region call for memory access analysis
      // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2: write, 3: read/write)
      // args[2..] = extents
      if (op->args.size() >= 2) {
        // Extract access type
        const auto* access_int = op->args[1].as<IntImmNode>();
        ICHECK(access_int);
        int access_type = access_int->value;

        // Extract buffer from BufferLoad
        if (const auto* buffer_load = op->args[0].as<BufferLoadNode>()) {
          Buffer buffer = buffer_load->buffer;
          std::vector<arith::IntSet> relaxed_region;

          // Assert that BufferLoad accesses a single element (no Ramp indices)
          for (size_t i = 0; i < buffer_load->indices.size(); ++i) {
            const PrimExpr& index = buffer_load->indices[i];
            // Check if index is a Ramp (vector access)
            if (index.as<RampNode>()) {
              LOG(FATAL) << "BufferLoad in tl.tileop.region should access a single element, "
                         << "but found Ramp index at dimension " << i;
            }
          }

          // Use provided extents if available, otherwise use buffer load indices
          size_t num_indices = buffer_load->indices.size();
          size_t buffer_ndim = buffer->shape.size();

          // Assert that indices count equals buffer dimension
          ICHECK_EQ(num_indices, buffer_ndim) << "BufferLoad indices count " << num_indices
                                              << " must equal buffer dimension " << buffer_ndim;

          if (op->args.size() > 2) {
            // args[2..] are extents for the region
            // Number of extents provided
            size_t num_extents = op->args.size() - 2;

            // Assert that extents count equals indices count
            ICHECK_EQ(num_extents, num_indices) << "Extents count " << num_extents
                                                << " must equal indices count " << num_indices;

            relaxed_region.reserve(num_indices);
            for (size_t i = 0; i < num_indices; ++i) {
              PrimExpr min = buffer_load->indices[i];
              PrimExpr extent = op->args[2 + i];

              // Create IntSet for range [min, min + extent)
              relaxed_region.push_back(arith::IntSet::FromRange(
                  Range::FromMinExtent(min, extent)));
            }
          } else {
            // No extents provided: each dimension is a single point at the index
            for (PrimExpr index : buffer_load->indices) {
              // Create IntSet for single point
              relaxed_region.push_back(RelaxAccessIndex(index));
            }
          }

          // Add to appropriate list based on access type
          if (access_type == 1 || access_type == 3) {  // read or read/write
            Update(&read_buffers_, &read_regions_, buffer, relaxed_region);
          }
          if (access_type == 2 || access_type == 3) {  // write or read/write
            Update(&write_buffers_, &write_regions_, buffer, relaxed_region);
          }
        } else {
          LOG(FATAL) << "First argument of tl.tileop.region should be a BufferLoad";
        }
      }
      return;
    }

    // Check for tl.tileop.reduce call
    if (op->op.same_as(reduce_op)) {
      // Handle tl.tileop.reduce call for memory access analysis
      // args[0] = input buffer region (read)
      // args[1] = output buffer region (write)
      // args[2] = reduce_type (string)
      // args[3] = dim (int)
      // args[4] = clear (bool)
      if (op->args.size() >= 2) {
        // Process first argument as read region
        ProcessBufferRegion(op->args[0], true);  // is_read = true
        // Process second argument as write region
        ProcessBufferRegion(op->args[1], false); // is_read = false
      }
      return;
    }

    // Handle other calls (e.g., builtin::tvm_access_ptr)
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      // Simplified: skip for now
      StmtExprVisitor::VisitExpr_(op);
      return;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  // Skip block-specific handling for now
  void VisitStmt_(const BlockRealizeNode* op) override {
    // Don't visit child blocks recursively
  }
};


// Latency estimator for H100 GPU
class LatencyEstimator {
public:
  // H100 latency parameters (in cycles)
  struct H100Params {
    // Base latencies
    int64_t global_memory_read = 400;   // Global memory read latency
    int64_t global_memory_write = 200;  // Global memory write latency (usually lower)
    int64_t shared_memory_read = 20;    // Shared memory read latency
    int64_t shared_memory_write = 20;   // Shared memory write latency
    int64_t register_access = 1;        // Register access latency
    int64_t cuda_core_operation = 4;    // Basic CUDA core operation (add, mul, etc.)
    int64_t tensor_core_operation = 64; // Tensor core operation (matrix multiply)
    int64_t tma_operation = 100;        // TMA operation latency

    // Bandwidth parameters (bytes per cycle)
    // H100: ~2TB/s global memory, 1.8GHz clock → ~1111 bytes/cycle
    // H100: ~19TB/s shared memory → ~10556 bytes/cycle
    int64_t global_memory_bandwidth = 1111;   // bytes per cycle
    int64_t shared_memory_bandwidth = 10556;  // bytes per cycle

    // Pipeline initiation capabilities
    int64_t max_memory_ops_per_cycle = 1;     // Max memory ops that can start per cycle
  };

  LatencyEstimator() = default;

  // Estimate latency for a TaskNode
  void Estimate(TaskNode* task) {
    int64_t total_latency = 0;
    int64_t memory_latency = 0;
    int64_t compute_latency = 0;

    // Count memory operations and track bytes by memory type
    int num_memory_ops = 0;
    int64_t global_memory_bytes = 0;
    int64_t shared_memory_bytes = 0;
    int64_t register_bytes = 0;

    // Estimate latency from memory accesses and track bandwidth usage
    for (const auto& region : task->GetReadRegions()) {
      int64_t region_latency = EstimateMemoryAccessLatency(region, true); // read
      memory_latency += region_latency;
      num_memory_ops++;

      // Track bandwidth usage by memory type
      const Buffer& buffer = region->buffer;
      String scope = buffer.scope();
      MemoryType mem_type = GetMemoryTypeFromScope(scope);
      int64_t access_bytes = CalculateAccessBytes(region);

      switch (mem_type) {
        case MemoryType::kGlobal:
          global_memory_bytes += access_bytes;
          break;
        case MemoryType::kShared:
          shared_memory_bytes += access_bytes;
          break;
        case MemoryType::kRegister:
          register_bytes += access_bytes;
          break;
        default:
          global_memory_bytes += access_bytes; // Conservative
          break;
      }
    }

    for (const auto& region : task->GetWriteRegions()) {
      int64_t region_latency = EstimateMemoryAccessLatency(region, false); // write
      memory_latency += region_latency;
      num_memory_ops++;

      // Track bandwidth usage by memory type
      const Buffer& buffer = region->buffer;
      String scope = buffer.scope();
      MemoryType mem_type = GetMemoryTypeFromScope(scope);
      int64_t access_bytes = CalculateAccessBytes(region);

      switch (mem_type) {
        case MemoryType::kGlobal:
          global_memory_bytes += access_bytes;
          break;
        case MemoryType::kShared:
          shared_memory_bytes += access_bytes;
          break;
        case MemoryType::kRegister:
          register_bytes += access_bytes;
          break;
        default:
          global_memory_bytes += access_bytes; // Conservative
          break;
      }
    }

    // Estimate compute latency based on resource usage
    if (task->UsesCUDACore()) {
      // Simple heuristic: assume some number of CUDA operations
      // For now, assume 1 operation per statement as a rough estimate
      compute_latency = params_.cuda_core_operation * std::max(1, static_cast<int>(task->stmts.size()));
    }

    if (task->UsesTensorCore()) {
      compute_latency = std::max(compute_latency, params_.tensor_core_operation);
    }

    if (task->UsesTMACore()) {
      compute_latency = std::max(compute_latency, params_.tma_operation);
    }

    // Total latency is sum of memory and compute (assuming sequential for now)
    total_latency = memory_latency + compute_latency;

    // Calculate initiation interval (II)
    int64_t ii = 1;  // Default minimum II

    if (task->UsesTMACore()) {
      // TMA operations (async memory copies): instruction latency can be hidden
      // II is determined by bandwidth constraints only
      if (global_memory_bytes > 0) {
        int64_t bandwidth_ii = (global_memory_bytes + params_.global_memory_bandwidth - 1) / params_.global_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
      if (shared_memory_bytes > 0) {
        int64_t bandwidth_ii = (shared_memory_bytes + params_.shared_memory_bandwidth - 1) / params_.shared_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
    } else {
      // Regular operations
      // According to requirements:
      // 1. If there's only one operation and it's a memory access, II = memory latency
      // 2. Otherwise, II = total latency
      ii = total_latency;

      if (num_memory_ops == 1 && task->stmts.size() == 1) {
        // Single operation that is a memory access
        // Check if this is likely a memory operation (has read/write regions)
        if (!task->GetReadRegions().empty() || !task->GetWriteRegions().empty()) {
          ii = memory_latency;
        }
      }

      // Additional II constraints from bandwidth limitations
      // II must be at least the time needed to transfer data based on bandwidth
      if (global_memory_bytes > 0) {
        int64_t bandwidth_ii = (global_memory_bytes + params_.global_memory_bandwidth - 1) / params_.global_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }

      if (shared_memory_bytes > 0) {
        int64_t bandwidth_ii = (shared_memory_bytes + params_.shared_memory_bandwidth - 1) / params_.shared_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
    }

    // II must be at least 1 cycle
    if (ii < 1) ii = 1;

    // Store results in task node
    task->SetLatency(total_latency);
    task->SetII(ii);
  }

private:
  H100Params params_;

  // Helper function to calculate total bytes accessed in a region
  int64_t CalculateAccessBytes(const BufferRegion& region) {
    const Buffer& buffer = region->buffer;
    const Region& ranges = region->region;

    // Calculate total number of elements
    int64_t total_elements = 1;
    for (const auto& range : ranges) {
      // Try to get constant extent if possible
      if (const auto* extent_int = range->extent.as<IntImmNode>()) {
        total_elements *= extent_int->value;
      } else {
        // For non-constant extents, use a conservative estimate
        // Assume at least 1 element per dimension
        // TODO: Better estimation for symbolic extents
        total_elements *= 1;
      }
    }

    // Get data type size in bytes
    DataType dtype(buffer->dtype);
    int64_t element_size = dtype.bytes();

    return total_elements * element_size;
  }

  // Estimate latency for a single memory access
  int64_t EstimateMemoryAccessLatency(const BufferRegion& region, bool is_read) {
    const Buffer& buffer = region->buffer;
    String scope = buffer.scope();
    MemoryType mem_type = GetMemoryTypeFromScope(scope);

    int64_t access_bytes = CalculateAccessBytes(region);

    switch (mem_type) {
      case MemoryType::kGlobal:
        // Global memory latency depends on data size
        // Base latency + bandwidth-limited component
        // Latency = base_latency + bytes / bytes_per_cycle
        // Subtract cache line size (32 bytes) since first cache line has base latency
        if (is_read) {
          // Base read latency + bandwidth component
          return params_.global_memory_read + std::max(0L, (access_bytes - 32) / params_.global_memory_bandwidth);
        } else {
          // Write latency usually lower
          return params_.global_memory_write + std::max(0L, (access_bytes - 32) / params_.global_memory_bandwidth);
        }
      case MemoryType::kShared:
        // Shared memory has high bandwidth, less sensitive to size
        // Subtract typical burst size (128 bytes) for base latency
        if (is_read) {
          return params_.shared_memory_read + std::max(0L, (access_bytes - 128) / params_.shared_memory_bandwidth);
        } else {
          return params_.shared_memory_write + std::max(0L, (access_bytes - 128) / params_.shared_memory_bandwidth);
        }
      case MemoryType::kRegister:
        // Register access latency is constant and very small
        return params_.register_access;
      default:
        // Unknown memory type, use global memory as conservative estimate
        if (is_read) {
          return params_.global_memory_read + std::max(0L, (access_bytes - 32) / params_.global_memory_bandwidth);
        } else {
          return params_.global_memory_write + std::max(0L, (access_bytes - 32) / params_.global_memory_bandwidth);
        }
    }
  }
};

// Builder that collects ScheduleUnits from IRStructure
class ScheduleUnitBuilder {
public:
  std::vector<ScheduleUnit> Build(IRStructure* root) {
    units_.clear();
    current_unit_ = nullptr;
    current_seq_ = nullptr;
    current_child_idx_ = 0;
    control_nesting_depth_ = 0;

    // Global warpgroup id assignment from the top level
    // This ensures register region constraints are respected across all ScheduleUnits
    LOG(INFO) << "Performing global warpgroup id assignment...";
    AssignWarpgroupIdsGlobal(root);

    Collect(root);
    // Finalize the last unit if any
    FinalizeCurrentUnit();

    // Apply scheduling (Z3-based) to each unit and update IRStructure
    for (auto& unit : units_) {
      ApplyScheduling(unit);
    }

    return std::move(units_);
  }

  // Z3-based scheduler that calls Python implementation via FFI
  std::vector<TaskNode*> Z3SchedulePython(const std::vector<TaskNode*>& tasks) {
    size_t n = tasks.size();
    if (n <= 1) {
      if (n == 1) {
        tasks[0]->SetStartTime(0);
      }
      return tasks;
    }

    LOG(INFO) << "Z3SchedulePython: Starting Python-based scheduling for " << n << " tasks";

    try {
      // Get the Python-registered function using ffi::Function::GetGlobal
      static std::optional<tvm::ffi::Function> z3_schedule_func =
          tvm::ffi::Function::GetGlobal("tl.transform.z3_schedule_python");
      if (!z3_schedule_func.has_value()) {
        LOG(WARNING) << "Python Z3 scheduler not registered, falling back to topological sort";
        return tasks;
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
        const TaskNode* task = tasks[i];
        latencies.push_back(task->GetLatency());
        iis.push_back(task->GetII());

        // Encode resource flags as bitmask
        int64_t flags = 0;
        if (task->UsesCUDACore()) flags |= 1;
        if (task->UsesTMACore()) flags |= 2;
        if (task->UsesTensorCore()) flags |= 4;
        resource_flags.push_back(flags);
      }

      // Collect data dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasDependency(tasks[i], tasks[j])) {
            data_deps.emplace_back(i, j);
          }
        }
      }

      // Collect resource dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasResourceDependency(tasks[i], tasks[j])) {
            resource_deps.emplace_back(i, j);
          }
        }
      }

      LOG(INFO) << "Z3SchedulePython: Calling Python function with "
                << data_deps.size() << " data dependencies and "
                << resource_deps.size() << " resource dependencies";

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
      for (const auto& dep : data_deps) {
        ffi::Array<int64_t> pair;
        pair.push_back(dep.first);
        pair.push_back(dep.second);
        tvm_data_deps.push_back(pair);
      }
      for (const auto& dep : resource_deps) {
        ffi::Array<int64_t> pair;
        pair.push_back(dep.first);
        pair.push_back(dep.second);
        tvm_resource_deps.push_back(pair);
      }

      // Extract results
      // Python function returns only start_times, C++ side will sort by start_time
      auto start_times = z3_schedule_func.value()(tvm_latencies, tvm_iis, tvm_resource_flags,
                                              tvm_data_deps, tvm_resource_deps).cast<ffi::Array<int64_t>>();

      if (start_times.size() != n) {
        LOG(WARNING) << "Python Z3 scheduler returned invalid results (size mismatch), falling back to topological sort";
        return tasks;
      }

      // Apply start times to tasks
      for (size_t i = 0; i < n; ++i) {
        tasks[i]->SetStartTime(start_times[i]);
      }

      // Create sorted task list based on start_time (and original index as tie-breaker)
      std::vector<std::pair<int64_t, size_t>> start_time_with_idx;
      start_time_with_idx.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        start_time_with_idx.emplace_back(start_times[i], i);
      }

      // Sort by start_time, then by original index
      std::sort(start_time_with_idx.begin(), start_time_with_idx.end(),
                [](const std::pair<int64_t, size_t>& a,
                   const std::pair<int64_t, size_t>& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;
                });

      // Create sorted task list
      std::vector<TaskNode*> sorted_tasks;
      sorted_tasks.reserve(n);
      for (const auto& p : start_time_with_idx) {
        sorted_tasks.push_back(tasks[p.second]);
      }

      LOG(INFO) << "Z3SchedulePython: Python scheduling completed successfully";
      return sorted_tasks;

    } catch (const std::exception& e) {
      LOG(WARNING) << "Python Z3 scheduler failed with exception: " << e.what()
                   << ", falling back to topological sort";
      return tasks;
    } catch (...) {
      LOG(WARNING) << "Python Z3 scheduler failed with unknown exception, falling back to topological sort";
      return tasks;
    }
  }

  // Z3-based scheduler for loops that calls Python implementation via FFI
  // with distance-aware dependencies
  std::vector<TaskNode*> Z3SchedulePythonLoop(const std::vector<TaskNode*>& tasks) {
    size_t n = tasks.size();
    if (n <= 1) {
      if (n == 1) {
        tasks[0]->SetStartTime(0);
      }
      return tasks;
    }

    LOG(INFO) << "Z3SchedulePythonLoop: Starting Python-based loop scheduling for " << n << " tasks";

    try {
      // Get the Python-registered function using ffi::Function::GetGlobal
      static std::optional<tvm::ffi::Function> z3_schedule_loop_func =
          tvm::ffi::Function::GetGlobal("tl.transform.z3_schedule_loop_python");
      if (!z3_schedule_loop_func.has_value()) {
        LOG(WARNING) << "Python Z3 loop scheduler not registered, falling back to topological sort";
        return tasks;
      }

      // Prepare input data
      std::vector<int64_t> latencies;
      std::vector<int64_t> iis;
      std::vector<int64_t> resource_flags;
      std::vector<std::tuple<int64_t, int64_t, int64_t>> data_deps;  // (i, j, distance)
      std::vector<std::pair<int64_t, int64_t>> resource_deps;

      latencies.reserve(n);
      iis.reserve(n);
      resource_flags.reserve(n);

      for (size_t i = 0; i < n; ++i) {
        const TaskNode* task = tasks[i];
        latencies.push_back(task->GetLatency());
        iis.push_back(task->GetII());

        // Encode resource flags as bitmask
        int64_t flags = 0;
        if (task->UsesCUDACore()) flags |= 1;
        if (task->UsesTMACore()) flags |= 2;
        if (task->UsesTensorCore()) flags |= 4;
        resource_flags.push_back(flags);
      }

      // Collect data dependencies with distance
      // distance = 0 if i < j (same iteration), distance = 1 if i > j (next iteration)
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          if (i == j) continue;  // Skip self-dependency
          if (HasDependency(tasks[i], tasks[j])) {
            // distance = 0 if i < j, 1 if i > j
            int64_t distance = (i < j) ? 0 : 1;
            data_deps.emplace_back(i, j, distance);
          }
        }
      }

      // Collect resource dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasResourceDependency(tasks[i], tasks[j])) {
            resource_deps.emplace_back(i, j);
          }
        }
      }

      LOG(INFO) << "Z3SchedulePythonLoop: Calling Python loop scheduler with "
                << data_deps.size() << " data dependencies and "
                << resource_deps.size() << " resource dependencies";

      // Convert vectors to TVM containers
      ffi::Array<int64_t> tvm_latencies;
      ffi::Array<int64_t> tvm_iis;
      ffi::Array<int64_t> tvm_resource_flags;
      ffi::Array<ffi::Array<int64_t>> tvm_data_deps;  // each element is [i, j, distance]
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
      for (const auto& dep : data_deps) {
        ffi::Array<int64_t> triple;
        triple.push_back(std::get<0>(dep));
        triple.push_back(std::get<1>(dep));
        triple.push_back(std::get<2>(dep));
        tvm_data_deps.push_back(triple);
      }
      for (const auto& dep : resource_deps) {
        ffi::Array<int64_t> pair;
        pair.push_back(dep.first);
        pair.push_back(dep.second);
        tvm_resource_deps.push_back(pair);
      }

      // Extract results
      // Python function returns (start_times, promotes) as a tuple of two arrays
      auto return_val = z3_schedule_loop_func.value()(tvm_latencies, tvm_iis, tvm_resource_flags,
                                                  tvm_data_deps, tvm_resource_deps).cast<ffi::Tuple<ffi::Array<int64_t>, ffi::Array<bool>>>();

      ffi::Array<int64_t> start_times = return_val.get<0>();
      ffi::Array<bool> promotes = return_val.get<1>();

      if (start_times.size() != n || promotes.size() != n) {
        LOG(WARNING) << "Python Z3 loop scheduler returned invalid results (size mismatch), falling back to topological sort";
        return tasks;
      }

      // Apply start times and promote flags to tasks
      size_t num_promoted = 0;
      for (size_t i = 0; i < n; ++i) {
        tasks[i]->SetStartTime(start_times[i]);
        bool promote = promotes[i] != 0;  // Convert int to bool
        tasks[i]->SetPromote(promote);
        if (promote) {
          num_promoted++;
        }
      }
      LOG(INFO) << "Z3SchedulePythonLoop: " << num_promoted << " tasks marked for promotion";
      if (num_promoted > 0) {
        LOG(INFO) << "Z3SchedulePythonLoop: Promote flags can be used for loop transformation (software pipelining)";
        // Promote transformation will be applied in IRStructureRewriter
        // For promoted tasks, they should be executed in previous iteration
        // Loop bounds will be extended by step (not II) as per user requirement
        // Example transformation:
        // for i = start to end:
        //   Task0 (not promoted)
        //   Task1 (promoted)
        // =>
        // for i = start to end+step:
        //   if i > start: Task1 with index i-step
        //   if i < end+step: Task0 with index i
      }

      // Create sorted task list based on start_time (and original index as tie-breaker)
      std::vector<std::pair<int64_t, size_t>> start_time_with_idx;
      start_time_with_idx.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        start_time_with_idx.emplace_back(start_times[i], i);
      }

      // Sort by start_time, then by original index
      std::sort(start_time_with_idx.begin(), start_time_with_idx.end(),
                [](const std::pair<int64_t, size_t>& a,
                   const std::pair<int64_t, size_t>& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;
                });

      // Create sorted task list
      std::vector<TaskNode*> sorted_tasks;
      sorted_tasks.reserve(n);
      for (const auto& p : start_time_with_idx) {
        sorted_tasks.push_back(tasks[p.second]);
      }

      LOG(INFO) << "Z3SchedulePythonLoop: Python loop scheduling completed successfully";
      return sorted_tasks;

    } catch (const std::exception& e) {
      LOG(WARNING) << "Python Z3 loop scheduler failed with exception: " << e.what()
                   << ", falling back to topological sort";
      return tasks;
    } catch (...) {
      LOG(WARNING) << "Python Z3 loop scheduler failed with unknown exception, falling back to topological sort";
      return tasks;
    }
  }

  // Apply promote transformation for software pipelining
  // This transforms loops with promoted tasks according to the pattern:
  // Original: for i = start to end: Task0 (promoted), Task1 (non-promoted)
  // Transformed: for i = start to end+step:
  //   if i > start: Task1 with i-step (non-promoted, adjusted index)
  //   if i < end+step: Task0 with i (promoted, original index)
  void ApplyPromoteTransformation(std::vector<TaskNode*>& tasks, ControlNode* control_node) {
    if (tasks.empty() || !control_node) {
      return;
    }

    // Count promoted tasks
    size_t num_promoted = 0;
    std::vector<TaskNode*> promoted_tasks;
    std::vector<TaskNode*> non_promoted_tasks;

    for (TaskNode* task : tasks) {
      if (task->GetPromote()) {
        num_promoted++;
        promoted_tasks.push_back(task);
      } else {
        non_promoted_tasks.push_back(task);
      }
    }

    if (num_promoted == 0) {
      LOG(INFO) << "ApplyPromoteTransformation: No promoted tasks, skipping transformation";
      return;
    }

    LOG(INFO) << "ApplyPromoteTransformation: Applying promote transformation for "
              << num_promoted << " promoted tasks out of " << tasks.size() << " total tasks";

    // Get loop step from control_node->control
    // ForNode has a 'step' member (ffi::Optional<PrimExpr>, default is 1)
    const ForNode* for_node = control_node->control.get();
    PrimExpr loop_step_expr = for_node->step.has_value() ? for_node->step.value() : IntImm(DataType::Int(32), 1);

    // Get step expression (no simplification)

    // Try to get constant step value for logging
    int64_t loop_step_value = 1;
    if (const auto* int_imm = loop_step_expr.as<IntImmNode>()) {
      loop_step_value = int_imm->value;
    } else {
      LOG(WARNING) << "ApplyPromoteTransformation: Loop step is not constant: " << loop_step_expr
                   << ", using default step = 1";
      loop_step_value = 1;
    }

    LOG(INFO) << "ApplyPromoteTransformation: Loop step = " << loop_step_value;

    // Get loop bounds from control_node->control
    PrimExpr loop_var = for_node->loop_var;
    PrimExpr loop_start = for_node->min;
    PrimExpr loop_extent = for_node->extent;

    // Calculate loop end
    PrimExpr loop_end = loop_start + loop_extent;

    LOG(INFO) << "ApplyPromoteTransformation: Loop bounds: var=" << loop_var
              << ", start=" << loop_start << ", extent=" << loop_extent
              << ", end=" << loop_end << ", step=" << loop_step_value;

    // Create new loop with extended bounds: end = loop_end + loop_step_expr
    PrimExpr new_loop_end = loop_end + loop_step_expr;
    LOG(INFO) << "ApplyPromoteTransformation: New loop end = " << new_loop_end
              << " (original end " << loop_end << " + step " << loop_step_value << ")";

    // Actually apply the promote transformation:
    // 1. Update ControlNode's loop bounds: extent = loop_extent + loop_step_expr
    // 2. For each promoted task: add condition if (loop_var < loop_end)
    // 3. For each non-promoted task: add condition if (loop_var > loop_start) and substitute loop_var with (loop_var - loop_step_expr)

    LOG(INFO) << "ApplyPromoteTransformation: Actually applying promote transformation";
    LOG(INFO) << "  - Promoted tasks: " << promoted_tasks.size();
    LOG(INFO) << "  - Non-promoted tasks: " << non_promoted_tasks.size();
    LOG(INFO) << "  - Loop extension: end + " << loop_step_value << " (step)";
    LOG(INFO) << "  - Promoted tasks execute with index: i (original index)";
    LOG(INFO) << "  - Non-promoted tasks execute with index: i - " << loop_step_value;

    // 1. Update ControlNode's loop bounds
    PrimExpr new_loop_extent = loop_extent + loop_step_expr;
    control_node->control.CopyOnWrite()->extent = new_loop_extent;
    LOG(INFO) << "ApplyPromoteTransformation: Updated ControlNode loop extent from " << loop_extent << " to " << new_loop_extent;

    // 2. Process promoted tasks
    for (TaskNode* task : promoted_tasks) {
      if (task->stmts.empty()) {
        continue;
      }

      // Combine multiple statements into SeqStmt if needed
      Stmt task_body;
      if (task->stmts.size() == 1) {
        task_body = task->stmts[0];
      } else {
        task_body = SeqStmt(task->stmts);
      }

      // Create condition: loop_var < loop_end (original loop boundary)
      PrimExpr condition = loop_var < loop_end;

      // Create conditional statement
      Stmt conditional_stmt = IfThenElse(condition, task_body);

      // Replace task statements with the conditional statement
      task->stmts.clear();
      task->stmts.push_back(conditional_stmt);

      LOG(INFO) << "ApplyPromoteTransformation: Added condition " << condition << " to promoted task";
    }

    // 3. Process non-promoted tasks
    for (TaskNode* task : non_promoted_tasks) {
      if (task->stmts.empty()) {
        continue;
      }

      // Combine multiple statements into SeqStmt if needed
      Stmt task_body;
      if (task->stmts.size() == 1) {
        task_body = task->stmts[0];
      } else {
        task_body = SeqStmt(task->stmts);
      }

      // Apply variable substitution: loop_var -> loop_var - loop_step_expr
      Map<Var, PrimExpr> substitution;
      substitution.Set(for_node->loop_var, loop_var - loop_step_expr);
      task_body = Substitute(task_body, substitution);

      // Create condition: loop_var > loop_start (i.e., not in first iteration)
      PrimExpr condition = loop_var > loop_start;

      // Create conditional statement
      Stmt conditional_stmt = IfThenElse(condition, task_body);

      // Replace task statements with the conditional statement
      task->stmts.clear();
      task->stmts.push_back(conditional_stmt);

      LOG(INFO) << "ApplyPromoteTransformation: Added condition " << condition
                << " and substituted " << loop_var << " -> " << loop_var << " - " << loop_step_expr
                << " to non-promoted task";
    }

    LOG(INFO) << "ApplyPromoteTransformation: Promote transformation completed successfully";
  }

  // Set thread index variable for warpgroup partition
  void SetThreadVar(IterVar thread_var) {
    thread_var_ = thread_var;
  }


private:
  std::vector<ScheduleUnit> units_;
  ScheduleUnit* current_unit_{nullptr};
  IterVar thread_var_;  // Thread index variable for warpgroup partition
  SequenceNode* current_seq_{nullptr};
  size_t current_child_idx_{0};
  int control_nesting_depth_{0};
  ControlNode* current_control_node_{nullptr};  // Track current ControlNode for promote transformation

  // Check if two regions refer to the same buffer
  bool SameBuffer(const BufferRegion& a, const BufferRegion& b) const {
    return a->buffer.same_as(b->buffer);
  }

  // Check if two TaskNodes have data dependency (excluding read-after-read)
  bool HasDependency(const TaskNode* a, const TaskNode* b) const {
    // Check all combinations of accesses
    // a writes, b reads (RAW)
    // a reads, b writes (WAR)
    // a writes, b writes (WAW)
    // a reads, b reads (RAR) - no dependency

    // For simplicity, we check if they access the same buffer
    // and at least one of them writes to that buffer
    for (const auto& write_region_a : a->GetWriteRegions()) {
      for (const auto& read_region_b : b->GetReadRegions()) {
        if (SameBuffer(write_region_a, read_region_b)) return true;
      }
      for (const auto& write_region_b : b->GetWriteRegions()) {
        if (SameBuffer(write_region_a, write_region_b)) return true;
      }
    }
    for (const auto& read_region_a : a->GetReadRegions()) {
      for (const auto& write_region_b : b->GetWriteRegions()) {
        if (SameBuffer(read_region_a, write_region_b)) return true;
      }
    }
    return false;
  }

  // Check if two TaskNodes have resource dependency (use same hardware resource)
  bool HasResourceDependency(const TaskNode* a, const TaskNode* b) const {
    // Resource dependencies occur when two tasks use the same hardware resource
    // that cannot be used simultaneously (or has limited throughput)

    // Check TMA core dependency
    if (a->UsesTMACore() && b->UsesTMACore()) {
      return true;  // Both use TMA core, cannot execute simultaneously
    }

    // Check Tensor core dependency
    if (a->UsesTensorCore() && b->UsesTensorCore()) {
      return true;  // Both use Tensor core, cannot execute simultaneously
    }

    // Check CUDA core dependency (more nuanced - CUDA cores can often be pipelined)
    // For now, we treat CUDA core as a shared resource with limited throughput
    // Could be refined based on actual hardware constraints
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

  // Apply scheduling (Z3-based) to a ScheduleUnit and update parent SequenceNode
  void ApplyScheduling(ScheduleUnit& unit) {
    if (unit.tasks.size() <= 1) {
      // No reordering needed for single task
      // Still assign warpgroup id if single task
      if (!unit.tasks.empty()) {
        unit.tasks[0]->SetWarpgroupId(0); // assign to warpgroup 0 by default
      }
      return;
    }


    if (!unit.parent_seq) {
      // No parent sequence to update (should not happen for tasks from SequenceNode)
      LOG(WARNING) << "ScheduleUnit has no parent SequenceNode, skipping reorder";
      return;
    }

    // Perform Z3-based scheduling using Python implementation
    // Use loop-aware scheduler if inside control node (loop)
    std::vector<TaskNode*> sorted_tasks;
    if (unit.inside_control_node) {
      LOG(INFO) << "ScheduleUnit is inside control node, using loop-aware scheduler";
      sorted_tasks = Z3SchedulePythonLoop(unit.tasks);
    } else {
      LOG(INFO) << "ScheduleUnit is not inside control node, using regular scheduler";
      sorted_tasks = Z3SchedulePython(unit.tasks);
    }

    // Check if order actually changed
    bool order_changed = false;
    for (size_t i = 0; i < sorted_tasks.size(); ++i) {
      if (sorted_tasks[i] != unit.tasks[i]) {
        order_changed = true;
        break;
      }
    }

    if (!order_changed) {
      // Order unchanged, nothing to do
      return;
    }

    // Update unit.tasks to sorted order
    unit.tasks = std::move(sorted_tasks);

    // Reorder children in parent SequenceNode
    size_t start_idx = unit.start_idx;
    size_t num_tasks = unit.tasks.size();

    // Extract the relevant children from parent sequence
    std::vector<std::unique_ptr<IRStructure>> task_children;
    task_children.reserve(num_tasks);

    // First, move the task children out of parent sequence
    for (size_t i = 0; i < num_tasks; ++i) {
      task_children.push_back(std::move(unit.parent_seq->children[start_idx + i]));
    }

    // Create a mapping from TaskNode pointer to its index in task_children
    std::unordered_map<TaskNode*, size_t> task_to_old_index;
    for (size_t i = 0; i < num_tasks; ++i) {
      // Each child should be a TaskNode
      IRStructure* child_ptr = task_children[i].get();
      if (child_ptr->IsTask()) {
        TaskNode* task = static_cast<TaskNode*>(child_ptr);
        task_to_old_index[task] = i;
      } else {
        LOG(FATAL) << "Expected TaskNode child in ScheduleUnit";
      }
    }

    // Reorder task_children according to sorted_tasks
    std::vector<std::unique_ptr<IRStructure>> reordered_children;
    reordered_children.reserve(num_tasks);

    for (TaskNode* task : unit.tasks) {
      auto it = task_to_old_index.find(task);
      if (it == task_to_old_index.end()) {
        LOG(FATAL) << "TaskNode not found in extracted children";
      }
      size_t old_idx = it->second;
      reordered_children.push_back(std::move(task_children[old_idx]));
    }

    // Move reordered children back to parent sequence
    for (size_t i = 0; i < num_tasks; ++i) {
      unit.parent_seq->children[start_idx + i] = std::move(reordered_children[i]);
    }

    LOG(INFO) << "Reordered " << num_tasks << " TaskNodes in SequenceNode";

    // Apply promote transformation if inside control node (loop)
    if (unit.inside_control_node) {
      LOG(INFO) << "ScheduleUnit is inside control node, checking for promoted tasks";

      // Check if any tasks are marked for promotion
      bool has_promoted_tasks = false;
      for (TaskNode* task : unit.tasks) {
        if (task->GetPromote()) {
          has_promoted_tasks = true;
          break;
        }
      }

      if (has_promoted_tasks) {
        LOG(INFO) << "Found promoted tasks, applying promote transformation";

        if (unit.control_node) {
          LOG(INFO) << "ControlNode available, applying promote transformation";
          ApplyPromoteTransformation(unit.tasks, unit.control_node);
        } else {
          LOG(WARNING) << "ControlNode not set in ScheduleUnit, cannot apply promote transformation";
          LOG(INFO) << "Promote transformation requires access to For loop step information";
        }
      }
    }

    // Apply warpgroup partition after scheduling
    // NOTE: Now applied at the entire IRStructure level, not per ScheduleUnit
    // ApplyWarpgroupPartition(unit, thread_var_);
  }

  void StartNewUnit() {
    FinalizeCurrentUnit();
    units_.emplace_back();
    current_unit_ = &units_.back();
    current_unit_->inside_control_node = (control_nesting_depth_ > 0);
    // Set control_node pointer if inside a ControlNode
    current_unit_->control_node = current_control_node_;
  }

  void FinalizeCurrentUnit() {
    if (current_unit_ && current_unit_->tasks.empty()) {
      // Remove empty unit
      units_.pop_back();
    }
    current_unit_ = nullptr;
  }

  void Collect(IRStructure* node) {
    if (!node) return;

    if (node->IsTask()) {
      TaskNode* task = static_cast<TaskNode*>(node);
      if (!current_unit_) {
        StartNewUnit();
        // Set parent sequence and start index for this unit
        current_unit_->parent_seq = current_seq_;
        current_unit_->start_idx = current_child_idx_;
      }
      current_unit_->tasks.push_back(task);
    } else if (node->IsSequence()) {
      SequenceNode* seq = static_cast<SequenceNode*>(node);
      // Save previous context
      SequenceNode* prev_seq = current_seq_;
      size_t prev_child_idx = current_child_idx_;

      // Set new context for this sequence
      current_seq_ = seq;
      current_child_idx_ = 0;

      for (auto& child : seq->children) {
        Collect(child.get());
        current_child_idx_++;
        // If child is a ControlNode, it will have finalized current_unit_
        // and set it to nullptr. So we need to ensure that after a ControlNode,
        // we don't continue adding to the previous unit.
        if (child->IsControl()) {
          // Already finalized by Collect
        }
      }

      // Restore previous context
      current_seq_ = prev_seq;
      current_child_idx_ = prev_child_idx;
    } else if (node->IsControl()) {
      ControlNode* ctrl = static_cast<ControlNode*>(node);
      // Finalize any current unit before entering control node
      FinalizeCurrentUnit();
      // Increase control nesting depth
      control_nesting_depth_++;
      // Set current ControlNode for promote transformation
      ControlNode* prev_control_node = current_control_node_;
      current_control_node_ = ctrl;
      // Process the body of the control node (creates new units internally)
      Collect(ctrl->child.get());
      // Restore previous ControlNode
      current_control_node_ = prev_control_node;
      // Decrease control nesting depth
      control_nesting_depth_--;
      // After control node, current_unit_ should remain nullptr
      FinalizeCurrentUnit();
    }
  }
};

// Helper function to print BufferRegion details
void PrintBufferRegion(const BufferRegion& region, const std::string& indent) {
  const Buffer& buffer = region->buffer;
  const Region& ranges = region->region;

  std::string buffer_name = buffer->name;
  if (buffer_name.empty()) buffer_name = "unnamed_buffer";

  LOG(INFO) << indent << "Buffer: " << buffer_name;

  // Get scope information
  String scope_str = buffer.scope();
  LOG(INFO) << indent << "  Scope: " << scope_str;

  // Build shape string
  std::ostringstream shape_ss;
  shape_ss << "[";
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (i > 0) shape_ss << ", ";
    shape_ss << buffer->shape[i];
  }
  shape_ss << "]";
  LOG(INFO) << indent << "  Shape: " << shape_ss.str();

  LOG(INFO) << indent << "  Region:";
  for (size_t i = 0; i < ranges.size(); ++i) {
    const Range& range = ranges[i];
    std::ostringstream range_ss;
    range_ss << "[" << range->min << ", " << range->min + range->extent
             << ") (extent=" << range->extent << ")";
    LOG(INFO) << indent << "    dim " << i << ": " << range_ss.str();
  }
}

// Helper function to print ScheduleUnits
void PrintScheduleUnits(const std::vector<ScheduleUnit>& units) {
  LOG(INFO) << "ScheduleUnits (" << units.size() << " units):";
  for (size_t i = 0; i < units.size(); i++) {
    const auto& unit = units[i];
    LOG(INFO) << "  Unit " << i << " contains " << unit.tasks.size() << " TaskNodes:";
    LOG(INFO) << "    Parent Sequence: " << (unit.parent_seq ? "present" : "none")
              << ", Start index: " << unit.start_idx
              << ", Inside ControlNode: " << (unit.inside_control_node ? "yes" : "no");
    for (size_t j = 0; j < unit.tasks.size(); j++) {
      const TaskNode* task = unit.tasks[j];
      LOG(INFO) << "    Task " << j << ": uses_cuda_core=" << task->UsesCUDACore()
                << ", uses_tma_core=" << task->UsesTMACore()
                << ", uses_tensor_core=" << task->UsesTensorCore()
                << ", read_regions=" << task->GetReadRegions().size()
                << ", write_regions=" << task->GetWriteRegions().size()
                << ", latency=" << task->GetLatency() << " cycles"
                << ", II=" << task->GetII() << " cycles"
                << ", warpgroup_id=" << task->GetWarpgroupId();

      // Print statements in this task
      if (!task->stmts.empty()) {
        LOG(INFO) << "      Statements (" << task->stmts.size() << "):";
        for (size_t k = 0; k < task->stmts.size(); ++k) {
          LOG(INFO) << "        Statement " << k << ":";
          LOG(INFO) << "          " << task->stmts[k];
        }
      }

      // Print read regions if any
      if (!task->GetReadRegions().empty()) {
        LOG(INFO) << "      Read regions:";
        for (size_t k = 0; k < task->GetReadRegions().size(); ++k) {
          LOG(INFO) << "      Region " << k << ":";
          PrintBufferRegion(task->GetReadRegions()[k], "        ");
        }
      }

      // Print write regions if any
      if (!task->GetWriteRegions().empty()) {
        LOG(INFO) << "      Write regions:";
        for (size_t k = 0; k < task->GetWriteRegions().size(); ++k) {
          LOG(INFO) << "      Region " << k << ":";
          PrintBufferRegion(task->GetWriteRegions()[k], "        ");
        }
      }
    }
  }
}

// Helper function to print all stmts in IRStructure
void PrintAllStmts(const IRStructure* node, int indent = 0) {
  if (!node) return;

  std::string indent_str(indent * 2, ' ');

  if (node->IsTask()) {
    const TaskNode* task = static_cast<const TaskNode*>(node);
    LOG(INFO) << indent_str << "TaskNode with " << task->stmts.size() << " statements:";
    for (size_t i = 0; i < task->stmts.size(); i++) {
      LOG(INFO) << indent_str << "  Statement " << i << ":";
      LOG(INFO) << indent_str + "    " << task->stmts[i];
    }
    LOG(INFO) << indent_str << "  Resource usage: CUDA=" << task->UsesCUDACore()
              << ", TMA=" << task->UsesTMACore()
              << ", Tensor=" << task->UsesTensorCore();
    LOG(INFO) << indent_str << "  Latency: " << task->GetLatency() << " cycles, II: " << task->GetII() << " cycles, warpgroup_id: " << task->GetWarpgroupId();
  } else if (node->IsControl()) {
    const ControlNode* control = static_cast<const ControlNode*>(node);
    LOG(INFO) << indent_str << "ControlNode (For loop):";
    // Print the For statement itself
    LOG(INFO) << indent_str << "  For statement:";
    LOG(INFO) << indent_str + "    " << control->control;

    // Recursively print child statements
    if (control->child) {
      LOG(INFO) << indent_str << "  Loop body:";
      PrintAllStmts(control->child.get(), indent + 2);
    }
  } else if (node->IsSequence()) {
    const SequenceNode* seq = static_cast<const SequenceNode*>(node);
    LOG(INFO) << indent_str << "SequenceNode with " << seq->children.size() << " children:";
    for (size_t i = 0; i < seq->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintAllStmts(seq->children[i].get(), indent + 2);
    }
  }
}

// Original helper function to print IRStructure (kept for backward compatibility)
void PrintIRStructure(const IRStructure* node, int indent = 0) {
  if (!node) return;

  std::string indent_str(indent * 2, ' ');

  if (node->IsTask()) {
    const TaskNode* task = static_cast<const TaskNode*>(node);
    LOG(INFO) << indent_str << "TaskNode:";
    LOG(INFO) << indent_str << "  stmts: " << task->stmts.size() << " statements";
    LOG(INFO) << indent_str << "  uses_cuda_core: " << task->UsesCUDACore();
    LOG(INFO) << indent_str << "  uses_tma_core: " << task->UsesTMACore();
    LOG(INFO) << indent_str << "  uses_tensor_core: " << task->UsesTensorCore();
    LOG(INFO) << indent_str << "  latency: " << task->GetLatency() << " cycles";
    LOG(INFO) << indent_str << "  II: " << task->GetII() << " cycles";
    LOG(INFO) << indent_str << "  warpgroup_id: " << task->GetWarpgroupId();
  } else if (node->IsControl()) {
    const ControlNode* control = static_cast<const ControlNode*>(node);
    LOG(INFO) << indent_str << "ControlNode (For loop):";
    // Could print loop info if needed
    if (control->child) {
      LOG(INFO) << indent_str << "  Child:";
      PrintIRStructure(control->child.get(), indent + 2);
    }
  } else if (node->IsSequence()) {
    const SequenceNode* seq = static_cast<const SequenceNode*>(node);
    LOG(INFO) << indent_str << "SequenceNode: " << seq->children.size() << " children";
    for (size_t i = 0; i < seq->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintIRStructure(seq->children[i].get(), indent + 2);
    }
  }
}


// Helper function to check if an IRStructure contains any promoted tasks
bool HasPromotedTasks(const IRStructure* node) {
  if (!node) return false;

  if (node->IsTask()) {
    const TaskNode* task = static_cast<const TaskNode*>(node);
    return task->GetPromote();
  } else if (node->IsSequence()) {
    const SequenceNode* seq = static_cast<const SequenceNode*>(node);
    for (const auto& child : seq->children) {
      if (HasPromotedTasks(child.get())) {
        return true;
      }
    }
    return false;
  } else if (node->IsControl()) {
    const ControlNode* control = static_cast<const ControlNode*>(node);
    return HasPromotedTasks(control->child.get());
  }

  return false;
}

// Helper function to collect all promoted tasks from an IRStructure
void CollectPromotedTasks(const IRStructure* node, std::vector<const TaskNode*>& promoted_tasks) {
  if (!node) return;

  if (node->IsTask()) {
    const TaskNode* task = static_cast<const TaskNode*>(node);
    if (task->GetPromote()) {
      promoted_tasks.push_back(task);
    }
  } else if (node->IsSequence()) {
    const SequenceNode* seq = static_cast<const SequenceNode*>(node);
    for (const auto& child : seq->children) {
      CollectPromotedTasks(child.get(), promoted_tasks);
    }
  } else if (node->IsControl()) {
    const ControlNode* control = static_cast<const ControlNode*>(node);
    CollectPromotedTasks(control->child.get(), promoted_tasks);
  }
}

// Rewriter to convert IRStructure back to TIR statements
class IRStructureRewriter {
public:
  // Build loop body with promote transformation
  Stmt BuildPromotedLoopBody(const IRStructure* node, const Var& loop_var,
                             const PrimExpr& loop_min, const PrimExpr& loop_step,
                             const PrimExpr& new_loop_end) {
    if (!node) return Stmt();

    // Calculate original loop end: end = new_loop_end - loop_step
    PrimExpr loop_end = new_loop_end - loop_step;

    // Recursive helper function
    std::function<Stmt(const IRStructure*)> build = [&](const IRStructure* n) -> Stmt {
      if (!n) return Stmt();

      if (n->IsTask()) {
        const TaskNode* task = static_cast<const TaskNode*>(n);
        bool promoted = task->GetPromote();

        // Get the task statements (may be multiple)
        if (task->stmts.empty()) {
          return Stmt();
        }

        // Combine multiple statements into SeqStmt if needed
        Stmt task_body;
        if (task->stmts.size() == 1) {
          task_body = task->stmts[0];
        } else {
          task_body = SeqStmt(task->stmts);
        }

        // Apply variable substitution for non-promoted tasks
        if (!promoted) {
          // Substitute loop_var with (loop_var - loop_step) for non-promoted tasks
          Map<Var, PrimExpr> substitution;
          substitution.Set(loop_var, loop_var - loop_step);
          task_body = Substitute(task_body, substitution);
        }

        // Build condition
        PrimExpr condition;
        if (promoted) {
          // Promoted tasks: execute when loop_var < loop_end (original loop boundary)
          condition = loop_var < loop_end;
        } else {
          // Non-promoted tasks: execute when loop_var > loop_min
          condition = loop_var > loop_min;
        }

        // Create conditional statement
        return IfThenElse(condition, task_body);

      } else if (n->IsSequence()) {
        const SequenceNode* seq = static_cast<const SequenceNode*>(n);

        // Collect tasks and other statements, but reorder: non-promoted first, then promoted
        std::vector<Stmt> non_promoted_stmts;
        std::vector<Stmt> promoted_stmts;
        std::vector<Stmt> other_stmts;  // for non-task statements

        for (const auto& child : seq->children) {
          if (child->IsTask()) {
            const TaskNode* task = static_cast<const TaskNode*>(child.get());
            bool promoted = task->GetPromote();

            if (task->stmts.empty()) {
              continue;
            }

            // Combine multiple statements into SeqStmt if needed
            Stmt task_body;
            if (task->stmts.size() == 1) {
              task_body = task->stmts[0];
            } else {
              task_body = SeqStmt(task->stmts);
            }

            // Apply variable substitution for non-promoted tasks
            if (!promoted) {
              // Substitute loop_var with (loop_var - loop_step) for non-promoted tasks
              // Non-promoted tasks execute in previous iteration with index i-step
              Map<Var, PrimExpr> substitution;
              substitution.Set(loop_var, loop_var - loop_step);
              task_body = Substitute(task_body, substitution);
            }

            // Create independent conditional statement for each task
            PrimExpr condition;
            if (promoted) {
              // Promoted tasks: execute when i < end (original loop boundary)
              // They execute in the current iteration with original index i
              condition = loop_var < loop_end;
            } else {
              // Non-promoted tasks: execute when i > start (i.e., not in first iteration)
              // They execute in previous iteration with adjusted index i-step
              condition = loop_var > loop_min;
            }

            Stmt conditional_stmt = IfThenElse(condition, task_body);

            if (promoted) {
              promoted_stmts.push_back(conditional_stmt);
            } else {
              non_promoted_stmts.push_back(conditional_stmt);
            }

          } else if (child->IsSequence() || child->IsControl()) {
            // For nested sequences or control nodes, recursively process them
            // They will maintain their own internal structure
            Stmt child_stmt = build(child.get());
            if (child_stmt.defined()) {
              // We don't know if they contain promoted tasks, so put them
              // in the middle (between non-promoted and promoted)
              other_stmts.push_back(child_stmt);
            }
          }
        }

        // Combine all statements in order: non-promoted first, then other statements,
        // then promoted tasks (following the user's example pattern)
        std::vector<Stmt> all_stmts;
        all_stmts.insert(all_stmts.end(), non_promoted_stmts.begin(), non_promoted_stmts.end());
        all_stmts.insert(all_stmts.end(), other_stmts.begin(), other_stmts.end());
        all_stmts.insert(all_stmts.end(), promoted_stmts.begin(), promoted_stmts.end());

        if (all_stmts.empty()) {
          return Stmt();
        } else if (all_stmts.size() == 1) {
          return all_stmts[0];
        } else {
          return SeqStmt(all_stmts);
        }

      } else if (n->IsControl()) {
        // For control nodes (nested loops), use regular Rewrite
        // Promote transformation only applies to tasks in the current loop
        const ControlNode* control = static_cast<const ControlNode*>(n);
        Stmt child_stmt = Rewrite(control->child.get());
        if (!child_stmt.defined()) {
          return Stmt();
        }
        // Reconstruct the control node with rewritten child
        return For(control->control->loop_var, control->control->min,
                   control->control->extent, control->control->kind,
                   child_stmt, control->control->thread_binding,
                   control->control->annotations);

      } else {
        LOG(WARNING) << "BuildPromotedLoopBody: Unknown IRStructure type";
        return Stmt();
      }
    };

    return build(node);
  }

  Stmt Rewrite(const IRStructure* node) {
    if (!node) {
      return Stmt();
    }

    if (node->IsTask()) {
      const TaskNode* task = static_cast<const TaskNode*>(node);
      if (task->stmts.empty()) {
        return Stmt();
      }
      // If there's only one statement, return it directly
      if (task->stmts.size() == 1) {
        return task->stmts[0];
      }
      // Otherwise, create a SeqStmt
      return SeqStmt(task->stmts);

    } else if (node->IsControl()) {
      const ControlNode* control = static_cast<const ControlNode*>(node);

      // Check if there are any promoted tasks in this loop
      bool has_promoted_tasks = HasPromotedTasks(control->child.get());

      if (has_promoted_tasks) {
        LOG(INFO) << "IRStructureRewriter: ControlNode has promoted tasks, applying promote transformation";

        // Get For loop information
        const ForNode* for_node = control->control.get();
        Var loop_var = for_node->loop_var;
        PrimExpr loop_min = for_node->min;
        PrimExpr loop_extent = for_node->extent;
        PrimExpr loop_step = for_node->step.has_value() ? for_node->step.value() : IntImm(DataType::Int(32), 1);

        // Use original expressions (no simplification)

        // Calculate new loop extent: extent + step
        PrimExpr new_loop_extent = loop_extent + loop_step;

        LOG(INFO) << "IRStructureRewriter: Promote transformation details:";
        LOG(INFO) << "  Original loop: for " << loop_var << " in [" << loop_min << ", "
                  << (loop_min + loop_extent) << ") with step " << loop_step;
        LOG(INFO) << "  New loop extent: " << new_loop_extent << " (original " << loop_extent << " + step " << loop_step << ")";

        // Collect promoted tasks for logging
        std::vector<const TaskNode*> promoted_tasks;
        CollectPromotedTasks(control->child.get(), promoted_tasks);
        LOG(INFO) << "  Found " << promoted_tasks.size() << " promoted tasks";

        // Build transformed loop body with conditional execution (promote transformation implemented)
        Stmt transformed_body = BuildPromotedLoopBody(control->child.get(), loop_var, loop_min,
                                                      loop_step, loop_min + new_loop_extent);

        if (!transformed_body.defined()) {
          LOG(WARNING) << "Failed to build promoted loop body, using original";
          Stmt body = Rewrite(control->child.get());
          if (!body.defined()) {
            return control->control;
          }
          return For(loop_var, loop_min, new_loop_extent, for_node->kind, body,
                     for_node->thread_binding, for_node->annotations);
        }

        LOG(INFO) << "IRStructureRewriter: Successfully built promoted loop body";
        // Create new For loop with extended extent and transformed body
        return For(loop_var, loop_min, new_loop_extent, for_node->kind, transformed_body,
                   for_node->thread_binding, for_node->annotations);
      } else {
        // No promoted tasks, rebuild normally
        Stmt body = Rewrite(control->child.get());
        if (!body.defined()) {
          LOG(WARNING) << "ControlNode body is undefined";
          return control->control;
        }
        // Create a new For loop with the same parameters but updated body
        return For(control->control->loop_var,
                   control->control->min,
                   control->control->extent,
                   control->control->kind,
                   body,
                   control->control->thread_binding,
                   control->control->annotations);
      }

    } else if (node->IsSequence()) {
      const SequenceNode* seq = static_cast<const SequenceNode*>(node);
      std::vector<Stmt> stmts;
      stmts.reserve(seq->children.size());

      for (const auto& child : seq->children) {
        Stmt child_stmt = Rewrite(child.get());
        if (child_stmt.defined()) {
          stmts.push_back(child_stmt);
        }
      }

      if (stmts.empty()) {
        return Stmt();
      } else if (stmts.size() == 1) {
        return stmts[0];
      } else {
        return SeqStmt(stmts);
      }
    }

    LOG(FATAL) << "Unknown IRStructure kind";
    return Stmt();
  }
};

// Mutator to update thread extent in AttrStmt nodes
// Used after warpgroup partition to double thread extent
class ThreadExtentUpdater : public StmtExprMutator {
public:
  explicit ThreadExtentUpdater(PrimExpr updated_extent)
      : updated_thread_extent_(updated_extent) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      auto iter_var = Downcast<IterVar>(op->node);
      if (iter_var->thread_tag == "threadIdx.x") {
        // Save the thread IterVar
        thread_iv_ = iter_var;

        // Visit the body first (to update any references)
        AttrStmt attr_stmt = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));

        // Update the thread extent
        LOG(INFO) << "ThreadExtentUpdater: Updating thread extent for " << thread_iv_->thread_tag
                  << " from " << thread_iv_->dom->extent << " to " << updated_thread_extent_;

        // Create new IterVar with updated domain
        Range new_dom = Range::FromMinExtent(thread_iv_->dom->min, updated_thread_extent_);

        // Update the AttrStmt with new IterVar and value
        attr_stmt.CopyOnWrite()->node = iter_var;
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

  void VisitStmt_(const BlockNode* op) override {
    if (op->name_hint == "tilelang_root") {
      LOG(INFO) << "Found tilelang_root block, extracting its body";
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

  Stmt VisitStmt_(const BlockNode* op) override {
    auto block = GetRef<Block>(op);
    if (op->name_hint == "tilelang_root") {
      LOG(INFO) << "Replacing body of tilelang_root block";
      // Keep all block attributes but replace the body
      return Block(op->iter_vars, op->reads, op->writes, op->name_hint, new_body_,
                   op->init, op->alloc_buffers, op->match_buffers, op->annotations);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  Stmt new_body_;
};

// Visitor to build IRStructure from TIR statements
class IRStructureBuilder : public StmtVisitor {
public:
  std::unique_ptr<IRStructure> Build(const Stmt& stmt) {
    VisitStmt(stmt);
    if (!root_) {
      LOG(WARNING) << "IRStructureBuilder: root_ is null after visiting statement. "
                   << "This may indicate an unhandled statement type.";
      // Return an empty TaskNode as fallback
      auto task_node = std::make_unique<TaskNode>();
      task_node->stmts.push_back(stmt);
      return task_node;
    }
    return std::move(root_);
  }

protected:

  void VisitStmt_(const SeqStmtNode* op) override {
    auto seq_node = std::make_unique<SequenceNode>();

    for (size_t i = 0; i < op->seq.size(); i++) {
      VisitStmt(op->seq[i]);
      if (root_) {
        seq_node->children.push_back(std::move(root_));
      }
    }
    root_ = std::move(seq_node);
  }

  void VisitStmt_(const ForNode* op) override {
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

  void VisitStmt_(const EvaluateNode* op) override {
    // Evaluate statement (usually a Call) -> TaskNode
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze the expression for resource usage
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const IfThenElseNode* op) override {
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

  void VisitStmt_(const LetStmtNode* op) override {
    // Let statement -> treat as TaskNode
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    AnalyzeResourceUsage(op->body, task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const WhileNode* op) override {
    LOG(INFO) << "VisitStmt_: WhileNode -> TaskNode";
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze condition and body for resource usage
    AnalyzeResourceUsage(Evaluate(op->condition), task_node.get());
    AnalyzeResourceUsage(op->body, task_node.get());

    LOG(INFO) << "  TaskNode resource usage: CUDA=" << task_node->UsesCUDACore()
              << ", TMA=" << task_node->UsesTMACore()
              << ", Tensor=" << task_node->UsesTensorCore();

    root_ = std::move(task_node);
  }

  void VisitStmt_(const BlockNode* op) override {
    // All blocks are treated as TaskNode
    // Note: tilelang_root block should have been extracted by TilelangRootBodyExtractor
    // If we encounter it here, it means we're processing the entire function body
    // (not extracted), which should only happen when there's no tilelang_root block
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());
    root_ = std::move(task_node);
  }


private:
  std::unique_ptr<IRStructure> root_;

  void AnalyzeResourceUsage(const Stmt& stmt, TaskNode* task_node) {
    // Recursively analyze statements to determine resource usage
    struct ResourceAnalyzer : public StmtExprVisitor {
      TaskNode* task_node;
      bool found_tma{false};
      bool found_tensor{false};
      bool found_cuda{false};

      ResourceAnalyzer(TaskNode* node) : task_node(node) {}

      void VisitExpr_(const CallNode* op) override {
        // Check for specific TileLang operations
        static const auto copy_op = Op::Get("tl.tileop.copy");
        static const auto gemm_py_op = Op::Get("tl.tileop.gemm_py");
        static const auto gemm_op = Op::Get("tl.tileop.gemm");
        static const auto reduce_op = Op::Get("tl.tileop.reduce");
        static const auto fill_op = Op::Get("tl.tileop.fill");
        static const auto region_op = Op::Get("tl.tileop.region");

        // Try to get operation name for logging
        std::string op_name = "unknown";
        if (const auto* op_ptr = op->op.as<OpNode>()) {
          op_name = op_ptr->name;
        }

        // Check if this is a TMA copy operation
        if (op->op.same_as(copy_op)) {
          found_tma = true;
        } else if (op->op.same_as(gemm_py_op) || op->op.same_as(gemm_op)) {
          found_tensor = true;
        } else if (op->op.same_as(reduce_op) || op->op.same_as(fill_op)) {
          // Reduce and fill operations use CUDA core
          found_cuda = true;
        } else if (op->op.same_as(region_op)) {
          // Handle tl.tileop.region call for memory access analysis
          // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2: write, 3: read/write)
          // args[2..] = extents
          if (op->args.size() >= 2) {
            // Extract access type
            if (const auto* access_int = op->args[1].as<IntImmNode>()) {
              int access_type = access_int->value;
              // For now, just mark as CUDA operation (memory access)
              found_cuda = true;
              // TODO: Extract buffer region and add to task_node->read_regions or write_regions
              // BufferLoad buffer_load = Downcast<BufferLoad>(op->args[0]);
              // Construct BufferRegion from buffer_load and extents
              // if (access_type == 1 || access_type == 3) task_node->read_regions.push_back(region);
              // if (access_type == 2 || access_type == 3) task_node->write_regions.push_back(region);
            }
          }
        } else {
          // Check for other known operations that use CUDA core
          // For now, assume any other call is a basic computation
          found_cuda = true;
        }

        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const AddNode* op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const SubNode* op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const MulNode* op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const DivNode* op) override {
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
    }
    // If neither TMA nor Tensor core was used, and CUDA operations were found, set CUDA core flag
    if (!analyzer.found_tma && !analyzer.found_tensor && analyzer.found_cuda) {
      task_node->SetUsesCUDACore(true);
    }

    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(stmt);
    std::vector<BufferRegion> read_regions = memory_detector.GetReadRegions();
    std::vector<BufferRegion> write_regions = memory_detector.GetWriteRegions();

    // Merge with existing regions (avoid duplicates)
    for (const auto& region : read_regions) {
      task_node->AddReadRegion(region);
    }

    for (const auto& region : write_regions) {
      task_node->AddWriteRegion(region);
    }

    // Estimate latency and initiation interval for this task
    LatencyEstimator latency_estimator;
    latency_estimator.Estimate(task_node);
  }
};

// The main pass function
tvm::transform::Pass AutoSchedule() {
  using namespace tir::transform;
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    LOG(INFO) << "AutoSchedule pass started for function: " << func->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or("unnamed");

    // Extract the body of tilelang_root block if it exists
    TilelangRootBodyExtractor extractor;
    extractor(func->body);
    Stmt body_to_schedule;
    bool has_tilelang_root = false;
    PrimExpr updated_thread_extent;  // Will be set if warpgroup partition doubles thread extent
    IterVar thread_var;  // Thread index variable for warpgroup partition

    if (extractor.body.defined()) {
      LOG(INFO) << "Found tilelang_root block, scheduling its body only";
      body_to_schedule = extractor.body;
      has_tilelang_root = true;
    } else {
      LOG(INFO) << "No tilelang_root block found, scheduling entire function body";
      body_to_schedule = func->body;
    }

    // Build IRStructure from the body to schedule
    LOG(INFO) << "Building IRStructure from body...";
    IRStructureBuilder builder;
    auto ir_structure = builder.Build(body_to_schedule);

    // Print the built IRStructure with all statements
    if (ir_structure) {
      LOG(INFO) << "IRStructure built successfully:";

      // First print the summary view
      LOG(INFO) << "IRStructure summary:";
      PrintIRStructure(ir_structure.get());

      // Then print all statements
      LOG(INFO) << "=================== IRStructure Statements ===================";
      PrintAllStmts(ir_structure.get());
      LOG(INFO) << "=================== End IRStructure Statements ===================";

      // Build ScheduleUnits from IRStructure
      LOG(INFO) << "Building ScheduleUnits...";
      ScheduleUnitBuilder unit_builder;
      // Get thread index variable for warpgroup partition
      // First try to get from body_to_schedule, if not found, try from the entire function body
      thread_var = ThreadTagChecker::GetThreadVar(body_to_schedule);
      if (!thread_var.defined()) {
        LOG(INFO) << "Thread variable not found in body_to_schedule, trying full function body";
        thread_var = ThreadTagChecker::GetThreadVar(func->body);
      }
      if (thread_var.defined()) {
        LOG(INFO) << "Found thread index variable: " << thread_var->thread_tag
                  << " with extent " << thread_var->dom->extent;
        unit_builder.SetThreadVar(thread_var);
      } else {
        LOG(WARNING) << "Could not find thread index variable, warpgroup partition will use default";
      }
      auto schedule_units = unit_builder.Build(ir_structure.get());
      PrintScheduleUnits(schedule_units);

      // Print the modified summary view
      LOG(INFO) << "IRStructure modified summary:";
      PrintIRStructure(ir_structure.get());

      // Apply warpgroup partition to entire IRStructure
      if (thread_var.defined()) {
        LOG(INFO) << "Applying warpgroup partition to entire IRStructure...";
        auto partitioned_ir_structure = ApplyWarpgroupPartitionToIRStructure(ir_structure.get(), thread_var);
        if (partitioned_ir_structure) {
          ir_structure = std::move(partitioned_ir_structure);
          LOG(INFO) << "Warpgroup partition applied successfully";
          LOG(INFO) << "IRStructure after warpgroup partition:";
          PrintIRStructure(ir_structure.get());

          // After warpgroup partition, thread extent should be doubled
          // ThreadExtentUpdater will update the AttrStmt node in the final IR
          Range old_dom = thread_var->dom;
          PrimExpr new_extent = old_dom->extent * 2;
          updated_thread_extent = new_extent;
          LOG(INFO) << "Thread extent will be updated from " << old_dom->extent
                    << " to " << new_extent << " (doubled for warpgroup partition)";
        } else {
          LOG(WARNING) << "Warpgroup partition returned null, keeping original IRStructure";
        }
      } else {
        LOG(WARNING) << "Thread variable not defined, skipping warpgroup partition";
      }

    } else {
      LOG(INFO) << "IRStructure is null (empty body?)";
    }

    // Rewrite body based on IRStructure
    Stmt new_body;
    if (ir_structure) {
      IRStructureRewriter rewriter;
      new_body = rewriter.Rewrite(ir_structure.get());
      LOG(INFO) << "Rewrote body based on IRStructure";
    } else {
      LOG(INFO) << "IRStructure is null, keeping original body";
      new_body = body_to_schedule;
    }


    // If we extracted from tilelang_root block, replace the body
    Stmt final_body;
    if (has_tilelang_root) {
      TilelangRootBodyReplacer replacer(new_body);
      final_body = replacer(func->body);
      // Apply thread extent update if warpgroup partition was applied
      if (updated_thread_extent.defined() && thread_var.defined()) {
        LOG(INFO) << "Applying thread extent update from " << thread_var->dom->extent
                  << " to " << updated_thread_extent;
        ThreadExtentUpdater extent_updater(updated_thread_extent);
        final_body = extent_updater(final_body);
        LOG(INFO) << "Thread extent update applied";
      }
      LOG(INFO) << "Replaced tilelang_root block body";
    } else {
      final_body = new_body;
    }

    // Create a new PrimFunc with the updated body
    auto new_func = PrimFunc(func->params, final_body, func->ret_type, func->buffer_map, func->attrs);
    LOG(INFO) << "AutoSchedule pass completed";
    return new_func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

// Helper function to collect all TaskNodes from an IRStructure
void CollectTaskNodesFromIRStructure(IRStructure* node, std::vector<TaskNode*>& tasks) {
  if (!node) return;
  if (node->IsTask()) {
    tasks.push_back(static_cast<TaskNode*>(node));
  } else if (node->IsSequence()) {
    SequenceNode* seq = static_cast<SequenceNode*>(node);
    for (const auto& child : seq->children) {
      if (child) {
        CollectTaskNodesFromIRStructure(child.get(), tasks);
      }
    }
  } else if (node->IsControl()) {
    ControlNode* ctrl = static_cast<ControlNode*>(node);
    if (ctrl->child) {
      CollectTaskNodesFromIRStructure(ctrl->child.get(), tasks);
    }
  }
}

// Helper function to clone IRStructure with warpgroup filter
std::unique_ptr<IRStructure> CloneIRStructureWithWarpgroupFilter(
    IRStructure* node, int warpgroup_id) {
  if (!node) return nullptr;

  if (node->IsTask()) {
    TaskNode* task = static_cast<TaskNode*>(node);
    if (task->GetWarpgroupId() == warpgroup_id) {
      return task->Clone();
    } else {
      // Create empty task node for other warpgroup
      auto new_task = std::make_unique<TaskNode>();
      // Copy metadata but no statements
      new_task->SetUsesCUDACore(task->UsesCUDACore());
      new_task->SetUsesTMACore(task->UsesTMACore());
      new_task->SetUsesTensorCore(task->UsesTensorCore());
      new_task->SetReadRegions(task->GetReadRegions());
      new_task->SetWriteRegions(task->GetWriteRegions());
      new_task->SetLatency(task->GetLatency());
      new_task->SetII(task->GetII());
      new_task->SetStartTime(task->GetStartTime());
      new_task->SetPromote(task->GetPromote());
      new_task->SetWarpgroupId(task->GetWarpgroupId());
      // Empty statements
      return new_task;
    }
  } else if (node->IsSequence()) {
    SequenceNode* seq = static_cast<SequenceNode*>(node);
    auto new_seq = std::make_unique<SequenceNode>();
    for (const auto& child : seq->children) {
      if (child) {
        new_seq->children.push_back(CloneIRStructureWithWarpgroupFilter(child.get(), warpgroup_id));
      } else {
        new_seq->children.push_back(nullptr);
      }
    }
    return new_seq;
  } else if (node->IsControl()) {
    ControlNode* ctrl = static_cast<ControlNode*>(node);
    auto new_ctrl = std::make_unique<ControlNode>();
    new_ctrl->control = ctrl->control;
    if (ctrl->child) {
      new_ctrl->child = CloneIRStructureWithWarpgroupFilter(ctrl->child.get(), warpgroup_id);
    }
    return new_ctrl;
  }
  return nullptr;
}

// Apply warpgroup partition to a ScheduleUnit
// Split tasks into two groups based on warpgroup id and insert conditional branching
// if tx < original_threads: execute warpgroup 0 tasks, else: execute warpgroup 1 tasks
// Note: cross-warpgroup dependencies are ignored for now (will be handled later with barriers)
void ApplyWarpgroupPartition(ScheduleUnit& unit, IterVar thread_var) {
  if (unit.tasks.size() <= 1) {
    // No partition needed for single task
    return;
  }

  // Check if tasks have mixed warpgroup ids
  bool has_warpgroup0 = false;
  bool has_warpgroup1 = false;
  for (const auto* task : unit.tasks) {
    int wg_id = task->GetWarpgroupId();
    if (wg_id == 0) has_warpgroup0 = true;
    else if (wg_id == 1) has_warpgroup1 = true;
  }

  // If all tasks belong to the same warpgroup, no partition needed
  if (!(has_warpgroup0 && has_warpgroup1)) {
    LOG(INFO) << "All tasks belong to the same warpgroup, skipping partition";
    return;
  }

  LOG(INFO) << "Applying warpgroup partition to ScheduleUnit with " << unit.tasks.size() << " tasks";

  // We need to modify the parent SequenceNode's children
  if (!unit.parent_seq) {
    LOG(WARNING) << "ScheduleUnit has no parent SequenceNode, cannot apply partition";
    return;
  }

  // Check if this unit is inside a ControlNode and the child is a ControlNode
  if (unit.inside_control_node && unit.control_node) {
    // Find the ControlNode in parent sequence
    // The ControlNode should be at position start_idx in parent_seq
    size_t start_idx = unit.start_idx;
    if (start_idx >= unit.parent_seq->children.size()) {
      LOG(WARNING) << "Start index out of bounds";
      // Fall back to regular TaskNode case
      unit.inside_control_node = false;
      unit.control_node = nullptr;
    } else {
      IRStructure* child_ptr = unit.parent_seq->children[start_idx].get();
      if (child_ptr->IsControl()) {
        // Found ControlNode, process it
        ControlNode* original_ctrl = static_cast<ControlNode*>(child_ptr);

        // Clone ControlNode for warpgroup 0 and warpgroup 1
        auto ctrl_wg0 = CloneIRStructureWithWarpgroupFilter(original_ctrl, 0);
        auto ctrl_wg1 = CloneIRStructureWithWarpgroupFilter(original_ctrl, 1);

        // Check if both clones have actual statements (not just empty tasks)
        auto has_actual_statements = [](IRStructure* node) -> bool {
          std::vector<TaskNode*> tasks;
          CollectTaskNodesFromIRStructure(node, tasks);
          for (TaskNode* task : tasks) {
            if (!task->stmts.empty()) {
              return true;
            }
          }
          return false;
        };

        bool wg0_has_stmts = has_actual_statements(ctrl_wg0.get());
        bool wg1_has_stmts = has_actual_statements(ctrl_wg1.get());

        // Use the provided thread index variable from ThreadTagChecker
        PrimExpr condition;
        if (thread_var.defined() && thread_var->dom.defined()) {
          // Extract thread variable and domain
          Var thread_idx_var = thread_var->var;
          Range thread_dom = thread_var->dom;
          // Calculate condition: thread_idx < thread_dom->extent
          // Assuming threads are doubled: original threads = thread_dom->extent,
          // after warpgroup partition, total threads = 2 * thread_dom->extent
          // Condition splits into two warpgroups: tx < original_thread_count
          PrimExpr original_threads = thread_dom->extent;
          condition = thread_idx_var < original_threads;
          LOG(INFO) << "Using thread index variable " << thread_idx_var
                    << " with original domain " << thread_dom->min << " to " << thread_dom->min + thread_dom->extent
                    << " (threads will be doubled for warpgroup partition)"
                    << ", condition: " << condition;
        } else {
          LOG(WARNING) << "Thread index variable not properly defined, falling back to default tx < 128";
          Var tx_var("tx", DataType::Int(32));
          condition = tx_var < IntImm(DataType::Int(32), 128);
        }

        // Create IfThenElse statement with ControlNode clones as bodies
        // We need to convert IRStructure to Stmt
        std::function<Stmt(IRStructure*)> irstructure_to_stmt;
        irstructure_to_stmt = [&irstructure_to_stmt](IRStructure* structure) -> Stmt {
          if (!structure) {
            return Evaluate(0);
          }

          if (structure->IsTask()) {
            TaskNode* task = static_cast<TaskNode*>(structure);
            if (task->stmts.empty()) {
              return Evaluate(0);
            } else if (task->stmts.size() == 1) {
              return task->stmts[0];
            } else {
              return SeqStmt(task->stmts);
            }
          } else if (structure->IsSequence()) {
            SequenceNode* seq = static_cast<SequenceNode*>(structure);
            std::vector<Stmt> stmts;
            for (const auto& child : seq->children) {
              if (child) {
                Stmt child_stmt = irstructure_to_stmt(child.get());
                stmts.push_back(child_stmt);
              }
            }
            if (stmts.empty()) {
              return Evaluate(0);
            } else if (stmts.size() == 1) {
              return stmts[0];
            } else {
              return SeqStmt(stmts);
            }
          } else if (structure->IsControl()) {
            ControlNode* ctrl = static_cast<ControlNode*>(structure);
            Stmt body = Evaluate(0);
            if (ctrl->child) {
              body = irstructure_to_stmt(ctrl->child.get());
            }
            return For(ctrl->control->loop_var, ctrl->control->min,
                      ctrl->control->extent, ctrl->control->kind,
                      body, ctrl->control->thread_binding,
                      ctrl->control->annotations);
          }

          LOG(WARNING) << "Failed to convert IRStructure to Stmt, returning empty statement";
          return Evaluate(0);
        };

        Stmt then_body = wg0_has_stmts ? irstructure_to_stmt(ctrl_wg0.get()) : Evaluate(0);
        Stmt else_body = wg1_has_stmts ? irstructure_to_stmt(ctrl_wg1.get()) : Evaluate(0);

        // Create IfThenElse statement
        Stmt if_then_else;
        if (wg0_has_stmts && wg1_has_stmts) {
          if_then_else = IfThenElse(condition, then_body, else_body);
        } else if (wg0_has_stmts) {
          // Only warpgroup 0 has statements, execute unconditionally
          if_then_else = then_body;
        } else if (wg1_has_stmts) {
          // Only warpgroup 1 has statements, execute unconditionally
          if_then_else = else_body;
        } else {
          LOG(WARNING) << "Both warpgroups have no statements, skipping partition";
          return;
        }

        // Create a new TaskNode containing the IfThenElse statement
        auto new_task_node = std::make_unique<TaskNode>();
        new_task_node->stmts.push_back(if_then_else);

        // Copy resource usage flags from original tasks (take union)
        bool uses_cuda_core = false;
        bool uses_tma_core = false;
        bool uses_tensor_core = false;
        int64_t total_latency = 0;
        int64_t max_ii = 0;
        for (TaskNode* task : unit.tasks) {
          uses_cuda_core = uses_cuda_core || task->UsesCUDACore();
          uses_tma_core = uses_tma_core || task->UsesTMACore();
          uses_tensor_core = uses_tensor_core || task->UsesTensorCore();
          total_latency += task->GetLatency();
          max_ii = std::max(max_ii, task->GetII());
        }
        new_task_node->SetUsesCUDACore(uses_cuda_core);
        new_task_node->SetUsesTMACore(uses_tma_core);
        new_task_node->SetUsesTensorCore(uses_tensor_core);
        new_task_node->SetLatency(total_latency);
        new_task_node->SetII(max_ii);
        new_task_node->SetWarpgroupId(-1);  // mixed

        // Also copy read/write regions from original tasks
        for (TaskNode* task : unit.tasks) {
          auto read_regions = task->GetReadRegions();
          for (const auto& region : read_regions) {
            new_task_node->AddReadRegion(region);
          }
          auto write_regions = task->GetWriteRegions();
          for (const auto& region : write_regions) {
            new_task_node->AddWriteRegion(region);
          }
        }

        // Replace the ControlNode with the new conditional task
        unit.parent_seq->children[start_idx] = std::move(new_task_node);

        // Update the ScheduleUnit
        unit.tasks.clear();
        TaskNode* new_task = static_cast<TaskNode*>(unit.parent_seq->children[start_idx].get());
        unit.tasks.push_back(new_task);

        LOG(INFO) << "Created warpgroup partition for ControlNode: if " << condition
                  << " execute warpgroup 0, else execute warpgroup 1";
        LOG(INFO) << "Replaced ControlNode with conditional task";
        return;
      } else {
        LOG(WARNING) << "Expected ControlNode at position " << start_idx << ", but found "
                     << (child_ptr->IsTask() ? "TaskNode" : "SequenceNode")
                     << ", falling back to regular TaskNode case";
        // Fall back to regular TaskNode case
        unit.inside_control_node = false;
        unit.control_node = nullptr;
      }
    }
  }

  // Regular TaskNode case (no ControlNode)
  {
    // Group tasks by warpgroup id
    std::vector<TaskNode*> warpgroup0_tasks;
    std::vector<TaskNode*> warpgroup1_tasks;
    for (TaskNode* task : unit.tasks) {
      if (task->GetWarpgroupId() == 0) {
        warpgroup0_tasks.push_back(task);
      } else if (task->GetWarpgroupId() == 1) {
        warpgroup1_tasks.push_back(task);
      } else {
        LOG(WARNING) << "Task has invalid warpgroup id " << task->GetWarpgroupId() << ", assigning to warpgroup 0";
        warpgroup0_tasks.push_back(task);
      }
    }

    // Extract the original children from parent sequence
    size_t start_idx = unit.start_idx;
    size_t num_tasks = unit.tasks.size();

    // First, move the task children out of parent sequence
    std::vector<std::unique_ptr<IRStructure>> task_children;
    task_children.reserve(num_tasks);
    for (size_t i = 0; i < num_tasks; ++i) {
      task_children.push_back(std::move(unit.parent_seq->children[start_idx + i]));
    }

    // Create mapping from TaskNode pointer to its child index
    std::unordered_map<TaskNode*, size_t> task_to_child_index;
    for (size_t i = 0; i < num_tasks; ++i) {
      IRStructure* child_ptr = task_children[i].get();
      if (child_ptr->IsTask()) {
        TaskNode* task = static_cast<TaskNode*>(child_ptr);
        task_to_child_index[task] = i;
      } else {
        LOG(FATAL) << "Expected TaskNode child in ScheduleUnit";
      }
    }

    // Collect statements for each warpgroup
    std::vector<Stmt> warpgroup0_stmts;
    std::vector<Stmt> warpgroup1_stmts;

    for (TaskNode* task : warpgroup0_tasks) {
      auto it = task_to_child_index.find(task);
      if (it == task_to_child_index.end()) {
        LOG(FATAL) << "TaskNode not found in extracted children";
      }
      TaskNode* task_node = static_cast<TaskNode*>(task_children[it->second].get());
      for (const auto& stmt : task_node->stmts) {
        warpgroup0_stmts.push_back(stmt);
      }
    }

    for (TaskNode* task : warpgroup1_tasks) {
      auto it = task_to_child_index.find(task);
      if (it == task_to_child_index.end()) {
        LOG(FATAL) << "TaskNode not found in extracted children";
      }
      TaskNode* task_node = static_cast<TaskNode*>(task_children[it->second].get());
      for (const auto& stmt : task_node->stmts) {
        warpgroup1_stmts.push_back(stmt);
      }
    }

    // Build statements for each warpgroup
    Stmt then_body;
    if (warpgroup0_stmts.empty()) {
      then_body = Evaluate(0);  // Empty statement
    } else if (warpgroup0_stmts.size() == 1) {
      then_body = warpgroup0_stmts[0];
    } else {
      then_body = SeqStmt(warpgroup0_stmts);
    }

    Stmt else_body;
    if (warpgroup1_stmts.empty()) {
      else_body = Evaluate(0);  // Empty statement
    } else if (warpgroup1_stmts.size() == 1) {
      else_body = warpgroup1_stmts[0];
    } else {
      else_body = SeqStmt(warpgroup1_stmts);
    }

    // Use the provided thread index variable from ThreadTagChecker
    PrimExpr condition;
    if (thread_var.defined() && thread_var->dom.defined()) {
      // Extract thread variable and domain
      Var thread_idx_var = thread_var->var;
      Range thread_dom = thread_var->dom;
      // Calculate condition: thread_idx < thread_dom->extent
      PrimExpr original_threads = thread_dom->extent;
      condition = thread_idx_var < original_threads;
      LOG(INFO) << "Using thread index variable " << thread_idx_var
                << " with original domain " << thread_dom->min << " to " << thread_dom->min + thread_dom->extent
                << ", condition: " << condition;
    } else {
      LOG(WARNING) << "Thread index variable not properly defined, falling back to default tx < 128";
      Var tx_var("tx", DataType::Int(32));
      condition = tx_var < IntImm(DataType::Int(32), 128);
    }

    // Create IfThenElse statement
    Stmt if_then_else = IfThenElse(condition, then_body, else_body);

    // Create a new TaskNode containing the IfThenElse statement
    auto new_task_node = std::make_unique<TaskNode>();
    new_task_node->stmts.push_back(if_then_else);

    // Copy resource usage flags from original tasks (take union)
    bool uses_cuda_core = false;
    bool uses_tma_core = false;
    bool uses_tensor_core = false;
    int64_t total_latency = 0;
    int64_t max_ii = 0;
    for (TaskNode* task : unit.tasks) {
      uses_cuda_core = uses_cuda_core || task->UsesCUDACore();
      uses_tma_core = uses_tma_core || task->UsesTMACore();
      uses_tensor_core = uses_tensor_core || task->UsesTensorCore();
      total_latency += task->GetLatency();
      max_ii = std::max(max_ii, task->GetII());
    }
    new_task_node->SetUsesCUDACore(uses_cuda_core);
    new_task_node->SetUsesTMACore(uses_tma_core);
    new_task_node->SetUsesTensorCore(uses_tensor_core);
    new_task_node->SetLatency(total_latency);
    new_task_node->SetII(max_ii);
    new_task_node->SetWarpgroupId(-1);  // mixed

    // Also copy read/write regions from original tasks
    for (TaskNode* task : unit.tasks) {
      auto read_regions = task->GetReadRegions();
      for (const auto& region : read_regions) {
        new_task_node->AddReadRegion(region);
      }
      auto write_regions = task->GetWriteRegions();
      for (const auto& region : write_regions) {
        new_task_node->AddWriteRegion(region);
      }
    }

    // Replace the original tasks with the new conditional task
    for (size_t i = 0; i < num_tasks; ++i) {
      if (i == 0) {
        unit.parent_seq->children[start_idx + i] = std::move(new_task_node);
      } else {
        unit.parent_seq->children[start_idx + i].reset();
      }
    }

    // Clean up empty slots
    std::vector<std::unique_ptr<IRStructure>> new_children;
    new_children.reserve(unit.parent_seq->children.size());
    for (size_t i = 0; i < unit.parent_seq->children.size(); ++i) {
      if (unit.parent_seq->children[i]) {
        new_children.push_back(std::move(unit.parent_seq->children[i]));
      }
    }
    unit.parent_seq->children.swap(new_children);

    // Update the ScheduleUnit
    unit.tasks.clear();
    if (start_idx < unit.parent_seq->children.size()) {
      TaskNode* new_task = static_cast<TaskNode*>(unit.parent_seq->children[start_idx].get());
      unit.tasks.push_back(new_task);
    } else {
      unit.start_idx = 0;
      if (!unit.parent_seq->children.empty()) {
        TaskNode* new_task = static_cast<TaskNode*>(unit.parent_seq->children[0].get());
        unit.tasks.push_back(new_task);
      }
    }

    LOG(INFO) << "Created warpgroup partition for TaskNodes: if " << condition
              << " execute " << warpgroup0_stmts.size() << " statements (warpgroup 0), else execute "
              << warpgroup1_stmts.size() << " statements (warpgroup 1)";
    LOG(INFO) << "Replaced " << num_tasks << " tasks with 1 conditional task";
  }
}

// Apply warpgroup partition to entire IRStructure (top-level IfThenElse)
std::unique_ptr<IRStructure> ApplyWarpgroupPartitionToIRStructure(IRStructure* root, IterVar thread_var) {
  if (!root) return nullptr;

  // Check if there are tasks with mixed warpgroup ids
  std::vector<TaskNode*> all_tasks;
  CollectTaskNodesFromIRStructure(root, all_tasks);

  bool has_warpgroup0 = false;
  bool has_warpgroup1 = false;
  for (TaskNode* task : all_tasks) {
    int wg_id = task->GetWarpgroupId();
    if (wg_id == 0) has_warpgroup0 = true;
    else if (wg_id == 1) has_warpgroup1 = true;
  }

  // If all tasks belong to the same warpgroup, no partition needed
  if (!(has_warpgroup0 && has_warpgroup1)) {
    LOG(INFO) << "All tasks belong to the same warpgroup, skipping partition for entire IRStructure";
    // Return a clone of the original structure
    if (root->IsTask()) {
      return static_cast<TaskNode*>(root)->Clone();
    } else if (root->IsSequence()) {
      return static_cast<SequenceNode*>(root)->Clone();
    } else if (root->IsControl()) {
      return static_cast<ControlNode*>(root)->Clone();
    }
    return nullptr;
  }

  LOG(INFO) << "Applying warpgroup partition to entire IRStructure with " << all_tasks.size() << " tasks";

  // Clone IRStructure for warpgroup 0 and warpgroup 1
  auto wg0_structure = CloneIRStructureWithWarpgroupFilter(root, 0);
  auto wg1_structure = CloneIRStructureWithWarpgroupFilter(root, 1);

  // Check if both clones have actual statements
  auto has_actual_statements = [](IRStructure* node) -> bool {
    std::vector<TaskNode*> tasks;
    CollectTaskNodesFromIRStructure(node, tasks);
    for (TaskNode* task : tasks) {
      if (!task->stmts.empty()) {
        return true;
      }
    }
    return false;
  };

  bool wg0_has_stmts = has_actual_statements(wg0_structure.get());
  bool wg1_has_stmts = has_actual_statements(wg1_structure.get());

  // Prepare condition: tx < original_threads
  PrimExpr condition;
  if (thread_var.defined() && thread_var->dom.defined()) {
    Var thread_idx_var = thread_var->var;
    Range thread_dom = thread_var->dom;
    PrimExpr original_threads = thread_dom->extent;
    condition = thread_idx_var < original_threads;
    LOG(INFO) << "Using thread index variable " << thread_idx_var
              << " with original domain " << thread_dom->min << " to " << thread_dom->min + thread_dom->extent
              << " (threads will be doubled for warpgroup partition)"
              << ", condition: " << condition;
  } else {
    LOG(WARNING) << "Thread index variable not properly defined, falling back to default tx < 128";
    Var tx_var("tx", DataType::Int(32));
    condition = tx_var < IntImm(DataType::Int(32), 128);
  }

  // Convert IRStructure to Stmt for IfThenElse
  std::function<Stmt(IRStructure*)> irstructure_to_stmt;
  irstructure_to_stmt = [&irstructure_to_stmt](IRStructure* structure) -> Stmt {
    if (!structure) {
      return Evaluate(0);
    }

    if (structure->IsTask()) {
      TaskNode* task = static_cast<TaskNode*>(structure);
      if (task->stmts.empty()) {
        return Evaluate(0);
      } else if (task->stmts.size() == 1) {
        return task->stmts[0];
      } else {
        return SeqStmt(task->stmts);
      }
    } else if (structure->IsSequence()) {
      SequenceNode* seq = static_cast<SequenceNode*>(structure);
      std::vector<Stmt> stmts;
      for (const auto& child : seq->children) {
        if (child) {
          Stmt child_stmt = irstructure_to_stmt(child.get());
          stmts.push_back(child_stmt);
        }
      }
      if (stmts.empty()) {
        return Evaluate(0);
      } else if (stmts.size() == 1) {
        return stmts[0];
      } else {
        return SeqStmt(stmts);
      }
    } else if (structure->IsControl()) {
      ControlNode* ctrl = static_cast<ControlNode*>(structure);
      Stmt body = Evaluate(0);
      if (ctrl->child) {
        body = irstructure_to_stmt(ctrl->child.get());
      }
      return For(ctrl->control->loop_var, ctrl->control->min,
                ctrl->control->extent, ctrl->control->kind,
                body, ctrl->control->thread_binding,
                ctrl->control->annotations);
    }

    LOG(WARNING) << "Failed to convert IRStructure to Stmt, returning empty statement";
    return Evaluate(0);
  };

  Stmt then_body = wg0_has_stmts ? irstructure_to_stmt(wg0_structure.get()) : Evaluate(0);
  Stmt else_body = wg1_has_stmts ? irstructure_to_stmt(wg1_structure.get()) : Evaluate(0);

  // Create IfThenElse statement
  Stmt if_then_else;
  if (wg0_has_stmts && wg1_has_stmts) {
    if_then_else = IfThenElse(condition, then_body, else_body);
  } else if (wg0_has_stmts) {
    // Only warpgroup 0 has statements, execute unconditionally
    if_then_else = then_body;
  } else if (wg1_has_stmts) {
    // Only warpgroup 1 has statements, execute unconditionally
    if_then_else = else_body;
  } else {
    LOG(WARNING) << "Both warpgroups have no statements, returning original structure";
    if (root->IsTask()) {
      return static_cast<TaskNode*>(root)->Clone();
    } else if (root->IsSequence()) {
      return static_cast<SequenceNode*>(root)->Clone();
    } else if (root->IsControl()) {
      return static_cast<ControlNode*>(root)->Clone();
    }
    return nullptr;
  }

  // Create appropriate IRStructure based on original node type
  if (root->IsControl()) {
    // For ControlNode, we need to preserve the outer For loop
    // but filter tasks inside based on warpgroup id
    const ControlNode* original_ctrl = static_cast<const ControlNode*>(root);

    // Clone and filter the child of ControlNode for each warpgroup
    std::unique_ptr<IRStructure> wg0_child = nullptr;
    std::unique_ptr<IRStructure> wg1_child = nullptr;
    if (original_ctrl->child) {
      wg0_child = CloneIRStructureWithWarpgroupFilter(original_ctrl->child.get(), 0);
      wg1_child = CloneIRStructureWithWarpgroupFilter(original_ctrl->child.get(), 1);
    }

    // Convert filtered children to Stmt
    Stmt then_body = wg0_child ? irstructure_to_stmt(wg0_child.get()) : Evaluate(0);
    Stmt else_body = wg1_child ? irstructure_to_stmt(wg1_child.get()) : Evaluate(0);

    // Create IfThenElse statement
    Stmt if_then_else_stmt;

    // Check if each warpgroup has actual statements
    bool wg0_has_stmts = wg0_child ? has_actual_statements(wg0_child.get()) : false;
    bool wg1_has_stmts = wg1_child ? has_actual_statements(wg1_child.get()) : false;

    if (wg0_has_stmts && wg1_has_stmts) {
      if_then_else_stmt = IfThenElse(condition, then_body, else_body);
    } else if (wg0_has_stmts) {
      // Only warpgroup 0 has statements, execute unconditionally
      if_then_else_stmt = then_body;
    } else if (wg1_has_stmts) {
      // Only warpgroup 1 has statements, execute unconditionally
      if_then_else_stmt = else_body;
    } else {
      LOG(WARNING) << "Both warpgroups have no statements, creating empty statement";
      if_then_else_stmt = Evaluate(0);
    }

    // Create new ControlNode with the same For loop parameters
    auto new_ctrl = std::make_unique<ControlNode>();
    new_ctrl->control = original_ctrl->control;

    // Create a TaskNode to hold the IfThenElse statement as the body
    auto body_task_node = std::make_unique<TaskNode>();
    body_task_node->stmts.push_back(if_then_else_stmt);

    // Copy resource usage flags from all tasks (take union)
    bool uses_cuda_core = false;
    bool uses_tma_core = false;
    bool uses_tensor_core = false;
    int64_t total_latency = 0;
    int64_t max_ii = 0;
    for (TaskNode* task : all_tasks) {
      uses_cuda_core = uses_cuda_core || task->UsesCUDACore();
      uses_tma_core = uses_tma_core || task->UsesTMACore();
      uses_tensor_core = uses_tensor_core || task->UsesTensorCore();
      total_latency += task->GetLatency();
      max_ii = std::max(max_ii, task->GetII());
    }
    body_task_node->SetUsesCUDACore(uses_cuda_core);
    body_task_node->SetUsesTMACore(uses_tma_core);
    body_task_node->SetUsesTensorCore(uses_tensor_core);
    body_task_node->SetLatency(total_latency);
    body_task_node->SetII(max_ii);
    body_task_node->SetWarpgroupId(-1);  // mixed

    // Also copy read/write regions from all tasks
    for (TaskNode* task : all_tasks) {
      auto read_regions = task->GetReadRegions();
      for (const auto& region : read_regions) {
        body_task_node->AddReadRegion(region);
      }
      auto write_regions = task->GetWriteRegions();
      for (const auto& region : write_regions) {
        body_task_node->AddWriteRegion(region);
      }
    }

    new_ctrl->child = std::move(body_task_node);
    LOG(INFO) << "Created warpgroup partition inside ControlNode: if " << condition
              << " execute warpgroup 0, else execute warpgroup 1";
    return new_ctrl;

  } else if (root->IsSequence()) {
    // Create a new SequenceNode with a single TaskNode containing the IfThenElse
    auto new_seq = std::make_unique<SequenceNode>();
    auto new_task_node = std::make_unique<TaskNode>();
    new_task_node->stmts.push_back(if_then_else);

    // Copy resource usage flags from all tasks (take union)
    bool uses_cuda_core = false;
    bool uses_tma_core = false;
    bool uses_tensor_core = false;
    int64_t total_latency = 0;
    int64_t max_ii = 0;
    for (TaskNode* task : all_tasks) {
      uses_cuda_core = uses_cuda_core || task->UsesCUDACore();
      uses_tma_core = uses_tma_core || task->UsesTMACore();
      uses_tensor_core = uses_tensor_core || task->UsesTensorCore();
      total_latency += task->GetLatency();
      max_ii = std::max(max_ii, task->GetII());
    }
    new_task_node->SetUsesCUDACore(uses_cuda_core);
    new_task_node->SetUsesTMACore(uses_tma_core);
    new_task_node->SetUsesTensorCore(uses_tensor_core);
    new_task_node->SetLatency(total_latency);
    new_task_node->SetII(max_ii);
    new_task_node->SetWarpgroupId(-1);  // mixed

    // Also copy read/write regions from all tasks
    for (TaskNode* task : all_tasks) {
      auto read_regions = task->GetReadRegions();
      for (const auto& region : read_regions) {
        new_task_node->AddReadRegion(region);
      }
      auto write_regions = task->GetWriteRegions();
      for (const auto& region : write_regions) {
        new_task_node->AddWriteRegion(region);
      }
    }

    new_seq->children.push_back(std::move(new_task_node));
    LOG(INFO) << "Created warpgroup partition in SequenceNode: if " << condition
              << " execute warpgroup 0, else execute warpgroup 1";
    return new_seq;

  } else {
    // TaskNode or unknown type - create a TaskNode containing the IfThenElse statement
    auto new_task_node = std::make_unique<TaskNode>();
    new_task_node->stmts.push_back(if_then_else);

    // Copy resource usage flags from all tasks (take union)
    bool uses_cuda_core = false;
    bool uses_tma_core = false;
    bool uses_tensor_core = false;
    int64_t total_latency = 0;
    int64_t max_ii = 0;
    for (TaskNode* task : all_tasks) {
      uses_cuda_core = uses_cuda_core || task->UsesCUDACore();
      uses_tma_core = uses_tma_core || task->UsesTMACore();
      uses_tensor_core = uses_tensor_core || task->UsesTensorCore();
      total_latency += task->GetLatency();
      max_ii = std::max(max_ii, task->GetII());
    }
    new_task_node->SetUsesCUDACore(uses_cuda_core);
    new_task_node->SetUsesTMACore(uses_tma_core);
    new_task_node->SetUsesTensorCore(uses_tensor_core);
    new_task_node->SetLatency(total_latency);
    new_task_node->SetII(max_ii);
    new_task_node->SetWarpgroupId(-1);  // mixed

    // Also copy read/write regions from all tasks
    for (TaskNode* task : all_tasks) {
      auto read_regions = task->GetReadRegions();
      for (const auto& region : read_regions) {
        new_task_node->AddReadRegion(region);
      }
      auto write_regions = task->GetWriteRegions();
      for (const auto& region : write_regions) {
        new_task_node->AddWriteRegion(region);
      }
    }

    LOG(INFO) << "Created warpgroup partition in TaskNode: if " << condition
              << " execute warpgroup 0, else execute warpgroup 1";

    return new_task_node;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm