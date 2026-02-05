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

#include <unordered_set>


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
#include <unordered_set>
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
std::unique_ptr<IRStructure> ApplyWarpgroupPartitionToIRStructure(IRStructure* root, IterVar thread_var);
void CollectTaskNodesFromIRStructure(IRStructure* node, std::vector<TaskNode*>& tasks);
void CollectIRStructureNodes(IRStructure* node, std::vector<IRStructure*>& nodes);
// Barrier synchronization helper functions
static Stmt InsertBarriersForNeutralSync(Stmt neutral_body, Stmt warpgroup_body, PrimExpr thread_count);
// Barrier dependency analysis functions
static void AnalyzeAndInsertBarriers(IRStructure* node, int& next_barrier_id);
static void AnalyzeSequenceNodeBarriers(SequenceNode* seq, int& next_barrier_id);
static void AnalyzeControlNodeBarriers(ControlNode* ctrl, int& next_barrier_id);


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



// Check if two IRStructures use the same register region
// Used for warpgroup specialization: different warpgroups cannot share registers




// Helper functions for barrier insertion (preliminary implementation)
// These functions implement barrier_arrive/barrier_wait using the underlying
// ptx_arrive_barrier and mbarrier_wait_parity operations.
static PrimExpr makeGetBarrier(PrimExpr barrier_id) {
  return Call(DataType::Handle(), get_mbarrier(), {std::move(barrier_id)});
}

// Create a barrier_arrive statement for the given barrier ID
// Equivalent to T.barrier_arrive(barrier_id) in Python
static Stmt makeBarrierArrive(PrimExpr barrier_id, int cta_id = -1,
                              const PrimExpr &pred = 1) {
  Array<PrimExpr> args = {makeGetBarrier(std::move(barrier_id))};
  if (cta_id != -1) {
    args.push_back(cta_id);
    args.push_back(pred);
  }
  return Evaluate(
      Call(DataType::Handle(), builtin::ptx_arrive_barrier(), args));
}

// Create a barrier_wait statement for the given barrier ID and parity
// Equivalent to T.barrier_wait(barrier_id, parity) in Python
static Stmt makeBarrierWait(PrimExpr barrier_id, PrimExpr parity) {
  auto call = Call(DataType::Handle(), mbarrier_wait_parity(),
                   {makeGetBarrier(std::move(barrier_id)), std::move(parity)});
  return Evaluate(call);
}

// Create a barrier allocation statement with arrive_count
// Equivalent to T.alloc_barrier(arrive_count) in Python
static Stmt makeAllocBarrier(PrimExpr arrive_count) {
  // Note: In the actual implementation, this would allocate a barrier buffer
  // For now, we'll create a placeholder that will be replaced by actual barrier allocation
  // The barrier ID will be assigned later
  return Evaluate(Call(DataType::Handle(), create_list_of_mbarrier(), {arrive_count}));
}

// Insert barriers between neutral tasks and warpgroup-specific work
// This ensures neutral tasks complete before any warpgroup-specific work begins
static Stmt InsertBarriersForNeutralSync(Stmt neutral_body, Stmt warpgroup_body, PrimExpr thread_count) {
  // If either body is empty, no barriers needed
  if (neutral_body.as<EvaluateNode>() && warpgroup_body.as<EvaluateNode>()) {
    return SeqStmt({neutral_body, warpgroup_body});
  }

  LOG(WARNING) << "Using preliminary barrier insertion for neutral-to-warpgroup synchronization";
  LOG(WARNING) << "Proper barrier analysis and insertion needed for shared/global region dependencies";

  // Allocate barrier buffer for neutral-to-warpgroup synchronization
  // Equivalent to Python: barrier = T.alloc_buffer((thread_count*2,), "uint64", scope="shared.barrier")
  // Using arrive_count = thread_count * 2 (number of threads * 2 for neutral-to-warpgroup synchronization)
  Stmt alloc_barrier = makeAllocBarrier(thread_count * IntImm(DataType::Int(32), 2));

  // Use barrier buffer 0 for neutral-to-warpgroup synchronization
  // Parity 0 for wait, parity 1 for arrive (simplified)
  Stmt arrive_barrier = makeBarrierArrive(0);
  Stmt wait_barrier = makeBarrierWait(0, 0);

  // Combine: neutral_body -> arrive_barrier -> wait_barrier -> warpgroup_body
  std::vector<Stmt> stmts;
  if (!neutral_body.as<EvaluateNode>()) {
    stmts.push_back(neutral_body);
  }
  stmts.push_back(arrive_barrier);
  stmts.push_back(wait_barrier);
  if (!warpgroup_body.as<EvaluateNode>()) {
    stmts.push_back(warpgroup_body);
  }

  Stmt sync_body;
  if (stmts.size() == 1) {
    sync_body = stmts[0];
  } else {
    sync_body = SeqStmt(stmts);
  }

  // Combine barrier allocation with the synchronization body
  std::vector<Stmt> all_stmts;
  all_stmts.push_back(alloc_barrier);
  all_stmts.push_back(sync_body);
  return SeqStmt(all_stmts);
}

// Helper function to insert a statement into TaskNode's stmts, handling IfThenElse if present
static void InsertStatementIntoTaskNode(TaskNode* task, const Stmt& stmt, bool at_beginning = false) {
  if (task->stmts.empty()) {
    // If no statements, just add the statement
    task->stmts.push_back(stmt);
    return;
  }

  // Check if the first statement is an IfThenElse
  if (const auto* if_node = task->stmts[0].as<IfThenElseNode>()) {
    // Get the IfThenElse statement
    Stmt if_stmt = task->stmts[0];

    // Extract then case (always present)
    Stmt then_case = if_node->then_case;

    // Extract else case (optional)
    Optional<Stmt> else_case = if_node->else_case;

    // Insert statement into the appropriate case
    if (at_beginning) {
      // Insert at beginning of then_case
      if (const auto* seq_node = then_case.as<SeqStmtNode>()) {
        // If then_case is a SeqStmt, insert at beginning
        Array<Stmt> new_seq_stmts;
        new_seq_stmts.push_back(stmt);
        for (const auto& s : seq_node->seq) {
          new_seq_stmts.push_back(s);
        }
        then_case = SeqStmt(new_seq_stmts);
      } else {
        // If then_case is a single statement, create a SeqStmt
        then_case = SeqStmt({stmt, then_case});
      }
    } else {
      // Insert at end of then_case
      if (const auto* seq_node = then_case.as<SeqStmtNode>()) {
        // If then_case is a SeqStmt, append to end
        Array<Stmt> new_seq_stmts;
        for (const auto& s : seq_node->seq) {
          new_seq_stmts.push_back(s);
        }
        new_seq_stmts.push_back(stmt);
        then_case = SeqStmt(new_seq_stmts);
      } else {
        // If then_case is a single statement, create a SeqStmt
        then_case = SeqStmt({then_case, stmt});
      }
    }

    // Create new IfThenElse statement with updated then_case
    Stmt new_if_stmt;
    if (else_case.defined()) {
      new_if_stmt = IfThenElse(if_node->condition, then_case, else_case.value());
    } else {
      new_if_stmt = IfThenElse(if_node->condition, then_case);
    }

    // Replace the old IfThenElse with the new one
    task->stmts[0] = new_if_stmt;
  } else {
    // No IfThenElse, just insert into stmts vector
    if (at_beginning) {
      task->stmts.insert(task->stmts.begin(), stmt);
    } else {
      task->stmts.push_back(stmt);
    }
  }
}

// Barrier dependency analysis implementation
static void AnalyzeAndInsertBarriers(IRStructure* node, int& next_barrier_id) {
  if (!node) return;

  if (node->IsSequence()) {
    AnalyzeSequenceNodeBarriers(static_cast<SequenceNode*>(node), next_barrier_id);
  } else if (node->IsControl()) {
    AnalyzeControlNodeBarriers(static_cast<ControlNode*>(node), next_barrier_id);
  }
  // For TaskNode, nothing to do at this level
}

static void AnalyzeSequenceNodeBarriers(SequenceNode* seq, int& next_barrier_id) {
  if (!seq) return;

  // Map from buffer to (task, warpgroup_id) of last write
  std::unordered_map<Buffer, TaskNode*, ObjectPtrHash, ObjectPtrEqual> last_write_map;

  // Process tasks in sequence order
  for (auto& child : seq->children) {
    if (!child || !child->IsTask()) {
      // If child is SequenceNode or ControlNode, recursively analyze it
      AnalyzeAndInsertBarriers(child.get(), next_barrier_id);
      continue;
    }

    TaskNode* task = static_cast<TaskNode*>(child.get());
    int wg_id = task->GetWarpgroupId();
    if (wg_id == -1) continue;

    // Check read regions for dependencies
    for (const auto& read_region : task->GetReadRegions()) {
      Buffer buffer = read_region->buffer;
      auto it = last_write_map.find(buffer);
      if (it != last_write_map.end()) {
        TaskNode* last_write_task = it->second;
        int last_wg_id = last_write_task->GetWarpgroupId();
        if (last_wg_id == -1) continue;

        // If warpgroup ids differ, insert barrier
        if (last_wg_id != wg_id) {
          // Allocate a new barrier ID
          int barrier_id = next_barrier_id++;
          LOG(INFO) << "Inserting barrier " << barrier_id << " between task in warpgroup " << last_wg_id
                    << " and task in warpgroup " << wg_id << " for buffer " << buffer;

          // Insert barrier_arrive at the end of last_write_task's statements
          Stmt arrive_stmt = makeBarrierArrive(barrier_id);
          InsertStatementIntoTaskNode(last_write_task, arrive_stmt, false);
          LOG(INFO) << "Added barrier_arrive(" << barrier_id << ") to task in warpgroup " << last_wg_id;

          // Insert barrier_wait at the beginning of task's statements
          Stmt wait_stmt = makeBarrierWait(barrier_id, 0); // parity = 0 for non-loop barriers
          InsertStatementIntoTaskNode(task, wait_stmt, true);
          LOG(INFO) << "Added barrier_wait(" << barrier_id << ", parity=0) to task in warpgroup " << wg_id;
        }

        // Remove from map after read (as per user instruction)
        last_write_map.erase(it);
      }
    }

    // Update write regions
    for (const auto& write_region : task->GetWriteRegions()) {
      Buffer buffer = write_region->buffer;
      last_write_map[buffer] = task;
    }
  }
}

static void AnalyzeControlNodeBarriers(ControlNode* ctrl, int& next_barrier_id) {
  if (!ctrl || !ctrl->child) return;

  // Get loop information
  const ForNode* for_node = ctrl->control.get();
  if (!for_node) return;

  PrimExpr loop_var = for_node->loop_var;
  PrimExpr loop_start = for_node->min;
  PrimExpr loop_step = for_node->step.has_value() ? for_node->step.value() : IntImm(DataType::Int(32), 1);


  // If child is a SequenceNode, we need special handling for promote/non-promote tasks
  if (ctrl->child->IsSequence()) {
    SequenceNode* seq = static_cast<SequenceNode*>(ctrl->child.get());

    // Separate promoted and non-promoted tasks
    std::vector<TaskNode*> promoted_tasks;
    std::vector<TaskNode*> non_promoted_tasks;

    // Collect all tasks from the sequence
    std::vector<TaskNode*> all_tasks;
    for (auto& child : seq->children) {
      if (child && child->IsTask()) {
        TaskNode* task = static_cast<TaskNode*>(child.get());
        all_tasks.push_back(task);
      }
    }

    // Separate by promote flag
    for (TaskNode* task : all_tasks) {
      if (task->GetPromote()) {
        promoted_tasks.push_back(task);
      } else {
        non_promoted_tasks.push_back(task);
      }
    }

    // Process in order: promoted tasks first, then non-promoted tasks
    // This matches the software pipelining order
    std::vector<TaskNode*> ordered_tasks;
    ordered_tasks.insert(ordered_tasks.end(), promoted_tasks.begin(), promoted_tasks.end());
    ordered_tasks.insert(ordered_tasks.end(), non_promoted_tasks.begin(), non_promoted_tasks.end());

    // Map from buffer to (task, warpgroup_id, is_promoted)
    std::unordered_map<Buffer, TaskNode*, ObjectPtrHash, ObjectPtrEqual> last_write_map;

    // Process tasks in the specified order
    for (TaskNode* task : ordered_tasks) {
      int wg_id = task->GetWarpgroupId();
      bool is_promoted = task->GetPromote();
      if (wg_id == -1) continue;

      // Check read regions for dependencies
      for (const auto& read_region : task->GetReadRegions()) {
        Buffer buffer = read_region->buffer;
        auto it = last_write_map.find(buffer);
        if (it != last_write_map.end()) {
          TaskNode* last_write_task = it->second;
          int last_wg_id = last_write_task->GetWarpgroupId();
          bool last_is_promoted = last_write_task->GetPromote();
          if (last_wg_id == -1) continue;

          // If warpgroup ids differ, insert barrier
          if (last_wg_id != wg_id || last_is_promoted != is_promoted) {
            // Allocate a new barrier ID
            int barrier_id = next_barrier_id++;

            // Calculate parity for barrier wait
            // parity = indexmod(loop_var - loop_start + promote + 1, 2)
            // where promote is 1 if the waiting task is promoted, 0 otherwise
            // The waiting task is 'task' (the reading task)
            int promote_int = is_promoted ? 1 : 0;
            PrimExpr parity_expr = tvm::indexmod(loop_var - loop_start + IntImm(DataType::Int(32), promote_int + 1), 2);

            LOG(INFO) << "Inserting barrier " << barrier_id << " in loop between task in warpgroup " << last_wg_id
                      << " and task in warpgroup " << wg_id << " for buffer " << buffer;
            LOG(INFO) << "  Reading task is " << (is_promoted ? "promoted" : "non-promoted") << " (promote=" << promote_int << ")";
            LOG(INFO) << "  Writing task was " << (last_is_promoted ? "promoted" : "non-promoted");
            LOG(INFO) << "  Parity expression: " << parity_expr;

            // Insert barrier_arrive at the end of last_write_task's statements
            Stmt arrive_stmt = makeBarrierArrive(barrier_id);
            InsertStatementIntoTaskNode(last_write_task, arrive_stmt, false);
            LOG(INFO) << "Added barrier_arrive(" << barrier_id << ") to task in warpgroup " << last_wg_id;

            // Insert barrier_wait at the beginning of task's statements
            Stmt wait_stmt = makeBarrierWait(barrier_id, parity_expr);
            InsertStatementIntoTaskNode(task, wait_stmt, true);
            LOG(INFO) << "Added barrier_wait(" << barrier_id << ", parity=" << parity_expr << ") to task in warpgroup " << wg_id;
          }

          // Remove from map after read (as per user instruction)
          last_write_map.erase(it);
        }
      }

      // Update write regions
      for (const auto& write_region : task->GetWriteRegions()) {
        Buffer buffer = write_region->buffer;
        last_write_map[buffer] = task;
      }
    }
  } else {
    AnalyzeAndInsertBarriers(ctrl->child.get(), next_barrier_id);
  }
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

// Helper function to collect all prefix tasks (consecutive tasks without register region at the beginning of sequences)
void CollectPrefixTasks(IRStructure* node, std::unordered_set<TaskNode*>& prefix_tasks, ControlNode* current_control_node) {
  if (!node) return;

  if (node->IsSequence()) {
    SequenceNode* seq = static_cast<SequenceNode*>(node);
    // Check if we're inside a ControlNode - prefix optimization is not applied inside ControlNode
    if (current_control_node == nullptr) {
      // Find consecutive tasks without register region at the beginning of this sequence
      bool found_reg = false;
      for (size_t i = 0; i < seq->children.size(); i++) {
        if (!seq->children[i]) continue;

        if (seq->children[i]->IsTask()) {
          if (!found_reg && CountRegisterRegions(seq->children[i].get()) == 0) {
            // This is a prefix task (consecutive task without register region at the beginning)
            TaskNode* task = static_cast<TaskNode*>(seq->children[i].get());
            prefix_tasks.insert(task);
          } else {
            // Found a task with register region or after a task with register region
            found_reg = true;
          }
        } else if (seq->children[i]->IsSequence() || seq->children[i]->IsControl()) {
          // If we encounter a nested Sequence or Control node, we need to recurse
          // But this breaks the consecutive sequence at the top level
          CollectPrefixTasks(seq->children[i].get(), prefix_tasks, current_control_node);
          // After encountering a non-task node, we can't have more prefix tasks at this level
          found_reg = true;
        }
      }
    }
  } else if (node->IsControl()) {
    ControlNode* ctrl = static_cast<ControlNode*>(node);
    // When entering a control node, update the current control context
    CollectPrefixTasks(ctrl->child.get(), prefix_tasks, ctrl);
  } else if (node->IsTask()) {
    // Task node outside of a sequence context - not a prefix task
    return;
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
  return latency_;
}

int64_t SequenceNode::GetII() const {
  return ii_;
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
  latency_ = latency;
}

void SequenceNode::SetII(int64_t ii) {
  ii_ = ii;
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
  // Copy latency and II
  new_ctrl->SetLatency(GetLatency());
  new_ctrl->SetII(GetII());
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

  // Collect all prefix tasks (consecutive tasks without register region at the beginning of sequences)
  std::unordered_set<TaskNode*> prefix_tasks;
  CollectPrefixTasks(root, prefix_tasks, nullptr);

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

  // Step 5: Calculate weighted latency for each component that has register regions
  std::vector<ComponentInfo> component_infos;
  for (const auto& kv : components) {
    int root = kv.first;
    const std::vector<int>& indices = kv.second;
    // Check if this component has any task with register regions
    bool has_register_region = false;
    for (int idx : indices) {
      if (CountRegisterRegions(all_tasks[idx].task) > 0) {
        has_register_region = true;
        break;
      }
    }
    // Skip components without any register region only if they are prefix tasks
    // (consecutive tasks without register region at the beginning of sequences)
    // Other tasks without register region should still participate in warpgroup specialize
    if (!has_register_region) {
      // Check if this component is a prefix task
      // A component is a prefix task if it contains exactly one task and that task is in prefix_tasks
      if (indices.size() == 1) {
        TaskNode* task = all_tasks[indices[0]].task;
        if (prefix_tasks.find(task) != prefix_tasks.end()) {
          // This is a prefix task, skip it (it won't participate in warpgroup specialize)
          continue;
        }
      }
      // If not a prefix task, we should include it in component_infos
    }
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

// Builder that collects ScheduleUnits from IRStructure
class ScheduleUnitBuilder {
public:
  void Build(IRStructure* root) {
    LOG(INFO) << "[Build] Starting ScheduleUnitBuilder::Build (new recursive version)";
    scheduled_control_nodes_.clear();
    LOG(INFO) << "[Build] Cleared scheduled control nodes set";

    // Global warpgroup id assignment from the top level
    LOG(INFO) << "Performing global warpgroup id assignment...";
    AssignWarpgroupIdsGlobal(root);

    LOG(INFO) << "[Build] Calling ScheduleRecursive on root IRStructure";
    ScheduleRecursive(root);
    LOG(INFO) << "[Build] ScheduleRecursive completed";

    LOG(INFO) << "[Build] Build completed, IRStructure has been scheduled in place";
  }

  // New recursive scheduling function that replaces Collect method
  // Directly schedules the entire IRStructure tree recursively in place
  void ScheduleRecursive(IRStructure* node);

  // Z3-based scheduler that calls Python implementation via FFI
  std::vector<IRStructure*> Z3SchedulePython(const std::vector<IRStructure*>& nodes) {
    size_t n = nodes.size();
    if (n <= 1) {
      if (n == 1) {
        // For TaskNode, set start time
        if (nodes[0]->IsTask()) {
          TaskNode* task = static_cast<TaskNode*>(nodes[0]);
          task->SetStartTime(0);
        }
      }
      return nodes;
    }

    try {
      // Get the Python-registered function using ffi::Function::GetGlobal
      static std::optional<tvm::ffi::Function> z3_schedule_func =
          tvm::ffi::Function::GetGlobal("tl.transform.z3_schedule_python");
      if (!z3_schedule_func.has_value()) {
        LOG(WARNING) << "Python Z3 scheduler not registered, falling back to topological sort";
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
        const IRStructure* node = nodes[i];
        latencies.push_back(node->GetLatency());
        iis.push_back(node->GetII());

        // Encode resource flags as bitmask
        int64_t flags = 0;
        if (node->UsesCUDACore()) flags |= 1;
        if (node->UsesTMACore()) flags |= 2;
        if (node->UsesTensorCore()) flags |= 4;
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
        return nodes;
      }

      // Apply start times to nodes
      for (size_t i = 0; i < n; ++i) {
        // Only TaskNode has SetStartTime method
        if (nodes[i]->IsTask()) {
          TaskNode* task = static_cast<TaskNode*>(nodes[i]);
          task->SetStartTime(start_times[i]);
        }
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

      // Create sorted node list
      std::vector<IRStructure*> sorted_nodes;
      sorted_nodes.reserve(n);
      for (const auto& p : start_time_with_idx) {
        sorted_nodes.push_back(nodes[p.second]);
      }

      return sorted_nodes;

    } catch (const std::exception& e) {
      LOG(WARNING) << "Python Z3 scheduler failed with exception: " << e.what()
                   << ", falling back to topological sort";
      return nodes;
    } catch (...) {
      LOG(WARNING) << "Python Z3 scheduler failed with unknown exception, falling back to topological sort";
      return nodes;
    }
  }

  // Z3-based scheduler for loops that calls Python implementation via FFI
  // with distance-aware dependencies
  std::pair<std::vector<IRStructure*>, int64_t> Z3SchedulePythonLoop(const std::vector<IRStructure*>& nodes) {
    size_t n = nodes.size();
    int64_t ii = 1; // default II
    if (n <= 1) {
      if (n == 1) {
        // Only TaskNode has SetStartTime method
        if (nodes[0]->IsTask()) {
          TaskNode* task = static_cast<TaskNode*>(nodes[0]);
          task->SetStartTime(0);
        }
      }
      return std::make_pair(nodes, ii);
    }

    try {
      // Get the Python-registered function using ffi::Function::GetGlobal
      static std::optional<tvm::ffi::Function> z3_schedule_loop_func =
          tvm::ffi::Function::GetGlobal("tl.transform.z3_schedule_loop_python");
      if (!z3_schedule_loop_func.has_value()) {
        LOG(WARNING) << "Python Z3 loop scheduler not registered, falling back to topological sort";
        return std::make_pair(nodes, ii);
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
        const IRStructure* node = nodes[i];
        latencies.push_back(node->GetLatency());
        iis.push_back(node->GetII());

        // Encode resource flags as bitmask
        int64_t flags = 0;
        if (node->UsesCUDACore()) flags |= 1;
        if (node->UsesTMACore()) flags |= 2;
        if (node->UsesTensorCore()) flags |= 4;
        resource_flags.push_back(flags);
      }

      // Collect data dependencies with distance
      // distance = 0 if i < j (same iteration), distance = 1 if i > j (next iteration)
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          if (i == j) continue;  // Skip self-dependency
          if (HasDependency(nodes[i], nodes[j])) {
            // distance = 0 if i < j, 1 if i > j
            int64_t distance = (i < j) ? 0 : 1;
            data_deps.emplace_back(i, j, distance);
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
                                                  tvm_data_deps, tvm_resource_deps).cast<ffi::Tuple<ffi::Array<int64_t>, ffi::Array<bool>, int64_t>>();

      ffi::Array<int64_t> start_times = return_val.get<0>();
      ffi::Array<bool> promotes = return_val.get<1>();
      ii = return_val.get<2>();


      if (start_times.size() != n || promotes.size() != n) {
        LOG(WARNING) << "Python Z3 loop scheduler returned invalid results (size mismatch), falling back to topological sort";
        return std::make_pair(nodes, ii);
      }

      // Apply start times and promote flags to nodes
      size_t num_promoted = 0;
      for (size_t i = 0; i < n; ++i) {
        // Only TaskNode has SetStartTime and SetPromote methods
        if (nodes[i]->IsTask()) {
          TaskNode* task = static_cast<TaskNode*>(nodes[i]);
          task->SetStartTime(start_times[i]);
          bool promote = promotes[i] != 0;  // Convert int to bool
          task->SetPromote(promote);
          if (promote) {
            num_promoted++;
          }
        }
      }
      if (num_promoted > 0) {
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
        start_time_with_idx.emplace_back(start_times[i] + promotes[i] * return_val.get<2>(), i);
      }

      // Sort by start_time, then by original index
      std::sort(start_time_with_idx.begin(), start_time_with_idx.end(),
                [](const std::pair<int64_t, size_t>& a,
                   const std::pair<int64_t, size_t>& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;
                });

      // Create sorted node list
      std::vector<IRStructure*> sorted_nodes;
      sorted_nodes.reserve(n);
      for (const auto& p : start_time_with_idx) {
        sorted_nodes.push_back(nodes[p.second]);
      }

      return std::make_pair(sorted_nodes, ii);

    } catch (const std::exception& e) {
      LOG(WARNING) << "Python Z3 loop scheduler failed with exception: " << e.what()
                   << ", falling back to topological sort";
      return std::make_pair(nodes, ii);
    } catch (...) {
      LOG(WARNING) << "Python Z3 loop scheduler failed with unknown exception, falling back to topological sort";
      return std::make_pair(nodes, ii);
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
  IterVar thread_var_;  // Thread index variable for warpgroup partition
  std::unordered_set<ControlNode*> scheduled_control_nodes_;  // Track already scheduled ControlNodes to avoid duplicate scheduling

  // Check if two regions refer to the same buffer
  bool SameBuffer(const BufferRegion& a, const BufferRegion& b) const {
    return a->buffer.same_as(b->buffer);
  }

  // Check if two IRStructures have data dependency (excluding read-after-read)
  bool HasDependency(const IRStructure* a, const IRStructure* b) const {
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

  // Check if an IRStructure has any register region
  bool HasRegisterRegion(const IRStructure* node) const {
    return CountRegisterRegions(node) > 0;
  }

  // Check if two IRStructures have resource dependency (use same hardware resource)
  bool HasResourceDependency(const IRStructure* a, const IRStructure* b) const {
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

  // Helper function to recursively schedule ControlNode internals
  void ScheduleControlNodeRecursive(ControlNode* ctrl) {
    if (!ctrl || !ctrl->child) return;

    LOG(INFO) << "[ScheduleControlNodeRecursive] Entering ControlNode at " << ctrl;

    // Check if this ControlNode has already been scheduled
    if (scheduled_control_nodes_.find(ctrl) != scheduled_control_nodes_.end()) {
      LOG(INFO) << "[ScheduleControlNodeRecursive] ControlNode at " << ctrl << " already scheduled, skipping";
      return;
    }
    // Mark as scheduled
    scheduled_control_nodes_.insert(ctrl);
    LOG(INFO) << "[ScheduleControlNodeRecursive] Marked ControlNode at " << ctrl << " as scheduled";

    // First, recursively schedule any nested ControlNodes inside this ControlNode
    // If child is a SequenceNode, process its children recursively
    if (ctrl->child->IsSequence()) {
      SequenceNode* seq = static_cast<SequenceNode*>(ctrl->child.get());
      LOG(INFO) << "[ScheduleControlNodeRecursive] Child is SequenceNode with " << seq->children.size() << " children";
      int control_child_count = 0;
      for (auto& child : seq->children) {
        if (child && child->IsControl()) {
          control_child_count++;
          LOG(INFO) << "[ScheduleControlNodeRecursive] Found nested ControlNode at " << child.get();
          ScheduleControlNodeRecursive(static_cast<ControlNode*>(child.get()));
        }
      }
      LOG(INFO) << "[ScheduleControlNodeRecursive] Processed " << control_child_count << " nested ControlNodes";
    } else if (ctrl->child->IsControl()) {
      // Child is itself a ControlNode (nested loop)
      LOG(INFO) << "[ScheduleControlNodeRecursive] Child is itself a ControlNode (nested loop) at " << ctrl->child.get();
      ScheduleControlNodeRecursive(static_cast<ControlNode*>(ctrl->child.get()));
    } else {
      LOG(INFO) << "[ScheduleControlNodeRecursive] Child is a TaskNode or other non-ControlNode type";
    }

    // Now schedule the direct children of this ControlNode
    // Collect direct child nodes (not recursively)
    std::vector<IRStructure*> direct_children;
    if (ctrl->child->IsSequence()) {
      SequenceNode* seq = static_cast<SequenceNode*>(ctrl->child.get());
      for (auto& child : seq->children) {
        if (child) {
          direct_children.push_back(child.get());
          LOG(INFO) << "[ScheduleControlNodeRecursive] Adding child type: "
                    << (child->IsTask() ? "TaskNode" : child->IsControl() ? "ControlNode" : "SequenceNode")
                    << " at " << child.get();
        }
      }
    } else {
      // Child is a single node (TaskNode or ControlNode)
      direct_children.push_back(ctrl->child.get());
      LOG(INFO) << "[ScheduleControlNodeRecursive] Single child type: "
                << (ctrl->child->IsTask() ? "TaskNode" : ctrl->child->IsControl() ? "ControlNode" : "SequenceNode")
                << " at " << ctrl->child.get();
    }
    LOG(INFO) << "[ScheduleControlNodeRecursive] Collected " << direct_children.size() << " direct children";

    if (direct_children.size() <= 1) {
      // No need to schedule if only one or zero direct children
      // Still need to check for promote transformation
      LOG(INFO) << "[ScheduleControlNodeRecursive] Only " << direct_children.size() << " direct children, skipping scheduling";
      CheckAndApplyPromoteTransformation(ctrl);
      return;
    }

    // Schedule the direct children using loop-aware scheduler
    LOG(INFO) << "[ScheduleControlNodeRecursive] Calling Z3SchedulePythonLoop with " << direct_children.size() << " nodes";
    auto result = Z3SchedulePythonLoop(direct_children);
    std::vector<IRStructure*> sorted_nodes = result.first;
    int64_t ii = result.second;
    LOG(INFO) << "[ScheduleControlNodeRecursive] Z3SchedulePythonLoop returned II=" << ii << " and " << sorted_nodes.size() << " sorted nodes";

    // Apply promote transformation if any tasks are marked for promotion
    CheckAndApplyPromoteTransformation(ctrl);

    // Compute overall latency: II * tripcount
    int64_t tripcount = 100; // default if not constant
    if (auto extent_int = ctrl->control.get()->extent.as<IntImm>()) {
      tripcount = extent_int.value()->value;
    }
    int64_t overall_latency = ii * tripcount;
    LOG(INFO) << "ControlNode internal scheduling: II=" << ii << ", tripcount=" << tripcount << ", latency=" << overall_latency;

    // Apply the scheduling result by reordering children if child is a SequenceNode
    if (ctrl->child->IsSequence()) {
      SequenceNode* seq = static_cast<SequenceNode*>(ctrl->child.get());

      // Check if order actually changed
      bool order_changed = false;
      for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        if (sorted_nodes[i] != direct_children[i]) {
          order_changed = true;
          break;
        }
      }

      if (order_changed) {
        // Create a mapping from IRStructure pointer to its index in seq->children
        std::unordered_map<IRStructure*, size_t> node_to_old_index;
        for (size_t i = 0; i < seq->children.size(); ++i) {
          IRStructure* child_ptr = seq->children[i].get();
          node_to_old_index[child_ptr] = i;
        }

        // Reorder seq->children according to sorted_nodes
        std::vector<std::unique_ptr<IRStructure>> reordered_children;
        reordered_children.reserve(seq->children.size());

        for (IRStructure* node : sorted_nodes) {
          auto it = node_to_old_index.find(node);
          if (it == node_to_old_index.end()) {
            LOG(FATAL) << "IRStructure not found in SequenceNode children";
          }
          size_t old_idx = it->second;
          reordered_children.push_back(std::move(seq->children[old_idx]));
        }

        // Replace seq->children with reordered children
        seq->children = std::move(reordered_children);

        LOG(INFO) << "Reordered " << sorted_nodes.size() << " children in ControlNode's SequenceNode";

        // For nested scheduling: encapsulate scheduled children as a single SequenceNode
        // This allows the scheduled unit to be treated as a whole in outer scheduling
        if (sorted_nodes.size() > 1) {
          LOG(INFO) << "Encapsulating scheduled children as SequenceNode for nested scheduling";

          // Create a new SequenceNode
          auto new_seq_node = std::make_unique<SequenceNode>();

          // Move all children into the new SequenceNode
          for (auto& child : seq->children) {
            new_seq_node->children.push_back(std::move(child));
          }

          // Replace the original SequenceNode with the new encapsulated one
          // Note: seq is a pointer to ctrl->child, which is a unique_ptr
          // We need to replace ctrl->child with new_seq_node
          ctrl->child = std::move(new_seq_node);

          LOG(INFO) << "Scheduled children encapsulated as SequenceNode inside ControlNode";
        }
      }
    }
    LOG(INFO) << "[ScheduleControlNodeRecursive] Completed scheduling for ControlNode at " << ctrl;
  }

  // Helper to check and apply promote transformation for a ControlNode
  void CheckAndApplyPromoteTransformation(ControlNode* ctrl) {
    if (!ctrl || !ctrl->child) return;

    // Collect all TaskNodes inside the ControlNode (recursively) for promote check
    std::vector<TaskNode*> tasks;
    CollectTaskNodesFromIRStructure(ctrl->child.get(), tasks);

    bool has_promoted_tasks = false;
    for (TaskNode* task : tasks) {
      if (task->GetPromote()) {
        has_promoted_tasks = true;
        break;
      }
    }

    if (has_promoted_tasks) {
      LOG(INFO) << "ControlNode has promoted tasks, applying promote transformation";
      ApplyPromoteTransformation(tasks, ctrl);
    }
  }





};

// Implementation of ScheduleRecursive function
void ScheduleUnitBuilder::ScheduleRecursive(IRStructure* node) {
  if (!node) return;

  LOG(INFO) << "[ScheduleRecursive] Processing node at " << node
            << " type: " << (node->IsTask() ? "TaskNode" : node->IsControl() ? "ControlNode" : "SequenceNode");

  if (node->IsTask()) {
    // TaskNode: no further scheduling needed
    LOG(INFO) << "[ScheduleRecursive] TaskNode, no scheduling needed";
    return;
  } else if (node->IsSequence()) {
    SequenceNode* seq = static_cast<SequenceNode*>(node);
    LOG(INFO) << "[ScheduleRecursive] SequenceNode with " << seq->children.size() << " children";

    // First, recursively schedule all children
    for (size_t i = 0; i < seq->children.size(); ++i) {
      LOG(INFO) << "[ScheduleRecursive] Recursively scheduling child " << i << " of SequenceNode";
      ScheduleRecursive(seq->children[i].get());
    }

    // Now collect child nodes for potential scheduling
    std::vector<IRStructure*> child_nodes;
    child_nodes.reserve(seq->children.size());
    for (const auto& child : seq->children) {
      child_nodes.push_back(child.get());
    }

    // Only schedule if there are multiple nodes
    if (child_nodes.size() > 1) {
      LOG(INFO) << "[ScheduleRecursive] Scheduling " << child_nodes.size() << " nodes in SequenceNode";
      std::vector<IRStructure*> sorted_nodes;
      LOG(INFO) << "[ScheduleRecursive] Not inside control node, using Z3SchedulePython";
      sorted_nodes = Z3SchedulePython(child_nodes);
      LOG(INFO) << "[ScheduleRecursive] Z3SchedulePython returned " << sorted_nodes.size() << " sorted nodes";

      // Check if order changed
      bool order_changed = false;
      for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        if (sorted_nodes[i] != child_nodes[i]) {
          order_changed = true;
          break;
        }
      }

      if (order_changed) {
        LOG(INFO) << "[ScheduleRecursive] Order changed, reordering children in SequenceNode";
        // Create mapping from IRStructure pointer to child index
        std::unordered_map<IRStructure*, size_t> node_to_index;
        for (size_t i = 0; i < child_nodes.size(); ++i) {
          node_to_index[child_nodes[i]] = i;
        }

        // Reorder children according to sorted_nodes
        std::vector<std::unique_ptr<IRStructure>> reordered_children;
        reordered_children.reserve(sorted_nodes.size());

        for (IRStructure* sorted_node : sorted_nodes) {
          auto it = node_to_index.find(sorted_node);
          if (it == node_to_index.end()) {
            LOG(FATAL) << "[ScheduleRecursive] IRStructure not found in children mapping";
          }
          size_t old_idx = it->second;
          reordered_children.push_back(std::move(seq->children[old_idx]));
        }

        // Move reordered children back
        seq->children = std::move(reordered_children);
        LOG(INFO) << "[ScheduleRecursive] Reordered " << sorted_nodes.size() << " children in SequenceNode";
      } else {
        LOG(INFO) << "[ScheduleRecursive] Order unchanged, no reordering needed";
      }
    } else {
      LOG(INFO) << "[ScheduleRecursive] Single child or empty, no scheduling needed";
    }

    return;
  } else if (node->IsControl()) {
    ControlNode* ctrl = static_cast<ControlNode*>(node);
    LOG(INFO) << "[ScheduleRecursive] ControlNode (For loop)";

    // First, recursively schedule the body (child) inside the control node
    if (ctrl->child && !ctrl->child->IsSequence()) {
      LOG(INFO) << "[ScheduleRecursive] Recursively scheduling ControlNode body";
      ScheduleRecursive(ctrl->child.get());
    }

    // Now schedule the ControlNode's internal tasks (if any) as a unit
    // The body should now be a SequenceNode containing the tasks
    if (ctrl->child && ctrl->child->IsSequence()) {
      SequenceNode* seq_body = static_cast<SequenceNode*>(ctrl->child.get());
      if (seq_body->children.size() > 1) {
        LOG(INFO) << "[ScheduleRecursive] Scheduling " << seq_body->children.size()
                  << " nodes inside ControlNode using Z3SchedulePythonLoop";

        // Collect child nodes
        std::vector<IRStructure*> body_nodes;
        body_nodes.reserve(seq_body->children.size());
        for (const auto& child : seq_body->children) {
          body_nodes.push_back(child.get());
        }

        // Call loop-aware scheduler
        auto result = Z3SchedulePythonLoop(body_nodes);
        std::vector<IRStructure*> sorted_nodes = result.first;
        int64_t ii = result.second;
        LOG(INFO) << "[ScheduleRecursive] Z3SchedulePythonLoop returned II=" << ii;

        // Reorder children if order changed
        bool order_changed = false;
        for (size_t i = 0; i < sorted_nodes.size(); ++i) {
          if (sorted_nodes[i] != body_nodes[i]) {
            order_changed = true;
            break;
          }
        }

        if (order_changed) {
          LOG(INFO) << "[ScheduleRecursive] Order changed inside ControlNode, reordering";
          std::unordered_map<IRStructure*, size_t> node_to_index;
          for (size_t i = 0; i < body_nodes.size(); ++i) {
            node_to_index[body_nodes[i]] = i;
          }

          std::vector<std::unique_ptr<IRStructure>> reordered_children;
          reordered_children.reserve(sorted_nodes.size());
          for (IRStructure* sorted_node : sorted_nodes) {
            auto it = node_to_index.find(sorted_node);
            if (it == node_to_index.end()) {
              LOG(FATAL) << "[ScheduleRecursive] IRStructure not found in children mapping";
            }
            reordered_children.push_back(std::move(seq_body->children[it->second]));
          }
          seq_body->children = std::move(reordered_children);
        }

        // Apply promote transformation if any tasks are marked for promotion
        CheckAndApplyPromoteTransformation(ctrl);

        // Estimate overall latency: II * tripcount
        // Get tripcount from For loop extent
        int64_t tripcount = 100; // default if not constant
        if (const auto* extent_int = ctrl->control->extent.as<IntImmNode>()) {
          tripcount = extent_int->value;
        }
        int64_t overall_latency = ii * tripcount;

        // Set II and latency on the ControlNode (which delegates to child)
        ctrl->SetII(overall_latency);
        ctrl->SetLatency(overall_latency);

        LOG(INFO) << "[ScheduleRecursive] ControlNode scheduled: II=" << ii
                  << ", tripcount=" << tripcount << ", overall_latency=" << overall_latency;
      } else {
        LOG(INFO) << "[ScheduleRecursive] ControlNode body has " << seq_body->children.size()
                  << " nodes, no internal scheduling needed";
        // Still check for promote transformation
        CheckAndApplyPromoteTransformation(ctrl);
      }
    } else {
      LOG(INFO) << "[ScheduleRecursive] ControlNode body is not a SequenceNode or empty";
      // Still check for promote transformation if there's a child
      if (ctrl->child) {
        CheckAndApplyPromoteTransformation(ctrl);
      }
    }

    LOG(INFO) << "[ScheduleRecursive] ControlNode processing complete";
    return;
  }

  LOG(FATAL) << "[ScheduleRecursive] Unknown IRStructure type";
}


// Rewriter to convert IRStructure back to TIR statements
class IRStructureRewriter {
public:
  Stmt Rewrite(const IRStructure* node) {
    if (!node) return Stmt();

    if (node->IsTask()) {
      const TaskNode* task = static_cast<const TaskNode*>(node);

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

      // Create conditional statement
      return task_body;

    } else if (node->IsSequence()) {
      const SequenceNode* seq = static_cast<const SequenceNode*>(node);

      std::vector<Stmt> all_stmts;
      for (const auto& child : seq->children) {
        all_stmts.push_back(Rewrite(child.get()));
      }
      if (all_stmts.empty()) {
        return Stmt();
      }
      if (all_stmts.size() == 1) {
        return all_stmts[0];
      }
      return SeqStmt(all_stmts);

    } else if (node->IsControl()) {
      // For control nodes (nested loops), use regular Rewrite
      // Promote transformation only applies to tasks in the current loop
      const ControlNode* control = static_cast<const ControlNode*>(node);
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
              << ", Tensor=" << task_node->UsesTensorCore()
              << ", promote: " << task_node->GetPromote();

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
      unit_builder.Build(ir_structure.get());

      // Print the modified summary view
      LOG(INFO) << "IRStructure modified summary:";
      PrintIRStructure(ir_structure.get());

      // Analyze buffer dependencies and insert barriers before warpgroup partition
      LOG(INFO) << "Analyzing buffer dependencies and inserting barriers...";
      int next_barrier_id = 1;
      AnalyzeAndInsertBarriers(ir_structure.get(), next_barrier_id);

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

// Helper function to collect all IRStructure nodes (not just TaskNodes) from an IRStructure
void CollectIRStructureNodes(IRStructure* node, std::vector<IRStructure*>& nodes) {
  if (!node) return;
  nodes.push_back(node);
  if (node->IsSequence()) {
    SequenceNode* seq = static_cast<SequenceNode*>(node);
    for (const auto& child : seq->children) {
      if (child) {
        CollectIRStructureNodes(child.get(), nodes);
      }
    }
  } else if (node->IsControl()) {
    ControlNode* ctrl = static_cast<ControlNode*>(node);
    if (ctrl->child) {
      CollectIRStructureNodes(ctrl->child.get(), nodes);
    }
  }
  // For TaskNode, no recursion needed
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

// Apply warpgroup partition to entire IRStructure (top-level IfThenElse)
std::unique_ptr<IRStructure> ApplyWarpgroupPartitionToIRStructure(IRStructure* root, IterVar thread_var) {
  if (!root) return nullptr;

  // Check if there are tasks with mixed warpgroup ids
  std::vector<TaskNode*> all_tasks;
  CollectTaskNodesFromIRStructure(root, all_tasks);

  bool has_warpgroup0 = false;
  bool has_warpgroup1 = false;
  bool has_warpgroup_neutral = false;
  for (TaskNode* task : all_tasks) {
    int wg_id = task->GetWarpgroupId();
    if (wg_id == 0) has_warpgroup0 = true;
    else if (wg_id == 1) has_warpgroup1 = true;
    else if (wg_id == -1) has_warpgroup_neutral = true;
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

  // Helper function to clone IRStructure filtering tasks with warpgroup_id == -1 (neutral tasks)
  std::function<std::unique_ptr<IRStructure>(IRStructure*)> clone_neutral_filter;
  clone_neutral_filter = [&clone_neutral_filter](IRStructure* node) -> std::unique_ptr<IRStructure> {
    if (!node) return nullptr;

    if (node->IsTask()) {
      TaskNode* task = static_cast<TaskNode*>(node);
      if (task->GetWarpgroupId() == -1) {
        return task->Clone();
      } else {
        // Create empty task node for tasks with warpgroup id 0 or 1
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
          new_seq->children.push_back(clone_neutral_filter(child.get()));
        } else {
          new_seq->children.push_back(nullptr);
        }
      }
      return new_seq;
    } else if (node->IsControl()) {
      return nullptr;
    }
    return nullptr;
  };

  // Clone IRStructure for warpgroup neutral, 0 and 1
  auto wg_neutral_structure = has_warpgroup_neutral ? clone_neutral_filter(root) : nullptr;
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

  bool wg_neutral_has_stmts = wg_neutral_structure ? has_actual_statements(wg_neutral_structure.get()) : false;
  bool wg0_has_stmts = has_actual_statements(wg0_structure.get());
  bool wg1_has_stmts = has_actual_statements(wg1_structure.get());

  // Prepare condition: tx < original_threads
  PrimExpr condition;
  PrimExpr original_threads;
  if (thread_var.defined() && thread_var->dom.defined()) {
    Var thread_idx_var = thread_var->var;
    Range thread_dom = thread_var->dom;
    original_threads = thread_dom->extent;
    condition = thread_idx_var < original_threads;
    LOG(INFO) << "Using thread index variable " << thread_idx_var
              << " with original domain " << thread_dom->min << " to " << thread_dom->min + thread_dom->extent
              << " (threads will be doubled for warpgroup partition)"
              << ", condition: " << condition;
  } else {
    LOG(WARNING) << "Thread index variable not properly defined, falling back to default tx < 128";
    Var tx_var("tx", DataType::Int(32));
    original_threads = IntImm(DataType::Int(32), 128);
    condition = tx_var < original_threads;
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

  Stmt neutral_body = wg_neutral_has_stmts ? irstructure_to_stmt(wg_neutral_structure.get()) : Evaluate(0);
  Stmt then_body = wg0_has_stmts ? irstructure_to_stmt(wg0_structure.get()) : Evaluate(0);
  Stmt else_body = wg1_has_stmts ? irstructure_to_stmt(wg1_structure.get()) : Evaluate(0);

  // Create IfThenElse statement with barrier synchronization if both warpgroups have statements
  Stmt if_then_else;
  if (wg0_has_stmts && wg1_has_stmts) {
    // Both warpgroups exist: insert barriers for cross-warpgroup synchronization
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
  // Add barrier synchronization between neutral tasks and warpgroup-specific work
  Stmt combined_stmt;
  if (wg_neutral_has_stmts) {
    if (!if_then_else.as<EvaluateNode>() && !neutral_body.as<EvaluateNode>()) {
      // Both have statements: insert barriers for neutral-to-warpgroup synchronization
      combined_stmt = InsertBarriersForNeutralSync(neutral_body, if_then_else, original_threads);
    } else if (!if_then_else.as<EvaluateNode>() || !neutral_body.as<EvaluateNode>()) {
      // Only one has actual statements
      std::vector<Stmt> stmts;
      if (!neutral_body.as<EvaluateNode>()) {
        stmts.push_back(neutral_body);
      }
      if (!if_then_else.as<EvaluateNode>()) {
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

  // Create appropriate IRStructure based on original node type
  auto new_task_node = std::make_unique<TaskNode>();
  new_task_node->stmts.push_back(combined_stmt);

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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm