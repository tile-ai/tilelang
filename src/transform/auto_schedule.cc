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
Stmt ApplyWarpgroupPartitionToIRStructure(
    IRStructure *root, IterVar thread_var, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, const bool enable_epi,
    PrimExpr thread_count[2]);
void CollectIRStructureNodes(IRStructure *node,
                             std::vector<IRStructure *> &nodes);
// Barrier synchronization helper functions
static Stmt InsertBarriersForNeutralSync(Stmt neutral_body, Stmt warpgroup_body,
                                         std::vector<Buffer> &barrier_buffers,
                                         Map<ObjectRef, ObjectRef> &barrier_map,
                                         PrimExpr thread_count[2]);
// Barrier dependency analysis functions
static void AnalyzeAndInsertBarriers(IRStructure *node, int &next_barrier_id,
                                     std::vector<Buffer> &barrier_buffers,
                                     Map<ObjectRef, ObjectRef> &barrier_map,
                                     PrimExpr thread_count[2]);
static void AnalyzeSequenceNodeBarriers(SequenceNode *seq, int &next_barrier_id,
                                        std::vector<Buffer> &barrier_buffers,
                                        Map<ObjectRef, ObjectRef> &barrier_map,
                                        PrimExpr thread_count[2]);
static void AnalyzeControlNodeBarriers(ControlNode *ctrl, int &next_barrier_id,
                                       std::vector<Buffer> &barrier_buffers,
                                       Map<ObjectRef, ObjectRef> &barrier_map,
                                       PrimExpr thread_count[2]);

// Helper function to check if two ranges overlap
bool RangesOverlap(const Range &a, const Range &b) {
  // Two ranges [a_min, a_min+a_extent) and [b_min, b_min+b_extent) overlap if:
  // max(a_min, b_min) < min(a_min+a_extent, b_min+b_extent)
  // Since min and extent might be symbolic, we use arithmetic simplification
  // For simplicity, assume they are constants or can be compared
  // Use tir::is_zero to check if max(a_min, b_min) - min(a_min+a_extent,
  // b_min+b_extent) < 0 Actually, we can check if they are provably
  // non-overlapping If we can't prove either way, assume they might overlap
  // (conservative)

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
bool RegionsOverlap(const Region &a, const Region &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!RangesOverlap(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

// Check if two IRStructures use the same register region
// Used for warpgroup specialization: different warpgroups cannot share
// registers

// Helper functions for barrier insertion (preliminary implementation)
// These functions implement barrier_arrive/barrier_wait using the underlying
// ptx_arrive_barrier and mbarrier_wait_parity operations.

// Create a barrier_arrive statement for the given barrier expression
// Equivalent to T.barrier_arrive(barrier_expr) in Python
// barrier_expr should be BufferLoad(barrier_buffer, {0}) where barrier_buffer
// is allocated with makeAllocBarrier
static Stmt makeBarrierArrive(PrimExpr barrier_expr, int cta_id = -1,
                              const PrimExpr &pred = 1) {
  Array<PrimExpr> args = {std::move(barrier_expr)};
  if (cta_id != -1) {
    args.push_back(cta_id);
    args.push_back(pred);
  }
  return Evaluate(
      Call(DataType::Handle(), builtin::ptx_arrive_barrier(), args));
}

// Create a barrier_wait statement for the given barrier expression and parity
// Equivalent to T.barrier_wait(barrier_expr, parity) in Python
// barrier_expr should be BufferLoad(barrier_buffer, {0}) where barrier_buffer
// is allocated with makeAllocBarrier
static Stmt makeBarrierWait(PrimExpr barrier_expr, PrimExpr parity) {
  auto call = Call(DataType::Handle(), mbarrier_wait_parity(),
                   {std::move(barrier_expr), std::move(parity)});
  return Evaluate(call);
}

// Create a barrier allocation statement with arrive_count
// Equivalent to T.alloc_barrier(arrive_count) in Python

// Create a barrier buffer allocation and return the Buffer object along with
// allocation statement Equivalent to Python: barrier =
// T.alloc_buffer((arrive_count,), "uint64", scope="shared.barrier")
static Buffer makeBarrierBuffer(PrimExpr arrive_count, const std::string &name,
                                Map<ObjectRef, ObjectRef> &barrier_map) {
  // Create buffer shape: (arrive_count,)
  Array<PrimExpr> shape = {1};

  // Create buffer data type: uint64
  DataType dtype = DataType::UInt(64);
  Type ptr_type = PointerType(PrimType(dtype), "shared.barrier");
  Var handle(name, ptr_type);
  barrier_map.Set(handle, Array<ObjectRef>{arrive_count});

  // Create buffer
  Buffer buffer =
      Buffer(handle, dtype, shape, {}, PrimExpr(), name, 0, 0, kDefault);

  return buffer;
}

// Insert barriers between neutral tasks and warpgroup-specific work
// This ensures neutral tasks complete before any warpgroup-specific work begins
static Stmt InsertBarriersForNeutralSync(Stmt neutral_body, Stmt warpgroup_body,
                                         std::vector<Buffer> &barrier_buffers,
                                         Map<ObjectRef, ObjectRef> &barrier_map,
                                         PrimExpr thread_count[2]) {
  // If either body is empty, no barriers needed
  if (IsEvaluateZero(neutral_body) || IsEvaluateZero(warpgroup_body)) {
    return SeqStmt({neutral_body, warpgroup_body});
  }

  // Allocate barrier buffer for neutral-to-warpgroup synchronization
  // Equivalent to Python: barrier = T.alloc_buffer((thread_count*2,), "uint64",
  // scope="shared.barrier") Using arrive_count = thread_count * 2 (number of
  // threads * 2 for neutral-to-warpgroup synchronization)
  PrimExpr arrive_count = thread_count[0] + thread_count[1];
  Buffer barrier_buffer =
      makeBarrierBuffer(arrive_count, "neutral_warpgroup_barrier", barrier_map);
  barrier_buffers.push_back(barrier_buffer);

  // Create BufferLoad expression for barrier[0]
  PrimExpr barrier_load = tir::BufferLoad(barrier_buffer, {0});

  // Use barrier buffer for neutral-to-warpgroup synchronization
  // Parity 0 for wait, parity 1 for arrive (simplified)
  Stmt arrive_barrier = makeBarrierArrive(barrier_load);
  Stmt wait_barrier = makeBarrierWait(barrier_load, 0);

  // Combine: neutral_body -> arrive_barrier -> wait_barrier -> warpgroup_body
  std::vector<Stmt> stmts;
  if (!IsEvaluateZero(neutral_body)) {
    stmts.push_back(neutral_body);
  }
  stmts.push_back(arrive_barrier);
  stmts.push_back(wait_barrier);
  if (!IsEvaluateZero(warpgroup_body)) {
    stmts.push_back(warpgroup_body);
  }

  Stmt sync_body;
  if (stmts.size() == 1) {
    sync_body = stmts[0];
  } else {
    sync_body = SeqStmt(stmts);
  }

  // Barrier buffer allocation will be handled by the barrier analysis pass
  // The buffer will be added to tilelang_root BlockNode's alloc_buffers
  return sync_body;
}

// Helper function to insert a statement into PromoteNode's stmts
static void InsertStatementIntoPromoteNode(ScheduleUnit *task, const Stmt &stmt,
                                           bool at_beginning = false) {
  if (at_beginning) {
    task->before.insert(task->before.begin(), stmt);
  } else {
    task->after.push_back(stmt);
  }
}

// Barrier dependency analysis implementation
static void AnalyzeAndInsertBarriers(IRStructure *node, int &next_barrier_id,
                                     std::vector<Buffer> &barrier_buffers,
                                     Map<ObjectRef, ObjectRef> &barrier_map,
                                     PrimExpr thread_count[2]) {
  if (!node)
    return;

  if (node->IsSequence()) {
    AnalyzeSequenceNodeBarriers(static_cast<SequenceNode *>(node),
                                next_barrier_id, barrier_buffers, barrier_map,
                                thread_count);
  } else if (node->IsControl()) {
    AnalyzeControlNodeBarriers(static_cast<ControlNode *>(node),
                               next_barrier_id, barrier_buffers, barrier_map,
                               thread_count);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    AnalyzeAndInsertBarriers(wrapper->child.get(), next_barrier_id,
                             barrier_buffers, barrier_map, thread_count);
  } else if (node->IsTask()) {
    // For TaskNode, nothing to do at this level
  } else {
    LOG(FATAL);
  }
}

static void AnalyzeSequenceNodeBarriers(SequenceNode *seq, int &next_barrier_id,
                                        std::vector<Buffer> &barrier_buffers,
                                        Map<ObjectRef, ObjectRef> &barrier_map,
                                        PrimExpr thread_count[2]) {
  if (!seq)
    return;

  // Map from (buffer, warpgroup_id) to task of last access
  std::unordered_map<Buffer, ScheduleUnit *, ObjectPtrHash, ObjectPtrEqual>
      last_access_map[2];
  std::unordered_map<Buffer, std::pair<ScheduleUnit *, int>, ObjectPtrHash,
                     ObjectPtrEqual>
      last_write_map;
  std::unordered_map<Buffer, std::pair<ScheduleUnit *, int>, ObjectPtrHash,
                     ObjectPtrEqual>
      last_wgmma_map[2];
  int wait_wgmma_id[2] = {}, total_wgmma[2] = {};

  // Process tasks in sequence order
  for (auto &promote_child : seq->children) {
    auto task = static_cast<ScheduleUnit *>(promote_child.get());
    if (task->child->IsSequence() || task->child->IsControl()) {
      // If child is SequenceNode or ControlNode, recursively analyze it
      AnalyzeAndInsertBarriers(task->child.get(), next_barrier_id,
                               barrier_buffers, barrier_map, thread_count);
      continue;
    }
    if (!task->isInnerTask())
      continue;
    int wg_id = task->GetWarpgroupId();
    if (wg_id == -1)
      continue;

    // Check regions for dependencies
    for (const auto &region_access : task->GetReadWriteRegions()) {
      auto &region = region_access.first;
      if (IsRegisterRegion(region)) {
        // if (task->UsesTensorCore()) continue;
        Buffer buffer = region->buffer;
        auto it = last_wgmma_map[wg_id].find(buffer);
        if (it == last_wgmma_map[wg_id].end())
          continue;
        if (it->second.second <= wait_wgmma_id[wg_id])
          continue;
        wait_wgmma_id[wg_id] = it->second.second;
        Stmt wait_stmt =
            Evaluate(Call(DataType::Handle(), wait_wgmma(),
                          {total_wgmma[wg_id] - wait_wgmma_id[wg_id]}));
        InsertStatementIntoPromoteNode(task, wait_stmt, true);
      } else {
        Buffer buffer = region->buffer;
        bool need_barrier = false;
        ScheduleUnit *last_access_task;
        int last_wg_id;
        bool is_async = task->UsesTensorCore() || task->UsesTMACore();
        if (!region_access.second) {
          auto it = last_write_map.find(buffer);
          if (it != last_write_map.end()) {
            last_access_task = it->second.first;
            last_wg_id = last_access_task->GetWarpgroupId();
            if (last_wg_id == -1)
              continue;
            if (it->second.second & (1 << wg_id))
              continue;
            bool last_async = last_access_task->UsesTensorCore() ||
                              last_access_task->UsesTMACore();

            if (last_wg_id != wg_id || is_async || last_async) {
              need_barrier = true;
            }
          }
        } else {
          auto it = last_access_map[!wg_id].find(buffer);
          if (it != last_access_map[!wg_id].end()) {
            last_access_task = it->second;
            last_wg_id = last_access_task->GetWarpgroupId();
            if (last_wg_id == -1)
              continue;
            if (last_wg_id != wg_id) {
              need_barrier = true;
            }
          }
        }
        // If warpgroup ids differ, insert barrier
        if (need_barrier) {
          // Allocate a new barrier ID and buffer
          int barrier_id = next_barrier_id++;

          Buffer barrier_buffer = makeBarrierBuffer(
              thread_count[last_wg_id], "barrier_" + std::to_string(barrier_id),
              barrier_map);

          // Collect the barrier buffer to be added to tilelang_root block's
          // alloc_buffers
          barrier_buffers.push_back(barrier_buffer);

          // Create BufferLoad expression for barrier[0]
          PrimExpr barrier_load = tir::BufferLoad(barrier_buffer, {0});

          // Insert barrier_arrive at the end of last_access_task's statements
          Stmt arrive_stmt = makeBarrierArrive(barrier_load);
          InsertStatementIntoPromoteNode(last_access_task, arrive_stmt, false);

          // Insert barrier_wait at the beginning of task's statements
          Stmt wait_stmt =
              makeBarrierWait(barrier_load,
                              0); // parity = 0 for non-loop barriers
          InsertStatementIntoPromoteNode(task, wait_stmt, true);
          // Remove from map (as per user instruction)
          if (!region_access.second) {
            auto it = last_write_map.find(buffer);
            it->second.second |= (1 << wg_id);
            if (it->second.second == 3) {
              last_write_map.erase(last_write_map.find(buffer));
            }
          } else {
            for (unsigned idx = 0; idx < 2; ++idx) {
              auto it = last_access_map[idx].find(buffer);
              if (it != last_access_map[idx].end()) {
                last_access_map[idx].erase(it);
              }
            }
            auto it = last_write_map.find(buffer);
            if (it != last_write_map.end()) {
              last_write_map.erase(it);
            }
          }
        }
      }
    }

    // Update regions
    bool found_wgmma = false;
    for (const auto &region_access : task->GetReadWriteRegions()) {
      auto &region = region_access.first;
      if (IsRegisterRegion(region)) {
        if (!task->UsesTensorCore())
          continue;
        Buffer buffer = region->buffer;
        if (!found_wgmma) {
          found_wgmma = true;
          ++total_wgmma[wg_id];
        }
        last_wgmma_map[wg_id][buffer] =
            std::make_pair(task, total_wgmma[wg_id]);
      } else {
        Buffer buffer = region->buffer;
        last_access_map[wg_id][buffer] = task;
        if (region_access.second) {
          last_write_map[buffer] = std::make_pair(task, 0);
        }
      }
    }
  }
}

static void AnalyzeControlNodeBarriers(ControlNode *ctrl, int &next_barrier_id,
                                       std::vector<Buffer> &barrier_buffers,
                                       Map<ObjectRef, ObjectRef> &barrier_map,
                                       PrimExpr thread_count[2]) {
  if (!ctrl || !ctrl->child)
    return;

  // Get loop information
  const ForNode *for_node = ctrl->control.get();
  if (!for_node)
    return;

  PrimExpr loop_var = for_node->loop_var;
  PrimExpr loop_start = for_node->min;
  PrimExpr loop_step = for_node->step.has_value()
                           ? for_node->step.value()
                           : IntImm(DataType::Int(32), 1);
  bool has_promoted_tasks = ctrl->hasPromote();

  // If child is a SequenceNode, we need special handling for
  // promote/non-promote tasks
  if (ctrl->child->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(ctrl->child.get());

    // Separate promoted and non-promoted tasks
    std::vector<ScheduleUnit *> promoted_tasks;
    std::vector<ScheduleUnit *> non_promoted_tasks;

    // Collect all tasks from the sequence
    std::vector<ScheduleUnit *> all_tasks;
    for (auto &child : seq->children) {
      auto task = static_cast<ScheduleUnit *>(child.get());
      if (task->child->IsSequence() || task->child->IsControl()) {
        // If child is SequenceNode or ControlNode, recursively analyze it
        AnalyzeAndInsertBarriers(task->child.get(), next_barrier_id,
                                 barrier_buffers, barrier_map, thread_count);
        continue;
      }
      all_tasks.push_back(task);
    }

    // Separate by promote flag
    for (ScheduleUnit *task : all_tasks) {
      if (task->GetPromote()) {
        promoted_tasks.push_back(task);
      } else {
        non_promoted_tasks.push_back(task);
      }
    }

    // Process in order: promoted tasks first, then non-promoted tasks
    // This matches the software pipelining order
    std::vector<ScheduleUnit *> ordered_tasks;
    ordered_tasks.insert(ordered_tasks.end(), promoted_tasks.begin(),
                         promoted_tasks.end());
    ordered_tasks.insert(ordered_tasks.end(), non_promoted_tasks.begin(),
                         non_promoted_tasks.end());

    // Map from (buffer, warpgroup_id) to task
    std::unordered_map<Buffer, ScheduleUnit *, ObjectPtrHash, ObjectPtrEqual>
        last_access_map[2];
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
        last_access_set[2];
    std::unordered_map<Buffer, std::pair<ScheduleUnit *, int>, ObjectPtrHash,
                       ObjectPtrEqual>
        last_write_map;
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> last_write_set;
    std::unordered_map<Buffer, std::pair<ScheduleUnit *, int>, ObjectPtrHash,
                       ObjectPtrEqual>
        last_wgmma_map[2];
    int wait_wgmma_id[2] = {}, total_wgmma[2] = {};

    // Process tasks in the specified order
    for (unsigned iter = 0; iter != 2; ++iter) {
      for (ScheduleUnit *task : ordered_tasks) {
        if (!task->isInnerTask())
          continue;
        int wg_id = task->GetWarpgroupId();
        bool is_promoted = task->GetPromote();
        if (wg_id == -1)
          continue;
        bool is_async = task->UsesTensorCore() || task->UsesTMACore();

        // Check regions for dependencies
        for (const auto &region_access : task->GetReadWriteRegions()) {
          auto &region = region_access.first;
          if (IsRegisterRegion(region)) {
            if (task->UsesTensorCore())
              continue;
            Buffer buffer = region->buffer;
            auto it = last_wgmma_map[wg_id].find(buffer);
            if (it == last_wgmma_map[wg_id].end())
              continue;
            if (it->second.second <= wait_wgmma_id[wg_id])
              continue;
            wait_wgmma_id[wg_id] = it->second.second;
            Stmt wait_stmt =
                Evaluate(Call(DataType::Handle(), wait_wgmma(),
                              {total_wgmma[wg_id] - wait_wgmma_id[wg_id]}));
            InsertStatementIntoPromoteNode(task, wait_stmt, true);
          } else {
            Buffer buffer = region->buffer;
            bool need_barrier = false;
            ScheduleUnit *last_access_task;
            int last_wg_id;
            bool last_is_promoted;
            if (!region_access.second) {
              if (iter == 1) {
                if (last_write_set.find(buffer) != last_write_set.end()) {
                  continue;
                }
                last_write_set.insert(buffer);
              }
              auto it = last_write_map.find(buffer);
              if (it != last_write_map.end()) {
                last_access_task = it->second.first;
                last_wg_id = last_access_task->GetWarpgroupId();
                last_is_promoted = last_access_task->GetPromote();
                if (last_wg_id == -1)
                  continue; // Allow barriers involving neutral tasks
                if (it->second.second & (1 << wg_id))
                  continue;

                bool last_async = last_access_task->UsesTensorCore() ||
                                  last_access_task->UsesTMACore();
                // If warpgroup ids differ or promotion status differs, insert
                // barrier
                if (last_wg_id != wg_id || last_is_promoted != is_promoted ||
                    is_async || last_async) {
                  need_barrier = true;
                }
              }
            } else {
              if (iter == 1) {
                if (last_access_set[!wg_id].find(buffer) !=
                    last_access_set[!wg_id].end()) {
                  continue;
                }
                last_access_set[!wg_id].insert(buffer);
              }
              auto it = last_access_map[!wg_id].find(buffer);
              if (it != last_access_map[!wg_id].end()) {
                last_access_task = it->second;
                last_wg_id = last_access_task->GetWarpgroupId();
                last_is_promoted = last_access_task->GetPromote();
                if (last_wg_id == -1)
                  continue; // Allow barriers involving neutral tasks

                // If warpgroup ids differ or promotion status differs, insert
                // barrier
                if (last_wg_id != wg_id || last_is_promoted != is_promoted) {
                  need_barrier = true;
                }
              }
            }
            // If warpgroup ids differ or promotion status differs, insert
            // barrier
            if (need_barrier) {
              // Allocate a new barrier ID and buffer
              int barrier_id = next_barrier_id++;

              // Calculate parity for barrier wait
              PrimExpr parity_expr = tvm::indexmod(
                  tvm::indexdiv(loop_var - loop_start, loop_step) +
                      IntImm(DataType::Int(32), iter + 2),
                  2);

              Buffer barrier_buffer = makeBarrierBuffer(
                  thread_count[last_wg_id],
                  "barrier_" + std::to_string(barrier_id), barrier_map);

              // Collect the barrier buffer to be added to tilelang_root block's
              // alloc_buffers
              barrier_buffers.push_back(barrier_buffer);

              // Create BufferLoad expression for barrier[0]
              PrimExpr barrier_load = tir::BufferLoad(barrier_buffer, {0});

              // Insert barrier_arrive at the end of last_access_task's
              // statements
              Stmt arrive_stmt = makeBarrierArrive(barrier_load);
              InsertStatementIntoPromoteNode(last_access_task, arrive_stmt,
                                             false);

              // Insert barrier_wait at the beginning of task's statements
              Stmt wait_stmt = makeBarrierWait(barrier_load, parity_expr);
              if (iter == 1) {
                wait_stmt = IfThenElse(loop_var != loop_start, wait_stmt);
              }
              InsertStatementIntoPromoteNode(task, wait_stmt, true);
              // Remove from map (as per user instruction)
              if (!region_access.second) {
                auto it = last_write_map.find(buffer);
                it->second.second |= (1 << wg_id);
                if (it->second.second == 3) {
                  last_write_map.erase(last_write_map.find(buffer));
                }
              } else {
                for (unsigned idx = 0; idx < 2; ++idx) {
                  auto it = last_access_map[idx].find(buffer);
                  if (it != last_access_map[idx].end()) {
                    last_access_map[idx].erase(it);
                  }
                }
                auto it = last_write_map.find(buffer);
                if (it != last_write_map.end()) {
                  last_write_map.erase(it);
                }
              }
              bool last_async = last_access_task->UsesTensorCore() ||
                                last_access_task->UsesTMACore();
            }
          }
        }

        if (iter == 0) {
          // Update regions
          bool found_wgmmap = false;
          for (const auto &region_access : task->GetReadWriteRegions()) {
            auto &region = region_access.first;
            if (IsRegisterRegion(region)) {
              if (!task->UsesTensorCore() || !region_access.second)
                continue;
              Buffer buffer = region->buffer;
              if (!found_wgmmap) {
                found_wgmmap = true;
                ++total_wgmma[wg_id];
              }
              last_wgmma_map[wg_id][buffer] =
                  std::make_pair(task, total_wgmma[wg_id]);
            } else {
              if (iter == 1)
                continue;
              Buffer buffer = region->buffer;
              last_access_map[wg_id][buffer] = task;
              if (region_access.second) {
                last_write_map[buffer] = std::make_pair(task, 0);
              }
            }
          }
        }
      }
    }
  } else {
    AnalyzeAndInsertBarriers(ctrl->child.get(), next_barrier_id,
                             barrier_buffers, barrier_map, thread_count);
  }
}

// Helper function to collect all TaskNodes with context information
void CollectAllTaskNodesWithContext(
    IRStructure *node, std::vector<TaskNodeWithContext> &all_tasks,
    ControlNode *current_control_node = nullptr) {
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
      if (const int64_t *extent_ptr = tir::as_const_int(loop_extent)) {
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
  new_unit->promote = promote;
  // Clone child if exists
  if (child) {
    new_unit->child = child->Clone();
  }
  // Copy latency and II
  new_unit->SetLatency(GetLatency());
  new_unit->SetII(GetII());
  return new_unit;
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
      if (prefix_tasks.find(all_tasks[idx].task) != prefix_tasks.end()) {
        // This is a prefix task, skip it (it won't participate in warpgroup
        // specialize)
        continue;
      }
      has_task = true;
      int64_t latency = all_tasks[idx].task->GetLatency();
      int64_t tripcount = all_tasks[idx].tripcount;
      total_weighted_latency += latency * tripcount;
      has_tma_core |= all_tasks[idx].task->UsesTMACore();
      has_tensor_core |= all_tasks[idx].task->UsesTensorCore();
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
      for (int idx : comp.task_indices) {
        all_tasks[idx].task->SetWarpgroupId(assigned_warpgroup);
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
      for (int idx : comp.task_indices) {
        all_tasks[idx].task->SetWarpgroupId(assigned_warpgroup);
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
      static std::optional<tvm::ffi::Function> z3_schedule_func =
          tvm::ffi::Function::GetGlobal("tl.transform.z3_schedule_python");
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

    static std::optional<tvm::ffi::Function> z3_schedule_loop_func =
        tvm::ffi::Function::GetGlobal("tl.transform.z3_schedule_loop_python");
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
    // Python function returns (start_times, promotes) as a tuple of two
    // arrays
    auto return_val =
        z3_schedule_loop_func
            .value()(tvm_latencies, tvm_iis, tvm_resource_flags, tvm_data_deps,
                     tvm_resource_deps)
            .cast<ffi::Tuple<ffi::Array<int64_t>, ffi::Array<bool>, int64_t>>();

    ffi::Array<int64_t> start_times = return_val.get<0>();
    ffi::Array<bool> promotes = return_val.get<1>();
    ii = return_val.get<2>();

    // Apply start times and promote flags to nodes
    std::map<IRStructure *, bool> promote_map;
    size_t num_promoted = 0;
    for (size_t i = 0; i < n; ++i) {
      nodes[i]->SetStartTime(start_times[i]);
      bool promote = promotes[i] != 0; // Convert int to bool
      promote_map[nodes[i]] = promote;
      if (promote) {
        num_promoted++;
      }
    }
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

    // Create sorted task list based on start_time (and original index as
    // tie-breaker)
    std::vector<std::pair<int64_t, size_t>> start_time_with_idx;
    start_time_with_idx.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      start_time_with_idx.emplace_back(
          start_times[i] + promotes[i] * return_val.get<2>(), i);
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
      auto promote_node = std::make_unique<ScheduleUnit>();
      promote_node->promote = promote_map[node.get()];
      promote_node->child = std::move(node);
      node = std::move(promote_node);
    }

    if (num_promoted > 0) {
      ctrl->SetPromote(true);
    }

    // Estimate overall latency: II * tripcount
    // Get tripcount from For loop extent
    int64_t tripcount = 100; // default if not constant
    if (const auto *extent_int = ctrl->control->extent.as<IntImmNode>()) {
      tripcount = extent_int->value;
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
      auto promote_node = std::make_unique<ScheduleUnit>();
      promote_node->promote = -1;
      promote_node->child = std::move(node);
      node = std::move(promote_node);
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

// Mutator to add alloc_buffers to tilelang_root block
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
                   op->body, op->init, new_alloc_buffers, op->match_buffers,
                   new_annotations);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  std::vector<Buffer> buffers_to_add_;
  Map<ObjectRef, ObjectRef> &barrier_map_;
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
    }
    // If neither TMA nor Tensor core was used, and CUDA operations were found,
    // set CUDA core flag
    if (!analyzer.found_tma && !analyzer.found_tensor && analyzer.found_cuda) {
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
    int next_barrier_id = 1;
    std::vector<Buffer> barrier_buffers;
    Map<ObjectRef, ObjectRef> barrier_map;
    // Determine thread count for barrier arrive_count calculations
    PrimExpr thread_count[2] = {thread_var->dom->extent,
                                double_thread ? thread_var->dom->extent
                                              : IntImm(DataType::Int(32), 128)};
    AnalyzeAndInsertBarriers(ir_structure.get(), next_barrier_id,
                             barrier_buffers, barrier_map, thread_count);

    // Print the modified summary view
    // PrintIRStructure(ir_structure.get());

    // Apply warpgroup partition to entire IRStructure
    Stmt new_body = ApplyWarpgroupPartitionToIRStructure(
        ir_structure.get(), thread_var, barrier_buffers, barrier_map,
        enable_epi, thread_count);
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
    // Add barrier buffers to tilelang_root block's alloc_buffers
    if (!barrier_buffers.empty()) {
      TilelangRootAllocBufferAdder adder(barrier_buffers, barrier_map);
      final_body = adder(final_body);
    }

    // Create a new PrimFunc with the updated body
    auto new_func = PrimFunc(func->params, final_body, func->ret_type,
                             func->buffer_map, func->attrs);
    return new_func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

// Helper function to collect all IRStructure nodes (not just TaskNodes) from an
// IRStructure
void CollectIRStructureNodes(IRStructure *node,
                             std::vector<IRStructure *> &nodes) {
  if (!node)
    return;
  nodes.push_back(node);
  if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (const auto &child : seq->children) {
      if (child) {
        CollectIRStructureNodes(child.get(), nodes);
      }
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);
    if (ctrl->child) {
      CollectIRStructureNodes(ctrl->child.get(), nodes);
    }
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    if (wrapper->child) {
      CollectIRStructureNodes(wrapper->child.get(), nodes);
    }
  }
  // For TaskNode, no recursion needed
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
    new_unit->before = unit->before;
    new_unit->after = unit->after;
    new_unit->promote = unit->promote;
    new_unit->child =
        CloneIRStructureWithWarpgroupFilter(unit->child.get(), warpgroup_id);
    return new_unit;
  }
  LOG(FATAL);
}

// Apply warpgroup partition to entire IRStructure (top-level IfThenElse)
Stmt ApplyWarpgroupPartitionToIRStructure(
    IRStructure *root, IterVar thread_var, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, const bool enable_epi,
    PrimExpr thread_count[2]) {
  if (!root)
    return Evaluate(0);

  if (root->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(root);
    Stmt body = Evaluate(0);
    if (wrapper->child) {
      body = ApplyWarpgroupPartitionToIRStructure(
          wrapper->child.get(), thread_var, barrier_buffers, barrier_map,
          enable_epi, thread_count);
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
                         enable_epi](IRStructure *structure) -> Stmt {
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
        for (auto &stmt : unit->before) {
          stmts.push_back(stmt);
        }
        Stmt child_stmt = irstructure_to_stmt(unit->child.get());
        stmts.push_back(child_stmt);
        for (auto &stmt : unit->after) {
          stmts.push_back(stmt);
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
      if (!ctrl->hasPromote() || !ctrl->child->IsSequence()) {
        std::vector<Stmt> stmts;
        if (ctrl->child->IsScheduleUnit()) {
          auto unit = static_cast<ScheduleUnit *>(ctrl->child.get());
          for (auto &stmt : unit->before) {
            stmts.push_back(stmt);
          }
          stmts.push_back(irstructure_to_stmt(unit->child.get()));
          for (auto &stmt : unit->after) {
            stmts.push_back(stmt);
          }
        } else if (ctrl->child->IsSequence()) {
          auto seq = static_cast<SequenceNode *>(ctrl->child.get());
          for (auto &child : seq->children) {
            ICHECK(child->IsScheduleUnit());
            auto unit = static_cast<ScheduleUnit *>(child.get());
            for (auto &stmt : unit->before) {
              stmts.push_back(stmt);
            }
            stmts.push_back(irstructure_to_stmt(unit->child.get()));
            for (auto &stmt : unit->after) {
              stmts.push_back(stmt);
            }
          }
        } else {
          LOG(FATAL);
        }
        Stmt body = SeqStmt::Flatten(stmts);
        return For(loop_var, loop_start, loop_extent, ctrl->control->kind, body,
                   ctrl->control->thread_binding, ctrl->control->annotations);
      }
      Stmt body = Evaluate(0);
      std::vector<Stmt> unit_stages[2];
      auto seq = static_cast<SequenceNode *>(ctrl->child.get());
      for (auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        std::vector<Stmt> stmts;
        for (auto &stmt : unit->before) {
          stmts.push_back(stmt);
        }
        stmts.push_back(irstructure_to_stmt(unit->child.get()));
        for (auto &stmt : unit->after) {
          stmts.push_back(stmt);
        }
        unit_stages[unit->promote].push_back(SeqStmt::Flatten(stmts));
      }
      Stmt prologue = Evaluate(0);
      bool enable_pro = true;
      if (enable_pro && !unit_stages[1].empty()) {
        prologue = SeqStmt::Flatten(unit_stages[1]);
        Map<Var, PrimExpr> substitution;
        substitution.Set(loop_var, loop_start);
        prologue = Substitute(prologue, substitution);
        prologue = IfThenElse(loop_extent > 0, prologue);
      }
      Stmt epilogue = Evaluate(0);
      if (enable_epi && !unit_stages[0].empty()) {
        epilogue = SeqStmt::Flatten(unit_stages[0]);
        Map<Var, PrimExpr> substitution;
        substitution.Set(loop_var, loop_start + loop_extent - loop_step);
        epilogue = Substitute(epilogue, substitution);
        epilogue = IfThenElse(loop_extent > 0, epilogue);
      }
      std::vector<Stmt> steady;

      bool remove_pro = enable_pro || unit_stages[1].empty();
      bool remove_epi = enable_epi || unit_stages[0].empty();
      for (auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        std::vector<Stmt> stmts;
        for (auto &stmt : unit->before) {
          stmts.push_back(stmt);
        }
        stmts.push_back(irstructure_to_stmt(unit->child.get()));
        for (auto &stmt : unit->after) {
          stmts.push_back(stmt);
        }
        if (unit->promote == 1) {
          if (remove_epi) {
            Map<Var, PrimExpr> substitution;
            substitution.Set(loop_var, loop_var + loop_step);
            Stmt new_stmt = Substitute(SeqStmt::Flatten(stmts), substitution);
            steady.push_back(new_stmt);
          } else {
            Map<Var, PrimExpr> substitution;
            substitution.Set(loop_var, loop_var + loop_step);
            Stmt new_stmt = IfThenElse(
                loop_var < loop_start + loop_extent - loop_step * remove_pro,
                Substitute(SeqStmt::Flatten(stmts), substitution));
            steady.push_back(new_stmt);
          }
        } else if (unit->promote == 0) {
          Map<Var, PrimExpr> substitution;
          substitution.Set(loop_var, loop_var - loop_step);
          if (remove_pro) {
            steady.push_back(SeqStmt::Flatten(stmts));
          } else {
            Stmt new_stmt =
                IfThenElse(loop_var > loop_start,
                           Substitute(SeqStmt::Flatten(stmts), substitution));
            steady.push_back(new_stmt);
          }
        } else {
          steady.push_back(SeqStmt::Flatten(stmts));
        }
      }
      Stmt new_body = SeqStmt::Flatten(steady);
      if (unit_stages[0].empty()) {
        Map<Var, PrimExpr> substitution;
        substitution.Set(loop_var, loop_var - loop_step);
        new_body = Substitute(new_body, substitution);
      }
      auto new_var = loop_var.copy_with_suffix("");
      Stmt new_for =
          For(new_var, loop_start,
              ctrl->control->extent + loop_step * (1 - remove_pro - remove_epi),
              ctrl->control->kind, new_body, ctrl->control->thread_binding,
              ctrl->control->annotations);
      {
        Map<Var, PrimExpr> substitution;
        substitution.Set(loop_var, new_var);
        new_for = Substitute(new_for, substitution);
      }
      return SeqStmt({prologue, new_for, epilogue});
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
      combined_stmt = InsertBarriersForNeutralSync(neutral_body, if_then_else,
                                                   barrier_buffers, barrier_map,
                                                   thread_count);
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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm
