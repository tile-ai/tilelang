#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include "./ir_structure.h"

namespace tvm {
namespace tl {

using namespace tir;

bool IsEvaluateZero(const tvm::tir::Stmt &stmt) {
  if (const EvaluateNode *eval_node = stmt.as<EvaluateNode>()) {
    if (is_const_int(eval_node->value, 0)) {
      return true;
    }
  }
  return false;
}

// Structure to store loop nesting information
struct LoopNestingInfo {
  std::vector<Var> loop_vars;
  std::vector<PrimExpr> loop_starts;
  std::vector<PrimExpr> loop_steps;
  std::vector<PrimExpr> loop_extents;

  // Add a loop to the nesting info
  void AddLoop(const ForNode *for_node) {
    loop_vars.push_back(for_node->loop_var);
    loop_starts.push_back(for_node->min);
    loop_steps.push_back(for_node->step.has_value()
                             ? for_node->step.value()
                             : IntImm(DataType::Int(32), 1));
    loop_extents.push_back(for_node->extent);
  }

  // Remove the innermost loop
  void PopLoop() {
    if (!loop_vars.empty()) {
      loop_vars.pop_back();
      loop_starts.pop_back();
      loop_steps.pop_back();
      loop_extents.pop_back();
    }
  }

  // Calculate parity expression considering all nested loops
  PrimExpr CalculateParityExpr(PrimExpr iter_offset = 0) const {
    if (loop_vars.empty()) {
      return IntImm(DataType::Int(32), 0);
    }

    PrimExpr total_iter = IntImm(DataType::Int(32), 0);
    PrimExpr total_multiplier = IntImm(DataType::Int(32), 1);

    // Build expression: outer_var * inner_constant + inner_var
    // For nested loops: (((outer_var * inner_extent) + inner_var) *
    // innermost_step) + ...
    for (int i = loop_vars.size() - 1; i >= 0; i--) {
      PrimExpr normalized_iter =
          indexdiv(loop_vars[i] - loop_starts[i], loop_steps[i]);

      if (i == static_cast<int>(loop_vars.size()) - 1) {
        // Innermost loop
        total_iter = normalized_iter;
      } else {
        // Outer loop: multiply by inner loop extent
        // Check if inner loop extent is constant
        if (const auto *extent_int = loop_extents[i + 1].as<IntImmNode>()) {
          total_iter =
              normalized_iter * IntImm(DataType::Int(32), extent_int->value) +
              total_iter;
        } else {
          // If inner loop extent is not constant, we cannot compute parity
          // This should have been caught earlier
          LOG(FATAL)
              << "Inner loop extent must be constant for parity calculation";
          return IntImm(DataType::Int(32), 0);
        }
      }
    }

    // Add iteration offset and calculate parity
    return indexmod(total_iter + iter_offset, 2);
  }

  // Check if at least one loop is not at its start iteration
  // Returns true if NOT all loops are at their start iteration
  PrimExpr NotAllLoopsAtStart() const {
    if (loop_vars.empty()) {
      return Bool(false);
    }

    // Build expression: (loop_var0 != loop_start0) || (loop_var1 !=
    // loop_start1) || ...
    PrimExpr condition = Bool(false);

    for (size_t i = 0; i < loop_vars.size(); ++i) {
      PrimExpr loop_not_at_start = (loop_vars[i] != loop_starts[i]);

      if (i == 0) {
        condition = loop_not_at_start;
      } else {
        condition = (condition || loop_not_at_start);
      }
    }

    return condition;
  }
};

// Barrier synchronization helper functions
static Stmt InsertBarriersForNeutralSync(Stmt neutral_body, Stmt warpgroup_body,
                                         std::vector<Buffer> &barrier_buffers,
                                         Map<ObjectRef, ObjectRef> &barrier_map,
                                         PrimExpr thread_count[2]);
// Barrier dependency analysis functions
static void AnalyzeAndInsertBarriers(tl::IRStructure *node,
                                     int &next_barrier_id,
                                     std::vector<Buffer> &barrier_buffers,
                                     Map<ObjectRef, ObjectRef> &barrier_map,
                                     PrimExpr thread_count[2],
                                     tl::LoopNestingInfo &loop_info);
static void AnalyzeSequenceNodeBarriers(tl::SequenceNode *seq,
                                        int &next_barrier_id,
                                        std::vector<Buffer> &barrier_buffers,
                                        Map<ObjectRef, ObjectRef> &barrier_map,
                                        PrimExpr thread_count[2],
                                        tl::LoopNestingInfo &loop_info);
static void AnalyzeControlNodeBarriers(tl::ControlNode *ctrl,
                                       int &next_barrier_id,
                                       std::vector<Buffer> &barrier_buffers,
                                       Map<ObjectRef, ObjectRef> &barrier_map,
                                       PrimExpr thread_count[2],
                                       tl::LoopNestingInfo &loop_info);

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
  PrimExpr barrier_load = BufferLoad(barrier_buffer, {0});

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

// Helper function to insert a statement into ScheduleUnit's stmts
static void InsertStatementIntoScheduleUnit(ScheduleUnit *task, const Stmt &stmt,
                                           bool at_beginning,
                                           int warpgroup_id) {
  if (at_beginning) {
    task->before[warpgroup_id].insert(task->before[warpgroup_id].begin(), stmt);
  } else {
    task->after[warpgroup_id].push_back(stmt);
  }
}

// Barrier dependency analysis implementation
static void AnalyzeAndInsertBarriers(IRStructure *node, int &next_barrier_id,
                                     std::vector<Buffer> &barrier_buffers,
                                     Map<ObjectRef, ObjectRef> &barrier_map,
                                     PrimExpr thread_count[2],
                                     LoopNestingInfo &loop_info) {
  if (!node)
    return;

  if (node->IsSequence()) {
    AnalyzeSequenceNodeBarriers(static_cast<SequenceNode *>(node),
                                next_barrier_id, barrier_buffers, barrier_map,
                                thread_count, loop_info);
  } else if (node->IsControl()) {
    AnalyzeControlNodeBarriers(static_cast<ControlNode *>(node),
                               next_barrier_id, barrier_buffers, barrier_map,
                               thread_count, loop_info);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    AnalyzeAndInsertBarriers(wrapper->child.get(), next_barrier_id,
                             barrier_buffers, barrier_map, thread_count,
                             loop_info);
  } else if (node->IsTask()) {
    // For TaskNode, nothing to do at this level
  } else {
    LOG(FATAL);
  }
}

static void AnalyzeSequenceNodeBarriers(SequenceNode *seq, int &next_barrier_id,
                                        std::vector<Buffer> &barrier_buffers,
                                        Map<ObjectRef, ObjectRef> &barrier_map,
                                        PrimExpr thread_count[2],
                                        LoopNestingInfo &loop_info) {
  if (!seq)
    return;

  // Map from (buffer, warpgroup_id) to task of last access
  std::unordered_map<Buffer, std::pair<ScheduleUnit *, int>, ObjectPtrHash,
                     ObjectPtrEqual>
      last_access_map[2];
  std::unordered_map<Buffer, std::pair<ScheduleUnit *, std::pair<int, int>>,
                     ObjectPtrHash, ObjectPtrEqual>
      last_write_map;
  std::unordered_map<Buffer, std::pair<ScheduleUnit *, int>, ObjectPtrHash,
                     ObjectPtrEqual>
      last_wgmma_map[2];
  std::unordered_map<ScheduleUnit *, Buffer> barrier_unit_map;
  int wait_wgmma_id[2] = {}, total_wgmma[2] = {};

  // Process tasks in sequence order
  for (auto &promote_child : seq->children) {
    auto task = static_cast<ScheduleUnit *>(promote_child.get());
    if (task->child->IsSequence() || task->child->IsControl()) {
      // If child is SequenceNode or ControlNode, recursively analyze it
      AnalyzeAndInsertBarriers(task->child.get(), next_barrier_id,
                               barrier_buffers, barrier_map, thread_count,
                               loop_info);
    }

    // Check regions for dependencies
    for (const auto &region_access : task->GetReadWriteRegions()) {
      int wg_id = region_access.warpgroup_id;
      if (wg_id == -1)
        continue;
      auto &region = region_access.region;
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
        InsertStatementIntoScheduleUnit(task, wait_stmt, true, wg_id);
      } else {
        Buffer buffer = region->buffer;
        bool need_barrier = false;
        ScheduleUnit *last_access_task;
        int last_wg_id;
        bool is_async = task->UsesTensorCore() || task->UsesTMACore();
        if (!region_access.is_write) {
          auto it = last_write_map.find(buffer);
          if (it != last_write_map.end()) {
            last_access_task = it->second.first;
            last_wg_id = it->second.second.second;
            if (last_wg_id == -1)
              continue;
            if (it->second.second.first & (1 << wg_id))
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
            last_access_task = it->second.first;
            last_wg_id = it->second.second;
            if (last_wg_id == -1)
              continue;
            if (last_wg_id != wg_id) {
              need_barrier = true;
            }
          }
        }
        if (last_access_task == task)
          continue;
        // If warpgroup ids differ, insert barrier
        if (need_barrier) {
          if (barrier_unit_map.find(last_access_task) ==
              barrier_unit_map.end()) {
            // Allocate a new barrier ID and buffer
            int barrier_id = next_barrier_id++;
            Buffer barrier_buffer = makeBarrierBuffer(
                thread_count[last_wg_id],
                "barrier_" + std::to_string(barrier_id), barrier_map);
            barrier_unit_map[last_access_task] = barrier_buffer;
            // Collect the barrier buffer to be added to tilelang_root block's
            // alloc_buffers
            barrier_buffers.push_back(barrier_buffer);
            // Create BufferLoad expression for barrier[0]
            PrimExpr barrier_load = BufferLoad(barrier_buffer, {0});
            // Insert barrier_arrive at the end of last_access_task's statements
            Stmt arrive_stmt = makeBarrierArrive(barrier_load);
            InsertStatementIntoScheduleUnit(last_access_task, arrive_stmt, false,
                                           last_wg_id);
          }
          PrimExpr barrier_load =
              BufferLoad(barrier_unit_map[last_access_task], {0});

          // Insert barrier_wait at the beginning of task's statements
          Stmt wait_stmt =
              makeBarrierWait(barrier_load,
                              0); // parity = 0 for non-loop barriers
          InsertStatementIntoScheduleUnit(task, wait_stmt, true, wg_id);
          // Remove from map (as per user instruction)
          if (!region_access.is_write) {
            auto it = last_write_map.find(buffer);
            it->second.second.first |= (1 << wg_id);
            if (it->second.second.first == 3) {
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
      int wg_id = region_access.warpgroup_id;
      if (wg_id == -1)
        continue;
      auto &region = region_access.region;
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
        last_access_map[wg_id][buffer] = std::make_pair(task, wg_id);
        if (region_access.is_write) {
          last_write_map[buffer] =
              std::make_pair(task, std::make_pair(0, wg_id));
        }
      }
    }
  }
}

static void AnalyzeControlNodeBarriers(ControlNode *ctrl, int &next_barrier_id,
                                       std::vector<Buffer> &barrier_buffers,
                                       Map<ObjectRef, ObjectRef> &barrier_map,
                                       PrimExpr thread_count[2],
                                       LoopNestingInfo &loop_info) {
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
  PrimExpr loop_extent = for_node->extent;
  bool has_promoted_tasks = ctrl->hasPromote();

  // Add this loop to nesting info
  loop_info.AddLoop(for_node);

  // Check if inner loops have constant extents (if any)
  // This check will be done when calculating parity expression

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
                                 barrier_buffers, barrier_map, thread_count,
                                 loop_info);
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
    std::unordered_map<Buffer, std::pair<ScheduleUnit *, int>, ObjectPtrHash,
                       ObjectPtrEqual>
        last_access_map[2];
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
        last_access_set[2];
    std::unordered_map<Buffer, std::pair<ScheduleUnit *, std::pair<int, int>>,
                       ObjectPtrHash, ObjectPtrEqual>
        last_write_map;
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> last_write_set;
    std::unordered_map<Buffer, std::pair<ScheduleUnit *, int>, ObjectPtrHash,
                       ObjectPtrEqual>
        last_wgmma_map[2];
    std::unordered_map<ScheduleUnit *, Buffer> barrier_unit_map;
    int wait_wgmma_id[2] = {}, total_wgmma[2] = {};

    // Process tasks in the specified order
    for (unsigned iter = 0; iter != 2; ++iter) {
      for (ScheduleUnit *task : ordered_tasks) {
        bool is_promoted = task->GetPromote();
        bool is_async = task->UsesTensorCore() || task->UsesTMACore();

        // Check regions for dependencies
        for (const auto &region_access : task->GetReadWriteRegions()) {
          int wg_id = region_access.warpgroup_id;
          if (wg_id == -1)
            continue;
          auto &region = region_access.region;
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
            InsertStatementIntoScheduleUnit(task, wait_stmt, true, wg_id);
          } else {
            Buffer buffer = region->buffer;
            bool need_barrier = false;
            ScheduleUnit *last_access_task;
            int last_wg_id;
            bool last_is_promoted;
            if (!region_access.is_write) {
              if (iter == 1) {
                if (last_write_set.find(buffer) != last_write_set.end()) {
                  continue;
                }
                last_write_set.insert(buffer);
              }
              auto it = last_write_map.find(buffer);
              if (it != last_write_map.end()) {
                last_access_task = it->second.first;
                last_wg_id = it->second.second.second;
                last_is_promoted = last_access_task->GetPromote();
                if (last_wg_id == -1)
                  continue; // Allow barriers involving neutral tasks
                if (it->second.second.first & (1 << wg_id))
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
                last_access_task = it->second.first;
                last_wg_id = it->second.second;
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
            if (last_access_task == task)
              continue;
            // If warpgroup ids differ or promotion status differs, insert
            // barrier
            if (need_barrier) {
              // Calculate parity for barrier wait considering all nested loops
              // Use loop_info to calculate parity expression: outer_var *
              // inner_constant + inner_var
              PrimExpr parity_expr = loop_info.CalculateParityExpr(
                  IntImm(DataType::Int(32), iter + 2));

              if (barrier_unit_map.find(last_access_task) ==
                  barrier_unit_map.end()) {
                // Allocate a new barrier ID and buffer
                int barrier_id = next_barrier_id++;
                Buffer barrier_buffer = makeBarrierBuffer(
                    thread_count[last_wg_id],
                    "barrier_" + std::to_string(barrier_id), barrier_map);
                barrier_unit_map[last_access_task] = barrier_buffer;

                // Collect the barrier buffer to be added to tilelang_root
                // block's alloc_buffers
                barrier_buffers.push_back(barrier_buffer);

                // Create BufferLoad expression for barrier[0]
                PrimExpr barrier_load = BufferLoad(barrier_buffer, {0});

                // Insert barrier_arrive at the end of last_access_task's
                // statements
                Stmt arrive_stmt = makeBarrierArrive(barrier_load);
                InsertStatementIntoScheduleUnit(last_access_task, arrive_stmt,
                                               false, last_wg_id);
              }
              PrimExpr barrier_load =
                  BufferLoad(barrier_unit_map[last_access_task], {0});

              // Insert barrier_wait at the beginning of task's statements
              Stmt wait_stmt = makeBarrierWait(barrier_load, parity_expr);
              if (iter == 1) {
                // Check if at least one loop is not at its start iteration
                // (not the first iteration of all nested loops)
                wait_stmt =
                    IfThenElse(loop_info.NotAllLoopsAtStart(), wait_stmt);
              }
              InsertStatementIntoScheduleUnit(task, wait_stmt, true, wg_id);
              // Remove from map (as per user instruction)
              if (!region_access.is_write) {
                auto it = last_write_map.find(buffer);
                it->second.second.first |= (1 << wg_id);
                if (it->second.second.first == 3) {
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
            int wg_id = region_access.warpgroup_id;
            if (wg_id == -1)
              continue;
            auto &region = region_access.region;
            if (IsRegisterRegion(region)) {
              if (!task->UsesTensorCore() || !region_access.is_write)
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
              last_access_map[wg_id][buffer] = std::make_pair(task, wg_id);
              if (region_access.is_write) {
                last_write_map[buffer] =
                    std::make_pair(task, std::make_pair(0, wg_id));
              }
            }
          }
        }
      }
    }
  } else {
    AnalyzeAndInsertBarriers(ctrl->child.get(), next_barrier_id,
                             barrier_buffers, barrier_map, thread_count,
                             loop_info);
  }

  // Remove this loop from nesting info when exiting
  loop_info.PopLoop();
}
} // namespace tl
} // namespace tvm
