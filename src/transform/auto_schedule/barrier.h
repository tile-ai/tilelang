#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../../op/builtin.h"
#include "../../op/utils.h"
#include "./ir_structure.h"
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using ffi::Array;
using ffi::Map;

// Helper function to rewrite alloc_buffer for multi-version support
Buffer RewriteAllocBuffer(const Buffer &buffer, int num_stages) {
  // Create a copy of the buffer
  ObjectPtr<BufferNode> new_buffer =
      tvm::ffi::make_object<BufferNode>(*(buffer.get()));

  // Add num_stages as first dimension
  new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_stages));

  // Update strides if they exist
  if (!new_buffer->strides.empty()) {
    ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
    PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
    new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
  }

  return Buffer(new_buffer);
}

// Barrier manager for create_list_of_mbarrier and get_mbarrier
class BarrierManager {
public:
  BarrierManager() : next_barrier_id_(0) {}

  // Add a barrier with arrive_count, returns barrier_id
  int AddBarrier(PrimExpr arrive_count) {
    int barrier_id = next_barrier_id_++;
    barrier_arrive_counts_.push_back(arrive_count);
    return barrier_id;
  }

  // Get barrier_id for a specific barrier (by index)
  int GetBarrierId(int index) const { return index; }

  // Get arrive_count for a barrier_id
  PrimExpr GetArriveCount(int barrier_id) const {
    if (barrier_id >= 0 &&
        static_cast<size_t>(barrier_id) < barrier_arrive_counts_.size()) {
      return barrier_arrive_counts_[barrier_id];
    }
    return IntImm(DataType::Int(32), 0);
  }

  // Get all arrive counts for create_list_of_mbarrier
  Array<PrimExpr> GetAllArriveCounts() const { return barrier_arrive_counts_; }

  // Check if we have any barriers
  bool HasBarriers() const { return !barrier_arrive_counts_.empty(); }

  // Create the create_list_of_mbarrier statement
  Stmt CreateListOfMBarrier() const {
    if (barrier_arrive_counts_.empty()) {
      return Evaluate(0);
    }
    return Evaluate(Call(DataType::Handle(), tl::create_list_of_mbarrier(),
                         barrier_arrive_counts_));
  }

  // Create get_mbarrier expression
  PrimExpr GetMBarrier(int barrier_id) const {
    return GetMBarrier(IntImm(DataType::Int(32), barrier_id));
  }

  // Create get_mbarrier expression
  PrimExpr GetMBarrier(PrimExpr barrier_id) const {
    return Call(DataType::Handle(), tl::get_mbarrier(), {barrier_id});
  }

  // Convert to barrier_map for annotations
  Map<ObjectRef, ObjectRef> ToBarrierMap() const {
    Map<ObjectRef, ObjectRef> barrier_map;
    for (size_t i = 0; i < barrier_arrive_counts_.size(); ++i) {
      barrier_map.Set(IntImm(DataType::Int(32), i), barrier_arrive_counts_[i]);
    }
    return barrier_map;
  }

private:
  int next_barrier_id_;
  Array<PrimExpr> barrier_arrive_counts_;
};

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

  PrimExpr CalculateIterationCount() const {
    ICHECK(!loop_vars.empty());
    PrimExpr total_iter = IntImm(DataType::Int(32), 0);
    PrimExpr total_multiplier = IntImm(DataType::Int(32), 1);

    // Build expression: outer_var * inner_tripcount + inner_var
    // For nested loops: (((outer_var * inner_tripcount) + inner_var) *
    // innermost_step) + ...
    for (int i = loop_vars.size() - 1; i >= 0; i--) {
      // Calculate normalized iteration: (loop_var - start) / step
      PrimExpr normalized_iter =
          indexdiv(loop_vars[i] - loop_starts[i], loop_steps[i]);

      if (i == static_cast<int>(loop_vars.size()) - 1) {
        total_iter = normalized_iter;
      } else {
        total_iter = total_iter + normalized_iter * total_multiplier;
      }
      total_multiplier = total_multiplier * loop_extents[i];
    }
    return total_iter;
  }

  // Calculate parity expression considering all nested loops
  PrimExpr CalculateParityExpr(PrimExpr iter_offset, int num_stages) const {
    PrimExpr total_iter = indexdiv(CalculateIterationCount(), num_stages);

    // Add iteration offset and calculate parity
    return indexmod(total_iter + iter_offset, 2);
  }
};

// Structure to store multi-version buffer information
struct MultiVersionBufferInfo {
  Buffer buffer;
  int num_stages;
  Buffer new_buffer;

  MultiVersionBufferInfo(Buffer buffer, int num_stages, Buffer new_buffer)
      : buffer(buffer), num_stages(num_stages), new_buffer(new_buffer) {}
};

// Barrier dependency analysis function declarations
static void
AnalyzeAndInsertBarriers(IRStructure *node, BarrierManager &barrier_manager,
                         PrimExpr thread_count[2], LoopNestingInfo &loop_info,
                         std::vector<MultiVersionBufferInfo> &buffer_infos);
static void
AnalyzeSequenceNodeBarriers(SequenceNode *seq, BarrierManager &barrier_manager,
                            PrimExpr thread_count[2],
                            LoopNestingInfo &loop_info,
                            std::vector<MultiVersionBufferInfo> &buffer_infos);
static void
AnalyzeControlNodeBarriers(ControlNode *ctrl, BarrierManager &barrier_manager,
                           PrimExpr thread_count[2], LoopNestingInfo &loop_info,
                           std::vector<MultiVersionBufferInfo> &buffer_infos);

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

// Create a barrier using create_list_of_mbarrier and get_mbarrier
// Equivalent to Python: T.create_list_of_mbarrier(arrive_counts...)
// and T.get_mbarrier(barrier_id)
static int AddBarrierToManager(BarrierManager &barrier_manager,
                               PrimExpr arrive_count, int num_stages = 1) {
  auto begin = barrier_manager.AddBarrier(arrive_count);
  for (int idx = 1; idx < num_stages; ++idx) {
    barrier_manager.AddBarrier(arrive_count);
  }
  return begin;
}

// Get mbarrier expression by barrier_id
static PrimExpr GetMBarrierExpr(const BarrierManager &barrier_manager,
                                int barrier_id) {
  return barrier_manager.GetMBarrier(barrier_id);
}

// Get mbarrier expression by barrier_id
static PrimExpr GetMBarrierExpr(const BarrierManager &barrier_manager,
                                PrimExpr barrier_id) {
  return barrier_manager.GetMBarrier(barrier_id);
}

// Create the create_list_of_mbarrier statement
static Stmt CreateListOfMBarrierStmt(const BarrierManager &barrier_manager) {
  return barrier_manager.CreateListOfMBarrier();
}

// Convert barrier manager to barrier_map for annotations
static Map<ObjectRef, ObjectRef>
BarrierManagerToMap(const BarrierManager &barrier_manager) {
  return barrier_manager.ToBarrierMap();
}

// Insert barriers between neutral tasks and warpgroup-specific work
// This ensures neutral tasks complete before any warpgroup-specific work
// begins
static Stmt InsertBarriersForNeutralSync(Stmt neutral_body, Stmt warpgroup_body,
                                         BarrierManager &barrier_manager,
                                         PrimExpr thread_count[2]) {
  // If either body is empty, no barriers needed
  if (IsEvaluateZero(neutral_body) || IsEvaluateZero(warpgroup_body)) {
    return SeqStmt({neutral_body, warpgroup_body});
  }

  // Create barrier for neutral-to-warpgroup synchronization
  // Using arrive_count = thread_count[0] + thread_count[1] (number of threads
  // for neutral-to-warpgroup synchronization)
  PrimExpr arrive_count = thread_count[0] + thread_count[1];
  int barrier_id = AddBarrierToManager(barrier_manager, arrive_count);
  PrimExpr barrier_expr = GetMBarrierExpr(barrier_manager, barrier_id);

  // Use barrier for neutral-to-warpgroup synchronization
  // Parity 0 for wait, parity 1 for arrive (simplified)
  Stmt arrive_barrier = makeBarrierArrive(barrier_expr);
  Stmt wait_barrier = makeBarrierWait(barrier_expr, 0);

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

// StmtExprMutator to rewrite BufferLoad/BufferStore for multi-version buffers
class MultiBufferAccessRewriter : public StmtExprMutator {
public:
  MultiBufferAccessRewriter(
      const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
          &multi_buffer,
      PrimExpr version_index)
      : multi_buffer_(multi_buffer), version_index_(version_index) {}

private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));

    // Check if this buffer is in multi_buffer
    if (multi_buffer_.find(load->buffer) != multi_buffer_.end()) {
      // Add version_index as first dimension
      auto *n = load.CopyOnWrite();
      n->buffer = multi_buffer_.at(load->buffer);
      n->indices.insert(n->indices.begin(), version_index_);
    }

    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));

    // Check if this buffer is in multi_buffer
    if (multi_buffer_.find(store->buffer) != multi_buffer_.end()) {
      // Add version_index as first dimension
      auto *n = store.CopyOnWrite();
      n->buffer = multi_buffer_.at(store->buffer);
      n->indices.insert(n->indices.begin(), version_index_);
    }

    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    // Check if this is a tl.tileop.region call
    static const auto region_op = Op::Get("tl.tileop.region");
    if (op->op.same_as(region_op)) {
      // Handle tl.tileop.region call for multi-version buffers
      // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2:
      // write, 3: read/write) args[2..] = extents
      if (op->args.size() >= 2) {
        // Check if the buffer in BufferLoad needs multi-version support
        if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
          auto it = multi_buffer_.find(buffer_load->buffer);
          if (it != multi_buffer_.end()) {
            // This buffer needs multi-version support
            // Create new arguments array
            Array<PrimExpr> new_args;

            // Add the updated BufferLoad (already processed by VisitExpr)
            new_args.push_back(VisitExpr(op->args[0]));

            // Add access_type (unchanged)
            new_args.push_back(VisitExpr(op->args[1]));

            // Add extent for version dimension (value = 1)
            new_args.push_back(IntImm(DataType::Int(32), 1));

            // Add existing extents (if any)
            for (size_t i = 2; i < op->args.size(); i++) {
              new_args.push_back(VisitExpr(op->args[i]));
            }

            // Create new Call node with updated arguments
            return Call(op->dtype, op->op, new_args, op->annotations, op->span);
          }
        }
      }
    }

    // For other Call nodes, use default processing
    return StmtExprMutator::VisitExpr_(op);
  }

  const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      &multi_buffer_;
  PrimExpr version_index_;
};

// Recursive function to rewrite BufferLoad/BufferStore in TaskNode stmts
static void RewriteTaskNodeBuffers(
    IRStructure *node,
    const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
        &multi_buffer,
    PrimExpr version_index) {
  if (!node)
    return;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);

    // Apply MultiBufferAccessRewriter to all stmts in the task
    MultiBufferAccessRewriter rewriter(multi_buffer, version_index);
    for (auto &stmt : task->stmts) {
      stmt = rewriter(stmt);
    }
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (auto &child : seq->children) {
      RewriteTaskNodeBuffers(child.get(), multi_buffer, version_index);
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);
    RewriteTaskNodeBuffers(ctrl->child.get(), multi_buffer, version_index);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    RewriteTaskNodeBuffers(wrapper->child.get(), multi_buffer, version_index);
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    RewriteTaskNodeBuffers(unit->child.get(), multi_buffer, version_index);
  }
}

// Helper function to insert a statement into ScheduleUnit's stmts
static void InsertStatementIntoScheduleUnit(ScheduleUnit *task,
                                            const Stmt &stmt, bool at_beginning,
                                            int warpgroup_id) {
  if (at_beginning) {
    task->before[warpgroup_id].insert(task->before[warpgroup_id].begin(), stmt);
  } else {
    task->after[warpgroup_id].push_back(stmt);
  }
}

// Barrier dependency analysis implementation
static void
AnalyzeAndInsertBarriers(IRStructure *node, BarrierManager &barrier_manager,
                         PrimExpr thread_count[2], LoopNestingInfo &loop_info,
                         std::vector<MultiVersionBufferInfo> &buffer_infos) {
  if (!node)
    return;

  if (node->IsSequence()) {
    AnalyzeSequenceNodeBarriers(static_cast<SequenceNode *>(node),
                                barrier_manager, thread_count, loop_info,
                                buffer_infos);
  } else if (node->IsControl()) {
    AnalyzeControlNodeBarriers(static_cast<ControlNode *>(node),
                               barrier_manager, thread_count, loop_info,
                               buffer_infos);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    AnalyzeAndInsertBarriers(wrapper->child.get(), barrier_manager,
                             thread_count, loop_info, buffer_infos);
  } else if (node->IsTask()) {
    // For TaskNode, nothing to do at this level
  } else {
    LOG(FATAL);
  }
}

static void
AnalyzeSequenceNodeBarriers(SequenceNode *seq, BarrierManager &barrier_manager,
                            PrimExpr thread_count[2],
                            LoopNestingInfo &loop_info,
                            std::vector<MultiVersionBufferInfo> &buffer_infos) {
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
  std::unordered_map<ScheduleUnit *, int> barrier_unit_map;
  int wait_wgmma_id[2] = {}, total_wgmma[2] = {};

  // Process tasks in sequence order
  for (auto &promote_child : seq->children) {
    auto task = static_cast<ScheduleUnit *>(promote_child.get());
    if (task->child->IsSequence() || task->child->IsControl()) {
      // If child is SequenceNode or ControlNode, recursively analyze it
      AnalyzeAndInsertBarriers(task->child.get(), barrier_manager, thread_count,
                               loop_info, buffer_infos);
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
            // Allocate a new barrier using BarrierManager
            int barrier_id =
                AddBarrierToManager(barrier_manager, thread_count[last_wg_id]);
            barrier_unit_map[last_access_task] = barrier_id;
            // Create get_mbarrier expression
            PrimExpr barrier_expr =
                GetMBarrierExpr(barrier_manager, barrier_id);
            // Insert barrier_arrive at the end of last_access_task's
            // statements
            Stmt arrive_stmt = makeBarrierArrive(barrier_expr);
            InsertStatementIntoScheduleUnit(last_access_task, arrive_stmt,
                                            false, last_wg_id);
          }
          int barrier_id = barrier_unit_map[last_access_task];
          PrimExpr barrier_expr = GetMBarrierExpr(barrier_manager, barrier_id);

          // Insert barrier_wait at the beginning of task's statements
          Stmt wait_stmt =
              makeBarrierWait(barrier_expr,
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

static void
AnalyzeControlNodeBarriers(ControlNode *ctrl, BarrierManager &barrier_manager,
                           PrimExpr thread_count[2], LoopNestingInfo &loop_info,
                           std::vector<MultiVersionBufferInfo> &buffer_infos) {
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
        AnalyzeAndInsertBarriers(task->child.get(), barrier_manager,
                                 thread_count, loop_info, buffer_infos);
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
    std::unordered_map<ScheduleUnit *, PrimExpr> barrier_unit_map;
    int wait_wgmma_id[2] = {}, total_wgmma[2] = {};
    auto num_stages = 1;
    auto num_stages_val = ctrl->control.get()->annotations.Get("num_stages");
    if (num_stages_val.has_value()) {
      num_stages = num_stages_val.value().cast<IntImm>()->value;
    }

    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
        multi_buffer;
    if (num_stages != 1) {
      for (auto &region : ctrl->GetReadRegions()) {
        auto &buffer = region.get()->buffer;
        if (!IsSharedBuffer(buffer))
          continue;
        if (multi_buffer.find(buffer) != multi_buffer.end())
          continue;
        multi_buffer[buffer] = RewriteAllocBuffer(buffer, num_stages);
      }
      for (auto &region : ctrl->GetWriteRegions()) {
        auto &buffer = region.get()->buffer;
        if (!IsSharedBuffer(buffer))
          continue;
        if (multi_buffer.find(buffer) != multi_buffer.end())
          continue;
        multi_buffer[buffer] = RewriteAllocBuffer(buffer, num_stages);
      }
      // Need multi-version buffer rewriter
      // Add collected buffers to buffer_infos
      for (auto &buffer : multi_buffer) {
        buffer_infos.emplace_back(buffer.first, num_stages, buffer.second);
      }

      // Rewrite BufferLoad/BufferStore in TaskNode stmts for multi-version
      // buffers Calculate version index:
      // indexmod(loop_info.CalculateIterationCount(), num_stages)
      PrimExpr version_index =
          indexmod(loop_info.CalculateIterationCount(), num_stages);

      // Recursively rewrite all TaskNode stmts
      RewriteTaskNodeBuffers(ctrl, multi_buffer, version_index);
    }

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
              // Calculate parity for barrier wait considering all nested
              // loops Use loop_info to calculate parity expression: outer_var
              // * inner_constant + inner_var
              PrimExpr iter_offset = IntImm(DataType::Int(32), iter);
              PrimExpr parity_expr =
                  loop_info.CalculateParityExpr(iter_offset, num_stages);

              if (barrier_unit_map.find(last_access_task) ==
                  barrier_unit_map.end()) {
                // Allocate a new barrier using BarrierManager
                int barrier_id = AddBarrierToManager(
                    barrier_manager, thread_count[last_wg_id], num_stages);
                PrimExpr barrier = IntImm(DataType::Int(32), barrier_id);
                barrier =
                    barrier +
                    indexmod(loop_info.CalculateIterationCount(), num_stages);
                barrier_unit_map[last_access_task] = barrier;
                // Create get_mbarrier expression
                PrimExpr barrier_expr =
                    GetMBarrierExpr(barrier_manager, barrier);
                // Insert barrier_arrive at the end of last_access_task's
                // statements
                Stmt arrive_stmt = makeBarrierArrive(barrier_expr);
                InsertStatementIntoScheduleUnit(last_access_task, arrive_stmt,
                                                false, last_wg_id);
              }
              PrimExpr barrier = barrier_unit_map[last_access_task];
              PrimExpr barrier_expr = GetMBarrierExpr(barrier_manager, barrier);

              // Insert barrier_wait at the beginning of task's statements
              Stmt wait_stmt = makeBarrierWait(barrier_expr, parity_expr);
              if (iter == 1) {
                // Check if at least one loop is not at its start iteration
                // (not the first iteration of all nested loops)
                wait_stmt =
                    IfThenElse(indexdiv(loop_info.CalculateIterationCount(),
                                        num_stages) != 0,
                               wait_stmt);
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
    AnalyzeAndInsertBarriers(ctrl->child.get(), barrier_manager, thread_count,
                             loop_info, buffer_infos);
  }

  // Remove this loop from nesting info when exiting
  loop_info.PopLoop();
}
} // namespace tl
} // namespace tvm
