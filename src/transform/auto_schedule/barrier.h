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

// Calculate the size in bytes of a buffer. Returns 0 if the size cannot be
// determined at compile time.
size_t GetBufferSize(const Buffer &buffer) {
  size_t elem_size = buffer->dtype.bits() * buffer->dtype.lanes() / 8;
  PrimExpr size = IntImm(DataType::Int(64), elem_size);
  for (const auto &dim : buffer->shape) {
    size *= dim;
  }
  arith::Analyzer analyzer;
  auto size_val = analyzer.Simplify(size);
  if (const auto *int_imm = size_val.as<IntImmNode>()) {
    return int_imm->value;
  } else {
    return 0;
  }
}

// Helper function to rewrite alloc_buffer for multi-version support
Buffer RewriteAllocBuffer(const Buffer &buffer, int num_versions) {
  // Create a copy of the buffer
  ObjectPtr<BufferNode> new_buffer =
      tvm::ffi::make_object<BufferNode>(*(buffer.get()));

  // Add num_versions as first dimension
  new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));

  // Update strides if they exist
  if (!new_buffer->strides.empty()) {
    ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
    PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
    new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
  }

  return Buffer(new_buffer);
}

// Create a barrier buffer allocation and return the Buffer object.
// shape = (num_versions,) so that each pipeline stage has its own barrier slot,
// accessed via BufferLoad(buffer, {stage_index}).
static Buffer makeBarrierBuffer(PrimExpr arrive_count, const std::string &name,
                                int num_versions,
                                std::vector<Buffer> &barrier_buffers,
                                Map<ObjectRef, ObjectRef> &barrier_map) {
  Array<PrimExpr> shape = {num_versions};
  DataType dtype = DataType::UInt(64);
  Type ptr_type = PointerType(PrimType(dtype), "shared.barrier");
  Var handle(name, ptr_type);
  Array<ObjectRef> arrive_counts;
  for (int i = 0; i < num_versions; ++i) {
    arrive_counts.push_back(arrive_count);
  }
  barrier_map.Set(handle, arrive_counts);
  Buffer buffer =
      Buffer(handle, dtype, shape, {}, PrimExpr(), name, 0, 0, kDefault);
  barrier_buffers.push_back(buffer);
  return buffer;
}

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
  PrimExpr CalculateParityExpr(PrimExpr iter_offset, int num_versions) const {
    PrimExpr total_iter = indexdiv(CalculateIterationCount(), num_versions);

    // Add iteration offset and calculate parity
    return indexmod(total_iter + iter_offset, 2);
  }
};

// Structure to store multi-version buffer information
struct MultiVersionBufferInfo {
  Buffer buffer;
  int num_versions;
  Buffer new_buffer;

  MultiVersionBufferInfo(Buffer buffer, int num_versions, Buffer new_buffer)
      : buffer(buffer), num_versions(num_versions), new_buffer(new_buffer) {}
};

// Barrier dependency analysis function declarations
static void
AnalyzeAndInsertBarriers(IRStructure *node, int &next_barrier_id,
                         std::vector<Buffer> &barrier_buffers,
                         Map<ObjectRef, ObjectRef> &barrier_map,
                         PrimExpr thread_count[2], LoopNestingInfo &loop_info,
                         std::vector<MultiVersionBufferInfo> &buffer_infos,
                         Buffer neutral_sync_shared_barrier);
static void
AnalyzeSequenceNodeBarriers(SequenceNode *seq, int &next_barrier_id,
                            std::vector<Buffer> &barrier_buffers,
                            Map<ObjectRef, ObjectRef> &barrier_map,
                            PrimExpr thread_count[2],
                            LoopNestingInfo &loop_info,
                            std::vector<MultiVersionBufferInfo> &buffer_infos,
                            Buffer neutral_sync_shared_barrier);
static void
AnalyzeControlNodeBarriers(ControlNode *ctrl, int &next_barrier_id,
                           std::vector<Buffer> &barrier_buffers,
                           Map<ObjectRef, ObjectRef> &barrier_map,
                           PrimExpr thread_count[2], LoopNestingInfo &loop_info,
                           std::vector<MultiVersionBufferInfo> &buffer_infos,
                           Buffer neutral_sync_shared_barrier);

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

static Stmt makeTcgen05MmaArrive(Buffer barrier_buffer) {
  Array<PrimExpr> access_ptr_args;
  access_ptr_args.push_back(tir::TypeAnnotation(DataType::UInt(64)));
  access_ptr_args.push_back(barrier_buffer->data);
  access_ptr_args.push_back(barrier_buffer->elem_offset);
  access_ptr_args.push_back(IntImm(DataType::Int(32), 1));
  access_ptr_args.push_back(IntImm(DataType::Int(32), 3));
  auto access_ptr =
      Call(DataType::Handle(), builtin::tvm_access_ptr(), access_ptr_args);
  return Evaluate(Call(DataType::Handle(), tcgen05_mma_arrive(), {access_ptr}));
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

static bool IsRegularSharedScope(const String &scope) {
  return scope == "shared" || scope == "shared.dyn";
}

static bool IsTmemScope(const String &scope) { return scope == "shared.tmem"; }

static bool HasWriteReadDependencyByScope(
    const IRStructure *producer, const IRStructure *consumer,
    const std::function<bool(const String &)> &scope_matcher) {
  if (!producer || !consumer) {
    return false;
  }

  auto producer_writes = producer->GetWriteRegions();
  auto consumer_reads = consumer->GetReadRegions();

  for (const auto &write_region : producer_writes) {
    const Buffer &write_buffer = write_region->buffer;
    if (!scope_matcher(write_buffer.scope())) {
      continue;
    }
    for (const auto &read_region : consumer_reads) {
      if (write_buffer.same_as(read_region->buffer)) {
        return true;
      }
    }
  }
  return false;
}

static bool HasSharedWriteReadDependency(const IRStructure *producer,
                                         const IRStructure *consumer) {
  return HasWriteReadDependencyByScope(
      producer, consumer,
      [](const String &scope) { return IsRegularSharedScope(scope); });
}

static bool HasTmemWriteReadDependency(const IRStructure *producer,
                                       const IRStructure *consumer) {
  return HasWriteReadDependencyByScope(
      producer, consumer,
      [](const String &scope) { return IsTmemScope(scope); });
}

static Stmt InsertBarriersForNeutralSyncWithDependency(
    Stmt producer_body, Stmt consumer_body,
    std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, PrimExpr total_thread_count,
    bool need_regular_barrier, bool need_tmem_barrier,
    Buffer neutral_sync_shared_barrier = Buffer(), Var thread_var = Var(),
    PrimExpr tensor_core_wg_start = PrimExpr(),
    PrimExpr tensor_core_wg_end = PrimExpr()) {
  if (IsEvaluateZero(producer_body) || IsEvaluateZero(consumer_body)) {
    return SeqStmt({producer_body, consumer_body});
  }

  if (!need_regular_barrier && !need_tmem_barrier) {
    return SeqStmt({producer_body, consumer_body});
  }

  std::vector<Stmt> arrive_stmts;
  std::vector<Stmt> wait_stmts;

  if (need_regular_barrier) {
    Buffer barrier_buffer =
        neutral_sync_shared_barrier.defined()
            ? neutral_sync_shared_barrier
            : makeBarrierBuffer(total_thread_count,
                                "neutral_sync_shared_barrier", 1,
                                barrier_buffers, barrier_map);
    PrimExpr barrier_load = BufferLoad(barrier_buffer, {0});
    arrive_stmts.push_back(makeBarrierArrive(barrier_load));
    wait_stmts.push_back(makeBarrierWait(barrier_load, 0));
  }

  if (need_tmem_barrier) {
    // TMEM barrier: arrive_count = 1 (only tensor core warp arrives)
    Buffer barrier_buffer = makeBarrierBuffer(IntImm(DataType::Int(32), 1),
                                              "neutral_sync_tmem_barrier", 1,
                                              barrier_buffers, barrier_map);
    PrimExpr barrier_load = BufferLoad(barrier_buffer, {0});

    // tcgen05_mma_arrive only in tensor core warpgroup
    Stmt tmem_arrive = makeTcgen05MmaArrive(barrier_buffer);
    if (tensor_core_wg_start.defined() && tensor_core_wg_end.defined()) {
      // Wrap in if condition for tensor core warpgroup only
      // Condition: tensor_core_wg_start <= thread_idx < tensor_core_wg_end
      tmem_arrive = IfThenElse((thread_var >= tensor_core_wg_start) &&
                                   (thread_var < tensor_core_wg_end),
                               tmem_arrive, Evaluate(0));
    }
    arrive_stmts.push_back(tmem_arrive);
    wait_stmts.push_back(makeBarrierWait(barrier_load, 0));
  }

  std::vector<Stmt> stmts;
  stmts.push_back(producer_body);
  stmts.insert(stmts.end(), arrive_stmts.begin(), arrive_stmts.end());
  stmts.insert(stmts.end(), wait_stmts.begin(), wait_stmts.end());
  stmts.push_back(consumer_body);
  return SeqStmt(stmts);
}

// Insert barriers between neutral tasks and warpgroup-specific work
// This ensures neutral tasks complete before any warpgroup-specific work begins
static Stmt InsertBarriersForNeutralSync(Stmt neutral_body, Stmt warpgroup_body,
                                         std::vector<Buffer> &barrier_buffers,
                                         Map<ObjectRef, ObjectRef> &barrier_map,
                                         PrimExpr total_thread_count,
                                         Buffer neutral_sync_shared_barrier) {
  return InsertBarriersForNeutralSyncWithDependency(
      neutral_body, warpgroup_body, barrier_buffers, barrier_map,
      total_thread_count, true, false, neutral_sync_shared_barrier);
}

// StmtExprMutator to rewrite BufferLoad/BufferStore for multi-version buffers
class MultiBufferAccessRewriter : public StmtExprMutator {
public:
  MultiBufferAccessRewriter(
      const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
          &multi_buffer,
      PrimExpr iteration)
      : multi_buffer_(multi_buffer), iteration_(iteration) {}

private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));

    // Check if this buffer is in multi_buffer
    if (multi_buffer_.find(load->buffer) != multi_buffer_.end()) {
      // Add version_index as first dimension
      auto *n = load.CopyOnWrite();
      n->buffer = multi_buffer_.at(load->buffer);
      auto num_versions = n->buffer->shape[0];
      n->indices.insert(n->indices.begin(), indexmod(iteration_, num_versions));
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
      auto num_versions = n->buffer->shape[0];
      n->indices.insert(n->indices.begin(), indexmod(iteration_, num_versions));
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
  PrimExpr iteration_;
};

// Recursive function to rewrite BufferLoad/BufferStore in TaskNode stmts
static void RewriteTaskNodeBuffers(
    IRStructure *node,
    const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
        &multi_buffer,
    PrimExpr iteration) {
  if (!node)
    return;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);

    // Apply MultiBufferAccessRewriter to all stmts in the task
    MultiBufferAccessRewriter rewriter(multi_buffer, iteration);
    for (auto &stmt : task->stmts) {
      stmt = rewriter(stmt);
    }
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (auto &child : seq->children) {
      RewriteTaskNodeBuffers(child.get(), multi_buffer, iteration);
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);
    RewriteTaskNodeBuffers(ctrl->child.get(), multi_buffer, iteration);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    RewriteTaskNodeBuffers(wrapper->child.get(), multi_buffer, iteration);
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    RewriteTaskNodeBuffers(unit->child.get(), multi_buffer, iteration);
  }
}

// Rewrite the gemm call's mbar argument (arg[16]) in a TaskNode to use the
// allocated barrier expression from BarrierManager.
// This is used for TCGEN05MMA where the gemm needs to reference the correct
// mbarrier for synchronization.
static void RewriteGemmMbar(TaskNode *task, PrimExpr mbar_expr) {
  static const auto gemm_py_op = Op::Get("tl.tileop.gemm_py");
  static const auto gemm_op = Op::Get("tl.tileop.gemm");

  class GemmMbarRewriter : public StmtExprMutator {
  public:
    GemmMbarRewriter(PrimExpr mbar_expr) : mbar_expr_(std::move(mbar_expr)) {}

  private:
    PrimExpr VisitExpr_(const CallNode *op) override {
      static const auto gemm_py_op = Op::Get("tl.tileop.gemm_py");
      static const auto gemm_op = Op::Get("tl.tileop.gemm");

      if ((op->op.same_as(gemm_py_op) || op->op.same_as(gemm_op)) &&
          op->args.size() > 16) {
        Array<PrimExpr> new_args;
        for (size_t i = 0; i < op->args.size(); ++i) {
          if (i == 16) {
            // Replace mbar argument with the barrier expression
            new_args.push_back(mbar_expr_);
          } else {
            new_args.push_back(VisitExpr(op->args[i]));
          }
        }
        return Call(op->dtype, op->op, new_args, op->annotations, op->span);
      }
      return StmtExprMutator::VisitExpr_(op);
    }

    PrimExpr mbar_expr_;
  };

  GemmMbarRewriter rewriter(mbar_expr);
  for (auto &stmt : task->stmts) {
    stmt = rewriter(stmt);
  }
}

static void RewriteCopyMbar(TaskNode *task, PrimExpr mbar_expr) {
  static const auto copy_op = Op::Get("tl.tileop.copy");
  static const auto tma_copy_op = Op::Get("tl.tileop.tma_copy");

  class CopyMbarRewriter : public StmtExprMutator {
  public:
    CopyMbarRewriter(PrimExpr mbar_expr) : mbar_expr_(std::move(mbar_expr)) {}

  private:
    PrimExpr VisitExpr_(const CallNode *op) override {
      if (op->op.same_as(copy_op)) {
        auto new_ann = op->annotations;
        new_ann.Set("barrier", mbar_expr_);
        return Call(op->dtype, tma_copy_op, op->args, new_ann, op->span);
      }
      return StmtExprMutator::VisitExpr_(op);
    }

    PrimExpr mbar_expr_;
  };

  CopyMbarRewriter rewriter(mbar_expr);
  for (auto &stmt : task->stmts) {
    stmt = rewriter(stmt);
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
AnalyzeAndInsertBarriers(IRStructure *node, int &next_barrier_id,
                         std::vector<Buffer> &barrier_buffers,
                         Map<ObjectRef, ObjectRef> &barrier_map,
                         PrimExpr thread_count[2], LoopNestingInfo &loop_info,
                         std::vector<MultiVersionBufferInfo> &buffer_infos,
                         Buffer neutral_sync_shared_barrier) {
  if (!node)
    return;

  if (node->IsSequence()) {
    AnalyzeSequenceNodeBarriers(static_cast<SequenceNode *>(node),
                                next_barrier_id, barrier_buffers, barrier_map,
                                thread_count, loop_info, buffer_infos,
                                neutral_sync_shared_barrier);
  } else if (node->IsControl()) {
    AnalyzeControlNodeBarriers(static_cast<ControlNode *>(node),
                               next_barrier_id, barrier_buffers, barrier_map,
                               thread_count, loop_info, buffer_infos,
                               neutral_sync_shared_barrier);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    AnalyzeAndInsertBarriers(
        wrapper->child.get(), next_barrier_id, barrier_buffers, barrier_map,
        thread_count, loop_info, buffer_infos, neutral_sync_shared_barrier);
  } else if (node->IsTask()) {
    // For TaskNode, nothing to do at this level
  } else {
    LOG(FATAL);
  }
}

static void
AnalyzeSequenceNodeBarriers(SequenceNode *seq, int &next_barrier_id,
                            std::vector<Buffer> &barrier_buffers,
                            Map<ObjectRef, ObjectRef> &barrier_map,
                            PrimExpr thread_count[2],
                            LoopNestingInfo &loop_info,
                            std::vector<MultiVersionBufferInfo> &buffer_infos,
                            Buffer neutral_sync_shared_barrier) {
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
      AnalyzeAndInsertBarriers(
          task->child.get(), next_barrier_id, barrier_buffers, barrier_map,
          thread_count, loop_info, buffer_infos, neutral_sync_shared_barrier);
    }

    // Allocate barrier for TCGEN05MMA and rewrite gemm mbar argument
    if (task->isInnerTask() && task->UsesTensorCore()) {
      auto child = static_cast<TaskNode *>(task->child.get());
      if (child->is_TCGEN05()) {
        int wg_id = child->GetWarpgroupId();
        int barrier_id = next_barrier_id++;
        Buffer barrier_buffer = makeBarrierBuffer(
            1, "tcgen05_barrier_" + std::to_string(barrier_id), 1,
            barrier_buffers, barrier_map);
        barrier_unit_map[task] = barrier_buffer;

        PrimExpr barrier_load = BufferLoad(barrier_buffer, {0});
        RewriteGemmMbar(child, barrier_load);
      }
    }

    // Allocate barrier for TMA
    if (task->isInnerTask() && task->UsesTMACore()) {
      auto child = static_cast<TaskNode *>(task->child.get());
      if (child->HasTMALoad()) {
        int wg_id = child->GetWarpgroupId();
        if (wg_id != -1) {
          int barrier_id = next_barrier_id++;
          Buffer barrier_buffer = makeBarrierBuffer(
              thread_count[wg_id], "tma_barrier_" + std::to_string(barrier_id),
              1, barrier_buffers, barrier_map);
          barrier_unit_map[task] = barrier_buffer;

          PrimExpr barrier_load = BufferLoad(barrier_buffer, {0});
          RewriteCopyMbar(child, barrier_load);
          Stmt arrive_stmt = makeBarrierArrive(barrier_load);
          InsertStatementIntoScheduleUnit(task, arrive_stmt, false, wg_id);
        } else {
          PrimExpr barrier_load = BufferLoad(neutral_sync_shared_barrier, {0});
          RewriteCopyMbar(child, barrier_load);
        }
      }
    }

    // Check regions for dependencies
    for (const auto &region_access : task->GetReadWriteRegions()) {
      int wg_id = region_access.warpgroup_id;
      if (wg_id == -1)
        continue;
      auto &region = region_access.region;
      if (IsRegisterRegion(region)) {
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
        ScheduleUnit *last_access_task = nullptr;
        int last_wg_id = -1;
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
            // Allocate a new barrier buffer (single stage for sequence)
            int barrier_id = next_barrier_id++;
            Buffer barrier_buffer =
                makeBarrierBuffer(thread_count[last_wg_id],
                                  "barrier_" + std::to_string(barrier_id), 1,
                                  barrier_buffers, barrier_map);
            barrier_unit_map[last_access_task] = barrier_buffer;
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
        if (!task->UsesTensorCore() || !region_access.is_write)
          continue;
        if (!task->isInnerTask())
          continue;
        auto child = static_cast<TaskNode *>(task->child.get());
        if (child->is_WGMMA()) {
          Buffer buffer = region->buffer;
          if (!found_wgmma) {
            found_wgmma = true;
            ++total_wgmma[wg_id];
          }
          last_wgmma_map[wg_id][buffer] =
              std::make_pair(task, total_wgmma[wg_id]);
        }
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
AnalyzeControlNodeBarriers(ControlNode *ctrl, int &next_barrier_id,
                           std::vector<Buffer> &barrier_buffers,
                           Map<ObjectRef, ObjectRef> &barrier_map,
                           PrimExpr thread_count[2], LoopNestingInfo &loop_info,
                           std::vector<MultiVersionBufferInfo> &buffer_infos,
                           Buffer neutral_sync_shared_barrier) {
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

    // Collect all tasks from the sequence
    std::vector<ScheduleUnit *> ordered_tasks;
    for (auto &child : seq->children) {
      auto task = static_cast<ScheduleUnit *>(child.get());
      if (task->child->IsSequence() || task->child->IsControl()) {
        // If child is SequenceNode or ControlNode, recursively analyze it
        AnalyzeAndInsertBarriers(
            task->child.get(), next_barrier_id, barrier_buffers, barrier_map,
            thread_count, loop_info, buffer_infos, neutral_sync_shared_barrier);
      }
      ordered_tasks.push_back(task);
    }

    // Process in order: sort by stage
    // This matches the software pipelining order
    std::stable_sort(
        ordered_tasks.begin(), ordered_tasks.end(),
        [](ScheduleUnit *a, ScheduleUnit *b) { return a->stage > b->stage; });

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
    std::map<std::pair<ScheduleUnit *, Buffer>, Buffer> barrier_unit_map;
    int wait_wgmma_id[2] = {}, total_wgmma[2] = {};
    auto num_stages = 1;
    auto num_stages_val = ctrl->control.get()->annotations.Get("num_stages");
    if (num_stages_val.has_value()) {
      num_stages = num_stages_val.value().cast<IntImm>()->value;
    }

    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
        multi_buffer;
    std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual>
        buffer_num_versions;
    if (num_stages != 1) {
      for (const auto &task : ordered_tasks) {
        for (const auto &region_access : task->GetReadWriteRegions()) {
          auto &buffer = region_access.region->buffer;
          if (!IsSharedBuffer(buffer))
            continue;
          for (const auto &other_task : ordered_tasks) {
            if (task == other_task)
              continue;
            int distance = task->child->GetStartTime() +
                           task->child->GetLatency() -
                           other_task->child->GetStartTime();
            if (distance <= 0)
              continue;
            distance = (distance - 1) / ctrl->GetIIperIter() + 1;
            for (const auto &other_region_access :
                 other_task->GetReadWriteRegions()) {
              auto &other_buffer = other_region_access.region->buffer;
              if (!buffer.same_as(other_buffer))
                continue;
              if (region_access.is_write || other_region_access.is_write) {
                auto &num_versions = buffer_num_versions[buffer];
                num_versions = std::max(num_versions, distance);
              }
            }
          }
        }
      }
      for (auto &region : ctrl->GetWriteRegions()) {
        auto &buffer = region.get()->buffer;
        if (!IsSharedBuffer(buffer))
          continue;
        if (multi_buffer.find(buffer) != multi_buffer.end())
          continue;
        auto it = buffer_num_versions.find(buffer);
        if (it == buffer_num_versions.end())
          continue;
        int num_versions = it->second;
        if (num_versions == 1)
          continue;
        auto new_buffer = RewriteAllocBuffer(buffer, num_versions);
        multi_buffer[buffer] = new_buffer;
        buffer_infos.emplace_back(buffer, num_versions, new_buffer);
      }

      // Rewrite BufferLoad/BufferStore in TaskNode stmts for multi-version
      // buffers
      PrimExpr iteration = loop_info.CalculateIterationCount();

      // Recursively rewrite all TaskNode stmts
      RewriteTaskNodeBuffers(ctrl, multi_buffer, iteration);
    }

    // Process tasks in the specified order
    for (unsigned iter = 0; iter != 2; ++iter) {
      for (ScheduleUnit *task : ordered_tasks) {
        int stage = task->GetStage();
        bool is_async = task->UsesTensorCore() || task->UsesTMACore();

        if (iter == 0 && task->isInnerTask() && task->UsesTensorCore()) {
          auto child = static_cast<TaskNode *>(task->child.get());
          if (child->is_TCGEN05()) {
            auto &&write_regions = child->GetWriteRegions();
            if (write_regions.size() == 1) {
              Buffer buffer = write_regions[0]->buffer;
              auto it = buffer_num_versions.find(buffer);
              int num_versions =
                  it != buffer_num_versions.end() ? it->second : 1;
              int wg_id = child->GetWarpgroupId();
              int barrier_id = next_barrier_id++;
              // Create a single barrier buffer with shape (num_versions,)
              Buffer barrier_buffer = makeBarrierBuffer(
                  1, "tcgen05_barrier_" + std::to_string(barrier_id),
                  num_versions, barrier_buffers, barrier_map);
              barrier_unit_map[std::make_pair(task, buffer)] = barrier_buffer;

              // Rewrite the gemm call's mbar argument (arg[16]) to use
              // BufferLoad(barrier_buffer, {version_index})
              PrimExpr version_index =
                  indexmod(loop_info.CalculateIterationCount(), num_versions);
              PrimExpr mbar_expr = BufferLoad(barrier_buffer, {version_index});
              RewriteGemmMbar(child, mbar_expr);
            } else {
              LOG(FATAL)
                  << "TCGEN05MMA tasks must have exactly one write region";
            }
          }
        }

        // Allocate barrier for TMA
        if (iter == 0 && task->isInnerTask() && task->UsesTMACore()) {
          auto child = static_cast<TaskNode *>(task->child.get());
          if (child->HasTMALoad()) {
            auto &&write_regions = child->GetWriteRegions();
            if (write_regions.size() == 1) {
              Buffer buffer = write_regions[0]->buffer;
              auto it = buffer_num_versions.find(buffer);
              int num_versions =
                  it != buffer_num_versions.end() ? it->second : 1;
              int wg_id = child->GetWarpgroupId();
              ICHECK(wg_id != -1) << "TMA loads must have valid warpgroup id";
              int barrier_id = next_barrier_id++;
              Buffer barrier_buffer =
                  makeBarrierBuffer(thread_count[wg_id],
                                    "tma_barrier_" + std::to_string(barrier_id),
                                    num_versions, barrier_buffers, barrier_map);
              barrier_unit_map[std::make_pair(task, buffer)] = barrier_buffer;

              PrimExpr version_index =
                  indexmod(loop_info.CalculateIterationCount(), num_versions);
              PrimExpr barrier_load =
                  BufferLoad(barrier_buffer, {version_index});
              RewriteCopyMbar(child, barrier_load);
              Stmt arrive_stmt = makeBarrierArrive(barrier_load);
              InsertStatementIntoScheduleUnit(task, arrive_stmt, false, wg_id);
            } else {
              LOG(FATAL) << "TMA loads must have exactly one write region";
            }
          }
        }

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
            auto it = buffer_num_versions.find(buffer);
            int num_versions = it != buffer_num_versions.end() ? it->second : 1;
            bool need_barrier = false;
            ScheduleUnit *last_access_task;
            int last_wg_id;
            int last_stage;
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
                last_stage = last_access_task->GetStage();
                if (last_wg_id == -1)
                  continue; // Allow barriers involving neutral tasks
                if (it->second.second.first & (1 << wg_id))
                  continue;

                bool last_async = last_access_task->UsesTensorCore() ||
                                  last_access_task->UsesTMACore();
                // If warpgroup ids differ or promotion status differs,
                // insert barrier
                if (last_wg_id != wg_id || last_stage != stage || is_async ||
                    last_async) {
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
                last_stage = last_access_task->GetStage();
                if (last_wg_id == -1)
                  continue; // Allow barriers involving neutral tasks

                // If warpgroup ids differ or promotion status differs,
                // insert barrier
                if (last_wg_id != wg_id || last_stage != stage) {
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
              // loops Use loop_info to calculate parity expression:
              // outer_var
              // * inner_constant + inner_var
              PrimExpr iter_offset = IntImm(DataType::Int(32), iter);
              PrimExpr parity_expr =
                  loop_info.CalculateParityExpr(iter_offset, num_versions);

              if (barrier_unit_map.find(std::make_pair(
                      last_access_task, buffer)) == barrier_unit_map.end()) {
                // Allocate a single barrier buffer with shape (num_versions,)
                int barrier_id = next_barrier_id++;
                Buffer barrier_buffer = makeBarrierBuffer(
                    thread_count[last_wg_id],
                    "barrier_" + std::to_string(barrier_id), num_versions,
                    barrier_buffers, barrier_map);
                barrier_unit_map[std::make_pair(last_access_task, buffer)] =
                    barrier_buffer;

                // Create BufferLoad with version-indexed offset
                PrimExpr version_index =
                    indexmod(loop_info.CalculateIterationCount(), num_versions);
                PrimExpr barrier_load =
                    BufferLoad(barrier_buffer, {version_index});
                // Insert barrier_arrive at the end of last_access_task's
                // statements
                Stmt arrive_stmt = makeBarrierArrive(barrier_load);
                InsertStatementIntoScheduleUnit(last_access_task, arrive_stmt,
                                                false, last_wg_id);
              }
              Buffer &barrier_buffer =
                  barrier_unit_map[std::make_pair(last_access_task, buffer)];
              PrimExpr version_index =
                  indexmod(loop_info.CalculateIterationCount(), num_versions);
              PrimExpr barrier_load =
                  BufferLoad(barrier_buffer, {version_index});

              // Insert barrier_wait at the beginning of task's statements
              Stmt wait_stmt = makeBarrierWait(barrier_load, parity_expr);
              // if (iter == 1) {
              //   // Check if at least one loop is not at its start
              //   iteration
              //   // (not the first iteration of all nested loops)
              //   wait_stmt =
              //       IfThenElse(indexdiv(loop_info.CalculateIterationCount(),
              //                           num_stages) != 0,
              //                  wait_stmt);
              // }
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
          bool found_wgmma = false;
          for (const auto &region_access : task->GetReadWriteRegions()) {
            int wg_id = region_access.warpgroup_id;
            if (wg_id == -1)
              continue;
            auto &region = region_access.region;
            if (IsRegisterRegion(region)) {
              if (!task->UsesTensorCore() || !region_access.is_write)
                continue;
              if (!task->isInnerTask())
                continue;
              auto child = static_cast<TaskNode *>(task->child.get());
              if (child->is_WGMMA()) {
                Buffer buffer = region->buffer;
                if (!found_wgmma) {
                  found_wgmma = true;
                  ++total_wgmma[wg_id];
                }
                last_wgmma_map[wg_id][buffer] =
                    std::make_pair(task, total_wgmma[wg_id]);
              }
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
    AnalyzeAndInsertBarriers(
        ctrl->child.get(), next_barrier_id, barrier_buffers, barrier_map,
        thread_count, loop_info, buffer_infos, neutral_sync_shared_barrier);
  }

  // Remove this loop from nesting info when exiting
  loop_info.PopLoop();
}
} // namespace tl
} // namespace tvm
