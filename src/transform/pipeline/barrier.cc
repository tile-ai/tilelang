#include "barrier.h"

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "../common/mbarrier.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/operator.h"

namespace tvm {
namespace tl {
namespace software_pipeline {

using namespace tirx;
using namespace ffi;

/*!
 * \brief Build the dependency graph among a array of blocks.
 * \param[in] blocks The array of blocks.
 * \param[out] dep_src2dst Optional, a map to store dependency edges from the
 * source to the destination. \param[out] dep_dst2src Optional, a map to store
 * dependency edges from the destination to the source.
 */
void BuildDependencyGraph(const Array<SBlock> &blocks,
                          BlockDependencyGraph *dep_src2dst,
                          BlockDependencyGraph *dep_dst2src) {
  std::unordered_map<Var, Array<SBlock>, ObjectPtrHash, ObjectPtrEqual>
      buffer_writers;

  for (const SBlock &block : blocks) {
    for (const BufferRegion &read : block->reads) {
      auto it = buffer_writers.find(read->buffer->data);
      if (it != buffer_writers.end()) {
        for (const SBlock &writer : it->second) {
          if (dep_src2dst != nullptr) {
            (*dep_src2dst)[writer].push_back(block);
          }
          if (dep_dst2src != nullptr) {
            (*dep_dst2src)[block].push_back(writer);
          }
        }
      }
    }
    for (const BufferRegion &write : block->writes) {
      buffer_writers[write->buffer->data].push_back(block);
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers for pipeline-level TMA barrier management
// ---------------------------------------------------------------------------

/*!
 * \brief Rewrite a block's body, converting tl.tileop.copy calls to
 *        tl.tileop.tma_copy with barrier and emit_arrive annotations.
 */
class CopyToTmaCopyRewriter : public StmtExprMutator {
public:
  CopyToTmaCopyRewriter(const Buffer &barrier_buf, PrimExpr barrier_id,
                        bool emit_arrive = true)
      : barrier_buf_(barrier_buf), barrier_id_(std::move(barrier_id)),
        emit_arrive_(emit_arrive) {}

  PrimExpr VisitExpr_(const CallNode *op) final {
    static const Op &copy_op = Op::Get("tl.tileop.copy");
    static const Op &tma_copy_op = Op::Get("tl.tileop.tma_copy");
    static const Op &im2col_op = Op::Get("tl.tileop.im2col");
    static const Op &deprecated_c2d_im2col_op = Op::Get("tl.tileop.c2d_im2col");
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(copy_op)) {
      auto new_annotations = call->annotations;
      new_annotations.Set("barrier", MakeBarrierRef(barrier_buf_, barrier_id_));
      new_annotations.Set("is_tma_copy", IntImm(DataType::Int(32), 1));
      new_annotations.Set("emit_arrive",
                          IntImm(DataType::Int(32), emit_arrive_ ? 1 : 0));
      return Call(call->dtype, tma_copy_op, call->args, new_annotations,
                  call->span);
    }
    // Annotate im2col with pipeline barrier so its Lower() uses it
    // instead of allocating a separate internal barrier.
    if (call->op.same_as(im2col_op) ||
        call->op.same_as(deprecated_c2d_im2col_op)) {
      auto new_annotations = call->annotations;
      new_annotations.Set("barrier", MakeBarrierRef(barrier_buf_, barrier_id_));
      new_annotations.Set("emit_arrive",
                          IntImm(DataType::Int(32), emit_arrive_ ? 1 : 0));
      return Call(call->dtype, call->op, call->args, new_annotations,
                  call->span);
    }
    return call;
  }

private:
  Buffer barrier_buf_;
  PrimExpr barrier_id_;
  bool emit_arrive_;
};

// ---------------------------------------------------------------------------
// ExpandPipelineBarriers - multi-version all barrier buffers for pipelining
// ---------------------------------------------------------------------------

/// Collect all shared.barrier Buffer objects referenced in a statement.
class BarrierBufferCollector : public StmtExprVisitor {
public:
  static std::vector<Buffer> Collect(const Array<SBlock> &blocks) {
    BarrierBufferCollector c;
    for (const auto &block : blocks) {
      c(block->body);
    }
    return {c.barriers_.begin(), c.barriers_.end()};
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (op->buffer.scope() == "shared.barrier" ||
        op->buffer.scope() == "shared.cluster_barrier") {
      if (!seen_.count(op->buffer)) {
        seen_.insert(op->buffer);
        barriers_.push_back(op->buffer);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->buffer.scope() == "shared.barrier" ||
        op->buffer.scope() == "shared.cluster_barrier") {
      if (!seen_.count(op->buffer)) {
        seen_.insert(op->buffer);
        barriers_.push_back(op->buffer);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // Also check barrier refs inside Call annotations (e.g., tma_copy barrier).
  void VisitExpr_(const CallNode *op) final {
    for (const auto &[key, val] : op->annotations) {
      if (auto load = val.as<BufferLoadNode>()) {
        if (load->buffer.scope() == "shared.barrier" ||
            load->buffer.scope() == "shared.cluster_barrier") {
          if (!seen_.count(load->buffer)) {
            seen_.insert(load->buffer);
            barriers_.push_back(load->buffer);
          }
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  BufferSet seen_;
  std::vector<Buffer> barriers_;
};

/// Rewrite barrier references: expand indices and rewrite parity.
class BarrierIndexRewriter : public StmtExprMutator {
public:
  BarrierIndexRewriter(const BufferMap &old_to_new,
                       const BufferShapeMap &old_shapes, PrimExpr stage_expr,
                       PrimExpr parity_cycle, Var loop_var, PrimExpr loop_min)
      : old_to_new_(old_to_new), old_shapes_(old_shapes),
        stage_expr_(std::move(stage_expr)),
        parity_cycle_(std::move(parity_cycle)), loop_var_(std::move(loop_var)),
        loop_min_(std::move(loop_min)) {}

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = old_to_new_.find(load->buffer);
    if (it != old_to_new_.end()) {
      auto *n = load.CopyOnWrite();
      PrimExpr old_size = old_shapes_.at(load->buffer);
      n->buffer = it->second;
      n->indices.Set(0, stage_expr_ * old_size + n->indices[0]);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = old_to_new_.find(store->buffer);
    if (it != old_to_new_.end()) {
      auto *n = store.CopyOnWrite();
      PrimExpr old_size = old_shapes_.at(store->buffer);
      n->buffer = it->second;
      n->indices.Set(0, stage_expr_ * old_size + n->indices[0]);
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));

    // Rewrite barrier refs inside annotations (e.g., tma_copy "barrier").
    bool anno_changed = false;
    Map<String, ObjectRef> new_annos = call->annotations;
    for (const auto &[key, val] : call->annotations) {
      if (auto load = val.as<BufferLoadNode>()) {
        auto it = old_to_new_.find(load->buffer);
        if (it != old_to_new_.end()) {
          PrimExpr old_size = old_shapes_.at(load->buffer);
          auto new_load = BufferLoad(
              it->second, {stage_expr_ * old_size + load->indices[0]});
          new_annos.Set(key, new_load);
          anno_changed = true;
        }
      }
    }
    if (anno_changed) {
      call = Call(call->dtype, call->op, call->args, new_annos, call->span);
    }

    // Rewrite mbarrier_wait_parity parity argument.
    if (call->op.same_as(mbarrier_wait_parity()) && call->args.size() >= 2) {
      if (auto load = call->args[0].as<BufferLoadNode>()) {
        // Check if the barrier ref (possibly already rewritten above)
        // targets one of our expanded barriers.
        bool is_expanded = false;
        for (const auto &kv : old_to_new_) {
          if (load->buffer.same_as(kv.second)) {
            is_expanded = true;
            break;
          }
        }
        if (is_expanded) {
          // Compute initial-phase offset from the user's original parity.
          arith::Analyzer analyzer;
          PrimExpr user_parity = call->args[1];
          PrimExpr user_parity_at_min = analyzer.Simplify(
              tirx::Substitute(user_parity, {{loop_var_, loop_min_}}));
          // New parity = (iteration_block + offset) % 2
          PrimExpr offset = IntImm(DataType::Int(32), 0);
          if (const int64_t *imm = as_const_int(user_parity_at_min)) {
            offset = IntImm(DataType::Int(32), *imm % 2);
          }
          PrimExpr new_parity = FloorMod(parity_cycle_ + offset, 2);
          Array<PrimExpr> new_args = call->args;
          new_args.Set(1, new_parity);
          return Call(call->dtype, call->op, new_args, call->annotations,
                      call->span);
        }
      }
    }
    return call;
  }

private:
  const BufferMap &old_to_new_;
  const BufferShapeMap &old_shapes_;
  PrimExpr stage_expr_;
  PrimExpr parity_cycle_;
  Var loop_var_;
  PrimExpr loop_min_;
};

/// Expand all shared.barrier buffers in the pipeline body from [N] to
/// [N * num_stages], rewrite barrier indices to include stage offset, and
/// rewrite mbarrier_wait_parity parity expressions.
///
/// This is the unified barrier multi-versioning path that replaces the old
/// late barrier-only fixup in OptimizeForTarget.
/// Returns a map of old-to-new barrier buffers for outer block alloc_buffers
/// update.
Map<Buffer, Buffer> ExpandPipelineBarriers(
    Array<SBlock> &original_order, PipelineInfo &pipeline_info,
    Map<Var, Buffer> &buffer_data_to_buffer, BufferSet &allocated_buffers,
    Array<Buffer> &block_local_allocs, Array<Buffer> &pipeline_allocs,
    Var loop_var, PrimExpr loop_min, int num_stages) {
  if (num_stages <= 1)
    return {};

  // Only expand barriers that have explicit ptx_arrive_barrier calls in the
  // loop body.  This distinguishes pipeline synchronization barriers (where
  // arrive/wait are user-managed and need per-stage slots) from barriers
  // whose arrival is managed internally by tile-ops (e.g., tcgen05 MMA
  // arrive barriers) - those should NOT be pipeline-expanded.
  // ISP-created pipeline_mbar is handled specially: it's always in
  // block_local_allocs and was just created, so include it too.
  BufferSet local_barrier_set;
  for (const Buffer &buf : block_local_allocs) {
    if (buf.scope() == "shared.barrier" ||
        buf.scope() == "shared.cluster_barrier")
      local_barrier_set.insert(buf);
  }

  // Find barriers that have explicit ptx_arrive_barrier calls.
  class ArriveBarrierDetector : public StmtExprVisitor {
  public:
    BufferSet arrived_;
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(builtin::ptx_arrive_barrier()) && !op->args.empty()) {
        if (auto load = op->args[0].as<BufferLoadNode>()) {
          arrived_.insert(load->buffer);
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  };
  ArriveBarrierDetector arrive_det;
  for (const auto &block : original_order) {
    arrive_det(block->body);
  }

  std::vector<Buffer> all_referenced =
      BarrierBufferCollector::Collect(original_order);
  std::vector<Buffer> barriers;
  for (const Buffer &buf : all_referenced) {
    // Include if: (a) it's an ISP-created local barrier, OR
    //             (b) it has explicit ptx_arrive_barrier calls.
    if (local_barrier_set.count(buf) || arrive_det.arrived_.count(buf)) {
      barriers.push_back(buf);
    }
  }
  if (barriers.empty())
    return {};

  PrimExpr ns = IntImm(DataType::Int(32), num_stages);
  PrimExpr stage_expr = FloorMod(loop_var - loop_min, ns);
  PrimExpr parity_cycle = FloorMod(FloorDiv(loop_var - loop_min, ns), 2);

  auto replace_in_array = [](Array<Buffer> &arr, const Buffer &old_buf,
                             const Buffer &new_buf) {
    for (size_t i = 0; i < arr.size(); ++i) {
      if (arr[i].same_as(old_buf)) {
        arr.Set(i, new_buf);
      }
    }
  };

  // Create expanded buffer for each barrier.
  BufferMap old_to_new;
  BufferShapeMap old_shapes;
  for (const Buffer &buf : barriers) {
    old_shapes[buf] = buf->shape[0];
    ObjectPtr<BufferNode> new_node = make_object<BufferNode>(*(buf.get()));
    new_node->shape = {PrimExpr(num_stages) * buf->shape[0]};
    Buffer new_buf(new_node);
    old_to_new[buf] = new_buf;

    // Update all maps and alloc arrays.
    buffer_data_to_buffer.Set(buf->data, new_buf);
    allocated_buffers.erase(buf);
    allocated_buffers.insert(new_buf);
    replace_in_array(block_local_allocs, buf, new_buf);
    replace_in_array(pipeline_allocs, buf, new_buf);
  }

  // Rewrite all blocks.
  BarrierIndexRewriter rewriter(old_to_new, old_shapes, stage_expr,
                                parity_cycle, loop_var, loop_min);
  for (size_t i = 0; i < original_order.size(); ++i) {
    SBlock old_block = original_order[i];
    Stmt new_body = rewriter(old_block->body);
    if (!new_body.same_as(old_block->body)) {
      // Also rewrite alloc_buffers in the block (barriers may be allocated
      // here).
      Array<Buffer> new_allocs;
      for (const Buffer &ab : old_block->alloc_buffers) {
        auto it = old_to_new.find(ab);
        new_allocs.push_back(it != old_to_new.end() ? it->second : ab);
      }
      SBlock new_block(old_block->iter_vars, old_block->reads,
                       old_block->writes, old_block->name_hint, new_body,
                       old_block->init, new_allocs, old_block->match_buffers,
                       old_block->annotations);
      PipelineAnnotation anno = pipeline_info.at(old_block);
      pipeline_info.erase(old_block);
      pipeline_info.emplace(new_block, anno);
      original_order.Set(i, new_block);
    }
  }

  // Return the old-to-new mapping for outer block alloc_buffers update.
  Map<Buffer, Buffer> result;
  for (const auto &[old_buf, new_buf] : old_to_new) {
    result.Set(old_buf, new_buf);
  }
  return result;
}

/*!
 * \brief Rewrite TMA-eligible copy blocks in the pipeline body for
 *        pipeline-level barrier management.
 *
 * For each TMA copy: convert tl.tileop.copy to tl.tileop.tma_copy with a
 * per-stage barrier slot and emit_arrive=1 so LowerTileOp emits arrive inside
 * the thread-0 guard.
 *
 * For the first consumer stage block: prepend mbarrier_wait_parity with
 * stage-indexed barrier reference and parity expression.
 *
 * \param original_order  In/out: blocks in original pipeline order.
 * \param pipeline_info   In/out: block to PipelineAnnotation mapping.
 * \param tma_copies      Per-statement TMA flag array from PipelinePlanning.
 * \param buffer_data_to_buffer  In/out: buffer var to Buffer mapping.
 * \param allocated_buffers      In/out: set of allocated buffers.
 * \param block_local_allocs     In/out: buffers allocated in the pipeline
 * block.
 * \return The newly created barrier buffer (undefined if no TMA copies).
 */
Buffer RewritePipelineTmaBarriers(
    Array<SBlock> &original_order, PipelineInfo &pipeline_info,
    const Array<Integer> &tma_copies, Map<Var, Buffer> &buffer_data_to_buffer,
    BufferSet &allocated_buffers, Array<Buffer> &block_local_allocs,
    Var loop_var, PrimExpr loop_min, int num_stages) {
  if (!std::any_of(tma_copies.begin(), tma_copies.end(),
                   [](const Integer &tc) { return !is_zero(tc); })) {
    return Buffer();
  }

  // Create pipeline barrier buffer with a single slot.  The generic
  // ExpandPipelineBarriers pass (called later) will expand it to
  // num_stages slots along with all other barrier buffers.
  Buffer barrier_buf = CreateMBarrierBuffer("pipeline_mbar", 1);
  buffer_data_to_buffer.Set(barrier_buf->data, barrier_buf);
  allocated_buffers.insert(barrier_buf);
  block_local_allocs.push_back(barrier_buf);

  // Find the index of the last TMA copy for arrive emission.
  int last_tma_idx = -1;
  for (size_t i = 0; i < original_order.size(); i++) {
    if (!is_zero(tma_copies[i]))
      last_tma_idx = static_cast<int>(i);
  }

  // Phase 1: Rewrite TMA copy blocks - all share barrier slot 0.
  // ExpandPipelineBarriers (called later) will rewrite indices to be
  // stage-dependent.  Only the last TMA copy emits arrive.
  for (size_t i = 0; i < original_order.size(); i++) {
    if (is_zero(tma_copies[i]))
      continue;

    bool is_last = (static_cast<int>(i) == last_tma_idx);
    SBlock old_block = original_order[i];
    CopyToTmaCopyRewriter rewriter(barrier_buf,
                                   /*barrier_id=*/IntImm(DataType::Int(32), 0),
                                   /*emit_arrive=*/is_last);
    Stmt new_body = rewriter(old_block->body);

    SBlock new_block(old_block->iter_vars, old_block->reads, old_block->writes,
                     old_block->name_hint, new_body, old_block->init,
                     old_block->alloc_buffers, old_block->match_buffers,
                     old_block->annotations);

    PipelineAnnotation anno = pipeline_info.at(old_block);
    pipeline_info.erase(old_block);
    pipeline_info.emplace(new_block, anno);
    original_order.Set(i, new_block);
  }

  // Phase 2: Insert waits in consumer blocks (blocks that depend on TMA data).
  // For simplicity, we insert waits before the first block whose stage > 0.
  bool waits_inserted = false;
  for (size_t i = 0; i < original_order.size(); i++) {
    if (waits_inserted)
      break;
    SBlock old_block = original_order[i];
    int stage = pipeline_info.at(old_block).stage;
    if (stage == 0)
      continue; // still in producer stage

    // Wait on barrier slot 0 with single-slot parity.
    // ExpandPipelineBarriers will rewrite index and parity for versioning.
    Array<Stmt> wait_stmts;
    {
      PrimExpr barrier_ref =
          MakeBarrierRef(barrier_buf, IntImm(DataType::Int(32), 0));
      PrimExpr ns = IntImm(DataType::Int(32), num_stages);
      PrimExpr parity = FloorMod(FloorDiv(loop_var - loop_min, ns), 2);
      wait_stmts.push_back(Evaluate(Call(
          DataType::Handle(), mbarrier_wait_parity(), {barrier_ref, parity})));
    }
    wait_stmts.push_back(old_block->body);
    Stmt new_body = SeqStmt(wait_stmts);

    SBlock new_block(old_block->iter_vars, old_block->reads, old_block->writes,
                     old_block->name_hint, new_body, old_block->init,
                     old_block->alloc_buffers, old_block->match_buffers,
                     old_block->annotations);

    PipelineAnnotation anno = pipeline_info.at(old_block);
    pipeline_info.erase(old_block);
    pipeline_info.emplace(new_block, anno);
    original_order.Set(i, new_block);
    waits_inserted = true;
  }

  return barrier_buf;
}

} // namespace software_pipeline
} // namespace tl
} // namespace tvm
