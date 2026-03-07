/*!
 * \file producer_consumer_ws.cc
 * \brief Producer-Consumer Warp Specialization pass for sm90+ TMA pipelines.
 *
 * Works with the inline mbarrier IR emitted by LowerBulkCopy:
 *   SeqStmt({
 *     AttrStmt("tl.tma_copy_write_buffer", buf, 1,
 *       IfThenElse(threadIdx.x == 0,
 *         SeqStmt({arrive_expect_tx(mbar, bytes), tma_load(...)}))),
 *     mbarrier_wait_parity(mbar, parity)
 *   })
 *
 * Splits the loop body into producer (TMA loads) and consumer (compute)
 * branches, adds back-pressure barriers for buffer reuse, and wraps in
 * an IfThenElse on threadIdx.x.
 */

#include "warp_specialized_rewriter.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace runtime;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct TmaCopyBlockInfo {
  Stmt producer_stmt; // AttrStmt("tl.tma_copy_write_buffer", ...) or IfThenElse
                      // with tma_load
  Stmt wait_stmt;     // Evaluate(mbarrier_wait_parity(...))
};

// ---------------------------------------------------------------------------
// Helpers (reused from warp_specialized_rewriter.cc patterns)
// ---------------------------------------------------------------------------

static PrimExpr makeGetBarrier(PrimExpr barrier_id) {
  return Call(DataType::Handle(), get_mbarrier(), {std::move(barrier_id)});
}

static Stmt makeArriveBarrier(PrimExpr barrier_id) {
  Array<PrimExpr> args = {makeGetBarrier(std::move(barrier_id))};
  return Evaluate(
      Call(DataType::Handle(), builtin::ptx_arrive_barrier(), args));
}

static Stmt makeParityWait(PrimExpr barrier_id, PrimExpr parity) {
  auto call = Call(DataType::Handle(), mbarrier_wait_parity(),
                   {makeGetBarrier(std::move(barrier_id)), std::move(parity)});
  return Evaluate(call);
}

// ---------------------------------------------------------------------------
// TmaCopyBlockExtractor
// ---------------------------------------------------------------------------

/*!
 * \brief Extract {producer, wait} pairs from a flattened loop body.
 *
 * Identifies consecutive pairs where a TMA producer statement is followed
 * by an mbarrier_wait_parity consumer sync. Two patterns are recognized:
 *
 *  Pattern 1: AttrStmt("tl.tma_copy_write_buffer", ...) + mbarrier_wait_parity
 *  Pattern 2: IfThenElse containing tma_load + mbarrier_wait_parity
 *
 * Everything else is classified as a compute statement.
 */
class TmaCopyBlockExtractor {
public:
  std::vector<TmaCopyBlockInfo> blocks;
  std::vector<Stmt> compute_stmts;

  void Extract(const Array<Stmt> &flat_stmts) {
    size_t i = 0;
    while (i < flat_stmts.size()) {
      if (i + 1 < flat_stmts.size() &&
          IsMbarrierWaitParity(flat_stmts[i + 1])) {
        // Check Pattern 1: AttrStmt("tl.tma_copy_write_buffer")
        auto *attr = flat_stmts[i].as<AttrStmtNode>();
        if (attr && attr->attr_key == "tl.tma_copy_write_buffer") {
          blocks.push_back({flat_stmts[i], flat_stmts[i + 1]});
          i += 2;
          continue;
        }
        // Check Pattern 2: IfThenElse containing tma_load
        if (ContainsTmaLoad(flat_stmts[i])) {
          blocks.push_back({flat_stmts[i], flat_stmts[i + 1]});
          i += 2;
          continue;
        }
      }
      compute_stmts.push_back(flat_stmts[i]);
      i++;
    }
  }

private:
  static bool IsMbarrierWaitParity(const Stmt &stmt) {
    if (auto *eval = stmt.as<EvaluateNode>()) {
      if (auto *call = eval->value.as<CallNode>()) {
        return call->op.same_as(mbarrier_wait_parity());
      }
    }
    return false;
  }

  static bool ContainsTmaLoad(const Stmt &stmt) {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (auto *call = node.as<CallNode>()) {
        if (call->op.same_as(tma_load()) ||
            call->op.same_as(tma_load_im2col())) {
          found = true;
        }
      }
    });
    return found;
  }
};

// ---------------------------------------------------------------------------
// ThreadIdxRewriter (from warp_specialized_rewriter.cc)
// ---------------------------------------------------------------------------

class PCThreadIdxRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt stmt, Var thread_var, PrimExpr replaced,
                      PrimExpr thread_extent, bool do_shuffle = false) {
    auto rewriter =
        PCThreadIdxRewriter(std::move(thread_var), std::move(replaced),
                            std::move(thread_extent), do_shuffle);
    return rewriter(std::move(stmt));
  }

private:
  PCThreadIdxRewriter(Var thread_var, PrimExpr replaced, PrimExpr thread_extent,
                      bool do_shuffle)
      : thread_var_(std::move(thread_var)), replaced_(std::move(replaced)),
        thread_extent_(std::move(thread_extent)), do_shuffle_(do_shuffle) {}

  PrimExpr VisitExpr_(const VarNode *var) final {
    if (var == thread_var_.get()) {
      return replaced_;
    }
    return StmtExprMutator::VisitExpr_(var);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    auto f_uses_thread = [=](const tvm::tir::VarNode *v) {
      return v == thread_var_.get();
    };
    maybe_thread_opt_ = false;
    if (!op->else_case.defined() && op->condition.as<EQNode>() &&
        UsesVar(op->condition, f_uses_thread) &&
        !(UsesVar(op->then_case, f_uses_thread))) {
      auto eq_op = Downcast<EQ>(op->condition);
      if (eq_op->a.as<VarNode>() == thread_var_.get() ||
          eq_op->b.as<VarNode>() == thread_var_.get()) {
        maybe_thread_opt_ = true;
      }
      auto then_case = StmtExprMutator::VisitStmt(op->then_case);
      maybe_thread_opt_ = do_shuffle_ && maybe_thread_opt_ && has_tma_op_;
      has_tma_op_ = false;
      if (maybe_thread_opt_) {
        return IfThenElse(
            Call(DataType::Bool(), tl_shuffle_elect(), {thread_extent_}),
            StmtExprMutator::VisitStmt(op->then_case), std::nullopt);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::tma_load()) ||
        op->op.same_as(tl::tma_load_im2col()) ||
        op->op.same_as(tl::tma_store())) {
      has_tma_op_ = true;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Var thread_var_;
  PrimExpr replaced_;
  PrimExpr thread_extent_;
  bool maybe_thread_opt_ = false;
  bool do_shuffle_;
  bool has_tma_op_ = false;
};

// ---------------------------------------------------------------------------
// MbarrierInitRemover: removes all create_list_of_mbarrier calls from a stmt
// ---------------------------------------------------------------------------

/*!
 * \brief Post-transform cleanup: remove any create_list_of_mbarrier calls
 *        that remain outside the transformed block (e.g., at the function
 *        body level where lower_tile_op.cc originally placed them).
 *        The new init is already emitted inside the block by the rewriter.
 */
class MbarrierInitRemover : public StmtExprMutator {
public:
  static Stmt Remove(Stmt stmt) {
    MbarrierInitRemover remover;
    return remover(std::move(stmt));
  }

private:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> new_seq;
    bool changed = false;
    for (const auto &s : op->seq) {
      if (IsCreateListOfMbarrier(s)) {
        changed = true;
        continue; // drop this statement
      }
      Stmt visited = VisitStmt(s);
      new_seq.push_back(visited);
      if (!visited.same_as(s))
        changed = true;
    }
    if (!changed)
      return GetRef<Stmt>(op);
    if (new_seq.size() == 1)
      return new_seq[0];
    return SeqStmt(new_seq);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (IsCreateListOfMbarrier(GetRef<Stmt>(op))) {
      // Return a no-op (should be caught by SeqStmt visitor above,
      // but handle standalone case too)
      return Evaluate(0);
    }
    return GetRef<Stmt>(op);
  }

  // Stop recursion at BlockRealize — the new init is inside the block
  // and we don't want to remove it.
  Stmt VisitStmt_(const BlockRealizeNode *op) final { return GetRef<Stmt>(op); }

  static bool IsCreateListOfMbarrier(const Stmt &stmt) {
    if (auto *eval = stmt.as<EvaluateNode>()) {
      if (auto *call = eval->value.as<CallNode>()) {
        return call->op.same_as(create_list_of_mbarrier());
      }
    }
    return false;
  }
};

// ---------------------------------------------------------------------------
// ProducerConsumerWSRewriter — main pass
// ---------------------------------------------------------------------------

class ProducerConsumerWSRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    // Check thread tags
    if (!ThreadTagChecker::HasOnlyThreadIdxX(f)) {
      LOG(WARNING) << "ProducerConsumerWS: disabled because program uses "
                      "thread tags other than threadIdx.x";
      return f;
    }

    ProducerConsumerWSRewriter T;
    f.CopyOnWrite()->body = T(f->body);

    // If WS was applied, remove any create_list_of_mbarrier calls that
    // remain OUTSIDE the block (e.g. at function body level from
    // lower_tile_op). The new init is already inside the block.
    if (T.ws_transformed_) {
      f.CopyOnWrite()->body = MbarrierInitRemover::Remove(f->body);
    }

    return f;
  }

private:
  // Locate the threadIdx.x binding
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent &&
        Downcast<IterVar>(op->node)->thread_tag == "threadIdx.x") {
      thread_iv_ = Downcast<IterVar>(op->node);
      need_update_thread_extent_ = false;
      AttrStmt attr_stmt = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
      if (need_update_thread_extent_) {
        thread_iv_.CopyOnWrite()->dom = {0, updated_thread_extent_.value()};
        attr_stmt.CopyOnWrite()->node = thread_iv_;
        attr_stmt.CopyOnWrite()->value = updated_thread_extent_.value();
      }
      thread_iv_ = {};
      return attr_stmt;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    if (!thread_iv_.defined())
      return StmtExprMutator::VisitStmt_(op);

    const Block &orig_block = op->block;

    // Find the pipelined loop with "num_stages" annotation
    const ForNode *pipeline_loop = FindPipelineLoop(orig_block->body);
    if (!pipeline_loop)
      return StmtExprMutator::VisitStmt_(op);

    int num_stages = 1;
    auto ns_anno = pipeline_loop->annotations.Get("num_stages");
    if (ns_anno) {
      num_stages = static_cast<int>(ns_anno.value().as<IntImmNode>()->value);
    }
    // Flatten the loop body
    Array<Stmt> flat_stmts;
    Stmt loop_body_root = pipeline_loop->body;
    if (auto *realize = pipeline_loop->body.as<BlockRealizeNode>()) {
      loop_body_root = realize->block->body;
    }
    FlattenSeqStmt(loop_body_root, &flat_stmts);

    // Extract TMA copy blocks
    TmaCopyBlockExtractor extractor;
    extractor.Extract(flat_stmts);

    if (extractor.blocks.empty()) {
      // No TMA loads found — fall through to standard pipeline
      return StmtExprMutator::VisitStmt_(op);
    }

    // Check if there are existing tl_pipeline_order/tl_pipeline_stage
    // with -1 values (WS+TMA enabled markers) — if so, use those
    auto order_anno = pipeline_loop->annotations.Get("tl_pipeline_order");
    auto stage_anno = pipeline_loop->annotations.Get("tl_pipeline_stage");
    if (order_anno && stage_anno) {
      auto order_array = Downcast<Array<Integer>>(order_anno.value());
      for (const auto &val : order_array) {
        if (val->value == -1) {
          // Already has WS pipeline annotations — skip
          return StmtExprMutator::VisitStmt_(op);
        }
      }
    }

    // ---------------------------------------------------------------
    // Build producer and consumer loop bodies
    // ---------------------------------------------------------------
    PrimExpr consumer_thread_extent = thread_iv_->dom->extent;
    consumer_thread_extent_ =
        consumer_thread_extent; // Store for RebuildBlockBody
    PrimExpr producer_thread_extent = IntImm(DataType::Int(32), 128);

    // Barrier layout — computed purely from structural information:
    //   num_tma_groups = number of TMA copy blocks detected
    //   num_stages     = from pipeline loop annotation
    //
    // Forward barriers (allocated by LowerBulkCopy):
    //   Group i → IDs [i*num_stages, (i+1)*num_stages)
    //   arrive_count = 1 (only TMA hardware arrives)
    //
    // Back-pressure barriers (allocated by this pass):
    //   Group i → IDs [num_fwd + i*num_stages, num_fwd + (i+1)*num_stages)
    //   arrive_count = consumer_threads (all consumer threads arrive)
    //
    int num_tma_groups = static_cast<int>(extractor.blocks.size());
    int num_fwd_barriers = num_tma_groups * num_stages;
    int num_bp_barriers = num_tma_groups * num_stages;
    int total_barriers = num_fwd_barriers + num_bp_barriers;

    std::vector<int> bp_bases;
    bp_bases.reserve(num_tma_groups);
    for (int i = 0; i < num_tma_groups; i++) {
      bp_bases.push_back(num_fwd_barriers + i * num_stages);
    }

    Var loop_var = pipeline_loop->loop_var;
    PrimExpr loop_extent = pipeline_loop->extent;
    PrimExpr loop_min = pipeline_loop->min;

    // Compute stage and parity expressions
    // stage_expr = (loop_var - min) % num_stages
    // parity_expr = ((loop_var - min) / num_stages) % 2
    PrimExpr linear_idx = loop_var - loop_min;
    PrimExpr stage_expr = FloorMod(linear_idx, num_stages);
    PrimExpr parity_expr = FloorMod(FloorDiv(linear_idx, num_stages), 2);

    // --- Build Producer Body ---
    Array<Stmt> producer_body_stmts;
    for (size_t ti = 0; ti < extractor.blocks.size(); ti++) {
      const auto &tma = extractor.blocks[ti];
      PrimExpr bp_id = IntImm(DataType::Int(32), bp_bases[ti]) + stage_expr;

      // Wait on back-pressure barrier (consumer signals buffer is free)
      // Producer waits with xor(parity, 1) — bootstraps correctly
      producer_body_stmts.push_back(
          makeParityWait(bp_id, bitwise_xor(parity_expr, 1)));

      // Execute the original producer statement (arrive_expect_tx + tma_load)
      producer_body_stmts.push_back(tma.producer_stmt);
    }
    Stmt producer_loop_body = SeqStmt(producer_body_stmts);

    // --- Build Consumer Body ---
    Array<Stmt> consumer_body_stmts;

    // First: wait on all forward barriers (TMA data ready)
    for (size_t ti = 0; ti < extractor.blocks.size(); ti++) {
      consumer_body_stmts.push_back(extractor.blocks[ti].wait_stmt);
    }

    // Then: execute all compute statements
    for (const auto &stmt : extractor.compute_stmts) {
      consumer_body_stmts.push_back(stmt);
    }

    // Finally: arrive on back-pressure barriers (signal buffer reuse OK)
    for (size_t ti = 0; ti < extractor.blocks.size(); ti++) {
      PrimExpr bp_id = IntImm(DataType::Int(32), bp_bases[ti]) + stage_expr;
      consumer_body_stmts.push_back(makeArriveBarrier(bp_id));
    }
    Stmt consumer_loop_body = SeqStmt(consumer_body_stmts);

    // --- Build the loops ---
    // Remove pipeline annotations since WS handles overlap directly
    Map<String, Any> loop_annos;
    for (const auto &[key, value] : pipeline_loop->annotations) {
      if (key != "num_stages" && key != "tl_pipeline_order" &&
          key != "tl_pipeline_stage" && key != "software_pipeline_order" &&
          key != "software_pipeline_stage") {
        loop_annos.Set(key, value);
      }
    }

    Stmt producer_loop =
        For(loop_var, loop_min, loop_extent, ForKind::kSerial,
            producer_loop_body, Optional<IterVar>(), loop_annos);
    Stmt consumer_loop =
        For(loop_var, loop_min, loop_extent, ForKind::kSerial,
            consumer_loop_body, Optional<IterVar>(), loop_annos);

    // Rewrite threadIdx.x in producer: threadIdx.x -> threadIdx.x -
    // consumer_threads Also converts `if (threadIdx.x == 0)` to `if
    // (tl_shuffle_elect(extent))`
    producer_loop = PCThreadIdxRewriter::Rewrite(
        producer_loop, thread_iv_->var,
        thread_iv_->var - consumer_thread_extent, producer_thread_extent,
        /*do_shuffle=*/true);
    consumer_loop = PCThreadIdxRewriter::Rewrite(
        consumer_loop, thread_iv_->var, thread_iv_->var, consumer_thread_extent,
        /*do_shuffle=*/true);

    // Wrap in IfThenElse: producer if threadIdx.x >= consumer_threads
    Stmt ws_body = IfThenElse(GE(thread_iv_->var, consumer_thread_extent),
                              producer_loop, consumer_loop);

    // Add warp specialization scope attribute
    Array<IntImm> ws_partition = {Downcast<IntImm>(producer_thread_extent),
                                  Downcast<IntImm>(consumer_thread_extent)};
    ws_body =
        AttrStmt(ws_partition, attr::kWarpSpecializationScope, 0, ws_body);

    // Build barrier init: forward barriers (arrive_count=1) +
    // back-pressure barriers (arrive_count=consumer_threads)
    Array<PrimExpr> barrier_arrive_counts;
    barrier_arrive_counts.reserve(total_barriers);
    for (int i = 0; i < num_fwd_barriers; i++) {
      barrier_arrive_counts.push_back(IntImm(DataType::Int(32), 1));
    }
    for (int i = 0; i < num_bp_barriers; i++) {
      barrier_arrive_counts.push_back(consumer_thread_extent);
    }
    Stmt init_barrier = Evaluate(Call(
        DataType::Handle(), create_list_of_mbarrier(), barrier_arrive_counts));

    // Reconstruct block body: replace the pipeline loop and
    // create_list_of_mbarrier with new init_barrier + ws_body.
    Stmt new_block_body = RebuildBlockBody(orig_block->body, pipeline_loop,
                                           init_barrier, ws_body);

    // Update thread extent
    updated_thread_extent_ = consumer_thread_extent + producer_thread_extent;
    need_update_thread_extent_ = true;
    ws_transformed_ = true;

    // Build the new Block and BlockRealize (without recursive mutation
    // since we've already transformed the body directly).
    Block new_block(orig_block->iter_vars, orig_block->reads,
                    orig_block->writes, orig_block->name_hint, new_block_body,
                    orig_block->init, orig_block->alloc_buffers,
                    orig_block->match_buffers, orig_block->annotations);
    return BlockRealize(op->iter_values, op->predicate, new_block);
  }

  // Handle ForNode with thread bindings
  Stmt VisitStmt_(const ForNode *op) final {
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (for_node->kind == ForKind::kThreadBinding && thread_iv_.defined()) {
      ICHECK(for_node->thread_binding.defined());
      String thread_tag = for_node->thread_binding.value()->thread_tag;
      if (thread_tag == "threadIdx.x") {
        Var thread_v = Downcast<Var>(for_node->loop_var);
        Stmt new_body = PCThreadIdxRewriter::Rewrite(for_node->body, thread_v,
                                                     thread_iv_->var, 0);
        return new_body;
      }
    }
    return for_node;
  }

  // ---------------------------------------------------------------------------
  // Utility methods
  // ---------------------------------------------------------------------------

  void FlattenSeqStmt(const Stmt &s, Array<Stmt> *out) {
    if (auto *seq = s.as<SeqStmtNode>()) {
      for (const auto &sub : seq->seq) {
        FlattenSeqStmt(sub, out);
      }
    } else {
      out->push_back(s);
    }
  }

  const ForNode *FindPipelineLoop(const Stmt &stmt) {
    if (auto *for_node = stmt.as<ForNode>()) {
      if (for_node->annotations.Get("num_stages")) {
        return for_node;
      }
    }
    // Look through SeqStmt, BlockRealize, Block etc.
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        auto *result = FindPipelineLoop(s);
        if (result)
          return result;
      }
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return FindPipelineLoop(realize->block->body);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return FindPipelineLoop(block->body);
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return FindPipelineLoop(attr->body);
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      return FindPipelineLoop(let_s->body);
    }
    return nullptr;
  }

  /*!
   * \brief Rebuild the block body, replacing the pipeline loop with
   *        init_barrier + ws_body, removing old create_list_of_mbarrier,
   *        and guarding post-loop statements with a consumer thread check.
   *
   *  Statements after the pipeline loop (e.g. epilogue, store) must only
   *  be executed by consumer threads, since producer threads don't compute.
   *  We wrap them in: if (threadIdx.x < consumer_threads) { ... }
   */
  Stmt RebuildBlockBody(const Stmt &body, const ForNode *target_loop,
                        const Stmt &init_barrier, const Stmt &ws_body) {
    // If this IS the target loop, replace it
    if (body.as<ForNode>() == target_loop) {
      return SeqStmt({init_barrier, ws_body});
    }

    if (auto *seq = body.as<SeqStmtNode>()) {
      Array<Stmt> new_seq;
      Array<Stmt> post_loop_stmts;
      bool found_loop = false;

      for (const auto &s : seq->seq) {
        // Remove existing create_list_of_mbarrier
        if (IsCreateListOfMbarrier(s))
          continue;

        if (!found_loop && ContainsLoop(s, target_loop)) {
          // Replace the pipeline loop
          Stmt rebuilt =
              RebuildBlockBody(s, target_loop, init_barrier, ws_body);
          new_seq.push_back(rebuilt);
          found_loop = true;
        } else if (found_loop) {
          // Collect statements after the pipeline loop
          post_loop_stmts.push_back(s);
        } else {
          // Statements before the pipeline loop (e.g. C_local clear)
          new_seq.push_back(s);
        }
      }

      // Guard post-loop statements so only consumer threads execute them
      if (!post_loop_stmts.empty()) {
        ICHECK(thread_iv_.defined());
        Stmt post_body = post_loop_stmts.size() == 1 ? post_loop_stmts[0]
                                                     : SeqStmt(post_loop_stmts);
        Stmt guarded =
            IfThenElse(LT(thread_iv_->var, consumer_thread_extent_), post_body);
        new_seq.push_back(guarded);
      }

      if (new_seq.size() == 1)
        return new_seq[0];
      return SeqStmt(new_seq);
    }

    // Walk through wrapper nodes
    if (auto *attr = body.as<AttrStmtNode>()) {
      if (ContainsLoop(attr->body, target_loop)) {
        Stmt new_body =
            RebuildBlockBody(attr->body, target_loop, init_barrier, ws_body);
        return AttrStmt(attr->node, attr->attr_key, attr->value, new_body);
      }
    }
    if (auto *let_s = body.as<LetStmtNode>()) {
      if (ContainsLoop(let_s->body, target_loop)) {
        Stmt new_body =
            RebuildBlockBody(let_s->body, target_loop, init_barrier, ws_body);
        return LetStmt(let_s->var, let_s->value, new_body);
      }
    }

    // Fallback: return unchanged
    return body;
  }

  bool ContainsLoop(const Stmt &stmt, const ForNode *target) {
    if (stmt.as<ForNode>() == target)
      return true;
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        if (ContainsLoop(s, target))
          return true;
      }
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return ContainsLoop(attr->body, target);
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      return ContainsLoop(let_s->body, target);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return ContainsLoop(realize->block->body, target);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return ContainsLoop(block->body, target);
    }
    return false;
  }

  bool IsCreateListOfMbarrier(const Stmt &stmt) {
    if (auto *eval = stmt.as<EvaluateNode>()) {
      if (auto *call = eval->value.as<CallNode>()) {
        return call->op.same_as(create_list_of_mbarrier());
      }
    }
    return false;
  }

  IterVar thread_iv_;
  PrimExpr
      consumer_thread_extent_; // Original thread extent (consumer warp count)
  Optional<PrimExpr> updated_thread_extent_;
  bool need_update_thread_extent_ = false;
  bool ws_transformed_ = false;
};

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

using namespace tir::transform;

// Check only for manual warp specialization ("warp_specialize" attr).
// Unlike WarpSpecializedDetector, we do NOT skip when TMA+mbarrier are
// both present, since that is the expected input pattern for this pass.
class ManualWSDetector : public StmtExprVisitor {
public:
  static bool HasManualWS(const Stmt &stmt) {
    ManualWSDetector d;
    d.VisitStmt(stmt);
    return d.has_manual_ws_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "warp_specialize" &&
        op->value.as<IntImmNode>()->value == 1) {
      has_manual_ws_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  bool has_manual_ws_ = false;
};

tvm::transform::Pass ProducerConsumerWarpSpecialized() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool disable_warp_specialized =
        ctx->GetConfig<Bool>(kDisableWarpSpecialized, Bool(false)).value();
    if (disable_warp_specialized)
      return f;

    // Skip if user has manual warp specialization
    if (ManualWSDetector::HasManualWS(f->body))
      return f;

    return ProducerConsumerWSRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ProducerConsumerWarpSpecialized",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ProducerConsumerWarpSpecialized",
                        ProducerConsumerWarpSpecialized);
}

} // namespace tl
} // namespace tvm
