/*!
 * \file producer_consumer_ws.cc
 * \brief Producer-consumer warp specialization for sm90+ async-copy pipelines.
 *
 * Works on the inline barrier IR emitted by lowering passes such as
 * LowerBulkCopy / LowerPTXAsyncCopy:
 *   SeqStmt({
 *     AttrStmt("tl.tma_copy_write_buffer", buf, 1,
 *       IfThenElse(threadIdx.x == 0,
 *         SeqStmt({arrive_expect_tx(mbar, bytes), tma_load(...)}))),
 *     mbarrier_wait_parity(mbar, parity)
 *   })
 *
 * The pass splits the pipelined loop into:
 *   producer: issues TMA / cp.async
 *   consumer: waits, computes, and releases buffers
 *
 * For pure-TMA loops we rewrite the forward-barrier protocol so the producer
 * releases the barrier after issuing the TMA copy:
 *   expect_transaction -> tma_load -> arrive
 */

#include "common/tma_copy_utils.h"
#include "warp_specialized_rewriter.h"

#include <algorithm>
#include <string>

namespace tvm {
namespace tl {

using namespace tir;
using namespace runtime;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

enum class AsyncProducerKind : uint8_t { kTma, kCpAsync };

struct AsyncCopyBlockInfo {
  AsyncProducerKind kind;
  Stmt producer_stmt;              // TMA issue or cp.async enqueue+commit
  Optional<Stmt> wait_stmt;        // Existing forward wait for TMA blocks
  Optional<Var> write_buffer_data; // shared buffer written by producer
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

static Stmt makeCpAsyncBarrierNoInc(PrimExpr barrier_id) {
  auto call = Call(DataType::Handle(), tl::ptx_cp_async_barrier_noinc(),
                   {makeGetBarrier(std::move(barrier_id))});
  return Evaluate(call);
}

static Stmt makeParityWait(PrimExpr barrier_id, PrimExpr parity) {
  auto call = Call(DataType::Handle(), mbarrier_wait_parity(),
                   {makeGetBarrier(std::move(barrier_id)), std::move(parity)});
  return Evaluate(call);
}

static bool IsTrivialNoOpStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *imm = eval->value.as<IntImmNode>()) {
      return imm->value == 0;
    }
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq) {
      if (!IsTrivialNoOpStmt(s)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// AsyncCopyBlockExtractor
// ---------------------------------------------------------------------------

/*!
 * \brief Extract async producer blocks from a flattened loop body.
 *
 * Recognized patterns:
 *
 *  Pattern 1: AttrStmt("tl.tma_copy_write_buffer", ...) + mbarrier_wait_parity
 *  Pattern 2: IfThenElse containing tma_load + mbarrier_wait_parity
 *  Pattern 3: cp_async-only stmt + commit_group + wait_group(0)
 *
 * Everything else is classified as a compute statement.
 */
class AsyncCopyBlockExtractor {
public:
  std::vector<AsyncCopyBlockInfo> blocks;
  std::vector<Stmt> compute_stmts;

  void Extract(const Array<Stmt> &flat_stmts) {
    size_t i = 0;
    while (i < flat_stmts.size()) {
      if (i + 1 < flat_stmts.size() &&
          IsMbarrierWaitParity(flat_stmts[i + 1])) {
        // Check Pattern 1: AttrStmt("tl.tma_copy_write_buffer")
        auto *attr = flat_stmts[i].as<AttrStmtNode>();
        if (attr && attr->attr_key == "tl.tma_copy_write_buffer") {
          Optional<Var> write_buffer_data = std::nullopt;
          if (auto *v = attr->node.as<VarNode>()) {
            write_buffer_data = GetRef<Var>(v);
          }
          blocks.push_back({AsyncProducerKind::kTma,
                            StripTmaCopyWriteBufferAttr(flat_stmts[i]),
                            Optional<Stmt>(flat_stmts[i + 1]),
                            write_buffer_data});
          i += 2;
          continue;
        }
        // Check Pattern 2: IfThenElse containing tma_load
        if (ContainsTmaLoad(flat_stmts[i])) {
          blocks.push_back({AsyncProducerKind::kTma,
                            StripTmaCopyWriteBufferAttr(flat_stmts[i]),
                            Optional<Stmt>(flat_stmts[i + 1]), std::nullopt});
          i += 2;
          continue;
        }
      }
      if (i + 2 < flat_stmts.size() && ContainsPtxCpAsync(flat_stmts[i]) &&
          IsPtxCommitGroup(flat_stmts[i + 1]) &&
          IsPtxWaitGroupZero(flat_stmts[i + 2])) {
        Array<Stmt> producer_seq{flat_stmts[i], flat_stmts[i + 1]};
        Stmt producer_stmt =
            producer_seq.size() == 1 ? producer_seq[0] : SeqStmt(producer_seq);
        blocks.push_back({AsyncProducerKind::kCpAsync, producer_stmt,
                          Optional<Stmt>(),
                          GetCpAsyncDstBufferData(flat_stmts[i])});
        i += 3;
        continue;
      }
      compute_stmts.push_back(flat_stmts[i]);
      i++;
    }
  }

private:
  static const CallNode *GetEvaluateCallInSimpleWrapper(const Stmt &stmt) {
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      return eval->value.as<CallNode>();
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        return GetEvaluateCallInSimpleWrapper(if_stmt->then_case);
      }
      return nullptr;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return GetEvaluateCallInSimpleWrapper(attr->body);
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return GetEvaluateCallInSimpleWrapper(let->body);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return GetEvaluateCallInSimpleWrapper(seq->seq[0]);
      }
      return nullptr;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return GetEvaluateCallInSimpleWrapper(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return GetEvaluateCallInSimpleWrapper(realize->block->body);
      }
      return nullptr;
    }
    return nullptr;
  }

  static bool IsMbarrierWaitParity(const Stmt &stmt) {
    const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
    return call && call->op.same_as(mbarrier_wait_parity());
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

  static bool ContainsPtxCpAsync(const Stmt &stmt) {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (found) {
        return;
      }
      if (const auto *call = node.as<CallNode>()) {
        if (call->op.same_as(builtin::ptx_cp_async()) ||
            call->op.same_as(tl::ptx_cp_async())) {
          found = true;
        }
      }
    });
    return found;
  }

  static bool IsPtxCommitGroup(const Stmt &stmt) {
    const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
    return call && call->op.same_as(builtin::ptx_commit_group());
  }

  static bool IsPtxWaitGroupZero(const Stmt &stmt) {
    const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
    if (!call || !call->op.same_as(builtin::ptx_wait_group()) ||
        call->args.size() != 1) {
      return false;
    }
    if (const auto *imm = call->args[0].as<IntImmNode>()) {
      return imm->value == 0;
    }
    return false;
  }

  static Optional<Var> AccessPtrBufferVar(const PrimExpr &ptr) {
    const auto *call = ptr.as<CallNode>();
    if (!call) {
      return Optional<Var>();
    }
    if (call->op.same_as(tl::access_ptr())) {
      if (call->args.size() != 3) {
        return Optional<Var>();
      }
      const auto *base_load = call->args[0].as<BufferLoadNode>();
      if (!base_load) {
        return Optional<Var>();
      }
      return base_load->buffer->data;
    }
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      if (call->args.size() != 5) {
        return Optional<Var>();
      }
      if (call->args[1].as<VarNode>()) {
        return Downcast<Var>(call->args[1]);
      }
    }
    return Optional<Var>();
  }

  static Optional<Var> GetCpAsyncDstBufferData(const Stmt &stmt) {
    Optional<Var> found = std::nullopt;
    bool multiple = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (multiple) {
        return;
      }
      const auto *call = node.as<CallNode>();
      if (!call) {
        return;
      }
      if (!(call->op.same_as(builtin::ptx_cp_async()) ||
            call->op.same_as(tl::ptx_cp_async())) ||
          call->args.empty()) {
        return;
      }
      Optional<Var> current = AccessPtrBufferVar(call->args[0]);
      if (!current.defined()) {
        return;
      }
      if (!found.defined()) {
        found = current;
      } else if (found.value().get() != current.value().get()) {
        multiple = true;
      }
    });
    if (multiple) {
      return Optional<Var>();
    }
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
        op->op.same_as(tl::tma_store()) ||
        op->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
        op->op.same_as(mbarrier_expect_tx())) {
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

    // Extract async producer blocks (TMA and cp.async)
    AsyncCopyBlockExtractor extractor;
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
    producer_thread_extent_ = producer_thread_extent;

    // Barrier layout has two modes:
    // 1) Mixed TMA + cp.async:
    //    keep existing TMA forward ids, append cp.async forward ids, then
    //    append back-pressure ids.
    // 2) Pure TMA:
    //    remap to [loop forward][back-pressure][preloop forward] so producer
    //    and consumer follow the same protocol as the hand-written WS kernels.
    int num_existing_tma_fwd_barriers = 0;
    int num_cp_async_groups = 0;
    for (const auto &block : extractor.blocks) {
      if (block.kind == AsyncProducerKind::kTma) {
        ++num_existing_tma_fwd_barriers;
      } else if (block.kind == AsyncProducerKind::kCpAsync) {
        ++num_cp_async_groups;
      }
    }
    std::vector<int> wait_insert_pos(extractor.blocks.size(), 0);
    std::vector<int> arrive_insert_pos(
        extractor.blocks.size(),
        static_cast<int>(extractor.compute_stmts.size()));
    for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
      if (!extractor.blocks[ti].write_buffer_data.defined()) {
        continue;
      }
      auto *target = extractor.blocks[ti].write_buffer_data.value().get();
      auto uses_target = [target](const VarNode *v) { return v == target; };
      int first_use = -1;
      int last_use = -1;
      for (size_t ci = 0; ci < extractor.compute_stmts.size(); ++ci) {
        if (UsesVar(extractor.compute_stmts[ci], uses_target)) {
          if (first_use < 0) {
            first_use = static_cast<int>(ci);
          }
          last_use = static_cast<int>(ci);
        }
      }
      if (first_use >= 0) {
        wait_insert_pos[ti] = first_use;
        arrive_insert_pos[ti] = last_use + 1;
      }
    }
    int num_existing_loop_fwd_barriers =
        num_existing_tma_fwd_barriers * num_stages;
    int original_num_existing_loop_fwd_barriers =
        num_existing_loop_fwd_barriers;
    int num_new_cp_async_fwd_barriers = num_cp_async_groups * num_stages;
    int inferred_existing_required =
        InferMinRequiredBarrierCount(orig_block->body);
    bool old_use_full_tma_forward_barrier_protocol =
        use_full_tma_forward_barrier_protocol_;
    bool old_remap_pure_tma_barriers = remap_pure_tma_barriers_;
    int old_pure_tma_preloop_fwd_base = pure_tma_preloop_fwd_base_;
    int old_pure_tma_preloop_fwd_count = pure_tma_preloop_fwd_count_;
    int old_pure_tma_preloop_fwd_cursor = pure_tma_preloop_fwd_cursor_;
    use_full_tma_forward_barrier_protocol_ = (num_cp_async_groups == 0);
    remap_pure_tma_barriers_ = use_full_tma_forward_barrier_protocol_;
    std::vector<Optional<PrimExpr>> producer_guards(extractor.blocks.size(),
                                                    std::nullopt);
    for (size_t i = 0; i < extractor.blocks.size(); ++i) {
      producer_guards[i] =
          ExtractNonThreadProducerGuard(extractor.blocks[i].producer_stmt);
    }

    std::vector<int> block_group(extractor.blocks.size(), 0);
    int num_block_groups = static_cast<int>(extractor.blocks.size());
    if (remap_pure_tma_barriers_ && !extractor.blocks.empty()) {
      StructuralEqual equal;
      auto same_guard = [&](size_t lhs, size_t rhs) {
        const Optional<PrimExpr>& guard_a = producer_guards[lhs];
        const Optional<PrimExpr>& guard_b = producer_guards[rhs];
        if (guard_a.defined() != guard_b.defined()) {
          return false;
        }
        return !guard_a.defined() || equal(guard_a.value(), guard_b.value());
      };
      int next_group = 0;
      block_group[0] = next_group++;
      for (size_t i = 1; i < extractor.blocks.size(); ++i) {
        bool merge_with_prev =
            extractor.blocks[i].kind == AsyncProducerKind::kTma &&
            extractor.blocks[i - 1].kind == AsyncProducerKind::kTma &&
            wait_insert_pos[i] == wait_insert_pos[i - 1] &&
            arrive_insert_pos[i] == arrive_insert_pos[i - 1] &&
            same_guard(i - 1, i);
        block_group[i] = merge_with_prev ? block_group[i - 1] : next_group++;
      }
      num_block_groups = next_group;
      num_existing_tma_fwd_barriers = num_block_groups;
      num_existing_loop_fwd_barriers =
          num_existing_tma_fwd_barriers * num_stages;
    }

    int num_existing_barriers = 0;
    int num_preloop_fwd_barriers = 0;
    if (remap_pure_tma_barriers_) {
      num_preloop_fwd_barriers =
          std::max(1, inferred_existing_required -
                          original_num_existing_loop_fwd_barriers);
      num_existing_barriers =
          num_existing_loop_fwd_barriers + num_preloop_fwd_barriers;
    } else {
      // Reserve one extra slot for non-pipelined forward barriers such as a
      // prologue TMA load that lives outside the main pipeline loop.
      int min_reserved_non_pipeline = num_existing_loop_fwd_barriers + 1;
      num_existing_barriers =
          std::max(min_reserved_non_pipeline, inferred_existing_required);
      num_preloop_fwd_barriers =
          num_existing_barriers - num_existing_loop_fwd_barriers;
    }
    int num_total_fwd_barriers = 0;
    int num_bp_barriers =
        (remap_pure_tma_barriers_ ? num_block_groups
                                  : static_cast<int>(extractor.blocks.size())) *
        num_stages;
    int total_barriers = 0;

    std::vector<int> tma_fwd_bases(extractor.blocks.size(), -1);
    std::vector<int> cp_async_fwd_bases(extractor.blocks.size(), -1);

    std::vector<int> bp_bases;
    bp_bases.reserve(extractor.blocks.size());

    if (remap_pure_tma_barriers_) {
      // Pure-TMA layout:
      //   [0, loop_fwd)                    : loop forward barriers
      //   [loop_fwd, loop_fwd + bp)       : back-pressure barriers
      //   [loop_fwd + bp, total_barriers) : preloop/prologue forward barriers
      int next_loop_fwd_base = 0;
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        if (extractor.blocks[i].kind == AsyncProducerKind::kTma) {
          if (i == 0 || block_group[i] != block_group[i - 1]) {
            tma_fwd_bases[i] = next_loop_fwd_base;
            next_loop_fwd_base += num_stages;
          } else {
            tma_fwd_bases[i] = tma_fwd_bases[i - 1];
          }
        }
      }
      num_total_fwd_barriers =
          num_existing_loop_fwd_barriers + num_preloop_fwd_barriers;
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        bp_bases.push_back(num_existing_loop_fwd_barriers +
                           block_group[i] * num_stages);
      }
      pure_tma_preloop_fwd_base_ =
          num_existing_loop_fwd_barriers + num_bp_barriers;
      pure_tma_preloop_fwd_count_ = num_preloop_fwd_barriers;
      pure_tma_preloop_fwd_cursor_ = 0;
      total_barriers = num_total_fwd_barriers + num_bp_barriers;
    } else {
      // Mixed path:
      //   [0, num_existing_barriers) : pre-existing forward barriers
      //   [existing, total_fwd)      : new cp.async forward barriers
      //   [total_fwd, total)         : back-pressure barriers
      num_total_fwd_barriers =
          num_existing_barriers + num_new_cp_async_fwd_barriers;
      int next_cp_async_fwd_base = num_existing_barriers;
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        if (extractor.blocks[i].kind == AsyncProducerKind::kCpAsync) {
          cp_async_fwd_bases[i] = next_cp_async_fwd_base;
          next_cp_async_fwd_base += num_stages;
        }
      }
      for (size_t i = 0; i < extractor.blocks.size(); i++) {
        bp_bases.push_back(num_total_fwd_barriers +
                           static_cast<int>(i) * num_stages);
      }
      total_barriers = num_total_fwd_barriers + num_bp_barriers;
      pure_tma_preloop_fwd_base_ = -1;
      pure_tma_preloop_fwd_count_ = 0;
      pure_tma_preloop_fwd_cursor_ = 0;
    }

    // Defensive check: ensure back-pressure barriers do not overlap
    // any existing (forward/prologue) barrier ids in the original IR.
    if (num_bp_barriers > 0 && !remap_pure_tma_barriers_) {
      int existing_last = inferred_existing_required - 1;
      int bp_begin = bp_bases.front();
      int bp_last = bp_begin + num_bp_barriers - 1;
      ICHECK(bp_begin > existing_last)
          << "ProducerConsumerWS: barrier id overlap detected. "
          << "existing_last=" << existing_last << ", bp_begin=" << bp_begin
          << ", bp_last=" << bp_last;
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
      bool is_first_in_group = ti == 0 || !remap_pure_tma_barriers_ ||
                               block_group[ti] != block_group[ti - 1];
      bool is_last_in_group = ti + 1 == extractor.blocks.size() ||
                              !remap_pure_tma_barriers_ ||
                              block_group[ti] != block_group[ti + 1];
      PrimExpr bp_id = IntImm(DataType::Int(32), bp_bases[ti]) + stage_expr;

      // Back-pressure wait: producer cannot reuse the stage buffer until the
      // consumer releases it. xor(parity, 1) bootstraps the first iteration.
      if (is_first_in_group) {
        producer_body_stmts.push_back(
            makeParityWait(bp_id, bitwise_xor(parity_expr, 1)));
      }

      Stmt producer_stmt = tma.producer_stmt;
      if (use_full_tma_forward_barrier_protocol_ &&
          tma.kind == AsyncProducerKind::kTma) {
        // Pure-TMA WS uses a full producer-side release protocol so the
        // consumer waits on a barrier owned by the producer branch.
        ICHECK_GE(tma_fwd_bases[ti], 0);
        PrimExpr barrier_id =
            IntImm(DataType::Int(32), tma_fwd_bases[ti]) + stage_expr;
        producer_stmt = RewriteTmaForwardProducerStmt(producer_stmt, barrier_id,
                                                      is_last_in_group);
      }
      if (tma.kind == AsyncProducerKind::kTma) {
        // Keep expect/load under the same elected lane when lowering has
        // emitted them as adjacent identical IfThenElse wrappers.
        producer_stmt = MergeAdjacentEquivalentIfs(producer_stmt);
      }

      // Execute the producer statement.
      producer_body_stmts.push_back(producer_stmt);
      if (tma.kind == AsyncProducerKind::kCpAsync) {
        ICHECK(cp_async_fwd_bases[ti] >= 0);
        PrimExpr fwd_id =
            IntImm(DataType::Int(32), cp_async_fwd_bases[ti]) + stage_expr;
        producer_body_stmts.push_back(makeCpAsyncBarrierNoInc(fwd_id));
      }
    }
    Stmt producer_loop_body =
        MergeAdjacentEquivalentIfs(SeqStmt(producer_body_stmts));

    // --- Build Consumer Body ---
    Array<Stmt> consumer_body_stmts;

    // Place forward waits at first use and back-pressure arrives at last use.
    // If we cannot prove the dependency, fall back to wait-at-head /
    // arrive-at-tail.
    std::vector<bool> arrive_emitted(extractor.blocks.size(), false);
    std::vector<Stmt> normalized_waits;
    normalized_waits.reserve(extractor.blocks.size());
    for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
      if (extractor.blocks[ti].kind == AsyncProducerKind::kTma &&
          remap_pure_tma_barriers_) {
        ICHECK_GE(tma_fwd_bases[ti], 0);
        PrimExpr fwd_id =
            IntImm(DataType::Int(32), tma_fwd_bases[ti]) + stage_expr;
        normalized_waits.push_back(makeParityWait(fwd_id, parity_expr));
      } else if (extractor.blocks[ti].kind == AsyncProducerKind::kTma) {
        ICHECK(extractor.blocks[ti].wait_stmt.defined());
        normalized_waits.push_back(NormalizeForwardWaitParity(
            extractor.blocks[ti].wait_stmt.value(), parity_expr));
      } else {
        ICHECK(cp_async_fwd_bases[ti] >= 0);
        PrimExpr fwd_id =
            IntImm(DataType::Int(32), cp_async_fwd_bases[ti]) + stage_expr;
        normalized_waits.push_back(makeParityWait(fwd_id, parity_expr));
      }
    }
    // Emit waits / compute / arrives according to insertion points.
    for (size_t ci = 0; ci < extractor.compute_stmts.size(); ++ci) {
      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        bool is_first_in_group = ti == 0 || !remap_pure_tma_barriers_ ||
                                 block_group[ti] != block_group[ti - 1];
        if (is_first_in_group && wait_insert_pos[ti] == static_cast<int>(ci)) {
          consumer_body_stmts.push_back(normalized_waits[ti]);
        }
      }
      consumer_body_stmts.push_back(extractor.compute_stmts[ci]);
      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        bool is_last_in_group = ti + 1 == extractor.blocks.size() ||
                                !remap_pure_tma_barriers_ ||
                                block_group[ti] != block_group[ti + 1];
        if (is_last_in_group &&
            arrive_insert_pos[ti] == static_cast<int>(ci + 1)) {
          PrimExpr bp_id = IntImm(DataType::Int(32), bp_bases[ti]) + stage_expr;
          consumer_body_stmts.push_back(makeArriveBarrier(bp_id));
          arrive_emitted[ti] = true;
        }
      }
    }

    // Handle degenerate loops with no compute statements.
    if (extractor.compute_stmts.empty()) {
      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        bool is_first_in_group = ti == 0 || !remap_pure_tma_barriers_ ||
                                 block_group[ti] != block_group[ti - 1];
        if (is_first_in_group) {
          consumer_body_stmts.push_back(normalized_waits[ti]);
        }
      }
    }

    // Emit loop-tail arrives (blocks with unknown deps or tail use).
    for (size_t ti = 0; ti < extractor.blocks.size(); ti++) {
      bool is_last_in_group = ti + 1 == extractor.blocks.size() ||
                              !remap_pure_tma_barriers_ ||
                              block_group[ti] != block_group[ti + 1];
      if (is_last_in_group && !arrive_emitted[ti] &&
          arrive_insert_pos[ti] ==
              static_cast<int>(extractor.compute_stmts.size())) {
        PrimExpr bp_id = IntImm(DataType::Int(32), bp_bases[ti]) + stage_expr;
        consumer_body_stmts.push_back(makeArriveBarrier(bp_id));
      }
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

    // Forward barriers are producer-owned; back-pressure barriers are released
    // by the full consumer partition.
    Array<PrimExpr> barrier_arrive_counts;
    barrier_arrive_counts.reserve(total_barriers);
    if (remap_pure_tma_barriers_) {
      for (int i = 0; i < num_existing_loop_fwd_barriers; ++i) {
        barrier_arrive_counts.push_back(IntImm(DataType::Int(32), 1));
      }
      for (int i = 0; i < num_bp_barriers; ++i) {
        barrier_arrive_counts.push_back(consumer_thread_extent);
      }
      for (int i = 0; i < num_preloop_fwd_barriers; ++i) {
        barrier_arrive_counts.push_back(IntImm(DataType::Int(32), 1));
      }
    } else {
      for (int i = 0; i < num_existing_barriers; i++) {
        barrier_arrive_counts.push_back(use_full_tma_forward_barrier_protocol_
                                            ? IntImm(DataType::Int(32), 1)
                                            : IntImm(DataType::Int(32), 1));
      }
      for (int i = 0; i < num_new_cp_async_fwd_barriers; i++) {
        barrier_arrive_counts.push_back(producer_thread_extent);
      }
      for (int i = 0; i < num_bp_barriers; i++) {
        barrier_arrive_counts.push_back(consumer_thread_extent);
      }
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
    use_full_tma_forward_barrier_protocol_ =
        old_use_full_tma_forward_barrier_protocol;
    remap_pure_tma_barriers_ = old_remap_pure_tma_barriers;
    pure_tma_preloop_fwd_base_ = old_pure_tma_preloop_fwd_base;
    pure_tma_preloop_fwd_count_ = old_pure_tma_preloop_fwd_count;
    pure_tma_preloop_fwd_cursor_ = old_pure_tma_preloop_fwd_cursor;

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
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        op->thread_binding.value()->thread_tag == "threadIdx.x" &&
        !thread_iv_.defined()) {
      thread_iv_ = op->thread_binding.value();
      need_update_thread_extent_ = false;
      For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
      if (need_update_thread_extent_) {
        auto n = for_node.CopyOnWrite();
        n->extent = updated_thread_extent_.value();
        IterVar new_thread_iv = n->thread_binding.value();
        new_thread_iv.CopyOnWrite()->dom =
            Range::FromMinExtent(Integer(0), updated_thread_extent_.value());
        n->thread_binding = new_thread_iv;
      }
      thread_iv_ = {};
      return for_node;
    }

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

  // Infer how many mbarriers are already referenced by this block body.
  // This prevents assigning back-pressure barriers that alias existing
  // forward barriers (e.g. prologue TMA copy barriers outside the pipeline).
  int InferMinRequiredBarrierCount(const Stmt &stmt) {
    class GetMbarrierMaxIdxCollector : public StmtExprVisitor {
    public:
      int max_idx{-1};
      bool has_unbounded{false};

    private:
      void VisitStmt_(const ForNode *op) final {
        // Bind loop variable range so expressions like (k + c) can be bounded.
        analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
        StmtExprVisitor::VisitStmt_(op);
      }

      void VisitExpr_(const CallNode *op) final {
        if (op->op.same_as(get_mbarrier()) && op->args.size() == 1) {
          auto bound = analyzer_.const_int_bound(op->args[0]);
          if (bound->max_value != arith::ConstIntBound::kPosInf &&
              bound->max_value != arith::ConstIntBound::kNegInf) {
            max_idx = std::max(max_idx, static_cast<int>(bound->max_value));
          } else {
            has_unbounded = true;
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      arith::Analyzer analyzer_;
    };

    GetMbarrierMaxIdxCollector collector;
    collector(stmt);
    ICHECK(!collector.has_unbounded)
        << "ProducerConsumerWS: cannot infer finite upper bound for existing "
        << "mbarrier id expressions. Refusing to allocate back-pressure "
        << "barriers to avoid id overlap.";
    return collector.max_idx + 1;
  }

  // Single source of truth for barrier/TMA control-like calls that should not
  // be moved across producer/consumer partition boundaries.
  bool IsBarrierOrTmaControlCall(const CallNode *call) {
    return call->op.same_as(create_list_of_mbarrier()) ||
           call->op.same_as(mbarrier_wait_parity()) ||
           call->op.same_as(builtin::ptx_arrive_barrier()) ||
           call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
           call->op.same_as(builtin::ptx_cp_async_barrier()) ||
           call->op.same_as(tl::ptx_cp_async_barrier_noinc()) ||
           call->op.same_as(tma_load()) ||
           call->op.same_as(tma_load_im2col()) ||
           call->op.same_as(tma_store()) ||
           call->op.same_as(tma_store_arrive()) ||
           call->op.same_as(tma_store_wait()) ||
           call->op.same_as(builtin::tvm_storage_sync());
  }

  bool IsMovableConsumerPrefixStmt(const Stmt &stmt) {
    bool has_disallowed = false;
    auto is_shared_scope = [](const String &scope) {
      std::string s = scope;
      return s.rfind("shared", 0) == 0;
    };
    auto is_global_scope = [](const String &scope) {
      return std::string(scope) == "global";
    };
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (has_disallowed) {
        return;
      }
      if (auto *call = node.as<CallNode>()) {
        if (IsBarrierOrTmaControlCall(call)) {
          has_disallowed = true;
          return;
        }
      }
      if (auto *ld = node.as<BufferLoadNode>()) {
        // Only move pure local init into the consumer prefix. If a stmt reads
        // global or shared memory, the producer may also depend on its result
        // (for example a mask controlling which async copies to issue).
        if (is_shared_scope(ld->buffer.scope()) ||
            is_global_scope(ld->buffer.scope())) {
          has_disallowed = true;
          return;
        }
      }
      if (auto *st = node.as<BufferStoreNode>()) {
        if (is_shared_scope(st->buffer.scope()) ||
            is_global_scope(st->buffer.scope())) {
          has_disallowed = true;
          return;
        }
      }
    });
    return !has_disallowed;
  }

  Optional<Stmt> TryPrependToConsumerBranch(const Stmt &stmt,
                                            const Stmt &prepend_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryPrependToConsumerBranch(new_seq.back(), prepend_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryPrependToConsumerBranch(attr->body, prepend_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryPrependToConsumerBranch(let_s->body, prepend_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryPrependToConsumerBranch(realize->block->body, prepend_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryPrependToConsumerBranch(block->body, prepend_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined()) {
        return std::nullopt;
      }
      Stmt new_else = SeqStmt({prepend_stmt, if_stmt->else_case.value()});
      return IfThenElse(if_stmt->condition, if_stmt->then_case, new_else);
    }
    return std::nullopt;
  }

  Optional<Stmt> TryPrependToProducerBranch(const Stmt &stmt,
                                            const Stmt &prepend_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryPrependToProducerBranch(new_seq.back(), prepend_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryPrependToProducerBranch(attr->body, prepend_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryPrependToProducerBranch(let_s->body, prepend_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryPrependToProducerBranch(realize->block->body, prepend_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryPrependToProducerBranch(block->body, prepend_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      Stmt new_then = SeqStmt({prepend_stmt, if_stmt->then_case});
      return IfThenElse(if_stmt->condition, new_then, if_stmt->else_case);
    }
    return std::nullopt;
  }

  Optional<Stmt> TryAppendToProducerBranch(const Stmt &stmt,
                                           const Stmt &append_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryAppendToProducerBranch(new_seq.back(), append_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryAppendToProducerBranch(attr->body, append_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryAppendToProducerBranch(let_s->body, append_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryAppendToProducerBranch(realize->block->body, append_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryAppendToProducerBranch(block->body, append_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      auto nested = TryAppendToProducerBranch(if_stmt->then_case, append_stmt);
      if (nested.defined()) {
        return IfThenElse(if_stmt->condition, nested.value(),
                          if_stmt->else_case);
      }
      Stmt new_then = SeqStmt({if_stmt->then_case, append_stmt});
      return IfThenElse(if_stmt->condition, new_then, if_stmt->else_case);
    }
    return std::nullopt;
  }

  Optional<Stmt> TryAppendToConsumerBranch(const Stmt &stmt,
                                           const Stmt &append_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryAppendToConsumerBranch(new_seq.back(), append_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryAppendToConsumerBranch(attr->body, append_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryAppendToConsumerBranch(let_s->body, append_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryAppendToConsumerBranch(realize->block->body, append_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryAppendToConsumerBranch(block->body, append_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined()) {
        return std::nullopt;
      }
      Stmt new_else = SeqStmt({if_stmt->else_case.value(), append_stmt});
      return IfThenElse(if_stmt->condition, if_stmt->then_case, new_else);
    }
    return std::nullopt;
  }

  bool IsMbarrierWaitParityStmt(const Stmt &stmt) {
    return ExtractWaitBarrierId(stmt).defined();
  }

  Optional<PrimExpr> ExtractWaitBarrierId(const Stmt &stmt) {
    auto extract_from_call = [](const CallNode *call) -> Optional<PrimExpr> {
      if (!call || !call->op.same_as(mbarrier_wait_parity()) ||
          call->args.size() != 2) {
        return std::nullopt;
      }
      if (auto *get = call->args[0].as<CallNode>()) {
        if (get->op.same_as(get_mbarrier()) && get->args.size() == 1) {
          return get->args[0];
        }
      }
      return std::nullopt;
    };

    if (auto *eval = stmt.as<EvaluateNode>()) {
      return extract_from_call(eval->value.as<CallNode>());
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        return ExtractWaitBarrierId(if_stmt->then_case);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return ExtractWaitBarrierId(attr->body);
    }
    if (auto *let_stmt = stmt.as<LetStmtNode>()) {
      return ExtractWaitBarrierId(let_stmt->body);
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return ExtractWaitBarrierId(seq->seq[0]);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return ExtractWaitBarrierId(block->body);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return ExtractWaitBarrierId(realize->block->body);
      }
    }
    return std::nullopt;
  }

  Stmt NormalizeForwardWaitParity(const Stmt &wait_stmt,
                                  const PrimExpr &normalized_parity) {
    auto barrier_id = ExtractWaitBarrierId(wait_stmt);
    if (!barrier_id.defined()) {
      return wait_stmt;
    }
    return makeParityWait(barrier_id.value(), normalized_parity);
  }

  bool ContainsTmaLoadStmt(const Stmt &stmt) {
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

  bool IsThreadOnlyPredicate(const PrimExpr &expr) const {
    bool uses_thread = false;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        if (thread_iv_.defined() && var == thread_iv_->var.get()) {
          uses_thread = true;
        }
      }
    });
    return uses_thread;
  }

  Optional<PrimExpr> ExtractNonThreadProducerGuard(const Stmt &stmt) const {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return ExtractNonThreadProducerGuard(attr->body);
    }
    if (const auto *let_s = stmt.as<LetStmtNode>()) {
      return ExtractNonThreadProducerGuard(let_s->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      return ExtractNonThreadProducerGuard(realize->block->body);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return ExtractNonThreadProducerGuard(block->body);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        auto guard = ExtractNonThreadProducerGuard(s);
        if (guard.defined()) {
          return guard;
        }
      }
      return std::nullopt;
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        if (!IsThreadOnlyPredicate(if_stmt->condition)) {
          return if_stmt->condition;
        }
        return ExtractNonThreadProducerGuard(if_stmt->then_case);
      }
    }
    return std::nullopt;
  }

  Stmt RewriteWaitBarrierId(const Stmt &wait_stmt,
                            const PrimExpr &new_barrier_id) {
    if (const auto *eval = wait_stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>()) {
        if (call->op.same_as(mbarrier_wait_parity()) &&
            call->args.size() == 2) {
          return makeParityWait(new_barrier_id, call->args[1]);
        }
      }
    }
    return wait_stmt;
  }

  Stmt MergeAdjacentEquivalentIfs(const Stmt &stmt) {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value,
                      MergeAdjacentEquivalentIfs(attr->body), attr->span);
    }
    if (const auto *let_stmt = stmt.as<LetStmtNode>()) {
      return LetStmt(let_stmt->var, let_stmt->value,
                     MergeAdjacentEquivalentIfs(let_stmt->body));
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return Block(block->iter_vars, block->reads, block->writes,
                   block->name_hint, MergeAdjacentEquivalentIfs(block->body),
                   block->init, block->alloc_buffers, block->match_buffers,
                   block->annotations);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      const Block &orig = realize->block;
      Block new_block(orig->iter_vars, orig->reads, orig->writes,
                      orig->name_hint, MergeAdjacentEquivalentIfs(orig->body),
                      orig->init, orig->alloc_buffers, orig->match_buffers,
                      orig->annotations);
      return BlockRealize(realize->iter_values, realize->predicate, new_block);
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      Optional<Stmt> else_case = std::nullopt;
      if (if_stmt->else_case.defined()) {
        else_case = MergeAdjacentEquivalentIfs(if_stmt->else_case.value());
      }
      return IfThenElse(if_stmt->condition,
                        MergeAdjacentEquivalentIfs(if_stmt->then_case),
                        else_case, if_stmt->span);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> merged;
      StructuralEqual equal;
      for (size_t i = 0; i < seq->seq.size();) {
        const auto *if0 = seq->seq[i].as<IfThenElseNode>();
        if (if0 && !if0->else_case.defined()) {
          Array<Stmt> then_stmts;
          then_stmts.push_back(if0->then_case);
          size_t j = i + 1;
          while (j < seq->seq.size()) {
            const auto *ifj = seq->seq[j].as<IfThenElseNode>();
            if (!ifj || ifj->else_case.defined() ||
                !equal(if0->condition, ifj->condition)) {
              break;
            }
            then_stmts.push_back(ifj->then_case);
            ++j;
          }
          if (then_stmts.size() == 1) {
            merged.push_back(seq->seq[i]);
          } else {
            Stmt merged_then = MergeAdjacentEquivalentIfs(
                then_stmts.size() == 1 ? then_stmts[0] : SeqStmt(then_stmts));
            merged.push_back(IfThenElse(if0->condition, merged_then,
                                        std::nullopt, if0->span));
          }
          i = j;
          continue;
        }
        merged.push_back(seq->seq[i]);
        ++i;
      }
      return merged.size() == 1 ? merged[0] : SeqStmt(merged, seq->span);
    }
    return stmt;
  }

  Stmt RewriteTmaForwardProducerStmt(const Stmt &stmt,
                                     const PrimExpr &barrier_id,
                                     bool append_arrive) {
    class TmaForwardBarrierStmtRewriter : public StmtExprMutator {
    public:
      explicit TmaForwardBarrierStmtRewriter(PrimExpr barrier_id)
          : barrier_id_(std::move(barrier_id)) {}

      PrimExpr VisitExpr_(const CallNode *op) final {
        auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
        if ((call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
             call->op.same_as(mbarrier_expect_tx())) &&
            call->args.size() == 2) {
          return Call(call->dtype, mbarrier_expect_tx(),
                      {makeGetBarrier(barrier_id_), call->args[1]},
                      call->annotations, call->span);
        }
        if (call->op.same_as(tma_load()) ||
            call->op.same_as(tma_load_im2col())) {
          bool is_1d_tma_load = false;
          if (const auto *arg0 = call->args[0].as<CallNode>()) {
            is_1d_tma_load = !arg0->op.same_as(create_tma_descriptor()) &&
                             call->op.same_as(tma_load());
          }
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(is_1d_tma_load ? 2 : 1,
                             makeGetBarrier(barrier_id_));
          return call;
        }
        return call;
      }

    private:
      PrimExpr barrier_id_;
    };

    // Rebind the producer-side barrier id and finish the stage with a normal
    // barrier arrival. Pure-TMA pipelines do not need cp.async.mbarrier.arrive.
    Stmt rewritten = MergeAdjacentEquivalentIfs(
        TmaForwardBarrierStmtRewriter(barrier_id)(stmt));
    if (!append_arrive) {
      return rewritten;
    }
    Stmt elect_arrive = IfThenElse(
        Call(DataType::Bool(), tl_shuffle_elect(), {producer_thread_extent_}),
        makeArriveBarrier(barrier_id), std::nullopt);
    return SeqStmt({rewritten, elect_arrive});
  }

  bool IsSharedDependentConsumerPreStmt(const Stmt &stmt) {
    bool has_shared_access = false;
    bool has_control_ops = false;
    auto is_shared_scope = [](const String &scope) {
      std::string s = scope;
      return s.rfind("shared", 0) == 0;
    };
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (has_control_ops) {
        return;
      }
      if (auto *call = node.as<CallNode>()) {
        if (IsBarrierOrTmaControlCall(call)) {
          has_control_ops = true;
          return;
        }
      }
      if (auto *ld = node.as<BufferLoadNode>()) {
        if (is_shared_scope(ld->buffer.scope())) {
          has_shared_access = true;
        }
      }
      if (auto *st = node.as<BufferStoreNode>()) {
        if (is_shared_scope(st->buffer.scope())) {
          has_shared_access = true;
        }
      }
    });
    return has_shared_access && !has_control_ops;
  }

  bool IsDuplicableBranchLocalPreStmt(const Stmt &stmt) {
    bool has_disallowed = false;
    auto is_shared_scope = [](const String &scope) {
      std::string s = scope;
      return s.rfind("shared", 0) == 0;
    };
    auto is_global_scope = [](const String &scope) {
      return std::string(scope) == "global";
    };
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (has_disallowed) {
        return;
      }
      if (const auto *call = node.as<CallNode>()) {
        if (IsBarrierOrTmaControlCall(call)) {
          has_disallowed = true;
          return;
        }
      }
      if (const auto *ld = node.as<BufferLoadNode>()) {
        if (is_shared_scope(ld->buffer.scope())) {
          has_disallowed = true;
          return;
        }
      }
      if (const auto *st = node.as<BufferStoreNode>()) {
        if (is_shared_scope(st->buffer.scope()) ||
            is_global_scope(st->buffer.scope())) {
          has_disallowed = true;
          return;
        }
      }
    });
    return !has_disallowed;
  }

  /*!
   * \brief Rebuild the block body, replacing the pipeline loop with
   *        init_barrier + ws_body and removing old create_list_of_mbarrier.
   *
   *  Statements after the pipeline loop (e.g. epilogue, store) should execute
   *  only on consumer threads. Prefer appending them into the consumer branch
   *  of the warp-specialized if/else to keep a single top-level partition.
   *  If that is not possible, fall back to an explicit consumer-thread guard.
   */
  Stmt RebuildBlockBody(const Stmt &body, const ForNode *target_loop,
                        const Stmt &init_barrier, const Stmt &ws_body) {
    // If this IS the target loop, replace it
    if (body.as<ForNode>() == target_loop) {
      return SeqStmt({init_barrier, ws_body});
    }

    if (auto *seq = body.as<SeqStmtNode>()) {
      Array<Stmt> new_seq;
      Array<Stmt> pre_loop_stmts;
      Array<Stmt> post_loop_stmts;
      bool found_loop = false;
      Optional<Stmt> rebuilt_loop = std::nullopt;

      for (const auto &s : seq->seq) {
        // Remove existing create_list_of_mbarrier
        if (IsCreateListOfMbarrier(s))
          continue;

        if (!found_loop && ContainsLoop(s, target_loop)) {
          // Replace the pipeline loop
          rebuilt_loop =
              RebuildBlockBody(s, target_loop, init_barrier, ws_body);
          found_loop = true;
        } else if (found_loop) {
          // Collect statements after the pipeline loop
          post_loop_stmts.push_back(s);
        } else {
          // Statements before the pipeline loop.
          pre_loop_stmts.push_back(s);
        }
      }

      // Move a movable suffix of pre-loop statements into consumer branch
      // (e.g. fragment initialization), keeping barriers/syncs outside.
      size_t movable_begin = pre_loop_stmts.size();
      while (movable_begin > 0 &&
             IsMovableConsumerPrefixStmt(pre_loop_stmts[movable_begin - 1])) {
        --movable_begin;
      }

      // Split non-movable pre-loop statements into:
      //   common statements kept outside the WS split
      //   producer-side async issues
      //   consumer-side waits / shared-dependent setup
      //
      // In pure-TMA mode we also duplicate branch-local scalar prefix code
      // (for example loop-bound computation) into both branches. Leaving those
      // statements outside the split can make producer and consumer observe
      // inconsistent values after register repartitioning.
      Array<Stmt> common_pre_stmts;
      Array<Stmt> producer_prefix_stmts;
      Array<Stmt> consumer_wait_prefix_stmts;
      Array<Stmt> consumer_shared_prefix_stmts;
      Array<Stmt> duplicated_prefix_stmts;
      size_t i = 0;
      while (i < movable_begin) {
        if (i + 1 < movable_begin && ContainsTmaLoadStmt(pre_loop_stmts[i]) &&
            IsMbarrierWaitParityStmt(pre_loop_stmts[i + 1])) {
          Stmt producer_prefix_stmt =
              StripTmaCopyWriteBufferAttr(pre_loop_stmts[i]);
          Stmt consumer_wait_stmt = pre_loop_stmts[i + 1];
          if (remap_pure_tma_barriers_) {
            ICHECK_GE(pure_tma_preloop_fwd_base_, 0);
            ICHECK_LT(pure_tma_preloop_fwd_cursor_,
                      pure_tma_preloop_fwd_count_);
            PrimExpr barrier_id =
                IntImm(DataType::Int(32), pure_tma_preloop_fwd_base_ +
                                              pure_tma_preloop_fwd_cursor_++);
            producer_prefix_stmt =
                RewriteTmaForwardProducerStmt(producer_prefix_stmt, barrier_id,
                                              /*append_arrive=*/true);
            consumer_wait_stmt =
                RewriteWaitBarrierId(consumer_wait_stmt, barrier_id);
          } else if (use_full_tma_forward_barrier_protocol_) {
            auto barrier_id = ExtractWaitBarrierId(pre_loop_stmts[i + 1]);
            ICHECK(barrier_id.defined())
                << "ProducerConsumerWS: failed to extract pre-loop TMA "
                   "forward barrier id";
            producer_prefix_stmt = RewriteTmaForwardProducerStmt(
                producer_prefix_stmt, barrier_id.value(),
                /*append_arrive=*/true);
          }
          producer_prefix_stmt =
              MergeAdjacentEquivalentIfs(producer_prefix_stmt);
          producer_prefix_stmts.push_back(producer_prefix_stmt);
          consumer_wait_prefix_stmts.push_back(consumer_wait_stmt);
          i += 2;
          continue;
        }
        if (remap_pure_tma_barriers_ &&
            IsDuplicableBranchLocalPreStmt(pre_loop_stmts[i])) {
          duplicated_prefix_stmts.push_back(pre_loop_stmts[i]);
          ++i;
          continue;
        }
        if (IsSharedDependentConsumerPreStmt(pre_loop_stmts[i])) {
          consumer_shared_prefix_stmts.push_back(pre_loop_stmts[i]);
        } else {
          common_pre_stmts.push_back(pre_loop_stmts[i]);
        }
        ++i;
      }
      for (const auto &s : common_pre_stmts) {
        new_seq.push_back(s);
      }

      auto MakeOptionalStmt = [](const Array<Stmt> &stmts) -> Optional<Stmt> {
        if (stmts.empty()) {
          return std::nullopt;
        }
        return stmts.size() == 1 ? Optional<Stmt>(stmts[0])
                                 : Optional<Stmt>(SeqStmt(stmts));
      };

      Array<Stmt> producer_prefix_all_stmts;
      for (const auto &s : duplicated_prefix_stmts) {
        producer_prefix_all_stmts.push_back(s);
      }
      for (const auto &s : producer_prefix_stmts) {
        producer_prefix_all_stmts.push_back(s);
      }

      Array<Stmt> consumer_prefix_stmts;
      for (const auto &s : duplicated_prefix_stmts) {
        consumer_prefix_stmts.push_back(s);
      }
      // Keep pure local init before waits to delay blocking until needed.
      for (size_t j = movable_begin; j < pre_loop_stmts.size(); ++j) {
        consumer_prefix_stmts.push_back(pre_loop_stmts[j]);
      }
      for (const auto &s : consumer_wait_prefix_stmts) {
        consumer_prefix_stmts.push_back(s);
      }
      for (const auto &s : consumer_shared_prefix_stmts) {
        consumer_prefix_stmts.push_back(s);
      }
      Optional<Stmt> consumer_prefix = MakeOptionalStmt(consumer_prefix_stmts);
      Optional<Stmt> producer_prefix =
          MakeOptionalStmt(producer_prefix_all_stmts);

      Optional<Stmt> ws_stmt = rebuilt_loop;
      Optional<Stmt> producer_guard = std::nullopt;
      Optional<Stmt> pre_guard = std::nullopt;
      Optional<Stmt> post_guard = std::nullopt;

      // Merge TMA-issue producer prefix into producer branch.
      if (producer_prefix.defined()) {
        ICHECK(thread_iv_.defined());
        Stmt rewritten = PCThreadIdxRewriter::Rewrite(
            producer_prefix.value(), thread_iv_->var,
            thread_iv_->var - consumer_thread_extent_, producer_thread_extent_,
            /*do_shuffle=*/true);
        if (ws_stmt.defined()) {
          auto merged = TryPrependToProducerBranch(ws_stmt.value(), rewritten);
          if (merged.defined()) {
            ws_stmt = merged.value();
          } else {
            producer_guard = IfThenElse(
                GE(thread_iv_->var, consumer_thread_extent_), rewritten);
          }
        } else {
          producer_guard = IfThenElse(
              GE(thread_iv_->var, consumer_thread_extent_), rewritten);
        }
      }

      // Merge movable pre-loop suffix into consumer branch when possible.
      if (consumer_prefix.defined()) {
        if (ws_stmt.defined()) {
          auto merged = TryPrependToConsumerBranch(ws_stmt.value(),
                                                   consumer_prefix.value());
          if (merged.defined()) {
            ws_stmt = merged.value();
          } else {
            ICHECK(thread_iv_.defined());
            pre_guard = IfThenElse(LT(thread_iv_->var, consumer_thread_extent_),
                                   consumer_prefix.value());
          }
        } else {
          ICHECK(thread_iv_.defined());
          pre_guard = IfThenElse(LT(thread_iv_->var, consumer_thread_extent_),
                                 consumer_prefix.value());
        }
      }

      // Keep post-loop statements on consumer threads.
      if (!post_loop_stmts.empty()) {
        Stmt post_body = post_loop_stmts.size() == 1 ? post_loop_stmts[0]
                                                     : SeqStmt(post_loop_stmts);
        bool merged = false;
        if (ws_stmt.defined()) {
          auto merged_stmt =
              TryAppendToConsumerBranch(ws_stmt.value(), post_body);
          if (merged_stmt.defined()) {
            ws_stmt = merged_stmt.value();
            merged = true;
          }
        }
        if (!merged) {
          ICHECK(thread_iv_.defined());
          post_guard = IfThenElse(LT(thread_iv_->var, consumer_thread_extent_),
                                  post_body);
        }
      }

      if (producer_guard.defined()) {
        new_seq.push_back(producer_guard.value());
      }
      if (pre_guard.defined()) {
        new_seq.push_back(pre_guard.value());
      }
      if (ws_stmt.defined()) {
        new_seq.push_back(ws_stmt.value());
      }
      if (post_guard.defined()) {
        new_seq.push_back(post_guard.value());
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
  PrimExpr producer_thread_extent_ = IntImm(DataType::Int(32), 128);
  Optional<PrimExpr> updated_thread_extent_;
  bool need_update_thread_extent_ = false;
  bool ws_transformed_ = false;
  bool use_full_tma_forward_barrier_protocol_ = false;
  bool remap_pure_tma_barriers_ = false;
  int pure_tma_preloop_fwd_base_ = -1;
  int pure_tma_preloop_fwd_count_ = 0;
  int pure_tma_preloop_fwd_cursor_ = 0;
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
