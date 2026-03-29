/*!
 * \file producer_consumer_ws_tiled.cc
 * \brief Warp-specialized producer/consumer rewriting at the tile-op level.
 *
 * This pass runs **before** LayoutInference and LowerTileOp, operating on
 * high-level tile ops (`tl.tileop.copy`, `tl.tileop.gemm`, etc.).
 * It reads the `tl_instruction_kind` annotations placed by
 * InstructionAnnotation and splits pipelined loops into warp-specialized
 * producer/consumer branches with explicit barrier synchronization.
 *
 * The output IR is equivalent to a hand-written warp-specialized kernel:
 *   - TMA-annotated copies become `tl.tileop.tma_copy` with barrier refs
 *   - Barriers (`mbarrier_wait_parity`, `ptx_arrive_barrier`) are inserted
 *   - The loop body is wrapped in `if (threadIdx.x >= consumer_extent)`
 *
 * Prerequisites:
 *   - InstructionAnnotation must have run (tile ops carry tl_instruction_kind)
 *   - MultiVersionBuffer must have run (shared buffers are already expanded
 *     for pipelining and accesses include the stage index)
 *
 * Limitations (v1):
 *   - Pure TMA pipelines only (no mixed TMA + cp.async)
 *   - No conditionally guarded loop bodies (phase counters)
 *   - Single pipelined loop per block
 *   - No pre-loop TMA prefetch / prologue optimizations
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/gemm.h"
#include "../op/operator.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "common/mbarrier.h"
#include "warp_specialized_rewriter.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

// ---------------------------------------------------------------------------
// Utility: flatten SeqStmt recursively
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

/// Annotation key marking that this function was transformed by the tiled WS
/// pass, so downstream passes can skip redundant transformations.
static constexpr const char *kTiledWSApplied = "tl_tiled_ws_applied";

// ---------------------------------------------------------------------------
// PhaseCounter: local counter for correct barrier parity in guarded loops
// ---------------------------------------------------------------------------
struct PhaseCounter {
  Buffer buf;

  static PhaseCounter Create(const std::string &name) {
    return {decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32), name,
                        "local")};
  }

  PrimExpr Load() const {
    return BufferLoad(buf, {IntImm(DataType::Int(32), 0)});
  }

  Stmt Init() const {
    return BufferStore(buf, IntImm(DataType::Int(32), 0),
                       {IntImm(DataType::Int(32), 0)});
  }

  Stmt Increment() const {
    return BufferStore(buf, Load() + 1, {IntImm(DataType::Int(32), 0)});
  }

  Stmt WrapLoopWithAlloc(Stmt loop) const {
    Stmt body = SeqStmt({Init(), std::move(loop)});
    body = DeclBuffer(buf, body);
    return Allocate(buf->data, buf->dtype, buf->shape, const_true(), body);
  }

  PrimExpr StageExpr(int num_stages) const {
    if (num_stages == 1)
      return IntImm(DataType::Int(32), 0);
    return FloorMod(Load(), num_stages);
  }

  PrimExpr ParityExpr(int num_stages) const {
    if (num_stages == 1)
      return FloorMod(Load(), 2);
    return FloorMod(FloorDiv(Load(), num_stages), 2);
  }
};

// ---------------------------------------------------------------------------
// StageExprReplacer: rewrite loop-var-based stage indexing to counter-based
// ---------------------------------------------------------------------------
class StageExprReplacer : public StmtExprMutator {
public:
  static Stmt Replace(const Stmt &stmt, Var loop_var, PrimExpr loop_min,
                      int num_stages, PrimExpr replacement) {
    StageExprReplacer r(std::move(loop_var), std::move(loop_min), num_stages,
                        std::move(replacement));
    return r.VisitStmt(stmt);
  }

private:
  StageExprReplacer(Var loop_var, PrimExpr loop_min, int num_stages,
                    PrimExpr replacement)
      : loop_var_(std::move(loop_var)), loop_min_(std::move(loop_min)),
        num_stages_(num_stages), replacement_(std::move(replacement)) {}

  PrimExpr VisitExpr_(const FloorModNode *op) final {
    if (is_const_int(op->b, num_stages_) && MatchLinearIdx(op->a)) {
      return replacement_;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  bool MatchLinearIdx(const PrimExpr &expr) const {
    if (expr.same_as(loop_var_))
      return true;
    if (const auto *sub = expr.as<SubNode>()) {
      if (sub->a.same_as(loop_var_)) {
        if (is_const_int(sub->b, 0))
          return true;
        if (sub->b.same_as(loop_min_))
          return true;
      }
    }
    return false;
  }

  Var loop_var_;
  PrimExpr loop_min_;
  int num_stages_;
  PrimExpr replacement_;
};

// ---------------------------------------------------------------------------
// Statement classification
// ---------------------------------------------------------------------------

enum class TileStmtKind {
  kTmaProducer,     // TMA load producer (global->shared)
  kCpAsyncProducer, // Explicit cp.async / commit / wait_group producer stmt
  kSimtProducer,    // Non-tile-op SIMT copy: For loop writing shared from global
  kConsumer,        // Compute (gemm, reduce, element-wise, etc.)
  kOther            // Unclassified
};

/// Detect if a statement is a pure SIMT global-to-shared memory copy.
/// Pattern: For loop containing BufferStore(shared, BufferLoad(global))
/// with no arithmetic (just direct element copy).
class SimtProducerDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    SimtProducerDetector d;
    d(stmt);
    return d.is_pure_copy_;
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsSharedBuffer(op->buffer)) {
      if (auto *load = op->value.as<BufferLoadNode>()) {
        if (IsGlobalBuffer(load->buffer)) {
          is_pure_copy_ = true;
          return;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool is_pure_copy_{false};
};

static const CallNode *GetEvaluateCallInSimpleWrapper(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    return eval->value.as<CallNode>();
  }
  if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
    if (!if_stmt->else_case.defined()) {
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
  if (const auto *block = stmt.as<BlockNode>()) {
    return GetEvaluateCallInSimpleWrapper(block->body);
  }
  if (const auto *realize = stmt.as<BlockRealizeNode>()) {
    return GetEvaluateCallInSimpleWrapper(realize->block->body);
  }
  return nullptr;
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

static bool IsPtxWaitGroup(const Stmt &stmt) {
  const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
  return call && call->op.same_as(builtin::ptx_wait_group());
}

/// Classify a tile-op copy as TMA load producer, cp.async producer, or consumer.
/// Replicates the coarse checks from InstructionAnnotation inline so that
/// the tiled WS pass does not depend on a prior annotation pass.
static TileStmtKind ClassifyCopy(const CopyNode *copy, Target target) {
  // Explicit T.tma_copy() is a load-side primitive: only treat valid
  // global->shared TMA loads as producers.  TMA stores consume previously
  // produced shared data and must stay on the consumer side to preserve
  // per-iteration ordering.
  if (copy->GetIsTmaCopy()) {
    arith::Analyzer analyzer;
    if (copy->CheckBulkLoad(target, &analyzer, /*check_last_dim=*/false)) {
      return TileStmtKind::kTmaProducer;
    }
    return TileStmtKind::kConsumer; // target doesn't support TMA
  }
  // Explicit T.async_copy()
  if (copy->GetIsAsyncCopy()) {
    return TileStmtKind::kCpAsyncProducer;
  }
  // Generic T.copy(): check if TMA is possible
  {
    arith::Analyzer analyzer;
    if (copy->CheckBulkLoad(target, &analyzer, /*check_last_dim=*/true)) {
      return TileStmtKind::kTmaProducer;
    }
  }
  return TileStmtKind::kConsumer;
}

/// Classify a single statement in the pipeline loop body.
TileStmtKind ClassifyStmt(const Stmt &stmt, Target target) {
  // Tile-op Calls: classify directly via CopyNode checks.
  if (auto *eval = stmt.as<EvaluateNode>()) {
    if (auto *call = eval->value.as<CallNode>()) {
      auto tile_op = ParseOperator(ffi::GetRef<Call>(call));
      if (tile_op.defined()) {
        if (auto *copy = tile_op.as<CopyNode>()) {
          return ClassifyCopy(copy, target);
        }
        return TileStmtKind::kConsumer; // non-copy tile-op
      }
    }
  }
  // Explicit cp.async producer-side statements are already low-level builtins.
  if (ContainsPtxCpAsync(stmt) || IsPtxCommitGroup(stmt) || IsPtxWaitGroup(stmt)) {
    return TileStmtKind::kCpAsyncProducer;
  }
  // Non-tile-op: check for SIMT global-to-shared copy.
  if (SimtProducerDetector::Detect(stmt)) {
    return TileStmtKind::kSimtProducer;
  }
  return TileStmtKind::kConsumer;
}

bool IsProducer(TileStmtKind kind) {
  return kind == TileStmtKind::kTmaProducer ||
         kind == TileStmtKind::kCpAsyncProducer ||
         kind == TileStmtKind::kSimtProducer;
}

// ---------------------------------------------------------------------------
// Helpers: create barrier IR nodes
// ---------------------------------------------------------------------------

static Stmt MakeParityWait(const Buffer &barrier_buf, PrimExpr barrier_id,
                           PrimExpr parity) {
  auto ref = MakeBarrierRef(barrier_buf, std::move(barrier_id));
  return Evaluate(
      Call(DataType::Handle(), mbarrier_wait_parity(), {ref, std::move(parity)}));
}

static Stmt MakeArriveBarrier(const Buffer &barrier_buf,
                              PrimExpr barrier_id) {
  auto ref = MakeBarrierRef(barrier_buf, std::move(barrier_id));
  return Evaluate(Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {ref}));
}

// ---------------------------------------------------------------------------
// Convert tl.tileop.copy → tl.tileop.tma_copy with barrier annotation
// ---------------------------------------------------------------------------

/// Rewrite a `tl.tileop.copy` Call into a `tl.tileop.tma_copy` Call with
/// barrier reference.  The args (src/dst regions) are preserved; only the op
/// and annotations change.
static PrimExpr RewriteCopyToTmaCopy(const Call &copy_call,
                                     const Buffer &barrier_buf,
                                     PrimExpr barrier_id) {
  static const Op &tma_copy_op = Op::Get("tl.tileop.tma_copy");
  auto new_annotations = copy_call->annotations;
  new_annotations.Set("barrier", MakeBarrierRef(barrier_buf, barrier_id));
  new_annotations.Set("is_tma_copy", IntImm(DataType::Int(32), 1));
  return Call(copy_call->dtype, tma_copy_op, copy_call->args, new_annotations,
              copy_call->span);
}

// ---------------------------------------------------------------------------
// Main rewriter
// ---------------------------------------------------------------------------

class ProducerConsumerWSTiledRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "ProducerConsumerWSTiled: target attribute is required";

    ProducerConsumerWSTiledRewriter T;
    T.target_ = target.value();
    f.CopyOnWrite()->body = T(f->body);

    if (T.ws_transformed_) {
      f = WithAttr(std::move(f), kTiledWSApplied, IntImm(DataType::Int(32), 1));
      if (T.ws_total_threads_.defined()) {
        f = WithAttr(std::move(f), "tl_ws_total_threads",
                     T.ws_total_threads_.value());
      }
    }
    return f;
  }

private:
  // --- Track threadIdx.x binding ---
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_iv_ = iv;
        Optional<PrimExpr> old_num_threads = num_threads_;
        num_threads_ = std::nullopt;
        AttrStmt attr =
            Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
        if (num_threads_.defined()) {
          PrimExpr nt = num_threads_.value();
          thread_iv_.CopyOnWrite()->dom = {0, nt};
          attr.CopyOnWrite()->node = thread_iv_;
          attr.CopyOnWrite()->value = nt;
        }
        num_threads_ = old_num_threads;
        thread_iv_ = {};
        return attr;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // --- Find the block containing the pipeline loop ---
  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    if (!thread_iv_.defined())
      return StmtExprMutator::VisitStmt_(op);

    const Block &orig_block = op->block;

    // Find the pipelined loop.
    const ForNode *pipeline_loop = FindPipelineLoop(orig_block->body);
    if (!pipeline_loop)
      return StmtExprMutator::VisitStmt_(op);

    auto num_stages_anno = pipeline_loop->annotations.Get("num_stages");
    if (!num_stages_anno)
      return StmtExprMutator::VisitStmt_(op);
    int num_stages =
        static_cast<int>(Downcast<Integer>(num_stages_anno.value())->value);
    if (num_stages < 1)
      return StmtExprMutator::VisitStmt_(op);

    // Flatten the loop body.
    Array<Stmt> flat_stmts;
    Stmt loop_body = pipeline_loop->body;
    if (auto *realize = loop_body.as<BlockRealizeNode>()) {
      loop_body = realize->block->body;
    }
    // Unwrap LetStmt chain
    std::vector<std::pair<Var, PrimExpr>> let_bindings;
    while (const auto *let = loop_body.as<LetStmtNode>()) {
      let_bindings.emplace_back(let->var, let->value);
      loop_body = let->body;
    }
    // Unwrap a single IfThenElse wrapper (no else branch) so that
    // TMA producers inside conditional loop bodies can be classified.
    // Also peel LetStmt chains inside the conditional and append them
    // to let_bindings so they're in scope for both WS branches.
    Optional<PrimExpr> loop_body_condition;
    if (const auto *if_stmt = loop_body.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined()) {
        // Peel LetStmt chain from inside the conditional body so
        // they're in scope for both WS branches.
        Stmt inner = if_stmt->then_case;
        while (const auto *let = inner.as<LetStmtNode>()) {
          let_bindings.emplace_back(let->var, let->value);
          inner = let->body;
        }
        loop_body_condition = if_stmt->condition;
        loop_body = inner;
      }
    }
    FlattenSeqStmt(loop_body, &flat_stmts);

    // Classify statements into producer (TMA/SIMT copy) and consumer.
    std::vector<TileStmtKind> kinds;
    int num_tma = 0;
    int num_simt = 0;
    for (const Stmt &s : flat_stmts) {
      auto k = ClassifyStmt(s, target_);
      kinds.push_back(k);
      if (k == TileStmtKind::kTmaProducer)
        ++num_tma;
      if (k == TileStmtKind::kSimtProducer)
        ++num_simt;
    }

    // Require at least one TMA producer.
    if (num_tma == 0)
      return StmtExprMutator::VisitStmt_(op);

    // --- Build the WS transformation ---
    return BuildWSBlock(op, orig_block, pipeline_loop, num_stages, flat_stmts,
                        kinds, let_bindings, loop_body_condition);
  }

  Stmt BuildWSBlock(const BlockRealizeNode *orig_realize,
                    const Block &orig_block, const ForNode *pipeline_loop,
                    int num_stages, const Array<Stmt> &flat_stmts,
                    const std::vector<TileStmtKind> &kinds,
                    const std::vector<std::pair<Var, PrimExpr>> &let_bindings,
                    Optional<PrimExpr> loop_body_condition = Optional<PrimExpr>()) {
    Var loop_var = pipeline_loop->loop_var;
    PrimExpr loop_min = pipeline_loop->min;
    PrimExpr loop_extent = pipeline_loop->extent;
    PrimExpr linear_idx = loop_var - loop_min;

    PrimExpr base_stage_expr = FloorMod(linear_idx, num_stages);
    PrimExpr base_parity_expr = FloorMod(FloorDiv(linear_idx, num_stages), 2);

    // When the loop body is conditionally guarded, use PhaseCounters
    // instead of the loop variable for barrier stage/parity.  This
    // ensures parity stays correct when iterations are skipped.
    bool needs_phase_counter = loop_body_condition.defined();
    Optional<PhaseCounter> producer_phase_counter;
    Optional<PhaseCounter> consumer_phase_counter;
    PrimExpr p_stage_expr = base_stage_expr;
    PrimExpr p_parity_expr = base_parity_expr;
    PrimExpr c_stage_expr = base_stage_expr;
    PrimExpr c_parity_expr = base_parity_expr;
    if (needs_phase_counter) {
      producer_phase_counter = PhaseCounter::Create("producer_phase_cnt");
      consumer_phase_counter = PhaseCounter::Create("consumer_phase_cnt");
      p_stage_expr = producer_phase_counter.value().StageExpr(num_stages);
      p_parity_expr = producer_phase_counter.value().ParityExpr(num_stages);
      c_stage_expr = consumer_phase_counter.value().StageExpr(num_stages);
      c_parity_expr = consumer_phase_counter.value().ParityExpr(num_stages);
    }

    PrimExpr consumer_extent = thread_iv_->dom->extent;
    // Producer warp group is always 128 threads (one warp group).
    // SIMT copies run on these 128 threads — they don't need the full
    // consumer extent because the copy loop iteration space is distributed
    // across the available threads via T.Parallel.
    PrimExpr producer_extent = IntImm(DataType::Int(32), 128);

    bool has_simt_producer = false;
    bool has_cp_async_producer = false;
    int num_producer_groups = 0;
    for (auto k : kinds) {
      if (k == TileStmtKind::kTmaProducer)
        ++num_producer_groups;
      if (k == TileStmtKind::kSimtProducer)
        has_simt_producer = true;
      if (k == TileStmtKind::kCpAsyncProducer)
        has_cp_async_producer = true;
    }

    // --- Barrier allocation ---
    // Layout: [fwd_0..fwd_{G*S-1}] [bp_0..bp_{G*S-1}]
    // where G = num_producer_groups (one per TMA copy), S = num_stages.
    // When SIMT producers are present, all producer types share the same
    // barrier group — the last forward arrive covers everything.
    int num_fwd = num_producer_groups * num_stages;
    int num_bp = num_producer_groups * num_stages;
    int total_barriers = num_fwd + num_bp;
    Buffer barrier_buf = CreateMBarrierBuffer(injected_mbarrier_name_,
                                              total_barriers);

    // Forward arrive_count:
    //   - Pure TMA: 1 (leader thread only issues expect_tx + tma_load + arrive)
    //   - Mixed TMA with SIMT/cp.async producers: producer_extent (128) —
    //     all producer threads arrive after all producer-side work completes.
    PrimExpr fwd_arrive_count =
        (has_simt_producer || has_cp_async_producer)
            ? producer_extent  // 128
            : IntImm(DataType::Int(32), 1);
    Array<PrimExpr> arrive_counts;
    for (int i = 0; i < num_fwd; ++i) {
      arrive_counts.push_back(fwd_arrive_count);
    }
    for (int i = 0; i < num_bp; ++i) {
      arrive_counts.push_back(consumer_extent);
    }

    // --- Build producer body ---
    // Producer structure:
    //   bp_wait → TMA copies (leader gated) → SIMT/cp.async producers
    //   (all threads) → fwd arrive.
    Array<Stmt> producer_stmts;
    int tma_idx = 0;
    int last_tma_idx = num_producer_groups - 1;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] == TileStmtKind::kTmaProducer) {
        int fwd_base = tma_idx * num_stages;
        int bp_base = num_fwd + tma_idx * num_stages;
        PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) + p_stage_expr;
        PrimExpr bp_id = IntImm(DataType::Int(32), bp_base) + p_stage_expr;

        // Back-pressure wait
        producer_stmts.push_back(MakeParityWait(
            barrier_buf, bp_id,
            bitwise_xor(p_parity_expr, IntImm(DataType::Int(32), 1))));
        // Convert copy → tma_copy with barrier
        const auto *eval = flat_stmts[i].as<EvaluateNode>();
        ICHECK(eval);
        Call copy_call = Downcast<Call>(eval->value);
        PrimExpr tma_call =
            RewriteCopyToTmaCopy(copy_call, barrier_buf, fwd_id);
        producer_stmts.push_back(Evaluate(tma_call));

        if (!has_simt_producer) {
          // Pure TMA: single-thread arrive after each copy
          producer_stmts.push_back(
              IfThenElse(EQ(thread_iv_->var, IntImm(DataType::Int(32), 0)),
                         MakeArriveBarrier(barrier_buf, fwd_id)));
        }
        ++tma_idx;
      } else if (kinds[i] == TileStmtKind::kSimtProducer) {
        // SIMT copy goes directly into producer body (all threads)
        producer_stmts.push_back(flat_stmts[i]);
      } else if (kinds[i] == TileStmtKind::kCpAsyncProducer) {
        producer_stmts.push_back(flat_stmts[i]);
      }
      // Consumer/Other statements are skipped in producer.
    }
    // When any producer-side work is not single-threaded pure-TMA, all
    // producer threads arrive on all forward barriers after finishing it.
    // For mixed groups with cp.async, use cp_async_barrier_noinc to let
    // the async copy own the arrival count (matching reference protocol).
    if (has_simt_producer || has_cp_async_producer) {
      for (int g = 0; g < num_producer_groups; ++g) {
        int fwd_base = g * num_stages;
        PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) + p_stage_expr;
        if (has_cp_async_producer) {
          auto call = Call(DataType::Handle(), tl::ptx_cp_async_barrier_noinc(),
                           {MakeBarrierRef(barrier_buf, fwd_id)});
          producer_stmts.push_back(Evaluate(call));
        } else {
          producer_stmts.push_back(MakeArriveBarrier(barrier_buf, fwd_id));
        }
      }
    }
    // Phase counter increment at end of producer guarded iteration
    if (needs_phase_counter) {
      producer_stmts.push_back(producer_phase_counter.value().Increment());
    }

    // --- Build consumer body ---
    Array<Stmt> consumer_stmts;
    // Forward waits before first consumer stmt.
    for (int g = 0; g < num_producer_groups; ++g) {
      int fwd_base = g * num_stages;
      PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) + c_stage_expr;
      consumer_stmts.push_back(
          MakeParityWait(barrier_buf, fwd_id, c_parity_expr));
    }
    // Consumer statements (everything not a producer)
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (!IsProducer(kinds[i])) {
        consumer_stmts.push_back(flat_stmts[i]);
      }
    }
    // Back-pressure arrives
    for (int g = 0; g < num_producer_groups; ++g) {
      int bp_base = num_fwd + g * num_stages;
      PrimExpr bp_id = IntImm(DataType::Int(32), bp_base) + c_stage_expr;
      consumer_stmts.push_back(MakeArriveBarrier(barrier_buf, bp_id));
    }
    // Phase counter increment at end of consumer guarded iteration
    if (needs_phase_counter) {
      consumer_stmts.push_back(consumer_phase_counter.value().Increment());
    }

    // --- Wrap with let bindings and optional condition ---
    auto wrap_lets = [&](Stmt body) -> Stmt {
      for (auto it = let_bindings.rbegin(); it != let_bindings.rend(); ++it) {
        body = LetStmt(it->first, it->second, body);
      }
      return body;
    };

    Stmt producer_body = wrap_lets(SeqStmt(producer_stmts));
    Stmt consumer_body = wrap_lets(SeqStmt(consumer_stmts));

    // Wrap in original condition if the loop body was guarded.
    if (loop_body_condition.defined()) {
      producer_body = IfThenElse(loop_body_condition.value(), producer_body);
      consumer_body = IfThenElse(loop_body_condition.value(), consumer_body);
    }

    // Rewrite shared-buffer stage indices from loop-var-based to
    // counter-based so they stay in sync with barrier parity.
    if (needs_phase_counter) {
      producer_body = StageExprReplacer::Replace(
          producer_body, loop_var, loop_min, num_stages,
          producer_phase_counter.value().StageExpr(num_stages));
      consumer_body = StageExprReplacer::Replace(
          consumer_body, loop_var, loop_min, num_stages,
          consumer_phase_counter.value().StageExpr(num_stages));
    }

    // --- Build loops (strip pipeline annotations) ---
    // WS handles pipeline overlap via barriers, so strip all pipeline-
    // related annotations to prevent PipelinePlanning / InjectSoftware-
    // Pipeline from re-pipelining the already WS-transformed loops.
    Map<String, Any> loop_annos;
    for (const auto &[key, value] : pipeline_loop->annotations) {
      if (key != "num_stages" && key != "tl_pipeline_order" &&
          key != "tl_pipeline_stage" && key != "software_pipeline_order" &&
          key != "software_pipeline_stage") {
        loop_annos.Set(key, value);
      }
    }

    // Wrap the producer body in a pipeline-context AttrStmt so that
    // LowerTileOp enables cp.async injection for SIMT producers
    // (pipelined_depth_ > 0) without triggering re-pipelining.
    Stmt producer_body_with_ctx = AttrStmt(
        StringImm("tl.pipeline_context_num_stages"),
        "tl.pipeline_context_num_stages", IntImm(DataType::Int(32), num_stages),
        producer_body);

    For producer_loop(loop_var, loop_min, loop_extent, ForKind::kSerial,
                      producer_body_with_ctx, Optional<IterVar>(), loop_annos);
    For consumer_loop(loop_var, loop_min, loop_extent, ForKind::kSerial,
                      consumer_body, Optional<IterVar>(), loop_annos);

    // Wrap loops with phase counter allocation when needed.
    Stmt final_producer_loop = producer_loop;
    Stmt final_consumer_loop = consumer_loop;
    if (needs_phase_counter) {
      final_producer_loop =
          producer_phase_counter.value().WrapLoopWithAlloc(producer_loop);
      final_consumer_loop =
          consumer_phase_counter.value().WrapLoopWithAlloc(consumer_loop);
    }

    // --- Rewrite threadIdx.x for producer partition ---
    // Producer: threadIdx.x - consumer_extent (maps to [0, producer_extent))
    Stmt rewritten_producer = PCThreadIdxRewriter::Rewrite(
        final_producer_loop, thread_iv_->var,
        thread_iv_->var - consumer_extent, producer_extent, false);
    // Consumer: threadIdx.x stays, but extent is consumer_extent
    Stmt rewritten_consumer = final_consumer_loop;

    // --- WS if/else structure ---
    Stmt ws_body = IfThenElse(GE(thread_iv_->var, consumer_extent),
                              rewritten_producer, rewritten_consumer);

    // Replace the original pipeline loop in-place so outer Let/Seq wrappers
    // are preserved.  Any statements after the replaced subtree run on the
    // consumer threads only.
    ReplaceResult replaced = ReplacePipelineLoopInStmt(
        orig_block->body, pipeline_loop, ws_body, consumer_extent);
    ICHECK(replaced.found)
        << "ProducerConsumerWSTiled: failed to replace pipeline loop";
    Stmt new_block_body = replaced.stmt;

    // --- Update block ---
    Block new_block = orig_block;
    auto *block_ptr = new_block.CopyOnWrite();
    block_ptr->body = new_block_body;

    // Add barrier buffer to alloc_buffers.
    block_ptr->alloc_buffers.push_back(barrier_buf);

    // Add barrier_init annotation.
    Map<Var, Array<PrimExpr>> barrier_init_map;
    barrier_init_map.Set(barrier_buf->data, arrive_counts);
    auto ann = block_ptr->annotations;
    if (ann.count("barrier_init")) {
      auto existing = Downcast<Map<Var, Array<PrimExpr>>>(
          ann.Get("barrier_init").value());
      for (auto [k, v] : existing) {
        barrier_init_map.Set(k, v);
      }
    }
    ann.Set("barrier_init", barrier_init_map);
    block_ptr->annotations = std::move(ann);

    // Update thread extent at the tiled WS level so LayoutInference sees
    // the producer branch as live and can analyze explicit TMA copies.
    num_threads_ = consumer_extent + producer_extent;
    ws_total_threads_ = consumer_extent + producer_extent;
    ws_transformed_ = true;

    // Rebuild BlockRealize.
    BlockRealize new_realize = GetRef<BlockRealize>(orig_realize);
    new_realize.CopyOnWrite()->block = new_block;
    return new_realize;
  }

  // --- Find the first For loop with num_stages annotation ---
  const ForNode *FindPipelineLoop(const Stmt &stmt) {
    if (auto *for_node = stmt.as<ForNode>()) {
      if (for_node->annotations.Get("num_stages")) {
        return for_node;
      }
    }
    // Walk through SeqStmt, LetStmt, etc.
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (auto *result = FindPipelineLoop(s)) {
          return result;
        }
      }
    }
    if (auto *let = stmt.as<LetStmtNode>()) {
      return FindPipelineLoop(let->body);
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
    return nullptr;
  }

  struct ReplaceResult {
    Stmt stmt;
    bool found{false};
  };

  Stmt GuardConsumerOnly(const Stmt &stmt, PrimExpr consumer_extent) {
    return IfThenElse(LT(thread_iv_->var, consumer_extent), stmt);
  }

  ReplaceResult ReplacePipelineLoopInStmt(const Stmt &stmt,
                                          const ForNode *pipeline_loop,
                                          const Stmt &ws_body,
                                          PrimExpr consumer_extent) {
    if (stmt.get() == pipeline_loop) {
      return {ws_body, true};
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> new_seq;
      bool found = false;
      // First pass: find which child contains the pipeline loop.
      int loop_idx = -1;
      for (int i = 0; i < static_cast<int>(seq->seq.size()); ++i) {
        ReplaceResult probe = ReplacePipelineLoopInStmt(
            seq->seq[i], pipeline_loop, ws_body, consumer_extent);
        if (probe.found) {
          loop_idx = i;
          break;
        }
      }
      if (loop_idx < 0) {
        return {stmt, false};
      }
      // Guard pre-loop consumer-only statements (fragment init, etc.).
      // Statements that classify as TMA/SIMT/cp.async producers are
      // shared (needed by producer); everything else is consumer-only.
      for (int i = 0; i < loop_idx; ++i) {
        auto kind = ClassifyStmt(seq->seq[i], target_);
        if (IsProducer(kind)) {
          new_seq.push_back(seq->seq[i]);  // shared
        } else {
          new_seq.push_back(GuardConsumerOnly(seq->seq[i], consumer_extent));
        }
      }
      // Replace the pipeline loop itself.
      ReplaceResult result = ReplacePipelineLoopInStmt(
          seq->seq[loop_idx], pipeline_loop, ws_body, consumer_extent);
      new_seq.push_back(result.stmt);
      // Guard post-loop siblings.
      for (int i = loop_idx + 1; i < static_cast<int>(seq->seq.size()); ++i) {
        new_seq.push_back(GuardConsumerOnly(seq->seq[i], consumer_extent));
      }
      return {new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq), true};
    }
    if (auto *let = stmt.as<LetStmtNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          let->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      return {LetStmt(let->var, let->value, result.stmt), true};
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          realize->block->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      Block block = realize->block;
      block.CopyOnWrite()->body = result.stmt;
      BlockRealize new_realize = GetRef<BlockRealize>(realize);
      new_realize.CopyOnWrite()->block = block;
      return {new_realize, true};
    }
    if (auto *block = stmt.as<BlockNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          block->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      Block new_block = GetRef<Block>(block);
      new_block.CopyOnWrite()->body = result.stmt;
      return {new_block, true};
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          attr->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      AttrStmt new_attr = GetRef<AttrStmt>(attr);
      new_attr.CopyOnWrite()->body = result.stmt;
      return {new_attr, true};
    }
    return {stmt, false};
  }

  // --- PCThreadIdxRewriter (simplified for tile-op level) ---
  class PCThreadIdxRewriter : public StmtExprMutator {
  public:
    static Stmt Rewrite(Stmt stmt, Var thread_var, PrimExpr replaced,
                        PrimExpr thread_extent, bool do_shuffle) {
      PCThreadIdxRewriter r(std::move(thread_var), std::move(replaced),
                            std::move(thread_extent));
      return r(std::move(stmt));
    }

  private:
    PCThreadIdxRewriter(Var thread_var, PrimExpr replaced,
                        PrimExpr thread_extent)
        : thread_var_(std::move(thread_var)),
          replaced_(std::move(replaced)),
          thread_extent_(std::move(thread_extent)) {}

    PrimExpr VisitExpr_(const VarNode *var) final {
      if (var == thread_var_.get()) {
        return replaced_;
      }
      return StmtExprMutator::VisitExpr_(var);
    }

    Var thread_var_;
    PrimExpr replaced_;
    PrimExpr thread_extent_;
  };

  // State
  Target target_;
  IterVar thread_iv_;
  Optional<PrimExpr> num_threads_;          // total (consumer + producer)
  Optional<PrimExpr> ws_total_threads_;     // same, stored as func attr
  bool ws_transformed_{false};
};

// ---------------------------------------------------------------------------
// Detect if manual WS is already present (skip if so)
// ---------------------------------------------------------------------------

class ManualWSDetector : public StmtExprVisitor {
public:
  static bool HasManualWS(const Stmt &stmt) {
    ManualWSDetector d;
    d(stmt);
    return d.found_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    // Detect both the T.ws() language-level attr ("warp_specialize") and
    // the compiler-level attr (kWarpSpecializationScope).
    if (op->attr_key == "warp_specialize" ||
        op->attr_key == attr::kWarpSpecializationScope) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool found_{false};
};

/// Quick pre-scan: check if the function contains a pipelined loop (num_stages
/// >= 1) with at least one TMA load producer tile op and no manual layout
/// annotations (which are incompatible with early MVB expansion).
class TiledWSCandidate : public StmtExprVisitor {
public:
  static bool Check(const Stmt &stmt, Target target) {
    TiledWSCandidate c;
    c.target_ = target;
    c(stmt);
    return c.has_pipeline_loop_ && c.has_tma_tile_op_ &&
           !c.has_manual_layout_;
  }

  /// Check if the function has TMA copies in a pipeline loop (even if
  /// other conditions prevent full WS candidacy).
  static bool HasTmaPipeline(const Stmt &stmt, Target target) {
    TiledWSCandidate c;
    c.target_ = target;
    c(stmt);
    return c.has_pipeline_loop_ && c.has_tma_tile_op_;
  }

private:
  void VisitStmt_(const ForNode *op) final {
    bool old = in_pipeline_;
    if (auto anno = op->annotations.Get("num_stages")) {
      if (auto *imm = anno->as<IntImmNode>()) {
        if (imm->value >= 1) {
          has_pipeline_loop_ = true;
          in_pipeline_ = true;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
    in_pipeline_ = old;
  }

  void VisitExpr_(const CallNode *op) final {
    if (in_pipeline_ && !has_tma_tile_op_) {
      auto tile_op = ParseOperator(ffi::GetRef<Call>(op));
      if (auto *copy = tile_op.as<CopyNode>()) {
        if (ClassifyCopy(copy, target_) == TileStmtKind::kTmaProducer) {
          has_tma_tile_op_ = true;
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    if (op->annotations.count("layout_map")) {
      has_manual_layout_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Target target_;
  bool in_pipeline_{false};
  bool has_pipeline_loop_{false};
  bool has_tma_tile_op_{false};
  bool has_manual_layout_{false};
};

} // namespace

// Forward-declare MultiVersionBuffer (defined in multi_version_buffer_rewriter.cc).
tvm::transform::Pass MultiVersionBuffer(bool barrier_only = false);

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

tvm::transform::Pass ProducerConsumerWarpSpecializedTiled() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Skip if disabled.
    if (ctx->GetConfig(kDisableWarpSpecialized, Optional<Bool>())
            .value_or(false)) {
      return f;
    }
    // Skip if the function already has manual WS.
    if (ManualWSDetector::HasManualWS(f->body)) {
      return f;
    }
    // Skip if TMA is not available.
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetHasBulkCopy(target.value())) {
      return f;
    }
    // Only apply MVB + WS if the function is a tiled WS candidate.
    if (!TiledWSCandidate::Check(f->body, target.value())) {
      LOG(WARNING) << "[TiledWS] skipped: not a candidate";
      // If the function has TMA copies in a pipeline loop but was
      // rejected (e.g., manual layout annotations), strip pipeline
      // annotations to prevent InjectSoftwarePipeline from generating
      // broken non-WS TMA pipeline code.
      if (TiledWSCandidate::HasTmaPipeline(f->body, target.value())) {
        LOG(WARNING) << "[TiledWS] stripping pipeline for rejected TMA kernel";
        class StripAnnotation : public tir::StmtExprMutator {
         public:
          tir::Stmt VisitStmt_(const tir::ForNode *op) final {
            auto stmt = tir::StmtExprMutator::VisitStmt_(op);
            const auto *n = stmt.as<tir::ForNode>();
            ICHECK(n);
            if (n->annotations.count("num_stages")) {
              tir::For new_for = Downcast<tir::For>(stmt);
              new_for.CopyOnWrite()->annotations.erase("num_stages");
              return std::move(new_for);
            }
            return stmt;
          }
        };
        StripAnnotation stripper;
        auto *fn = f.CopyOnWrite();
        fn->body = stripper(f->body);
      }
      return f;
    }
    LOG(WARNING) << "[TiledWS] candidate found, applying MVB + WS";
    // Expand shared buffers for pipelining before the WS split.
    // Keep the original so we can fall back if the WS rewriter
    // doesn't fire (e.g. non-tile-op consumers in the loop body).
    PrimFunc original_f = f;
    {
      IRModule tmp_mod;
      tmp_mod->Add(GlobalVar("main"), f);
      tmp_mod = MultiVersionBuffer()(tmp_mod);
      f = Downcast<PrimFunc>(tmp_mod->Lookup("main"));
    }
    PrimFunc result =
        ProducerConsumerWSTiledRewriter::Substitute(std::move(f));
    if (!result->HasNonzeroAttr(kTiledWSApplied)) {
      LOG(WARNING) << "[TiledWS] rewriter did not fire, falling back";
      // The TMA kernel needs warp specialization for correct pipelined
      // execution.  Since the tiled rewriter could not apply WS (e.g.
      // conditional loop body), strip pipeline annotations so that
      // PipelinePlanning / InjectSoftwarePipeline do not generate
      // broken non-WS TMA pipeline code.
      class StripPipelineAnnotation : public tir::StmtExprMutator {
       public:
        tir::Stmt VisitStmt_(const tir::ForNode *op) final {
          auto stmt = tir::StmtExprMutator::VisitStmt_(op);
          const auto *for_node = stmt.as<tir::ForNode>();
          ICHECK(for_node);
          if (for_node->annotations.count("num_stages")) {
            tir::For new_for = Downcast<tir::For>(stmt);
            auto *n = new_for.CopyOnWrite();
            n->annotations.erase("num_stages");
            return std::move(new_for);
          }
          return stmt;
        }
      };
      StripPipelineAnnotation stripper;
      auto stripped = stripper(original_f->body);
      auto *fn = original_f.CopyOnWrite();
      fn->body = stripped;
      return original_f;
    }
    LOG(WARNING) << "[TiledWS] transformation applied successfully";
    return result;
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.ProducerConsumerWarpSpecializedTiled", {});
}

// ---------------------------------------------------------------------------
// RestoreWSThreadExtent: update AttrStmt(threadIdx.x) to the WS total.
// Must run after LowerTileOp but before the first Simplify in
// OptimizeForTarget, otherwise the analyzer proves the producer branch
// (threadIdx.x >= consumer_extent) is dead code.
// ---------------------------------------------------------------------------

tvm::transform::Pass RestoreWSThreadExtent() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto attr = f->GetAttr<PrimExpr>("tl_ws_total_threads");
    if (!attr.defined())
      return f;
    auto *imm = attr.value().as<IntImmNode>();
    if (!imm)
      return f;

    int64_t total = imm->value;

    // Walk the body:
    // 1. Update threadIdx.x AttrStmt extent to the WS total.
    // 2. Strip tl_ws_split wrapper (no longer needed after lowering).
    class Updater : public StmtExprMutator {
    public:
      int64_t total_;
      explicit Updater(int64_t t) : total_(t) {}
      Stmt VisitStmt_(const AttrStmtNode *op) final {
        if (op->attr_key == tir::attr::thread_extent) {
          IterVar iv = Downcast<IterVar>(op->node);
          if (iv->thread_tag == "threadIdx.x") {
            AttrStmt a =
                Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
            PrimExpr nt = IntImm(DataType::Int(32), total_);
            iv.CopyOnWrite()->dom = {0, nt};
            a.CopyOnWrite()->node = iv;
            a.CopyOnWrite()->value = nt;
            return a;
          }
        }
        return StmtExprMutator::VisitStmt_(op);
      }
    };

    Updater u(total);
    f.CopyOnWrite()->body = u(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RestoreWSThreadExtent", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ProducerConsumerWarpSpecializedTiled",
                        ProducerConsumerWarpSpecializedTiled);
  refl::GlobalDef().def("tl.transform.RestoreWSThreadExtent",
                        RestoreWSThreadExtent);
}

} // namespace tl
} // namespace tvm
