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

/// Annotation key written by InstructionAnnotation.
static constexpr const char *kInstructionKind = "tl_instruction_kind";

/// Annotation key marking that this function was transformed by the tiled WS
/// pass, so downstream passes can skip redundant transformations.
static constexpr const char *kTiledWSApplied = "tl_tiled_ws_applied";

// ---------------------------------------------------------------------------
// Statement classification
// ---------------------------------------------------------------------------

enum class TileStmtKind {
  kTmaProducer,     // TMA copy tile op (annotated "tma")
  kCpAsyncProducer, // cp.async copy tile op (annotated "cp_async")
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
    // Check: store to shared, value is a direct BufferLoad from global
    // (no computation on the value).
    if (IsSharedBuffer(op->buffer)) {
      if (auto *load = op->value.as<BufferLoadNode>()) {
        if (IsGlobalBuffer(load->buffer)) {
          is_pure_copy_ = true;
          return; // Don't recurse further
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool is_pure_copy_{false};
};

/// Classify a single statement in the pipeline loop body.
TileStmtKind ClassifyStmt(const Stmt &stmt) {
  // Tile-op Calls: use InstructionAnnotation.
  if (auto *eval = stmt.as<EvaluateNode>()) {
    if (auto *call = eval->value.as<CallNode>()) {
      if (auto kind_anno = call->annotations.Get(kInstructionKind)) {
        if (auto *str_imm = kind_anno->as<StringImmNode>()) {
          if (str_imm->value == "tma")
            return TileStmtKind::kTmaProducer;
          if (str_imm->value == "cp_async")
            return TileStmtKind::kCpAsyncProducer;
        }
      }
      // Tile-op with no producer annotation → consumer.
      if (ParseOperator(ffi::GetRef<Call>(call)).defined()) {
        return TileStmtKind::kConsumer;
      }
    }
  }
  // Non-tile-op statements (SIMT loops, scalar ops, etc.) are treated as
  // consumer.  SIMT global-to-shared copies are NOT promoted to producer
  // at this stage — they remain in the consumer branch and the old WS pass
  // (ProducerConsumerWarpSpecialized in OptimizeForTarget) can convert them
  // to cp.async if appropriate.
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
  // Remove the instruction annotation — it has been consumed.
  new_annotations.erase(kInstructionKind);
  return Call(copy_call->dtype, tma_copy_op, copy_call->args, new_annotations,
              copy_call->span);
}

/// Strip `tl_instruction_kind` from a statement's Call annotations.
static Stmt StripInstructionKind(const Stmt &stmt) {
  if (auto *eval = stmt.as<EvaluateNode>()) {
    if (auto *call = eval->value.as<CallNode>()) {
      if (call->annotations.count(kInstructionKind)) {
        auto new_annotations = call->annotations;
        new_annotations.erase(kInstructionKind);
        auto new_call =
            Call(call->dtype, call->op, call->args, new_annotations, call->span);
        return Evaluate(new_call);
      }
    }
  }
  return stmt;
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
    FlattenSeqStmt(loop_body, &flat_stmts);

    // Classify statements into producer (TMA/SIMT copy) and consumer.
    std::vector<TileStmtKind> kinds;
    int num_tma = 0;
    int num_cp_async = 0;
    int num_simt = 0;
    for (const Stmt &s : flat_stmts) {
      auto k = ClassifyStmt(s);
      kinds.push_back(k);
      if (k == TileStmtKind::kTmaProducer)
        ++num_tma;
      if (k == TileStmtKind::kCpAsyncProducer)
        ++num_cp_async;
      if (k == TileStmtKind::kSimtProducer)
        ++num_simt;
    }

    // Check that all consumer stmts are tile-op Calls.  If the pipeline
    // body contains non-tile-op consumer statements (SIMT parallel loops,
    // scalar ops, etc.), fall back to the old WS path which runs after
    // LowerTileOp and can handle arbitrary lowered IR.
    bool has_non_tileop_consumer = false;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] == TileStmtKind::kConsumer) {
        if (auto *eval = flat_stmts[i].as<EvaluateNode>()) {
          if (auto *call = eval->value.as<CallNode>()) {
            if (ParseOperator(ffi::GetRef<Call>(call)).defined()) {
              continue; // tile-op consumer — OK
            }
          }
        }
        has_non_tileop_consumer = true;
      }
    }

    // Require at least one TMA producer, no cp.async, no non-tile-op consumers.
    if (num_tma == 0 || num_cp_async > 0 || has_non_tileop_consumer)
      return StmtExprMutator::VisitStmt_(op);

    // --- Build the WS transformation ---
    return BuildWSBlock(op, orig_block, pipeline_loop, num_stages, flat_stmts,
                        kinds, let_bindings);
  }

  Stmt BuildWSBlock(const BlockRealizeNode *orig_realize,
                    const Block &orig_block, const ForNode *pipeline_loop,
                    int num_stages, const Array<Stmt> &flat_stmts,
                    const std::vector<TileStmtKind> &kinds,
                    const std::vector<std::pair<Var, PrimExpr>> &let_bindings) {
    Var loop_var = pipeline_loop->loop_var;
    PrimExpr loop_min = pipeline_loop->min;
    PrimExpr loop_extent = pipeline_loop->extent;
    PrimExpr linear_idx = loop_var - loop_min;

    PrimExpr stage_expr = FloorMod(linear_idx, num_stages);
    PrimExpr parity_expr = FloorMod(FloorDiv(linear_idx, num_stages), 2);

    PrimExpr consumer_extent = thread_iv_->dom->extent;
    // Producer warp group is always 128 threads (one warp group).
    // SIMT copies run on these 128 threads — they don't need the full
    // consumer extent because the copy loop iteration space is distributed
    // across the available threads via T.Parallel.
    PrimExpr producer_extent = IntImm(DataType::Int(32), 128);

    bool has_simt_producer = false;
    int num_producer_groups = 0;
    for (auto k : kinds) {
      if (k == TileStmtKind::kTmaProducer)
        ++num_producer_groups;
      if (k == TileStmtKind::kSimtProducer)
        has_simt_producer = true;
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
    //   - Mixed TMA+SIMT: producer_extent (128) — all producer threads arrive
    //     after both TMA and SIMT copies complete
    PrimExpr fwd_arrive_count =
        has_simt_producer ? producer_extent  // 128
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
    //   bp_wait → TMA copies (leader gated) → SIMT copies (all threads) → fwd arrive
    // We group all producers under the LAST TMA group's barrier.
    Array<Stmt> producer_stmts;
    int tma_idx = 0;
    int last_tma_idx = num_producer_groups - 1;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] == TileStmtKind::kTmaProducer) {
        int fwd_base = tma_idx * num_stages;
        int bp_base = num_fwd + tma_idx * num_stages;
        PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) + stage_expr;
        PrimExpr bp_id = IntImm(DataType::Int(32), bp_base) + stage_expr;

        // Back-pressure wait
        producer_stmts.push_back(MakeParityWait(
            barrier_buf, bp_id,
            bitwise_xor(parity_expr, IntImm(DataType::Int(32), 1))));
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
      }
      // Consumer/Other statements are skipped in producer.
    }
    // When SIMT producers present: all producer threads arrive on all fwd barriers
    if (has_simt_producer) {
      for (int g = 0; g < num_producer_groups; ++g) {
        int fwd_base = g * num_stages;
        PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) + stage_expr;
        producer_stmts.push_back(MakeArriveBarrier(barrier_buf, fwd_id));
      }
    }

    // --- Build consumer body ---
    Array<Stmt> consumer_stmts;
    // Forward waits before first consumer stmt.
    for (int g = 0; g < num_producer_groups; ++g) {
      int fwd_base = g * num_stages;
      PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) + stage_expr;
      consumer_stmts.push_back(
          MakeParityWait(barrier_buf, fwd_id, parity_expr));
    }
    // Consumer statements (everything not a producer)
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (!IsProducer(kinds[i])) {
        consumer_stmts.push_back(StripInstructionKind(flat_stmts[i]));
      }
    }
    // Back-pressure arrives
    for (int g = 0; g < num_producer_groups; ++g) {
      int bp_base = num_fwd + g * num_stages;
      PrimExpr bp_id = IntImm(DataType::Int(32), bp_base) + stage_expr;
      consumer_stmts.push_back(MakeArriveBarrier(barrier_buf, bp_id));
    }

    // --- Wrap with let bindings ---
    auto wrap_lets = [&](Stmt body) -> Stmt {
      for (auto it = let_bindings.rbegin(); it != let_bindings.rend(); ++it) {
        body = LetStmt(it->first, it->second, body);
      }
      return body;
    };

    Stmt producer_body = wrap_lets(SeqStmt(producer_stmts));
    Stmt consumer_body = wrap_lets(SeqStmt(consumer_stmts));

    // --- Build loops (strip pipeline annotations) ---
    Map<String, Any> loop_annos;
    for (const auto &[key, value] : pipeline_loop->annotations) {
      if (key != "num_stages" && key != "tl_pipeline_order" &&
          key != "tl_pipeline_stage" && key != "software_pipeline_order" &&
          key != "software_pipeline_stage") {
        loop_annos.Set(key, value);
      }
    }

    For producer_loop(loop_var, loop_min, loop_extent, ForKind::kSerial,
                      producer_body, Optional<IterVar>(), loop_annos);
    For consumer_loop(loop_var, loop_min, loop_extent, ForKind::kSerial,
                      consumer_body, Optional<IterVar>(), loop_annos);

    // --- Rewrite threadIdx.x for producer partition ---
    // Producer: threadIdx.x - consumer_extent (maps to [0, producer_extent))
    Stmt rewritten_producer = PCThreadIdxRewriter::Rewrite(
        producer_loop, thread_iv_->var,
        thread_iv_->var - consumer_extent, producer_extent, false);
    // Consumer: threadIdx.x stays, but extent is consumer_extent
    Stmt rewritten_consumer = consumer_loop;

    // --- WS if/else structure ---
    Stmt ws_body = IfThenElse(GE(thread_iv_->var, consumer_extent),
                              rewritten_producer, rewritten_consumer);

    // --- Handle pre-loop and post-loop statements ---
    // Split the block body into pre-loop, loop, post-loop.
    Array<Stmt> pre_loop, post_loop;
    bool found_loop = false;
    Array<Stmt> block_stmts;
    FlattenSeqStmt(orig_block->body, &block_stmts);
    for (const Stmt &s : block_stmts) {
      if (s.get() == pipeline_loop) {
        found_loop = true;
      } else if (!found_loop) {
        pre_loop.push_back(s);
      } else {
        post_loop.push_back(s);
      }
    }

    // Post-loop runs on consumer threads only.
    Array<Stmt> final_stmts;
    for (const auto &s : pre_loop) {
      final_stmts.push_back(s);
    }
    final_stmts.push_back(ws_body);
    if (!post_loop.empty()) {
      Stmt post = post_loop.size() == 1 ? post_loop[0] : SeqStmt(post_loop);
      final_stmts.push_back(
          IfThenElse(LT(thread_iv_->var, consumer_extent), post));
    }

    Stmt new_block_body =
        final_stmts.size() == 1 ? final_stmts[0] : SeqStmt(final_stmts);

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

    // --- Update thread extent ---
    num_threads_ = consumer_extent + producer_extent;
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
    return nullptr;
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
  Optional<PrimExpr> num_threads_;
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
    if (op->attr_key == attr::kWarpSpecializationScope) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool found_{false};
};

/// Quick pre-scan: check if the function contains a pipelined loop (num_stages
/// > 1) with at least one TMA-annotated tile op and no manual layout
/// annotations (which are incompatible with early MVB expansion).
class TiledWSCandidate : public StmtExprVisitor {
public:
  static bool Check(const Stmt &stmt) {
    TiledWSCandidate c;
    c(stmt);
    return c.has_pipeline_loop_ && c.has_tma_tile_op_ &&
           !c.has_manual_layout_;
  }

private:
  void VisitStmt_(const ForNode *op) final {
    bool old = in_pipeline_;
    if (auto anno = op->annotations.Get("num_stages")) {
      if (auto *imm = anno->as<IntImmNode>()) {
        if (imm->value > 1) {
          has_pipeline_loop_ = true;
          in_pipeline_ = true;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
    in_pipeline_ = old;
  }

  void VisitExpr_(const CallNode *op) final {
    if (in_pipeline_) {
      if (auto kind = op->annotations.Get(kInstructionKind)) {
        if (auto *str = kind->as<StringImmNode>()) {
          if (str->value == "tma") {
            has_tma_tile_op_ = true;
          }
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

  bool in_pipeline_{false};
  bool has_pipeline_loop_{false};
  bool has_tma_tile_op_{false};
  bool has_manual_layout_{false};
};

} // namespace

// Forward-declare MultiVersionBuffer (defined in multi_version_buffer_rewriter.cc).
tvm::transform::Pass MultiVersionBuffer();

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
    // Only apply MultiVersionBuffer + WS if the function actually has a
    // pipelined loop with TMA-annotated tile ops.  Running MultiVersionBuffer
    // on other functions would break manually annotated layouts.
    if (!TiledWSCandidate::Check(f->body)) {
      return f;
    }
    // Expand shared buffers for pipelining before the WS split.
    // We call MultiVersionBuffer here (not at the pipeline level) to avoid
    // expanding buffers in functions that won't be WS-transformed.
    //
    // Keep the original function so we can fall back if the WS
    // transformation doesn't actually fire (e.g. the loop body has
    // statements the rewriter can't handle).  Returning a
    // multi-versioned-but-not-WS-transformed function would cause
    // OptimizeForTarget's MultiVersionBuffer to double-expand buffers.
    PrimFunc original_f = f;
    {
      IRModule tmp_mod;
      tmp_mod->Add(GlobalVar("main"), f);
      tmp_mod = MultiVersionBuffer()(tmp_mod);
      f = Downcast<PrimFunc>(tmp_mod->Lookup("main"));
    }
    PrimFunc result =
        ProducerConsumerWSTiledRewriter::Substitute(std::move(f));
    // Check if the rewriter actually transformed the function.
    if (!result->HasNonzeroAttr(kTiledWSApplied)) {
      return original_f;
    }
    return result;
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.ProducerConsumerWarpSpecializedTiled", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ProducerConsumerWarpSpecializedTiled",
                        ProducerConsumerWarpSpecializedTiled);
}

} // namespace tl
} // namespace tvm
