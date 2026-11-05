/*!
 * \file producer_consumer_ws.cc
 * \brief Warp-specialized producer/consumer rewriting at the tile-op level.
 *
 * This pass runs **before** LayoutInference and LowerTileOp, operating on
 * high-level tile ops (`tl.tileop.copy`, `tl.tileop.gemm`, etc.).
 * It recognizes pipelined producer/consumer structure directly from tile-op
 * semantics and splits eligible loops into warp-specialized branches with
 * explicit barrier synchronization.
 *
 * The output IR is equivalent to a hand-written warp-specialized kernel:
 *   - TMA-annotated copies become `tl.tileop.tma_copy` with barrier refs
 *   - Barriers (`mbarrier_wait_parity`, `ptx_arrive_barrier`) are inserted
 *   - The loop body is wrapped in `if (threadIdx.x < producer_extent)`
 *
 * Guarded and persistent pipelines keep per-role phase counters. Mixed
 * TMA/SIMT/cp.async producers share the same completion protocol, while
 * pre-loop TMA copies wait at their first consumer. One pipelined loop is
 * selected per kernel; explicit WS regions retain their separate scheduler.
 */

#include "support/check.h"
#include <algorithm>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <utility>

#include "backend/common/target_utils.h"
#include "cuda/op/builtin.h"
#include "cuda/op/copy.h"
#include "layout/cute_layout.h"
#include "multi_version_buffer_rewriter.h"
#include "op/copy.h"
#include "op/fill.h"
#include "op/gemm.h"
#include "op/gemm_blockscaled.h"
#include "op/operator.h"
#include "op/utils.h"
#include "transform/common/mbarrier.h"
#include "ws_analysis.h"
#include "ws_native.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;

namespace {
SeqStmt PCSequence(const Stmt &stmt) { return WSSequence({stmt}, stmt->loc); }

If PCIf(PrimExpr condition, Stmt then_case,
        Optional<Stmt> else_case = std::nullopt, Location loc = UnknownLoc()) {
  Optional<SeqStmt> other;
  if (else_case)
    other = PCSequence(*else_case);
  return If(std::move(condition), PCSequence(then_case), other, loc);
}

// ---------------------------------------------------------------------------
// Utility: flatten SeqStmt recursively
// ---------------------------------------------------------------------------
void FlattenSeqStmt(const Stmt &s, Array<Stmt> *out) {
  if (auto *seq = s.as<SeqStmtObj>()) {
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
  TensorVar buf;
  static PhaseCounter Create(const std::string &name) {
    return {
        WSLocalTensor({IntImm::Int32(1)}, PrimType::Int(32), name, "local")};
  }
  PrimExpr Load() const { return TensorLoad(buf, {IntImm::Int32(0)}); }
  Stmt Init() const {
    return TensorStore(buf, {IntImm::Int32(0)}, IntImm::Int32(0));
  }
  Stmt Increment() const {
    return TensorStore(buf, {IntImm::Int32(0)}, Load() + 1);
  }
  Stmt WrapLoopWithAlloc(Stmt loop) const {
    return WSSequence({WSAllocate(buf), Init(), std::move(loop)});
  }
  Stmt AllocOnly() const { return WSAllocate(buf); }
  Stmt InitStmt() const { return Init(); }
  PrimExpr StageExpr(int stages) const { return floormod(Load(), stages); }
  PrimExpr ParityExpr(int stages) const {
    return floormod(floordiv(Load(), stages), 2);
  }
};

// ---------------------------------------------------------------------------
// MVBStageIndexReplacer: rewrite compiler-generated version indices
// ---------------------------------------------------------------------------
class MVBStageIndexReplacer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit MVBStageIndexReplacer(Optional<PrimExpr> replacement)
      : replacement_(std::move(replacement)) {}
  static Stmt Replace(const Stmt &stmt, Optional<PrimExpr> replacement) {
    auto rewriter = make_object<MVBStageIndexReplacer>(std::move(replacement));
    return rewriter->Mutate(stmt).ValueOrUnchanged(stmt);
  }

private:
  UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    if (!call->op.same_as(mvb_stage_index()))
      return StmtExprMutator::Mutate_(call, mode);
    ICHECK_EQ(call->args.size(), 1U);
    if (replacement_)
      return cast(*replacement_, call->ty.as_or_throw<PrimType>());
    return Mutate(call->args[0]).ValueOrUnchanged(call->args[0]);
  }
  Optional<PrimExpr> replacement_;
};

// ---------------------------------------------------------------------------
// Statement classification
// ---------------------------------------------------------------------------

// BufferLayoutMap / TileStmtKind and the statement classifiers live in
// ws_analysis.h, shared with the automatic schedulers.
struct WSMemoryContext {
  const PipelineMemoryAnalysis *memory{nullptr};
  Target target{ffi::UnsafeInit{}};
};
using TensorVarSet =
    std::unordered_set<TensorVar, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using VarExprMap = std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual>;
using StmtRewriteMap =
    std::unordered_map<Stmt, Stmt, ObjectPtrHash, ObjectPtrEqual>;

struct LocalAccessSummary {
  TensorVarSet read_buffers, write_buffers;
  VarSet read_vars, def_vars;
  bool HasTrackedDefs() const {
    return !write_buffers.empty() || !def_vars.empty();
  }
};

struct LocalLiveSet {
  TensorVarSet buffers;
  VarSet vars;
  bool NeedsAnyDef(const LocalAccessSummary &summary) const {
    for (const auto &buffer : summary.write_buffers)
      if (buffers.count(buffer))
        return true;
    for (const auto &var : summary.def_vars)
      if (vars.count(var))
        return true;
    return false;
  }
  void AddUses(const LocalAccessSummary &summary) {
    buffers.insert(summary.read_buffers.begin(), summary.read_buffers.end());
    vars.insert(summary.read_vars.begin(), summary.read_vars.end());
  }
};

static void MergeLocalAccessSummary(LocalAccessSummary *dst,
                                    const LocalAccessSummary &src) {
  dst->read_buffers.insert(src.read_buffers.begin(), src.read_buffers.end());
  dst->write_buffers.insert(src.write_buffers.begin(), src.write_buffers.end());
  dst->read_vars.insert(src.read_vars.begin(), src.read_vars.end());
  dst->def_vars.insert(src.def_vars.begin(), src.def_vars.end());
}

static TensorVar CloneBranchPrivateTensorVar(const TensorVar &tensor,
                                             const std::string &suffix) {
  auto type = make_object<TensorTypeObj>(*tensor.ty().get());
  if (IsFragmentBuffer(tensor))
    type->scope = runtime::MemoryScope::Get("local");
  return TensorVar(tensor->name + suffix, TensorType(type), tensor->loc);
}

class LocalAccessCollector : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  explicit LocalAccessCollector(const WSMemoryContext &context)
      : context_(context) {}
  static LocalAccessSummary Collect(const Stmt &stmt,
                                    const WSMemoryContext &context) {
    auto collector = make_object<LocalAccessCollector>(context);
    collector->Visit(stmt);
    auto accesses =
        make_object<PipelineAccessCollector>(context.memory, context.target);
    accesses->Visit(stmt);
    auto add = [&](const Array<TensorRegion> &regions, TensorVarSet *into) {
      for (const auto &region : regions) {
        auto tensor = RegionTensor(context.memory->CanonicalRegion(region));
        if (IsFragmentBuffer(tensor) || IsLocalBuffer(tensor, true))
          into->insert(tensor);
      }
    };
    add(accesses->GetReads(), &collector->summary_.read_buffers);
    add(accesses->GetWrites(), &collector->summary_.write_buffers);
    return std::move(collector->summary_);
  }

private:
  Optional<VisitInterrupt> Visit_(const SeqStmtObj *node) final {
    auto saved = bound_;
    for (const auto &stmt : node->seq)
      Visit(stmt);
    bound_ = std::move(saved);
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const BindObj *node) final {
    Visit(node->value);
    Visit(node->var->ty);
    summary_.def_vars.insert(node->var);
    bound_.insert(node->var);
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const ForObj *node) final {
    Visit(node->begin);
    Visit(node->end);
    Visit(node->step);
    Visit(node->var->ty);
    bound_.insert(node->var);
    Visit(node->body);
    bound_.erase(node->var);
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const RegionStmtObj *node) final {
    for (const auto &arg : node->args)
      Visit(arg);
    auto saved = bound_;
    for (const auto &var : node->body_params) {
      Visit(var->ty);
      bound_.insert(var);
    }
    Visit(node->body);
    bound_ = std::move(saved);
    for (const auto &var : node->result_vars) {
      Visit(var->ty);
      bound_.insert(var);
      summary_.def_vars.insert(var);
    }
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const VarObj *node) final {
    Var var = GetRef<Var>(node);
    if (bound_.count(var))
      return std::nullopt;
    auto root = context_.memory->Root(var);
    // Allocation/parameter tensors are memory objects; aliases still carry
    // a scalar definition dependency that must move with their consumer.
    if (!(root && root.value().same_as(var)))
      summary_.read_vars.insert(var);
    return StmtExprVisitor::Visit_(node);
  }
  const WSMemoryContext &context_;
  LocalAccessSummary summary_;
  VarSet bound_;
};

enum class PreludeStmtPlacement : uint8_t {
  kKeepSharedPrelude,
  kProducerOnly,
  kConsumerOnly,
  kDuplicateToBoth,
};

static PreludeStmtPlacement
ClassifyPreludeStmt(const Stmt &stmt, const WSMemoryContext &buffer_map,
                    const LocalLiveSet &shared_live_seed,
                    const LocalLiveSet &producer_live_seed,
                    const LocalLiveSet &consumer_live_seed) {
  LocalAccessSummary summary = LocalAccessCollector::Collect(stmt, buffer_map);
  if (!summary.HasTrackedDefs()) {
    return PreludeStmtPlacement::kKeepSharedPrelude;
  }

  if (shared_live_seed.NeedsAnyDef(summary)) {
    return PreludeStmtPlacement::kKeepSharedPrelude;
  }

  bool producer_needs = producer_live_seed.NeedsAnyDef(summary);
  bool consumer_needs = consumer_live_seed.NeedsAnyDef(summary);
  if (producer_needs && consumer_needs) {
    return PreludeStmtPlacement::kDuplicateToBoth;
  }
  if (producer_needs) {
    return PreludeStmtPlacement::kProducerOnly;
  }
  if (consumer_needs) {
    return PreludeStmtPlacement::kConsumerOnly;
  }
  return PreludeStmtPlacement::kKeepSharedPrelude;
}

static bool HasGlobalToSharedCopyShape(const CopyNode *copy) {
  return copy != nullptr && IsGlobalBuffer(copy->src) &&
         IsSharedBuffer(copy->dst) && copy->src.dtype() == copy->dst.dtype();
}

static bool CheckPipelineManagedCPAsyncCopy(const CopyNode *copy,
                                            Target target) {
  if (copy == nullptr) {
    return false;
  }
  return cuda::IsPipelineManagedCPAsyncCopy(*copy, target);
}

static bool IsSyncGlobalToSharedCopyLikeStmt(const Stmt &stmt, Target target) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!call.has_value()) {
    return false;
  }
  auto tile_op = ParseOperator(call.value());
  if (!tile_op.defined()) {
    return false;
  }
  const auto *copy = tile_op.as<CopyNode>();
  if (copy == nullptr) {
    return false;
  }

  cuda::CopyInstSelection result =
      cuda::ClassifyWarpSpecializedCopy(*copy, target);
  return HasGlobalToSharedCopyShape(copy) && result.supported &&
         !cuda::CopyInstIsTMALoad(result.inst) &&
         !cuda::CopyInstIsTMAStore(result.inst) &&
         !cuda::CopyInstIsCPAsync(result.inst);
}

static bool IsProducerMovableLoopPrefixStmt(const Stmt &stmt, Target target) {
  if (IsSyncGlobalToSharedCopyLikeStmt(stmt, target)) {
    return true;
  }

  bool has_allowed_work = false;
  bool has_disallowed = false;
  VisitNativePostOrder(stmt, [&](const ObjectRef &node) {
    if (has_disallowed) {
      return;
    }
    if (const auto *call = node.as<CallObj>()) {
      if ((call->op.same_as(Op::Get("tirx.gpu_storage_sync")) ||
           call->op.same_as(Op::Get("tl.storage_sync")))) {
        const auto *scope = call->args[0].as<StringImmObj>();
        if (!scope ||
            (scope->value != "shared" && scope->value != "shared.dyn")) {
          has_disallowed = true;
          return;
        }
        has_allowed_work = true;
        return;
      }
      if (IsBarrierOrTmaControlCall(call)) {
        has_disallowed = true;
        return;
      }
    }
    if (const auto *ld = node.as<TensorLoadObj>()) {
      if (IsSharedBuffer(ld->source.as_or_throw<TensorVar>()) ||
          IsLocalBuffer(ld->source.as_or_throw<TensorVar>(), true)) {
        has_disallowed = true;
        return;
      }
      if (IsGlobalBuffer(ld->source.as_or_throw<TensorVar>())) {
        has_allowed_work = true;
      }
    }
    if (const auto *st = node.as<TensorStoreObj>()) {
      if (IsSharedBuffer(st->dest.as_or_throw<TensorVar>())) {
        has_allowed_work = true;
        return;
      }
      has_disallowed = true;
    }
  });
  return has_allowed_work && !has_disallowed;
}

// ---------------------------------------------------------------------------
// Helpers: create barrier IR nodes
// ---------------------------------------------------------------------------

static Stmt MakeParityWait(const TensorVar &barrier_buf, PrimExpr barrier_id,
                           PrimExpr parity) {
  auto ref = MakeBarrierRef(barrier_buf, std::move(barrier_id));
  return Evaluate(
      Call(PrimType::Void(), mbarrier_wait_parity(), {ref, std::move(parity)}));
}

static Stmt MakeArriveBarrier(const TensorVar &barrier_buf,
                              PrimExpr barrier_id) {
  auto ref = MakeBarrierRef(barrier_buf, std::move(barrier_id));
  return Evaluate(Call(PrimType::Void(), tl::ptx_arrive_barrier(), {ref}));
}

// ---------------------------------------------------------------------------
// Convert tl.tileop.copy → tl.tileop.tma_copy with barrier annotation
// ---------------------------------------------------------------------------

/// Rewrite a `tl.tileop.copy` Call into a `tl.tileop.tma_copy` Call with
/// barrier reference.  The args (src/dst regions) are preserved; only the op
/// and annotations change.
static Expr RewriteCopyToTmaCopy(const Call &copy_call,
                                 const TensorVar &barrier_buf,
                                 PrimExpr barrier_id) {
  auto annotations = ReadWSCall(copy_call.get()).annotations;
  annotations.Set("barrier", MakeBarrierRef(barrier_buf, barrier_id));
  annotations.Set("is_tma_copy", IntImm::Int32(1));
  return RewriteWSCall(copy_call.get(), annotations,
                       Op::Get("tl.tileop.tma_copy"));
}

/// Annotate SIMT producer statements so the enclosing transform owns cp.async
/// synchronization.
/// - ForNodes get `kParallelAsyncWithoutAsyncCommitWait = true` so
///   InjectPTXAsyncCopy does not emit commit_group + wait_group(0).
/// - Tile-op copy calls get `kAsyncCopyNoImplicitCommitWait` so copy.cc does
///   not emit its own implicit commit/wait either.
/// This allows the WS pass to emit its own commit_group +
/// async arrival, tying cp.async completion to the forward mbarrier.
class SimtProducerAnnotator : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit SimtProducerAnnotator(Target target) : target_(std::move(target)) {}
  static Stmt Annotate(const Stmt &stmt, Target target) {
    auto mutator = make_object<SimtProducerAnnotator>(std::move(target));
    return mutator->Mutate(stmt).ValueOrUnchanged(stmt);
  }

private:
  UnchangedOr<Stmt> Mutate_(const ForObj *loop, InplaceMode mode) final {
    auto rewritten = StmtExprMutator::Mutate_(loop, mode)
                         .ValueOrUnchanged(GetRef<For>(loop))
                         .as_or_throw<For>();
    auto annotations = rewritten->annotations;
    annotations.Set(attr::kParallelAsyncWithoutAsyncCommitWait, true);
    return For(rewritten->var, rewritten->begin, rewritten->end,
               rewritten->step, rewritten->kind, rewritten->body, annotations,
               rewritten->loc);
  }
  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    Call call = StmtExprMutator::Mutate_(op, mode)
                    .ValueOrUnchanged(GetRef<Call>(op))
                    .as_or_throw<Call>();
    if (!call->op.same_as(Op::Get("tl.tileop.copy")))
      return call;
    auto operation = ParseOperator(call);
    auto copy = operation.as<CopyNode>();
    if (!copy || !CheckPipelineManagedCPAsyncCopy(copy, target_))
      return call;
    auto annotations = ReadWSCall(call.get()).annotations;
    annotations.Set(attr::kAsyncCopyNoImplicitCommitWait, IntImm::Int32(1));
    return RewriteWSCall(call.get(), annotations);
  }
  Target target_;
};

class TileOpMbarPhaseAnnotator : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit TileOpMbarPhaseAnnotator(PrimExpr phase)
      : phase_(std::move(phase)) {}
  static Stmt Annotate(const Stmt &stmt, PrimExpr phase) {
    auto mutator = make_object<TileOpMbarPhaseAnnotator>(std::move(phase));
    return mutator->Mutate(stmt).ValueOrUnchanged(stmt);
  }

private:
  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    Call call = StmtExprMutator::Mutate_(op, mode)
                    .ValueOrUnchanged(GetRef<Call>(op))
                    .as_or_throw<Call>();
    auto operation = ParseOperator(call);
    if (!operation.defined() ||
        !(operation.as<CopyNode>() || operation.as<Im2ColOpNode>() ||
          operation.as<GemmNode>()))
      return call;
    auto annotations = ReadWSCall(call.get()).annotations;
    if (annotations.count(attr::kPipelineMbarPhaseExpr))
      return call;
    annotations.Set(attr::kPipelineMbarPhaseExpr, phase_);
    return RewriteWSCall(call.get(), annotations);
  }
  PrimExpr phase_;
};

static Expr AnnotateTileOpBarrier(const Call &call, const TensorVar &barrier,
                                  PrimExpr slot) {
  auto annotations = ReadWSCall(call.get()).annotations;
  annotations.Set("barrier", MakeBarrierRef(barrier, slot));
  return RewriteWSCall(call.get(), annotations);
}

struct TensorVarDataAccessInfo {
  bool read{false};
  bool write{false};

  bool HasAnyAccess() const { return read || write; }
};

struct PreludeTmaLoadPlan {
  Stmt stmt;
  int wait_pos{-1};
};

static TensorVarDataAccessInfo
AnalyzeTensorVarDataAccess(const Stmt &stmt, const Var &buffer,
                           const WSMemoryContext &context) {
  auto collector =
      make_object<PipelineAccessCollector>(context.memory, context.target);
  collector->Visit(stmt);
  TensorVarDataAccessInfo result;
  auto root = context.memory->Root(buffer);
  auto matches = [&](const TensorRegion &region) {
    auto other = context.memory->Root(region->source);
    return !root || !other || root.value().same_as(*other);
  };
  for (const auto &read : collector->GetReads())
    result.read |= matches(read);
  for (const auto &write : collector->GetWrites())
    result.write |= matches(write);
  result.read |= collector->HasUnknownAccess();
  result.write |= collector->HasUnknownAccess();
  return result;
}

class PreludeStmtsToPipelineLoopCollector
    : public StmtFunctor<Optional<Array<Stmt>>(const Stmt &)> {
public:
  using Base = StmtFunctor<Optional<Array<Stmt>>(const Stmt &)>;

  explicit PreludeStmtsToPipelineLoopCollector(For pipeline_loop)
      : pipeline_loop_(std::move(pipeline_loop)) {}

  Optional<Array<Stmt>> Dispatch(const Stmt &stmt) final {
    if (stmt.same_as(pipeline_loop_)) {
      return Array<Stmt>();
    }
    return Base::Dispatch(stmt);
  }

  Optional<Array<Stmt>> Dispatch_(const SeqStmtObj *op) final {
    for (int i = 0; i < static_cast<int>(op->seq.size()); ++i) {
      Optional<Array<Stmt>> nested_prelude = Dispatch(op->seq[i]);
      if (!nested_prelude.has_value()) {
        continue;
      }

      Array<Stmt> prelude_stmts;
      for (int j = 0; j < i; ++j) {
        prelude_stmts.push_back(op->seq[j]);
      }
      prelude_stmts.insert(prelude_stmts.end(), nested_prelude.value().begin(),
                           nested_prelude.value().end());
      return prelude_stmts;
    }
    return Optional<Array<Stmt>>();
  }

  Optional<Array<Stmt>> Dispatch_(const RegionStmtObj *op) final {
    return Dispatch(op->body);
  }

  Optional<Array<Stmt>> Dispatch_(const IfObj *op) final {
    Optional<Array<Stmt>> then_prelude = Dispatch(op->then_case);
    if (then_prelude.has_value()) {
      return then_prelude;
    }
    if (op->else_case.has_value()) {
      return Dispatch(op->else_case.value());
    }
    return Optional<Array<Stmt>>();
  }

  Optional<Array<Stmt>> DispatchDefault_(const Object *) final {
    return Optional<Array<Stmt>>();
  }

private:
  For pipeline_loop_;
};

static Array<Stmt> CollectPreludeStmtsToPipelineLoop(const Stmt &stmt,
                                                     const For &pipeline_loop) {
  PreludeStmtsToPipelineLoopCollector collector(pipeline_loop);
  Optional<Array<Stmt>> prelude_stmts = collector(stmt);
  return prelude_stmts.has_value() ? prelude_stmts.value() : Array<Stmt>();
}

static Optional<Var> ExtractProducerWriteTensorVarData(const Stmt &stmt) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!call.has_value()) {
    return Optional<Var>();
  }
  auto tile_op = ParseOperator(call.value());
  if (!tile_op.defined()) {
    return Optional<Var>();
  }
  if (const auto *copy = tile_op.as<CopyNode>()) {
    if (IsSharedBuffer(copy->dst)) {
      return copy->dst;
    }
  }
  if (const auto *im2col = tile_op.as<Im2ColOpNode>()) {
    if (IsSharedBuffer(im2col->dst_)) {
      return im2col->dst_;
    }
  }
  return Optional<Var>();
}

static int FindFirstAsyncProducerConsumerRead(const Stmt &producer,
                                              const Array<Stmt> &consumers,
                                              const WSMemoryContext &context) {
  auto collector =
      make_object<PipelineAccessCollector>(context.memory, context.target);
  collector->Visit(producer);
  if (collector->HasUnknownAccess())
    return 0;
  int earliest = consumers.size();
  for (const auto &region : collector->GetWrites()) {
    auto tensor = RegionTensor(context.memory->CanonicalRegion(region));
    if (!IsSharedBuffer(tensor))
      continue;
    for (int i = 0; i < earliest; ++i)
      if (AnalyzeTensorVarDataAccess(consumers[i], tensor, context).read) {
        earliest = i;
        break;
      }
  }
  return earliest;
}

static Stmt RewritePreludeTmaProducerStmt(const Stmt &stmt,
                                          const TensorVar &barrier,
                                          PrimExpr slot) {
  class Rewriter : public StmtExprMutator {
  public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    Rewriter(TensorVar barrier, PrimExpr slot)
        : barrier_(std::move(barrier)), slot_(std::move(slot)) {}
    UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
      Call call = StmtExprMutator::Mutate_(op, mode)
                      .ValueOrUnchanged(GetRef<Call>(op))
                      .as_or_throw<Call>();
      if (rewritten_)
        return call;
      auto operation = ParseOperator(call);
      if (!operation.defined() ||
          !(operation.as<CopyNode>() || operation.as<Im2ColOpNode>()))
        return call;
      Expr rewritten = operation.as<CopyNode>()
                           ? RewriteCopyToTmaCopy(call, barrier_, slot_)
                           : AnnotateTileOpBarrier(call, barrier_, slot_);
      Call next = rewritten.as_or_throw<Call>();
      auto annotations = ReadWSCall(next.get()).annotations;
      annotations.Set("emit_arrive", IntImm::Int32(1));
      rewritten_ = true;
      return RewriteWSCall(next.get(), annotations);
    }
    TensorVar barrier_;
    PrimExpr slot_;
    bool rewritten_ = false;
  };
  auto rewriter = make_object<Rewriter>(barrier, slot);
  return rewriter->Mutate(stmt).ValueOrUnchanged(stmt);
}

// ---------------------------------------------------------------------------
// Main rewriter
// ---------------------------------------------------------------------------

class ProducerConsumerWSRewriter {
public:
  static tirx::Function Substitute(tirx::Function function) {
    bool changed = false;
    auto result = RewriteNativeWSKernels(
        function,
        [&](NativeWSKernel kernel, PrimVar thread,
            Target target) -> Optional<NativeWSResult> {
          ProducerConsumerWSRewriter rewriter(target, thread, kernel.threads,
                                              kernel.memory);
          auto output = rewriter.RewriteKernel(std::move(kernel));
          changed |= output.has_value();
          return output;
        });
    if (changed) {
      auto attrs = result->attrs->dict;
      attrs.Set(kTiledWSApplied, int64_t{1});
      result.CopyOnWrite()->attrs = DictAttrs(attrs);
      result = tirx::RenewDef(result);
    }
    return result;
  }

private:
  ProducerConsumerWSRewriter(Target target, PrimVar thread, PrimExpr extent,
                             const PipelineMemoryAnalysis *memory)
      : target_(target), thread_var_(thread), thread_extent_(extent),
        buffer_data_to_buffer_{memory, target} {}

  Optional<NativeWSResult> RewriteKernel(NativeWSKernel kernel) {
    auto loop_result = FindPipelineLoopWithNesting(kernel.body);
    if (!loop_result.pipeline_loop)
      return std::nullopt;
    For pipeline_loop = *loop_result.pipeline_loop;
    bool pipeline_under_outer_for = loop_result.under_outer_for;
    auto count = GetPipelineNumStages(pipeline_loop.get());
    if (!count || *count < 1)
      return std::nullopt;
    int num_stages = *count;

    // Flatten the loop body.
    Array<Stmt> flat_stmts;
    Stmt loop_body = pipeline_loop->body;
    loop_body = SingleStatement(loop_body);
    // Peel leading BindNodes that dominate the whole loop body.
    std::vector<std::pair<Var, Expr>> outer_leading_bindings;
    if (const auto *seq = loop_body.as<SeqStmtObj>()) {
      size_t start = 0;
      while (start < seq->seq.size()) {
        if (const auto *bind = seq->seq[start].as<BindObj>()) {
          outer_leading_bindings.emplace_back(bind->var, bind->value);
          ++start;
        } else {
          break;
        }
      }
      if (start > 0) {
        if (start + 1 == seq->seq.size()) {
          loop_body = seq->seq[start];
        } else {
          Array<Stmt> remaining(seq->seq.begin() + start, seq->seq.end());
          loop_body = SeqStmt(remaining);
        }
      }
    }
    // Unwrap a single If wrapper (no else branch) so that
    // TMA producers inside conditional loop bodies can be classified.
    // Keep leading BindNodes inside the conditional separate so they stay
    // dominated by the original guard after rebuilding WS branches.
    Optional<PrimExpr> loop_body_condition;
    std::vector<std::pair<Var, Expr>> inner_leading_bindings;
    if (const auto *if_stmt = loop_body.as<IfObj>()) {
      if (!if_stmt->else_case.has_value()) {
        // Peel leading BindNodes from inside the conditional body. These
        // bindings must remain inside the guarded region.
        Stmt inner = if_stmt->then_case;
        if (const auto *seq = inner.as<SeqStmtObj>()) {
          size_t start = 0;
          while (start < seq->seq.size()) {
            if (const auto *bind = seq->seq[start].as<BindObj>()) {
              inner_leading_bindings.emplace_back(bind->var, bind->value);
              ++start;
            } else {
              break;
            }
          }
          if (start > 0) {
            if (start + 1 == seq->seq.size()) {
              inner = seq->seq[start];
            } else {
              Array<Stmt> remaining(seq->seq.begin() + start, seq->seq.end());
              inner = SeqStmt(remaining);
            }
          }
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
      return std::nullopt;

    // --- Build the WS transformation ---
    return BuildWSBlock(kernel, pipeline_loop, num_stages, flat_stmts, kinds,
                        outer_leading_bindings, inner_leading_bindings,
                        loop_body_condition, pipeline_under_outer_for);
  }

  NativeWSResult
  BuildWSBlock(NativeWSKernel kernel, const For &pipeline_loop, int num_stages,
               const Array<Stmt> &flat_stmts,
               const std::vector<TileStmtKind> &kinds,
               const std::vector<std::pair<Var, Expr>> &outer_leading_bindings,
               const std::vector<std::pair<Var, Expr>> &inner_leading_bindings,
               Optional<PrimExpr> loop_body_condition = Optional<PrimExpr>(),
               bool pipeline_under_outer_for = false) {
    PrimVar loop_var = pipeline_loop->var;
    PrimExpr loop_min = pipeline_loop->begin;
    PrimExpr loop_extent = LoopExtent(pipeline_loop.get());
    PrimExpr linear_idx =
        WSPipelineIterationModulo({pipeline_loop.get()}, num_stages);
    PrimExpr stages = IntImm(linear_idx.ty(), num_stages);
    PrimExpr base_stage_expr =
        cast(floormod(linear_idx, stages), PrimType::Int(32));
    PrimExpr two = IntImm(linear_idx.ty(), 2);
    PrimExpr base_parity_expr =
        cast(floormod(floordiv(linear_idx, stages), two), PrimType::Int(32));

    // When to switch from loop-var-based to counter-based stage/parity:
    //   (a) `loop_body_condition.has_value()`: the pipeline body is guarded
    //       and some iterations may skip barrier arrive/wait entirely, so
    //       the loop variable no longer counts real transactions.
    //   (b) `pipeline_under_outer_for`: the pipeline loop is nested
    //       inside an outer `T.Persistent` / `T.serial` scheduler.  The
    //       block-scoped mbarrier phase persists across outer iterations
    //       while the inner loop variable resets every iteration, so a
    //       counter that is live across the whole kernel is required
    //       to keep barrier parity consistent (see PR discussion).
    // When (b) triggers, the counter's allocation is hoisted out of the
    // inner pipeline loop into the kernel's native allocation prelude and
    // its `Init()` becomes a one-shot statement before the outer loop — see the
    // assembly block near the end of this function.
    bool needs_phase_counter =
        loop_body_condition.has_value() || pipeline_under_outer_for;
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

    PrimExpr consumer_extent = thread_extent_;
    PrimExpr producer_extent = IntImm(PrimType::Int(32), 128);
    common_prelude_rewrites_.clear();

    bool has_simt_producer = false;
    bool has_cp_async_producer = false;
    int num_producer_groups = 0;
    for (auto k : kinds) {
      if (k == TileStmtKind::kTmaProducer)
        ++num_producer_groups;
      if (k == TileStmtKind::kSimtProducer)
        has_simt_producer = true;
      if (k == TileStmtKind::kCpAsyncProducer || k == TileStmtKind::kCpAsyncRaw)
        has_cp_async_producer = true;
    }

    // --- Barrier allocation ---
    // Layout: [fwd_0..fwd_{G*S-1}] [bp_0..bp_{G*S-1}]
    // where G = num_producer_groups (one per TMA copy), S = num_stages.
    // When SIMT producers are present, all producer types share the same
    // barrier group — the last forward arrive covers everything.
    int num_fwd = num_producer_groups * num_stages;
    int num_bp = num_producer_groups * num_stages;

    Array<Stmt> consumer_compute_stmts;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (!IsProducer(kinds[i])) {
        consumer_compute_stmts.push_back(flat_stmts[i]);
      }
    }

    Array<Stmt> prelude_stmts =
        CollectPreludeStmtsToPipelineLoop(kernel.body, pipeline_loop);
    std::vector<PreludeTmaLoadPlan> prelude_tma_plans;
    for (const Stmt &stmt : prelude_stmts) {
      if (ClassifyStmt(stmt, target_) != TileStmtKind::kTmaProducer) {
        continue;
      }
      Optional<Var> write_buffer_data = ExtractProducerWriteTensorVarData(stmt);
      if (!write_buffer_data.has_value()) {
        continue;
      }
      int first_read = -1;
      for (size_t ci = 0; ci < consumer_compute_stmts.size(); ++ci) {
        TensorVarDataAccessInfo access = AnalyzeTensorVarDataAccess(
            consumer_compute_stmts[ci], write_buffer_data.value(),
            buffer_data_to_buffer_);
        if (access.read) {
          first_read = static_cast<int>(ci);
          break;
        }
      }
      if (first_read < 0) {
        continue;
      }
      prelude_tma_plans.push_back({stmt, first_read});
    }

    int total_barriers = num_fwd + num_bp + prelude_tma_plans.size();
    TensorVar barrier_buf =
        CreateMBarrierBuffer(injected_mbarrier_name_, total_barriers);
    // arrive_counts are computed later (after producer_extent is finalized).

    std::vector<int> wait_insert_pos(num_producer_groups, 0);
    std::vector<int> arrive_insert_pos(
        num_producer_groups, static_cast<int>(consumer_compute_stmts.size()));
    int access_group_idx = 0;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] != TileStmtKind::kTmaProducer) {
        continue;
      }
      Optional<Var> write_buffer_data =
          ExtractProducerWriteTensorVarData(flat_stmts[i]);
      if (write_buffer_data.has_value()) {
        int first_read = -1;
        int last_access = -1;
        for (size_t ci = 0; ci < consumer_compute_stmts.size(); ++ci) {
          TensorVarDataAccessInfo access = AnalyzeTensorVarDataAccess(
              consumer_compute_stmts[ci], write_buffer_data.value(),
              buffer_data_to_buffer_);
          if (access.read && first_read < 0) {
            first_read = static_cast<int>(ci);
          }
          if (access.HasAnyAccess()) {
            last_access = static_cast<int>(ci);
          }
        }
        if (first_read >= 0) {
          wait_insert_pos[access_group_idx] = first_read;
          arrive_insert_pos[access_group_idx] = last_access + 1;
        } else if (last_access >= 0) {
          wait_insert_pos[access_group_idx] = 0;
          arrive_insert_pos[access_group_idx] = last_access + 1;
        }
      }
      ++access_group_idx;
    }

    // --- Adjust wait positions for SIMT/cp.async producers ---
    // SIMT and cp.async producers tie their completion to all forward barriers.
    // If a consumer reads any such shared destination before the first TMA
    // read, pull all waits earlier so the async producer is also covered.
    if (has_simt_producer || has_cp_async_producer) {
      int earliest_async_read = static_cast<int>(consumer_compute_stmts.size());
      for (size_t i = 0; i < flat_stmts.size(); ++i) {
        if (kinds[i] != TileStmtKind::kSimtProducer &&
            kinds[i] != TileStmtKind::kCpAsyncProducer &&
            kinds[i] != TileStmtKind::kCpAsyncRaw) {
          continue;
        }
        int first_read = FindFirstAsyncProducerConsumerRead(
            flat_stmts[i], consumer_compute_stmts, buffer_data_to_buffer_);
        earliest_async_read = std::min(earliest_async_read, first_read);
      }
      // Pull all wait positions earlier if needed.
      for (int g = 0; g < num_producer_groups; ++g) {
        wait_insert_pos[g] = std::min(wait_insert_pos[g], earliest_async_read);
      }
    }

    // --- Determine if TMA barriers can be merged ---
    // When all pure-TMA producers wait at the same consumer position and
    // release at the same position, forward and back-pressure barriers can
    // be shared across all TMA copies, reducing from 2*G*S to 2*S barriers.
    bool can_merge_tma_barriers = (num_producer_groups > 1) &&
                                  !has_simt_producer && !has_cp_async_producer;
    if (can_merge_tma_barriers) {
      for (int g = 1; g < num_producer_groups; ++g) {
        if (wait_insert_pos[g] != wait_insert_pos[0] ||
            arrive_insert_pos[g] != arrive_insert_pos[0]) {
          can_merge_tma_barriers = false;
          break;
        }
      }
    }
    if (can_merge_tma_barriers) {
      // Re-compute barrier layout with a single merged group.
      num_fwd = num_stages;
      num_bp = num_stages;
      total_barriers = num_fwd + num_bp + prelude_tma_plans.size();
      barrier_buf =
          CreateMBarrierBuffer(injected_mbarrier_name_, total_barriers);
    }

    std::vector<Array<Stmt>> producer_loop_prefix_stmts(num_producer_groups);
    std::vector<bool> moved_compute_stmts(consumer_compute_stmts.size(), false);
    std::vector<LocalAccessSummary> consumer_compute_summaries;
    consumer_compute_summaries.reserve(consumer_compute_stmts.size());
    for (const auto &stmt : consumer_compute_stmts) {
      consumer_compute_summaries.push_back(
          LocalAccessCollector::Collect(stmt, buffer_data_to_buffer_));
    }
    LocalLiveSet producer_body_live;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (IsProducer(kinds[i])) {
        producer_body_live.AddUses(LocalAccessCollector::Collect(
            flat_stmts[i], buffer_data_to_buffer_));
      }
    }
    int compute_cursor = 0;
    for (int ti = 0; ti < num_producer_groups; ++ti) {
      int wait_pos = wait_insert_pos[ti];
      if (wait_pos <= compute_cursor) {
        compute_cursor = std::max(compute_cursor, wait_pos);
        continue;
      }
      bool all_movable = true;
      LocalLiveSet movable_probe_live = producer_body_live;
      for (int ci = compute_cursor; ci < wait_pos; ++ci) {
        const LocalAccessSummary &summary = consumer_compute_summaries[ci];
        if (consumer_compute_stmts[ci].as<BindObj>()) {
          if (!movable_probe_live.NeedsAnyDef(summary)) {
            all_movable = false;
            break;
          }
          movable_probe_live.AddUses(summary);
          continue;
        }
        if (!IsProducerMovableLoopPrefixStmt(consumer_compute_stmts[ci],
                                             target_)) {
          all_movable = false;
          break;
        }
        movable_probe_live.AddUses(summary);
      }
      if (all_movable) {
        std::vector<bool> add_to_producer(consumer_compute_stmts.size(), false);
        LocalLiveSet producer_live = producer_body_live;
        LocalLiveSet consumer_live;
        for (int ci = wait_pos;
             ci < static_cast<int>(consumer_compute_stmts.size()); ++ci) {
          if (!moved_compute_stmts[ci]) {
            consumer_live.AddUses(consumer_compute_summaries[ci]);
          }
        }
        for (int ci = wait_pos - 1; ci >= compute_cursor; --ci) {
          const LocalAccessSummary &summary = consumer_compute_summaries[ci];
          const bool is_bind =
              consumer_compute_stmts[ci].as<BindObj>() != nullptr;
          if (!is_bind) {
            add_to_producer[ci] = true;
            moved_compute_stmts[ci] = true;
            producer_live.AddUses(summary);
            continue;
          }

          bool producer_needs = producer_live.NeedsAnyDef(summary);
          bool consumer_needs = consumer_live.NeedsAnyDef(summary);
          if (producer_needs) {
            add_to_producer[ci] = true;
            producer_live.AddUses(summary);
          }
          if (!producer_needs || consumer_needs) {
            consumer_live.AddUses(summary);
          } else {
            moved_compute_stmts[ci] = true;
          }
        }
        for (int ci = compute_cursor; ci < wait_pos; ++ci) {
          if (add_to_producer[ci]) {
            producer_loop_prefix_stmts[ti].push_back(
                consumer_compute_stmts[ci]);
          }
        }
      }
      compute_cursor = wait_pos;
    }

    bool producer_needs_full_thread_extent = false;
    for (size_t i = 0;
         i < flat_stmts.size() && !producer_needs_full_thread_extent; ++i) {
      if (kinds[i] == TileStmtKind::kSimtProducer ||
          IsSyncGlobalToSharedCopyLikeStmt(flat_stmts[i], target_)) {
        producer_needs_full_thread_extent = true;
      }
    }
    if (!producer_needs_full_thread_extent) {
      for (const auto &prefix_stmts : producer_loop_prefix_stmts) {
        for (const auto &stmt : prefix_stmts) {
          if (IsSyncGlobalToSharedCopyLikeStmt(stmt, target_)) {
            producer_needs_full_thread_extent = true;
            break;
          }
        }
        if (producer_needs_full_thread_extent) {
          break;
        }
      }
    }
    if (producer_needs_full_thread_extent) {
      // LowerTileOp will materialize these producer-side sync copies into
      // explicit SIMT global->shared loops. Keep the producer partition at the
      // original thread extent so the lowered thread mapping stays valid.
      producer_extent = consumer_extent;
    }

    // --- Compute arrive_counts (after producer_extent is finalized) ---
    bool has_async_producer = has_simt_producer || has_cp_async_producer;
    bool has_producer_prefix =
        !outer_leading_bindings.empty() || !inner_leading_bindings.empty();
    for (const auto &prefix : producer_loop_prefix_stmts)
      has_producer_prefix |= !prefix.empty();
    bool producer_is_single_warp = !has_async_producer &&
                                   !has_producer_prefix &&
                                   !loop_body_condition.has_value();
    PrimExpr fwd_arrive_count = producer_is_single_warp
                                    ? IntImm(PrimType::Int(32), 1)
                                    : producer_extent;
    Array<PrimExpr> arrive_counts;
    for (int i = 0; i < num_fwd; ++i) {
      arrive_counts.push_back(fwd_arrive_count);
    }
    for (int i = 0; i < num_bp; ++i) {
      arrive_counts.push_back(consumer_extent);
    }
    for (size_t i = 0; i < prelude_tma_plans.size(); ++i) {
      arrive_counts.push_back(IntImm(PrimType::Int(32), 1));
    }

    std::vector<Array<Stmt>> prelude_waits_before_consumer(
        consumer_compute_stmts.size());
    PrimExpr prelude_wait_guard{ffi::UnsafeInit{}};
    if (needs_phase_counter) {
      prelude_wait_guard = EQ(consumer_phase_counter.value().Load(),
                              IntImm(PrimType::Int(32), 0));
    } else {
      prelude_wait_guard = EQ(loop_var, cast(loop_min, loop_var.ty()));
    }
    int prelude_barrier_base = num_fwd + num_bp;
    for (size_t i = 0; i < prelude_tma_plans.size(); ++i) {
      PrimExpr barrier_id = IntImm(PrimType::Int(32), prelude_barrier_base + i);
      Stmt rewritten_prelude = RewritePreludeTmaProducerStmt(
          prelude_tma_plans[i].stmt, barrier_buf, barrier_id);
      common_prelude_rewrites_.emplace(prelude_tma_plans[i].stmt,
                                       rewritten_prelude);
      int wait_pos = prelude_tma_plans[i].wait_pos;
      ICHECK_GE(wait_pos, 0);
      ICHECK_LT(wait_pos, static_cast<int>(consumer_compute_stmts.size()));
      prelude_waits_before_consumer[wait_pos].push_back(PCIf(
          prelude_wait_guard, MakeParityWait(barrier_buf, barrier_id,
                                             IntImm(PrimType::Int(32), 0))));
    }

    // --- Build producer body ---
    // Producer structure (mixed TMA + SIMT/cp.async):
    //   bp_wait → SIMT copies (all threads, async) → TMA copies (leader) →
    //   completion tracking.
    // SIMT copies are placed after bp_wait but before TMA so cp.async
    // and TMA can overlap.

    // First pass: collect SIMT/cp.async producer stmts separately.
    Array<Stmt> simt_producer_stmts;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] == TileStmtKind::kSimtProducer) {
        // Annotate ForNodes with kParallelAsyncWithoutAsyncCommitWait so
        // InjectPTXAsyncCopy (called from LowerTileOp) does not insert
        // commit+wait — the WS pass will emit its own commit+barrier_noinc.
        simt_producer_stmts.push_back(
            SimtProducerAnnotator::Annotate(flat_stmts[i], target_));
      } else if (kinds[i] == TileStmtKind::kCpAsyncProducer ||
                 kinds[i] == TileStmtKind::kCpAsyncRaw) {
        simt_producer_stmts.push_back(flat_stmts[i]);
      }
    }

    // Second pass: build the producer body with correct ordering.
    Array<Stmt> producer_stmts;
    int tma_idx = 0;
    int last_tma_idx = num_producer_groups - 1;
    bool simt_stmts_emitted = false;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] == TileStmtKind::kTmaProducer) {
        int barrier_group = can_merge_tma_barriers ? 0 : tma_idx;
        int fwd_base = barrier_group * num_stages;
        int bp_base = num_fwd + barrier_group * num_stages;
        PrimExpr fwd_id = IntImm(PrimType::Int(32), fwd_base) + p_stage_expr;
        PrimExpr bp_id = IntImm(PrimType::Int(32), bp_base) + p_stage_expr;

        // Back-pressure wait (only once when barriers are merged)
        if (!can_merge_tma_barriers || tma_idx == 0) {
          producer_stmts.push_back(MakeParityWait(
              barrier_buf, bp_id,
              bitwise_xor(p_parity_expr, IntImm(PrimType::Int(32), 1))));
        }

        // After the first bp_wait, emit all SIMT/cp.async producers
        // followed immediately by commit_group so the hardware can start
        // the async transfers as early as possible, overlapping with TMA.
        if (!simt_stmts_emitted && !simt_producer_stmts.empty()) {
          for (const auto &s : simt_producer_stmts) {
            producer_stmts.push_back(s);
          }
          // Commit cp.async group right after issuing — the earlier the
          // commit, the more overlap with subsequent TMA loads.
          if (has_simt_producer || has_cp_async_producer) {
            producer_stmts.push_back(
                Evaluate(Call(PrimType::Void(), tl::ptx_commit_group(), {})));
          }
          simt_stmts_emitted = true;
        }

        for (const auto &stmt : producer_loop_prefix_stmts[tma_idx]) {
          producer_stmts.push_back(stmt);
        }
        // Convert copy → tma_copy with barrier, or annotate non-copy
        // TMA tile-ops (e.g. im2col) with barrier reference.
        const auto *eval = flat_stmts[i].as<EvaluateObj>();
        ICHECK(eval);
        Call tile_call = eval->value.as_or_throw<Call>();
        auto tile_op = ParseOperator(tile_call);
        Expr tma_call{ffi::UnsafeInit{}};
        // For pure TMA, tell LowerTileOp to emit arrive inside the same
        // tl_shuffle_elect block (via emit_arrive annotation), producing
        // arrive_and_expect_tx instead of separate expect_tx + arrive.
        // When merged barriers, only the last TMA copy should arrive.
        bool emit_arrive_on_this =
            producer_is_single_warp &&
            (!can_merge_tma_barriers || tma_idx == last_tma_idx);

        if (tile_op.defined() && tile_op.as<CopyNode>()) {
          tma_call = RewriteCopyToTmaCopy(tile_call, barrier_buf, fwd_id);
        } else {
          // Non-copy TMA producer (e.g. Im2ColOp): annotate with
          // barrier so Lower() uses the WS barrier instead of its own.
          tma_call = AnnotateTileOpBarrier(tile_call, barrier_buf, fwd_id);
        }
        if (emit_arrive_on_this) {
          auto call = tma_call.as_or_throw<Call>();
          auto annos = ReadWSCall(call.get()).annotations;
          annos.Set("emit_arrive", IntImm(PrimType::Int(32), 1));
          tma_call = RewriteWSCall(call.get(), annos);
        }
        producer_stmts.push_back(Evaluate(tma_call));
        ++tma_idx;
      }
      // SIMT/cp.async producers are handled above (after first bp_wait).
      // Consumer/Other statements are skipped in producer.
    }
    // Fallback: if there were no TMA producers to anchor the bp_wait,
    // emit SIMT stmts now (shouldn't happen in the mixed path).
    if (!simt_stmts_emitted && !simt_producer_stmts.empty()) {
      for (const auto &s : simt_producer_stmts) {
        producer_stmts.push_back(s);
      }
    }
    // Async-only producers already contribute one completion per thread.
    // With a prefix, also publish any synchronous writes by a regular release.
    if (!producer_is_single_warp) {
      for (int fwd_base = 0; fwd_base < num_fwd; fwd_base += num_stages) {
        PrimExpr fwd_id = IntImm(PrimType::Int(32), fwd_base) + p_stage_expr;
        if (!has_async_producer) {
          // No async copy: each thread publishes its writes and arrives.
          producer_stmts.push_back(MakeArriveBarrier(barrier_buf, fwd_id));
        } else if (!has_producer_prefix) {
          // Async copy only: completion consumes the thread's arrival credit.
          producer_stmts.push_back(
              Evaluate(Call(PrimType::Void(), tl::ptx_cp_async_barrier_noinc(),
                            {MakeBarrierRef(barrier_buf, fwd_id)})));
        } else {
          // Async copy with prefix: reserve completion, then publish writes.
          producer_stmts.push_back(
              Evaluate(Call(PrimType::Void(), tl::ptx_cp_async_barrier(),
                            {MakeBarrierRef(barrier_buf, fwd_id)})));
          producer_stmts.push_back(MakeArriveBarrier(barrier_buf, fwd_id));
        }
      }
    }
    // Phase counter increment at end of producer guarded iteration
    if (needs_phase_counter) {
      producer_stmts.push_back(producer_phase_counter.value().Increment());
    }

    // --- Build consumer body ---
    // When barriers are merged, iterate over a single effective group.
    int consumer_barrier_groups =
        can_merge_tma_barriers ? 1 : num_producer_groups;
    Array<Stmt> consumer_stmts;
    std::vector<bool> arrive_emitted(consumer_barrier_groups, false);
    for (size_t ci = 0; ci < consumer_compute_stmts.size(); ++ci) {
      for (const auto &stmt : prelude_waits_before_consumer[ci]) {
        consumer_stmts.push_back(stmt);
      }
      for (int g = 0; g < consumer_barrier_groups; ++g) {
        if (wait_insert_pos[g] == static_cast<int>(ci)) {
          int fwd_base = g * num_stages;
          PrimExpr fwd_id = IntImm(PrimType::Int(32), fwd_base) + c_stage_expr;
          consumer_stmts.push_back(
              MakeParityWait(barrier_buf, fwd_id, c_parity_expr));
        }
      }
      if (!moved_compute_stmts[ci]) {
        consumer_stmts.push_back(consumer_compute_stmts[ci]);
      }
      for (int g = 0; g < consumer_barrier_groups; ++g) {
        if (arrive_insert_pos[g] == static_cast<int>(ci + 1)) {
          int bp_base = num_fwd + g * num_stages;
          PrimExpr bp_id = IntImm(PrimType::Int(32), bp_base) + c_stage_expr;
          consumer_stmts.push_back(MakeArriveBarrier(barrier_buf, bp_id));
          arrive_emitted[g] = true;
        }
      }
    }
    if (consumer_compute_stmts.empty()) {
      for (int g = 0; g < consumer_barrier_groups; ++g) {
        int fwd_base = g * num_stages;
        PrimExpr fwd_id = IntImm(PrimType::Int(32), fwd_base) + c_stage_expr;
        consumer_stmts.push_back(
            MakeParityWait(barrier_buf, fwd_id, c_parity_expr));
      }
    }
    for (int g = 0; g < consumer_barrier_groups; ++g) {
      if (!arrive_emitted[g] &&
          arrive_insert_pos[g] ==
              static_cast<int>(consumer_compute_stmts.size())) {
        int bp_base = num_fwd + g * num_stages;
        PrimExpr bp_id = IntImm(PrimType::Int(32), bp_base) + c_stage_expr;
        consumer_stmts.push_back(MakeArriveBarrier(barrier_buf, bp_id));
      }
    }
    // Phase counter increment at end of consumer guarded iteration
    if (needs_phase_counter) {
      consumer_stmts.push_back(consumer_phase_counter.value().Increment());
    }

    // --- Prepend leading bindings and optional condition ---
    auto prepend_bindings =
        [&](Stmt body,
            const std::vector<std::pair<Var, Expr>> &bindings) -> Stmt {
      for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
        body = SeqStmt({Bind(it->first, it->second), body});
      }
      return body;
    };

    Stmt producer_body =
        prepend_bindings(SeqStmt(producer_stmts), inner_leading_bindings);
    Stmt consumer_body =
        prepend_bindings(SeqStmt(consumer_stmts), inner_leading_bindings);

    // Wrap in original condition if the loop body was guarded.
    if (loop_body_condition.has_value()) {
      producer_body = PCIf(loop_body_condition.value(), producer_body);
      consumer_body = PCIf(loop_body_condition.value(), consumer_body);
    }

    producer_body = prepend_bindings(producer_body, outer_leading_bindings);
    consumer_body = prepend_bindings(consumer_body, outer_leading_bindings);

    // Rewrite only the shared-buffer version indices marked by MVB. User
    // expressions that happen to depend on the loop variable must retain their
    // original semantics.
    Optional<PrimExpr> producer_stage_replacement;
    Optional<PrimExpr> consumer_stage_replacement;
    if (needs_phase_counter) {
      producer_stage_replacement =
          producer_phase_counter.value().StageExpr(num_stages);
      consumer_stage_replacement =
          consumer_phase_counter.value().StageExpr(num_stages);
    }
    producer_body = MVBStageIndexReplacer::Replace(
        producer_body, std::move(producer_stage_replacement));
    consumer_body = MVBStageIndexReplacer::Replace(
        consumer_body, std::move(consumer_stage_replacement));
    producer_body =
        TileOpMbarPhaseAnnotator::Annotate(producer_body, p_parity_expr);
    consumer_body =
        TileOpMbarPhaseAnnotator::Annotate(consumer_body, c_parity_expr);

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

    For producer_loop(loop_var, loop_min, pipeline_loop->end,
                      pipeline_loop->step, ForKind::kDefault,
                      PCSequence(producer_body), loop_annos,
                      pipeline_loop->loc);
    For consumer_loop(loop_var, loop_min, pipeline_loop->end,
                      pipeline_loop->step, ForKind::kDefault,
                      PCSequence(consumer_body), loop_annos,
                      pipeline_loop->loc);

    // Wrap loops with phase counter allocation when needed.
    Stmt final_producer_loop = producer_loop;
    Stmt final_consumer_loop = consumer_loop;
    if (needs_phase_counter && !pipeline_under_outer_for) {
      // Local scope: alloc + init live right around the pipeline loop.
      final_producer_loop =
          producer_phase_counter.value().WrapLoopWithAlloc(producer_loop);
      final_consumer_loop =
          consumer_phase_counter.value().WrapLoopWithAlloc(consumer_loop);
    }

    // --- Rewrite threadIdx.x for WS partitions ---
    // Producer owns the low partition and already sees [0, producer_extent).
    Stmt rewritten_producer = final_producer_loop;
    if (producer_is_single_warp) {
      rewritten_producer = PCIf(LT(thread_var_, IntImm(thread_var_.ty(), 32)),
                                rewritten_producer);
    }
    // Consumer owns the high partition and is remapped back to
    // [0, consumer_extent) for existing tile-op layout assumptions.
    Stmt rewritten_consumer = PCThreadIdxRewriter::Rewrite(
        final_consumer_loop, thread_var_, thread_var_ - producer_extent,
        consumer_extent, false);

    shared_prelude_live_seed_ = {};
    producer_prelude_live_seed_ = {};
    consumer_prelude_live_seed_ = {};
    producer_prelude_live_seed_.AddUses(LocalAccessCollector::Collect(
        rewritten_producer, buffer_data_to_buffer_));
    consumer_prelude_live_seed_.AddUses(LocalAccessCollector::Collect(
        rewritten_consumer, buffer_data_to_buffer_));

    // Move pre-loop branch-private initialization next to the branch that
    // consumes it. Classification is based on downstream producer/consumer
    // uses of the values defined by each prelude statement.
    extracted_producer_init_ = {};
    extracted_consumer_init_ = {};

    Array<IntImm> ws_partition = {producer_extent.as_or_throw<IntImm>(),
                                  consumer_extent.as_or_throw<IntImm>()};

    // First pass: find and extract consumer-only pre-loop statements
    // by doing a dry replacement that populates extracted_consumer_init_.
    Stmt dummy_producer = rewritten_producer;
    const Stmt &dummy_consumer = rewritten_consumer;
    Stmt dummy_ws =
        PCIf(LT(thread_var_, producer_extent), dummy_producer, dummy_consumer);
    dummy_ws = MakeNativeAttribute(ws_partition, attr::kWarpSpecializationScope,
                                   0, PCSequence(dummy_ws));
    Optional<Stmt> replaced = ReplacePipelineLoopInStmt(
        kernel.body, pipeline_loop, dummy_ws, producer_extent, consumer_extent);
    ICHECK(replaced.has_value())
        << "ProducerConsumerWS: failed to replace pipeline loop";
    Stmt replaced_stmt = replaced.value();

    // Producer and consumer partitions cannot safely share the same block-level
    // local/fragment buffers after tiled WS is introduced before
    // LayoutInference: a single fragment layout cannot represent both thread
    // ranges. Clone every branch-private buffer touched by the producer so
    // LayoutInference can infer an independent producer-side thread range.
    Map<Var, Expr> producer_buffer_remap;
    Array<TensorVar> producer_private_buffers;
    {
      TensorVarSet block_alloc_buffers;
      for (const auto &buffer : kernel.allocations) {
        block_alloc_buffers.insert(buffer);
      }
      LocalAccessSummary producer_access = LocalAccessCollector::Collect(
          rewritten_producer, buffer_data_to_buffer_);
      for (const auto &stmt : extracted_producer_init_) {
        MergeLocalAccessSummary(
            &producer_access,
            LocalAccessCollector::Collect(stmt, buffer_data_to_buffer_));
      }
      auto maybe_clone = [&](const TensorVar &buffer) {
        if (!(IsFragmentBuffer(buffer) || IsLocalBuffer(buffer)) ||
            !block_alloc_buffers.count(buffer) ||
            producer_buffer_remap.count(buffer)) {
          return;
        }
        TensorVar cloned = CloneBranchPrivateTensorVar(buffer, "_producer_ws");
        producer_buffer_remap.Set(buffer, cloned);
        producer_private_buffers.push_back(cloned);
      };
      for (const auto &buffer : producer_access.read_buffers) {
        maybe_clone(buffer);
      }
      for (const auto &buffer : producer_access.write_buffers) {
        maybe_clone(buffer);
      }
    }
    if (!producer_buffer_remap.empty()) {
      rewritten_producer =
          WSSubstitute(rewritten_producer, producer_buffer_remap);
      Array<Stmt> remapped_producer_init;
      for (const auto &stmt : extracted_producer_init_) {
        remapped_producer_init.push_back(
            WSSubstitute(stmt, producer_buffer_remap));
      }
      extracted_producer_init_ = remapped_producer_init;
    }

    // If branch-local prelude init/copy was extracted, rebuild with it inside
    // the corresponding WS branch so each branch initializes its own local
    // state before entering the pipelined loop.
    if (!extracted_producer_init_.empty() ||
        !extracted_consumer_init_.empty() || !producer_buffer_remap.empty()) {
      Stmt enriched_producer = rewritten_producer;
      if (!extracted_producer_init_.empty()) {
        Array<Stmt> producer_parts;
        for (const auto &s : extracted_producer_init_) {
          producer_parts.push_back(s);
        }
        producer_parts.push_back(rewritten_producer);
        enriched_producer = producer_parts.size() == 1
                                ? producer_parts[0]
                                : SeqStmt(producer_parts);
      }
      Array<Stmt> consumer_parts;
      for (const auto &s : extracted_consumer_init_) {
        consumer_parts.push_back(PCThreadIdxRewriter::Rewrite(
            s, thread_var_, thread_var_ - producer_extent, consumer_extent,
            false));
      }
      consumer_parts.push_back(rewritten_consumer);
      Stmt enriched_consumer = consumer_parts.size() == 1
                                   ? consumer_parts[0]
                                   : SeqStmt(consumer_parts);
      Stmt scoped_producer = enriched_producer;
      const Stmt &scoped_consumer = enriched_consumer;
      Stmt ws_body = PCIf(LT(thread_var_, producer_extent), scoped_producer,
                          scoped_consumer);
      ws_body = MakeNativeAttribute(
          ws_partition, attr::kWarpSpecializationScope, 0, PCSequence(ws_body));
      // Second pass: replace again with the enriched WS body.
      // extracted_consumer_init_ is already empty (stmts were removed
      // from the prelude in the first pass result).
      // We need to replace in the ALREADY-modified body from pass 1.
      // The pipeline loop has already been replaced by dummy_ws in that
      // tree, so do a direct substitution of the placeholder WS body.
      // Since dummy_ws appears exactly once in replaced.stmt, do a
      // simple statement replacement on the full placeholder stmt.
      class SubstWsBody : public StmtExprMutator {
      public:
        using StmtExprMutator::Mutate;
        using StmtExprMutator::Mutate_;
        SubstWsBody(Stmt old_body, Stmt new_body)
            : old_(old_body), new_(new_body) {}
        UnchangedOr<Any> Mutate(AnyView value, InplaceMode mode) final {
          if (auto stmt = value.as<Stmt>(); stmt && stmt->same_as(old_))
            return Any(new_);
          return StmtExprMutator::Mutate(value, mode);
        }
        Stmt old_, new_;
      };
      auto substitute = make_object<SubstWsBody>(dummy_ws, ws_body);
      replaced_stmt =
          substitute->Mutate(replaced_stmt).ValueOrUnchanged(replaced_stmt);
    }
    Stmt new_block_body = SinkGuardedConsumerPostlude::Rewrite(
        replaced_stmt, thread_var_, producer_extent);
    new_block_body = FreshenWSBranchDefinitions::Rewrite(new_block_body);

    if (needs_phase_counter && pipeline_under_outer_for) {
      Array<Stmt> block_prelude;
      block_prelude.push_back(producer_phase_counter.value().InitStmt());
      block_prelude.push_back(consumer_phase_counter.value().InitStmt());
      block_prelude.push_back(new_block_body);
      new_block_body = SeqStmt(block_prelude);
    }

    Array<TensorVar> extra_allocations = producer_private_buffers;
    extra_allocations.push_back(barrier_buf);
    if (needs_phase_counter && pipeline_under_outer_for) {
      extra_allocations.push_back(producer_phase_counter.value().buf);
      extra_allocations.push_back(consumer_phase_counter.value().buf);
    }
    Map<Var, Array<PrimExpr>> initializers;
    initializers.Set(barrier_buf, arrive_counts);
    if (auto existing = kernel.annotations.Get("barrier_init"))
      for (const auto &[key, value] : existing->cast<Map<Any, Any>>()) {
        Array<PrimExpr> counts;
        for (const auto &count : value.cast<Array<Any>>())
          counts.push_back(count.as<PrimExpr>().value_or(
              IntImm::Int32(GetIntegerAnnotation(count))));
        initializers.Set(key.cast<Var>(), counts);
      }
    kernel.annotations.Set("barrier_init", initializers);
    PrimExpr total = consumer_extent + producer_extent;
    auto total_threads = TryConstInt(total);
    TVM_FFI_CHECK(total_threads, ValueError)
        << "PC warp specialization requires a static thread count";
    return {
        RebuildNativeWSKernel(kernel, new_block_body, {}, extra_allocations),
        static_cast<int>(*total_threads)};
  }

  class PipelineLoopFinder : public StmtExprVisitor {
  public:
    using StmtExprVisitor::Visit;
    using StmtExprVisitor::Visit_;
    struct Result {
      Optional<For> pipeline_loop;
      bool under_outer_for = false;
    };
    static Result Find(const Stmt &stmt) {
      auto finder = make_object<PipelineLoopFinder>();
      finder->Visit(stmt);
      return {finder->pipeline_loop_, finder->under_outer_for_};
    }

  private:
    Optional<VisitInterrupt> Visit_(const ForObj *loop) final {
      if (pipeline_loop_)
        return VisitInterrupt();
      if (auto stages = GetPipelineNumStages(loop); stages && *stages >= 1) {
        pipeline_loop_ = GetRef<For>(loop);
        return VisitInterrupt();
      }
      bool old = under_outer_for_;
      under_outer_for_ = true;
      auto result = StmtExprVisitor::Visit_(loop);
      if (!pipeline_loop_)
        under_outer_for_ = old;
      return result;
    }
    Optional<For> pipeline_loop_;
    bool under_outer_for_ = false;
  };

  Optional<For> FindPipelineLoop(const Stmt &stmt) {
    return PipelineLoopFinder::Find(stmt).pipeline_loop;
  }

  PipelineLoopFinder::Result FindPipelineLoopWithNesting(const Stmt &stmt) {
    return PipelineLoopFinder::Find(stmt);
  }

  class SinkGuardedConsumerPostlude : public StmtExprMutator {
  public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    SinkGuardedConsumerPostlude(PrimVar thread, PrimExpr producer_extent)
        : thread_(std::move(thread)),
          producer_extent_(std::move(producer_extent)) {}

    static Stmt Rewrite(const Stmt &stmt, PrimVar thread,
                        PrimExpr producer_extent) {
      auto rewriter =
          make_object<SinkGuardedConsumerPostlude>(thread, producer_extent);
      return rewriter->Mutate(stmt).ValueOrUnchanged(stmt);
    }

  private:
    Optional<If> Branch(const Stmt &stmt) const {
      Stmt inner = SingleStatement(stmt);
      if (const auto *region = inner.as<RegionStmtObj>()) {
        auto attributes = WSAttribute(region);
        if (!attributes || attributes->at("key").cast<String>() !=
                               attr::kWarpSpecializationScope)
          return std::nullopt;
        inner = SingleStatement(region->body);
      }
      const auto *branch = inner.as<IfObj>();
      if (!branch || !branch->else_case)
        return std::nullopt;
      const auto *condition = branch->condition.as<LTObj>();
      if (!condition || !condition->a.same_as(thread_) ||
          !ffi::StructuralEqual()(condition->b, producer_extent_))
        return std::nullopt;
      return GetRef<If>(branch);
    }

    Optional<SeqStmt> GuardedConsumer(const Stmt &stmt) const {
      const auto *branch = SingleStatement(stmt).as<IfObj>();
      if (!branch || branch->else_case)
        return std::nullopt;
      const auto *condition = branch->condition.as<GEObj>();
      if (!condition || !condition->a.same_as(thread_) ||
          !ffi::StructuralEqual()(condition->b, producer_extent_))
        return std::nullopt;
      return branch->then_case;
    }

    Stmt AppendConsumer(const Stmt &container, const If &branch,
                        const Array<Stmt> &postlude) const {
      Array<Stmt> parts{branch->else_case.value()};
      parts.insert(parts.end(), postlude.begin(), postlude.end());
      If next(branch->condition, branch->then_case, WSSequence(parts),
              branch->loc);
      const auto *region = SingleStatement(container).as<RegionStmtObj>();
      if (!region)
        return next;
      return RegionStmt(region->op, region->args, region->body_params,
                        region->attrs, SeqStmt(next), region->result_vars,
                        region->loc);
    }

    UnchangedOr<Stmt> Mutate_(const SeqStmtObj *sequence,
                              InplaceMode mode) final {
      Array<Stmt> visited, result;
      for (const auto &statement : sequence->seq)
        visited.push_back(Mutate(statement, mode).ValueOrUnchanged(statement));
      for (size_t i = 0; i < visited.size(); ++i) {
        auto branch = Branch(visited[i]);
        if (!branch) {
          result.push_back(visited[i]);
          continue;
        }
        Array<Stmt> postlude;
        size_t end = i + 1;
        for (; end < visited.size(); ++end) {
          auto body = GuardedConsumer(visited[end]);
          if (!body)
            break;
          postlude.push_back(*body);
        }
        result.push_back(postlude.empty()
                             ? visited[i]
                             : AppendConsumer(visited[i], *branch, postlude));
        i = end - 1;
      }
      return WSSequence(result, sequence->loc);
    }

    PrimVar thread_;
    PrimExpr producer_extent_;
  };

  // Both roles start from the same source loop and may share a copied prelude.
  // Native definitions are unique across the whole function, even when their
  // lexical scopes do not overlap. Renew each complete role after postludes
  // have been joined so all uses (including tensor types and layout metadata)
  // follow the role's fresh definitions, while captured shared tensors remain
  // shared. RenewDef over the combined function cannot split one old identity
  // into distinct producer and consumer definitions.
  class FreshenWSBranchDefinitions : public StmtExprMutator {
  public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;

    static Stmt Rewrite(const Stmt &stmt) {
      auto rewriter = make_object<FreshenWSBranchDefinitions>();
      return rewriter->Mutate(stmt).ValueOrUnchanged(stmt);
    }

    UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                              InplaceMode mode) final {
      auto attributes = WSAttribute(region);
      if (!attributes || attributes->at("key").cast<String>() !=
                             attr::kWarpSpecializationScope) {
        return StmtExprMutator::Mutate_(region, mode);
      }
      auto branch = SingleStatement(region->body).as_or_throw<If>();
      auto freshen = [](SeqStmt body) {
        return tirx::RenewDef(tirx::Function({}, std::move(body)))->body;
      };
      If fresh(branch->condition, freshen(branch->then_case),
               freshen(branch->else_case.value()), branch->loc);
      return RegionStmt(region->op, region->args, region->body_params,
                        region->attrs, SeqStmt(fresh), region->result_vars,
                        region->loc);
    }
  };

  class PipelineLoopContainmentChecker {
  public:
    static bool Contains(const Stmt &stmt, const For &pipeline) {
      if (stmt.same_as(pipeline))
        return true;
      if (const auto *sequence = stmt.as<SeqStmtObj>()) {
        for (const auto &child : sequence->seq)
          if (Contains(child, pipeline))
            return true;
      } else if (const auto *region = stmt.as<RegionStmtObj>()) {
        return Contains(region->body, pipeline);
      } else if (const auto *loop = stmt.as<ForObj>()) {
        return Contains(loop->body, pipeline);
      } else if (const auto *branch = stmt.as<IfObj>()) {
        return Contains(branch->then_case, pipeline) ||
               (branch->else_case && Contains(*branch->else_case, pipeline));
      }
      return false;
    }
  };

  class PipelineLoopInStmtReplacer {
  public:
    PipelineLoopInStmtReplacer(ProducerConsumerWSRewriter *rewriter,
                               For pipeline, Stmt replacement,
                               PrimExpr producer, PrimExpr consumer)
        : rewriter_(rewriter), pipeline_loop_(std::move(pipeline)),
          ws_body_(std::move(replacement)),
          producer_extent_(std::move(producer)),
          consumer_extent_(std::move(consumer)) {}

    Optional<Stmt> operator()(const Stmt &stmt) { return Rewrite(stmt); }

  private:
    Optional<Stmt> Rewrite(const Stmt &stmt) {
      if (stmt.same_as(pipeline_loop_))
        return ws_body_;
      if (const auto *sequence = stmt.as<SeqStmtObj>()) {
        int position = -1;
        for (size_t i = 0; i < sequence->seq.size(); ++i)
          if (PipelineLoopContainmentChecker::Contains(sequence->seq[i],
                                                       pipeline_loop_)) {
            position = static_cast<int>(i);
            break;
          }
        if (position < 0)
          return std::nullopt;
        auto child = Rewrite(sequence->seq[position]);
        ICHECK(child)
            << "ProducerConsumerWS: failed to replace pipeline loop child";
        PropagatePreludeLiveness(sequence, position);
        Array<Stmt> result;
        AppendClassifiedPrelude(sequence, position, &result);
        result.push_back(*child);
        for (size_t i = position + 1; i < sequence->seq.size(); ++i)
          result.push_back(GuardConsumerOnly(sequence->seq[i]));
        return WSSequence(result, sequence->loc);
      }
      if (const auto *region = stmt.as<RegionStmtObj>()) {
        auto body = Rewrite(region->body);
        if (!body)
          return std::nullopt;
        return RegionStmt(region->op, region->args, region->body_params,
                          region->attrs, PCSequence(*body), region->result_vars,
                          region->loc);
      }
      if (const auto *branch = stmt.as<IfObj>()) {
        auto then_case = Rewrite(branch->then_case);
        auto else_case = branch->else_case;
        bool found = then_case.has_value();
        if (!found && else_case) {
          auto next = Rewrite(*else_case);
          if (next) {
            else_case = PCSequence(*next);
            found = true;
          }
        }
        if (!found)
          return std::nullopt;
        return If(branch->condition,
                  then_case ? PCSequence(*then_case) : branch->then_case,
                  else_case, branch->loc);
      }
      if (const auto *loop = stmt.as<ForObj>()) {
        auto body = Rewrite(loop->body);
        if (!body)
          return std::nullopt;
        return For(loop->var, loop->begin, loop->end, loop->step, loop->kind,
                   PCSequence(*body), loop->annotations, loop->loc);
      }
      return std::nullopt;
    }

    Stmt GuardConsumerOnly(const Stmt &stmt) const {
      auto rewritten = PCThreadIdxRewriter::Rewrite(
          stmt, rewriter_->thread_var_,
          rewriter_->thread_var_ - producer_extent_, consumer_extent_, false);
      return PCIf(rewriter_->thread_var_ >= producer_extent_, rewritten);
    }

    // Kept verbatim from the baseline algorithm by the implementation helper:
    // recursively propagate nested postlude liveness before splitting preludes.
    void PropagatePreludeLiveness(const SeqStmtObj *op, int loop_idx) {
      LocalLiveSet shared_live = rewriter_->shared_prelude_live_seed_;
      LocalLiveSet producer_live = rewriter_->producer_prelude_live_seed_;
      LocalLiveSet consumer_live = rewriter_->consumer_prelude_live_seed_;
      for (int i = loop_idx + 1; i < static_cast<int>(op->seq.size()); ++i) {
        // Post-loop siblings are later sunk into the consumer branch by
        // SinkGuardedConsumerPostlude.  Keep scalar/index dependencies in
        // the enclosing prelude so existing shared-prelude index math stays
        // common, but treat branch-private buffer uses as consumer-only so
        // their local/fragment initialization does not leak into producer.
        LocalAccessSummary summary = LocalAccessCollector::Collect(
            op->seq[i], rewriter_->buffer_data_to_buffer_);
        shared_live.vars.insert(summary.read_vars.begin(),
                                summary.read_vars.end());
        consumer_live.buffers.insert(summary.read_buffers.begin(),
                                     summary.read_buffers.end());
      }
      for (int i = loop_idx - 1; i >= 0; --i) {
        LocalAccessSummary summary = LocalAccessCollector::Collect(
            op->seq[i], rewriter_->buffer_data_to_buffer_);
        if (!summary.HasTrackedDefs()) {
          shared_live.AddUses(summary);
          continue;
        }
        bool shared_needs = shared_live.NeedsAnyDef(summary);
        bool producer_needs = producer_live.NeedsAnyDef(summary);
        bool consumer_needs = consumer_live.NeedsAnyDef(summary);
        if (shared_needs || (!producer_needs && !consumer_needs)) {
          shared_live.AddUses(summary);
        }
        if (producer_needs) {
          producer_live.AddUses(summary);
        }
        if (consumer_needs) {
          consumer_live.AddUses(summary);
        }
      }
      rewriter_->shared_prelude_live_seed_ = shared_live;
      rewriter_->producer_prelude_live_seed_ = producer_live;
      rewriter_->consumer_prelude_live_seed_ = consumer_live;
    }

    void AppendClassifiedPrelude(const SeqStmtObj *op, int loop_idx,
                                 Array<Stmt> *new_seq) {
      // Classify pre-loop statements using branch-private def/use sets.
      // Shared-prelude statements stay in place; branch-private definitions
      // move next to the branch that consumes them, or are duplicated when
      // both producer and consumer need the same definition.
      for (int i = 0; i < loop_idx; ++i) {
        switch (ClassifyPreludeStmt(op->seq[i],
                                    rewriter_->buffer_data_to_buffer_,
                                    rewriter_->shared_prelude_live_seed_,
                                    rewriter_->producer_prelude_live_seed_,
                                    rewriter_->consumer_prelude_live_seed_)) {
        case PreludeStmtPlacement::kProducerOnly:
          rewriter_->extracted_producer_init_.push_back(op->seq[i]);
          break;
        case PreludeStmtPlacement::kConsumerOnly:
          rewriter_->extracted_consumer_init_.push_back(op->seq[i]);
          break;
        case PreludeStmtPlacement::kDuplicateToBoth:
          rewriter_->extracted_producer_init_.push_back(op->seq[i]);
          rewriter_->extracted_consumer_init_.push_back(op->seq[i]);
          break;
        case PreludeStmtPlacement::kKeepSharedPrelude:
          if (auto it = rewriter_->common_prelude_rewrites_.find(op->seq[i]);
              it != rewriter_->common_prelude_rewrites_.end()) {
            new_seq->push_back(it->second);
          } else {
            new_seq->push_back(op->seq[i]);
          }
          break;
        }
      }
    }

    ProducerConsumerWSRewriter *rewriter_;
    For pipeline_loop_;
    Stmt ws_body_;
    PrimExpr producer_extent_, consumer_extent_;
  };

  Optional<Stmt> ReplacePipelineLoopInStmt(const Stmt &stmt,
                                           const For &pipeline_loop,
                                           const Stmt &ws_body,
                                           PrimExpr producer_extent,
                                           PrimExpr consumer_extent) {
    PipelineLoopInStmtReplacer replacer(this, pipeline_loop, ws_body,
                                        std::move(producer_extent),
                                        std::move(consumer_extent));
    return replacer(stmt);
  }

  class PCThreadIdxRewriter {
  public:
    static Stmt Rewrite(Stmt stmt, PrimVar thread, PrimExpr value,
                        PrimExpr extent, bool shuffle) {
      return WSSubstitute(stmt, Map<Var, Expr>{{thread, value}});
    }
  };

  Target target_;
  PrimVar thread_var_;
  PrimExpr thread_extent_;
  WSMemoryContext buffer_data_to_buffer_;
  StmtRewriteMap common_prelude_rewrites_;
  LocalLiveSet shared_prelude_live_seed_, producer_prelude_live_seed_,
      consumer_prelude_live_seed_;
  Array<Stmt> extracted_producer_init_, extracted_consumer_init_;
};

// ---------------------------------------------------------------------------
// Detect if manual WS is already present (skip if so)
// ---------------------------------------------------------------------------

/// Quick pre-scan: check if the function contains a pipelined loop (num_stages
/// >= 1) with at least one TMA load producer tile op and no manual layout
/// annotations (which are incompatible with early MVB expansion).
class TiledWSCandidate : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  explicit TiledWSCandidate(Target target) : target_(target) {}
  static bool Check(const Stmt &body, Target target) {
    auto candidate = make_object<TiledWSCandidate>(target);
    CollectAnnotatedLayouts(body, candidate->layout_map_);
    candidate->Visit(body);
    return candidate->has_tma_;
  }

private:
  Optional<VisitInterrupt> Visit_(const ForObj *loop) final {
    bool old = in_pipeline_;
    if (auto count = GetPipelineNumStages(loop); count && *count >= 1)
      in_pipeline_ = true;
    auto result = StmtExprVisitor::Visit_(loop);
    in_pipeline_ = old;
    return result;
  }
  Optional<VisitInterrupt> Visit_(const CallObj *call) final {
    if (in_pipeline_ && !has_tma_) {
      auto operation = ParseOperator(GetRef<Call>(call));
      if (auto copy = operation.as<CopyNode>();
          copy && ClassifyCopy(copy, target_) == TileStmtKind::kTmaProducer) {
        auto layout = layout_map_.find(copy->dst);
        has_tma_ = layout == layout_map_.end() ||
                   (layout->second.second.defined() &&
                    IsTmaCompatibleLayout(layout->second.second,
                                          layout->second.first));
      }
    }
    return StmtExprVisitor::Visit_(call);
  }
  Target target_;
  bool in_pipeline_ = false, has_tma_ = false;
  BufferLayoutMap layout_map_;
};

} // namespace

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

tvm::transform::Pass ProducerConsumerWarpSpecialized() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext context) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false) ||
        context->GetConfig<bool>(kDisableWarpSpecialized).value_or(false))
      return function;
    if (HasManualWarpSpecialization(function->body))
      return function;
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    if (!target || !TargetHasBulkCopy(*target) ||
        !TiledWSCandidate::Check(function->body, *target))
      return function;
    auto rewritten = ProducerConsumerWSRewriter::Substitute(
        ApplyMultiVersionBufferRewriter(function));
    if (!rewritten->GetAttr<int64_t>(kTiledWSApplied).value_or(0)) {
      class Stripper : public StmtExprMutator {
      public:
        using StmtExprMutator::Mutate;
        using StmtExprMutator::Mutate_;
        UnchangedOr<Stmt> Mutate_(const ForObj *loop, InplaceMode mode) final {
          auto result = StmtExprMutator::Mutate_(loop, mode)
                            .ValueOrUnchanged(GetRef<For>(loop))
                            .as_or_throw<For>();
          auto annotations = result->annotations;
          if (!annotations.count("num_stages"))
            return result;
          annotations.erase("num_stages");
          return For(result->var, result->begin, result->end, result->step,
                     result->kind, result->body, annotations, result->loc);
        }
      };
      auto stripper = make_object<Stripper>();
      auto body = function->body;
      function.CopyOnWrite()->body =
          stripper->Mutate(body).ValueOrUnchanged(body);
      return function;
    }
    auto body = MVBStageIndexReplacer::Replace(rewritten->body, std::nullopt);
    rewritten.CopyOnWrite()->body = PCSequence(body);
    return rewritten;
  };
  return tirx::transform::CreateFunctionPass(
      pass, 0, "tl.ProducerConsumerWarpSpecialized");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.cuda.transform.ProducerConsumerWarpSpecialized",
                        ProducerConsumerWarpSpecialized);
}

} // namespace tl
} // namespace tvm
