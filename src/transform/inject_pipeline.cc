/*!
 * \file inject_pipeline.cc
 * \brief Transform annotated loops into pipelined one that parallelize
 * producers and consumers.
 */
#include "ascend/ir/block.h"
#include "backend/common/target_utils.h"
#include "common/bind_utils.h"
#include "common/pipeline_access.h"
#include "common/pipeline_tensor.h"
#include "common/pipeline_utils.h"
#include "cuda/op/builtin.h"
#include "ir/analyzer.h"
#include "ir/lexical_scope.h"
#include "ir/loop_math.h"
#include "ir/stmt_functor.h"
#include "layout/expr_utils.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/utils.h"
#include "span_utils.h"
#include "support/check.h"
#include "transform/common/mbarrier.h"
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;
using namespace prim;
namespace software_pipeline {

using namespace tirx;
using namespace ffi;
using namespace prim;

// Scheduling-only data; it is never a statement in the output IR.
class PipelineStageNode : public ffi::Object {
public:
  Stmt body{ffi::UnsafeInit{}};
  Array<TensorRegion> reads, writes;
  String name_hint;
  Optional<RegionStmt> scope;
  Location loc = UnknownLoc();
  bool unknown_access{false};
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.PipelineStageInfo", PipelineStageNode,
                                    ffi::Object);
};
class PipelineStage : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PipelineStage, ffi::ObjectRef,
                                             PipelineStageNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PipelineStageNode);
};

namespace {
String ScopeKey(const RegionStmtObj *region) {
  if (!region || region->op->name != "tl.attr")
    return "";
  return ReadAnnotations(region).at("key").cast<String>();
}
Any ScopeNode(const RegionStmtObj *region) {
  return ReadAnnotations(region).at("node");
}
PrimExpr ScopeValue(const RegionStmtObj *region) {
  Any value = ReadAnnotations(region).at("value");
  if (auto expr = value.as<PrimExpr>())
    return *expr;
  return IntImm::Int32(GetIntegerAnnotation(value));
}
RegionStmt ScopeStmt(Any node, String key, Any value, Stmt body,
                     Location loc = UnknownLoc()) {
  return MakeAttribute(node, key, value, SeqStmt(body), loc);
}
bool TransparentScope(const RegionStmtObj *region) {
  return region && region->body_params.empty() && region->result_vars.empty() &&
         (region->op->name == "tl.attr" || ascend::IsAscendBlock(region) ||
          region->op.same_as(tl::lexical_scope_op()));
}
Stmt WithScopeBody(const RegionStmtObj *region, Stmt body) {
  return RegionStmt(region->op, region->args, region->body_params,
                    region->attrs, SeqStmt(body), region->result_vars,
                    region->loc);
}
const RegionStmtObj *SingleRegion(const SeqStmt &body) {
  return body->seq.size() == 1 ? body->seq[0].as<RegionStmtObj>() : nullptr;
}
Array<int64_t> IntegerArray(const Any &value) {
  Array<int64_t> result;
  for (const Any &item : value.cast<Array<Any>>())
    result.push_back(GetIntegerAnnotation(item));
  return result;
}
PrimExpr MakeZero(PrimType type) { return IntImm(type, 0); }
} // namespace

using TensorSet = std::unordered_set<TensorVar, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using BufferMap =
    std::unordered_map<TensorVar, TensorVar, ObjectPtrHash, ObjectPtrEqual>;
using BufferShapeMap =
    std::unordered_map<TensorVar, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
using BufferCommitGroupMap =
    std::unordered_map<TensorVar, int, ObjectPtrHash, ObjectPtrEqual>;
using BlockDependencyGraph =
    std::unordered_map<PipelineStage, Array<PipelineStage>, ObjectPtrHash,
                       ObjectPtrEqual>;

/*! Structure that represents the provided annotation per block or loop. */
struct PipelineAnnotation {
  int stage;
  int order;
  bool async{false};
  int async_group_id{-1};
};

using PipelineInfo = std::unordered_map<PipelineStage, PipelineAnnotation,
                                        ObjectPtrHash, ObjectPtrEqual>;

struct BufferAccessInfo {
  int def = -1; // the defining stage of the buffer
  int use = -1; // the last using stage of the buffer
};

struct PipelineRewriteResult {
  Stmt pipeline{ffi::UnsafeInit{}};
  Map<TensorVar, TensorVar> buffer_remap;
};

namespace {

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmObj>()) {
      return !IsZero(GetRef<IntImm>(int_val));
    }
  }
  return false;
}

bool GetIsTmaCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "is_tma_copy");
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  return GetBoolAnnotation(op, "force_cp_async");
}

bool CheckTargetIndependentAsyncCopyPreconditions(const CopyNode &op) {
  if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
    return false;
  }
  if (op.src.dtype() != op.dst.dtype()) {
    return false;
  }
  return true;
}

bool CheckPipelineManagedCPAsyncCopy(const CopyNode &op,
                                     Optional<Target> target) {
  if (GetIsTmaCopy(op) || GetIsAsyncCopy(op) ||
      !CheckTargetIndependentAsyncCopyPreconditions(op)) {
    return false;
  }
  return !target.has_value() || TargetHasAsyncCopy(target.value());
}

bool ShapesEqual(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs,
                 sym::AnalyzerObj *analyzer) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!analyzer->CanProveEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

Layout
ExpandAnnotatedLayoutForMultiVersionedBuffer(const Layout &layout,
                                             const TensorVar &old_buffer,
                                             const TensorVar &new_buffer) {
  if (!layout.defined() ||
      new_buffer.shape().size() <= old_buffer.shape().size()) {
    return Layout();
  }

  sym::Analyzer analyzer;
  if (!ShapesEqual(layout->InputShape(), old_buffer.shape(), analyzer.get())) {
    return Layout();
  }

  size_t leading_ndim = new_buffer.shape().size() - old_buffer.shape().size();
  Array<PrimExpr> trailing_shape;
  Array<PrimExpr> leading_shape;
  for (size_t i = 0; i < leading_ndim; ++i) {
    leading_shape.push_back(new_buffer.shape()[i]);
  }
  for (size_t i = 0; i < old_buffer.shape().size(); ++i) {
    trailing_shape.push_back(new_buffer.shape()[leading_ndim + i]);
  }
  if (!ShapesEqual(trailing_shape, old_buffer.shape(), analyzer.get())) {
    return Layout();
  }

  return layout->Expand(leading_shape);
}

class SimtProducerAnnotator : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate;
  using LayoutMapMutator::Mutate_;
  explicit SimtProducerAnnotator(Optional<Target> target)
      : target_(std::move(target)) {}
  static Stmt Annotate(const Stmt &stmt,
                       Optional<Target> target = std::nullopt) {
    auto mutator = make_object<SimtProducerAnnotator>(std::move(target));
    return mutator->Mutate(stmt).ValueOrUnchanged(stmt);
  }
  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    auto annotations = op->annotations;
    annotations.Set(attr::kParallelAsyncWithoutAsyncCommitWait, true);
    return For(op->var, op->begin, op->end, op->step, op->kind, body,
               annotations, op->loc);
  }
  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    auto call = LayoutMapMutator::Mutate_(op, mode)
                    .ValueOrUnchanged(GetRef<Call>(op))
                    .as_or_throw<Call>();
    if (!call->op.same_as(Op::Get("tl.tileop.copy")))
      return call;
    auto tile = ParseOperator(call);
    const auto *copy = tile.as<CopyNode>();
    if (!copy || !CheckPipelineManagedCPAsyncCopy(*copy, target_))
      return call;
    auto fields = DecodeTileOpArguments(call.get());
    fields.annotations.Set(attr::kAsyncCopyNoImplicitCommitWait, true);
    return MakeAnnotatedCall(call->ty, call->op.as_or_throw<Op>(),
                             fields.operands, fields.annotations, call->loc);
  }

private:
  Optional<Target> target_;
};

class TileOpMbarPhaseAnnotator : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate;
  using LayoutMapMutator::Mutate_;
  explicit TileOpMbarPhaseAnnotator(PrimExpr phase)
      : phase_(std::move(phase)) {}
  static Stmt Annotate(const Stmt &stmt, PrimExpr phase) {
    auto mutator = make_object<TileOpMbarPhaseAnnotator>(std::move(phase));
    return mutator->Mutate(stmt).ValueOrUnchanged(stmt);
  }
  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    auto call = LayoutMapMutator::Mutate_(op, mode)
                    .ValueOrUnchanged(GetRef<Call>(op))
                    .as_or_throw<Call>();
    auto tile = ParseOperator(call);
    bool consumes = false;
    if (const auto *gemm = tile.as<GemmNode>())
      consumes = gemm->mbar_.has_value() && !gemm->isTcgen05_;
    if (const auto *copy = tile.as<CopyNode>())
      consumes = !GetIsTmaCopy(*copy) && copy->annotations.count("barrier");
    if (!consumes)
      return call;
    auto fields = DecodeTileOpArguments(call.get());
    if (fields.annotations.count(attr::kPipelineMbarPhaseExpr))
      return call;
    fields.annotations.Set(attr::kPipelineMbarPhaseExpr, phase_);
    return MakeAnnotatedCall(call->ty, call->op.as_or_throw<Op>(),
                             fields.operands, fields.annotations, call->loc);
  }

private:
  PrimExpr phase_;
};

class AsyncCommitWaitAttrLowerer : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate;
  using LayoutMapMutator::Mutate_;
  static Stmt Lower(const Stmt &stmt) {
    auto mutator = make_object<AsyncCommitWaitAttrLowerer>();
    return mutator->Mutate(stmt).ValueOrUnchanged(stmt);
  }
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    String key = ScopeKey(op);
    if (key == "async_commit_queue_scope") {
      auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
      Stmt commit =
          Evaluate(Call(PrimType::Void(), Op::Get("tl.ptx_commit_group"), {}));
      if (IsNoOp(body))
        return commit;
      auto seq = body->seq;
      seq.push_back(commit);
      return SeqStmt(seq, op->loc);
    }
    if (key == "async_wait_queue_scope") {
      const auto *inner = SingleRegion(op->body);
      TVM_FFI_CHECK(inner && ScopeKey(inner) == "async_wait_inflight_count",
                    ValueError)
          << "An async wait scope requires an explicit inflight count";
      auto body = Mutate(inner->body, mode).ValueOrUnchanged(inner->body);
      Stmt wait = Evaluate(Call(PrimType::Void(), Op::Get("tl.ptx_wait_group"),
                                {ScopeValue(inner)}));
      auto seq = body->seq;
      seq.insert(seq.begin(), wait);
      return SeqStmt(seq, op->loc);
    }
    if (key == "async_wait_inflight_count")
      return Mutate(op->body, mode).ValueOrUnchanged(op->body);
    return LayoutMapMutator::Mutate_(op, mode);
  }
};

} // namespace

bool IsReplayableScalarBindBlock(const PipelineStage &block,
                                 const TensorSet &writes,
                                 const PipelineMemoryAnalysis *memory) {
  return !block->unknown_access &&
         tl::IsReplayablePipelineBind(
             block->body, block->reads, writes,
             [memory](const TensorVar &a, const TensorVar &b) {
               return memory->MayAlias(a, b);
             });
}

TensorSet CollectPipelineWriteBuffers(const Array<PipelineStage> &blocks) {
  TensorSet writes;
  for (const auto &block : blocks)
    for (const auto &write : block->writes)
      writes.insert(RegionTensor(write));
  return writes;
}

PipelineStage MakeBlock(const Stmt &body, const PipelineMemoryAnalysis *memory,
                        const Optional<Target> &target) {
  memory->ObserveDefinitions(body);
  auto stage = make_object<PipelineStageNode>();
  stage->body = body;
  stage->loc = body->loc;
  if (const auto *region = SingleStatement(body).as<RegionStmtObj>();
      ascend::IsAscendBlock(region)) {
    stage->scope = GetRef<RegionStmt>(region);
    stage->body = SingleStatement(region->body);
    stage->name_hint = region->attrs.GetAttr<ffi::String>("name").value();
  } else if (IsLexicalScope(region)) {
    stage->scope = GetRef<RegionStmt>(region);
    stage->body = SingleStatement(region->body);
  }
  auto collector = make_object<PipelineAccessCollector>(
      memory, target.value_or(Target("c")));
  collector->Visit(stage->body);
  stage->reads = collector->GetReads().Map(
      [memory](const TensorRegion &r) { return memory->CanonicalRegion(r); });
  stage->writes = collector->GetWrites().Map(
      [memory](const TensorRegion &r) { return memory->CanonicalRegion(r); });
  stage->unknown_access = collector->HasUnknownAccess();
  return PipelineStage(stage);
}

Stmt ReifyStage(const PipelineStage &stage,
                PrimExpr predicate = IntImm::Bool(true)) {
  Stmt body = stage->body;
  if (stage->scope)
    body = WithScopeBody(stage->scope.value().get(), body);
  if (!IsOne(predicate))
    body = If(predicate, SeqStmt(body), std::nullopt, stage->loc);
  return body;
}

PipelineStage RemakeStage(const PipelineStage &stage, Stmt body,
                          const PipelineMemoryAnalysis *memory,
                          const Optional<Target> &target) {
  auto result = MakeBlock(body, memory, target);
  result.CopyOnWrite()->scope = stage->scope;
  result.CopyOnWrite()->name_hint = stage->name_hint;
  return result;
}

Array<TensorVar> CollectUsedPipelineBuffers(
    const Stmt &stmt, const PipelineMemoryAnalysis *memory,
    const TensorSet &allocated, const Optional<Target> &target) {
  auto stage = MakeBlock(stmt, memory, target);
  TensorSet used;
  for (const auto &regions : {stage->reads, stage->writes})
    for (const auto &region : regions) {
      TensorVar tensor = RegionTensor(region);
      if (allocated.count(tensor))
        used.insert(tensor);
    }
  return Array<TensorVar>(used.begin(), used.end());
}

bool ContainsPipelineAsyncControlAttrs(const Stmt &stmt) {
  bool found = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      stmt, [&](const ObjectRef &obj) -> ffi::Expected<ffi::WalkResult> {
        if (found) {
          return ffi::WalkResult::Advance();
        }
        if (const auto *attr = obj.as<RegionStmtObj>()) {
          if (ScopeKey(attr) == "async_commit_queue_scope" ||
              ScopeKey(attr) == "async_wait_queue_scope" ||
              ScopeKey(attr) == "async_wait_inflight_count") {
            found = true;
            return ffi::WalkResult::Advance();
          }
        }
        return ffi::WalkResult::Advance();
      });
  return found;
}

Stmt AnnotateSimtProducer(const Stmt &stmt, Optional<Target> target) {
  return SimtProducerAnnotator::Annotate(stmt, std::move(target));
}

Stmt AnnotateTileOpMbarPhase(const Stmt &stmt, PrimExpr phase_expr) {
  return TileOpMbarPhaseAnnotator::Annotate(stmt, std::move(phase_expr));
}

Stmt LowerAsyncCommitWaitAttrs(const Stmt &stmt) {
  return AsyncCommitWaitAttrLowerer::Lower(stmt);
}

/*!
 * \brief Build the dependency graph among a array of blocks.
 * \param[in] blocks The array of blocks.
 * \param[out] dep_src2dst Optional, a map to store dependency edges from the
 * source to the destination. \param[out] dep_dst2src Optional, a map to store
 * dependency edges from the destination to the source.
 * \param[in] thread_domains Domains used to compare accesses across threads.
 */

void BuildDependencyGraph(const Array<PipelineStage> &blocks,
                          BlockDependencyGraph *src_to_dst,
                          BlockDependencyGraph *dst_to_src,
                          const Map<Var, sym::IntSet> &threads,
                          const PipelineMemoryAnalysis *memory) {
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (size_t j = i + 1; j < blocks.size(); ++j) {
      bool depends = false;
      auto conflict = [&](const Array<TensorRegion> &a,
                          const Array<TensorRegion> &b) {
        for (const auto &left : a)
          for (const auto &right : b)
            if (memory->MayConflict(left, right, threads))
              return true;
        return false;
      };
      depends = conflict(blocks[i]->writes, blocks[j]->reads) ||
                conflict(blocks[i]->reads, blocks[j]->writes) ||
                conflict(blocks[i]->writes, blocks[j]->writes);
      if (depends) {
        if (src_to_dst)
          (*src_to_dst)[blocks[i]].push_back(blocks[j]);
        if (dst_to_src)
          (*dst_to_src)[blocks[j]].push_back(blocks[i]);
      }
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

class CopyToTmaCopyRewriter : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate;
  using LayoutMapMutator::Mutate_;
  CopyToTmaCopyRewriter(TensorVar barrier, PrimExpr id, bool arrive)
      : barrier_(std::move(barrier)), id_(std::move(id)), arrive_(arrive) {}
  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    auto call = LayoutMapMutator::Mutate_(op, mode)
                    .ValueOrUnchanged(GetRef<Call>(op))
                    .as_or_throw<Call>();
    auto name = call->op.as<Op>();
    if (!name || (name->get()->name != "tl.tileop.copy" &&
                  name->get()->name != "tl.tileop.im2col" &&
                  name->get()->name != "tl.tileop.c2d_im2col"))
      return call;
    auto fields = DecodeTileOpArguments(call.get());
    fields.annotations.Set("barrier", MakeBarrierRef(barrier_, id_));
    fields.annotations.Set("emit_arrive", arrive_);
    Op result_op = *name;
    if (name->get()->name == "tl.tileop.copy") {
      result_op = Op::Get("tl.tileop.tma_copy");
      fields.annotations.Set("is_tma_copy", true);
    }
    return MakeAnnotatedCall(call->ty, result_op, fields.operands,
                             fields.annotations, call->loc);
  }

private:
  TensorVar barrier_;
  PrimExpr id_;
  bool arrive_;
};

Map<TensorVar, TensorVar>
ExpandPipelineBarriers(Array<PipelineStage> &stages, PipelineInfo &info,
                       const PipelineMemoryAnalysis *memory,
                       TensorSet &allocated, Array<TensorVar> &local_allocs,
                       Array<TensorVar> &pipeline_allocs, PrimVar loop_var,
                       PrimExpr loop_min, int count,
                       const Optional<Target> &target) {
  if (count <= 1)
    return {};
  TensorSet local(local_allocs.begin(), local_allocs.end()), arrived,
      referenced;
  for (const auto &stage : stages) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        stage->body,
        [&](const ObjectRef &obj) -> ffi::Expected<ffi::WalkResult> {
          if (auto tensor = obj.as<TensorVar>();
              tensor && (tensor->scope().name() == "shared.barrier" ||
                         tensor->scope().name() == "shared.cluster_barrier"))
            referenced.insert(*tensor);
          if (const auto *call = obj.as<CallObj>();
              call && call->op.same_as(tl::ptx_arrive_barrier()) &&
              !call->args.empty())
            if (const auto *load = call->args[0].as<TensorLoadObj>())
              arrived.insert(load->source.as_or_throw<TensorVar>());
          return ffi::WalkResult::Advance();
        });
  }
  Map<TensorVar, TensorVar> replacements;
  for (const auto &tensor : referenced) {
    if (!local.count(tensor) && !arrived.count(tensor))
      continue;
    TensorVar replacement = ExpandPipelineTensor(tensor, count, true);
    replacements.Set(tensor, replacement);
    allocated.erase(tensor);
    allocated.insert(replacement);
    for (auto *array : {&local_allocs, &pipeline_allocs})
      for (size_t i = 0; i < array->size(); ++i)
        if ((*array)[i].same_as(tensor))
          array->Set(i, replacement);
  }
  if (replacements.empty())
    return {};
  PrimExpr epoch = loop_var - loop_min;
  PrimExpr slot = floormod(epoch, count);
  PrimExpr phase = floormod(floordiv(epoch, count), 2);
  auto rewrite = make_object<PipelineTensorRewriter>(
      replacements, PipelineTensorRewriter::Mode::kBarrier, slot, phase,
      loop_var, loop_min);
  for (size_t i = 0; i < stages.size(); ++i) {
    auto old = stages[i];
    auto next = RemakeStage(old, rewrite->Rewrite(old->body), memory, target);
    auto annotation = info.at(old);
    info.erase(old);
    info.emplace(next, annotation);
    stages.Set(i, next);
  }
  return replacements;
}

Optional<TensorVar> RewritePipelineTmaBarriers(
    Array<PipelineStage> &stages, PipelineInfo &info, const Array<int64_t> &tma,
    const PipelineMemoryAnalysis *memory, TensorSet &allocated,
    Array<TensorVar> &local_allocs, PrimVar loop_var, PrimExpr loop_min,
    int count, const Optional<Target> &target) {
  int last = -1;
  for (size_t i = 0; i < tma.size(); ++i)
    if (tma[i])
      last = static_cast<int>(i);
  if (last < 0)
    return std::nullopt;
  TensorVar barrier = CreateMBarrierBuffer("pipeline_mbar", 1);
  local_allocs.push_back(barrier);
  allocated.insert(barrier);
  for (size_t i = 0; i < stages.size(); ++i) {
    if (!tma[i])
      continue;
    auto old = stages[i];
    auto rewrite = make_object<CopyToTmaCopyRewriter>(
        barrier, IntImm::Int32(0), i == static_cast<size_t>(last));
    auto next =
        RemakeStage(old, rewrite->Mutate(old->body).ValueOrUnchanged(old->body),
                    memory, target);
    auto annotation = info.at(old);
    info.erase(old);
    info.emplace(next, annotation);
    stages.Set(i, next);
  }
  for (size_t i = 0; i < stages.size(); ++i) {
    auto old = stages[i];
    if (info.at(old).stage == 0)
      continue;
    Stmt wait =
        Evaluate(Call(PrimType::Void(), mbarrier_wait_parity(),
                      {MakeBarrierRef(barrier, IntImm::Int32(0)),
                       floormod(floordiv(loop_var - loop_min, count), 2)}));
    auto next = RemakeStage(old, SeqStmt({wait, old->body}), memory, target);
    auto annotation = info.at(old);
    info.erase(old);
    info.emplace(next, annotation);
    stages.Set(i, next);
    break;
  }
  return barrier;
}

class PipelineBodyRewriter {
public:
  PipelineBodyRewriter(const PipelineMemoryAnalysis *memory,
                       const Map<TensorVar, TensorVar> &remap, const For &loop,
                       bool access_all_versions, const Optional<Target> &target)
      : memory_(memory), target_(target),
        mutator_(make_object<PipelineTensorRewriter>(
            remap, PipelineTensorRewriter::Mode::kVersion,
            loop->var - loop->begin, std::nullopt, std::nullopt, std::nullopt,
            memory)) {}
  Stmt operator()(const Stmt &stmt) { return mutator_->Rewrite(stmt); }
  PipelineStage operator()(const PipelineStage &stage) {
    return RemakeStage(stage, (*this)(stage->body), memory_, target_);
  }

private:
  const PipelineMemoryAnalysis *memory_;
  Optional<Target> target_;
  ObjectPtr<PipelineTensorRewriter> mutator_;
};

/*!
 * \brief Rewriter for the software pipeline that rewrite a loop into a
 * pipelined one.
 */
class PipelineRewriter {
public:
  /*!
   * \brief Constructor of PipelineRewriter.
   * \param buffer_data_to_buffer The map from buffer data to buffer.
   * \param pipeline_allocs All buffers that need multi-versioning in the
   * pipeline. This includes buffers allocated in the pipeline block and
   * buffers allocated in outer blocks that are used in the pipeline.
   * \param local_allocs Buffers that are allocated in the pipeline block
   * itself. These buffers will be re-allocated in the rewritten block.
   * Buffers in pipeline_allocs but not in local_allocs are allocated in outer
   * blocks and should not be re-allocated.
   * \param pipeline_loop The original loop to be software pipelined.
   * \param pipeline_info The pipeline annotation information.
   * \param scalar_binding_blocks Replayable scalar Bind statements from the
   * pipeline body.
   * \param target The compilation target, when available.
   * \param thread_domains Domains used to compare accesses across threads.
   */
  PipelineRewriter(const PipelineMemoryAnalysis *memory,
                   const Array<TensorVar> &pipeline_allocs,
                   const Array<TensorVar> &local_allocs,
                   const For &pipeline_loop, const PipelineInfo &pipeline_info,
                   const Array<PipelineStage> &scalar_binding_blocks,
                   Optional<Target> target,
                   Map<Var, sym::IntSet> thread_domains)
      : memory_(memory), pipeline_allocs_(pipeline_allocs),
        local_allocs_(local_allocs), pipeline_loop_(pipeline_loop),
        pipeline_info_(pipeline_info),
        scalar_binding_blocks_(scalar_binding_blocks),
        target_(std::move(target)), thread_domains_(std::move(thread_domains)) {
  }

  Stmt BuildPipeline() {
    // Step 1: Analyze accesses to the buffers in the pipeline and compute the
    // number of versions need to maintain for each buffer.
    std::unordered_map<TensorVar, BufferAccessInfo, ObjectPtrHash,
                       ObjectPtrEqual>
        infos = GetBufferAccessInfo();
    for (const TensorVar &buffer : pipeline_allocs_) {
      auto it = infos.find(buffer);
      if (it == infos.end()) {
        // TensorVar is not accessed in the pipeline blocks, skip it
        continue;
      }
      int num_versions = ComputeBufferVersions(buffer, it->second);
      if (num_versions > 1) {
        buffer_remap_.Set(buffer, RewriteAllocBuffer(buffer, num_versions));
      }
    }
    std::vector<std::pair<int, PipelineStage>> ordered_blocks;
    for (const auto &[block, anno] : pipeline_info_) {
      ordered_blocks.emplace_back(anno.order, block);
    }
    std::sort(
        ordered_blocks.begin(), ordered_blocks.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (const auto &[_, block] : ordered_blocks) {
      ordered_stmts_.push_back(block);
    }
    CollectScalarBindings();

    // Step 2: Emit the pipeline prologue, body and epilogue.
    Optional<int64_t> pipeline_num_stages =
        GetPipelineNumStages(pipeline_loop_.get());
    Stmt prologue = EmitImpl(pipeline_loop_->begin,
                             pipeline_loop_->begin + max_stage_, true, true);
    Stmt body = EmitImpl(
        pipeline_loop_->begin + max_stage_,
        pipeline_loop_->begin + LoopExtent(pipeline_loop_.get()), false, false);
    Stmt epilogue = EmitImpl(
        pipeline_loop_->begin + LoopExtent(pipeline_loop_.get()),
        pipeline_loop_->begin + LoopExtent(pipeline_loop_.get()) + max_stage_,
        true, true);

    Array<Stmt> pipeline_parts;
    for (const Stmt &part : {prologue, body, epilogue}) {
      for (const Stmt &stmt : FlattenTopLevelSeq(part)) {
        pipeline_parts.push_back(stmt);
      }
    }

    Stmt stmt = pipeline_parts.size() == 1 ? pipeline_parts[0]
                                           : SeqStmt(pipeline_parts);
    auto relaxer = make_object<AsyncPipelineLoopWaitRelaxer>(this);
    stmt = relaxer->Mutate(stmt).ValueOrUnchanged(stmt);
    Array<Stmt> relaxed_pipeline_parts = FlattenTopLevelSeq(stmt);
    relaxed_pipeline_parts =
        RelaxTrailingConsumerWaits(std::move(relaxed_pipeline_parts),
                                   PipelinedRetainGroups(pipeline_num_stages));
    stmt = relaxed_pipeline_parts.size() == 1 ? relaxed_pipeline_parts[0]
                                              : SeqStmt(relaxed_pipeline_parts);

    // Step 3: Make a new block only when the rewritten pipeline owns local
    // allocations.  When all buffers are already allocated by an outer block
    // (the common TileLang kernel-root case), an extra synthetic scope would
    // only carry a very large inferred read/write region and make the IR hard
    // to inspect.
    if (local_allocs_.empty()) {
      return stmt;
    }

    Array<Stmt> allocation_body;
    for (const TensorVar &allocation : local_allocs_) {
      auto tensor = buffer_remap_.Get(allocation).value_or(allocation);
      allocation_body.push_back(AllocatePipelineTensor(tensor, memory_));
    }
    for (const Stmt &part : FlattenTopLevelSeq(stmt))
      allocation_body.push_back(part);
    return RegionStmt(tl::lexical_scope_op(), {}, {}, DictAttrs(),
                      SeqStmt(allocation_body), {});
  }

  /*!
   * \brief Get the buffer remapping created during pipeline rewriting.
   * This is used to update alloc_buffers in outer blocks.
   */
  const Map<TensorVar, TensorVar> &GetBufferRemap() const {
    return buffer_remap_;
  }

private:
  struct ScalarBinding {
    Var var;
    Expr value;
    Location loc;
  };

  using ScalarBindingMap =
      std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual>;

  void CollectScalarBindings() {
    scalar_bindings_.clear();
    scalar_binding_map_.clear();
    for (const PipelineStage &block : scalar_binding_blocks_) {
      if (const auto *bind = block->body.as<BindObj>()) {
        if (!scalar_binding_map_.count(bind->var)) {
          scalar_binding_map_.emplace(bind->var, scalar_bindings_.size());
          scalar_bindings_.push_back({bind->var, bind->value, bind->loc});
        }
      }
    }
  }

  VarSet FindScalarBindingUses(const Array<Var> &undefined_vars) const {
    VarSet uses;
    for (const Var &var : undefined_vars) {
      if (scalar_binding_map_.count(var)) {
        uses.insert(var);
      }
    }
    return uses;
  }

  VarSet FindScalarBindingUses(const Stmt &stmt) const {
    return FindScalarBindingUses(UndefinedVars(stmt, Array<Var>{}));
  }

  VarSet FindScalarBindingUses(const Expr &expr) const {
    return FindScalarBindingUses(UndefinedVars(expr));
  }

  void AppendScalarBinding(size_t binding_index, VarSet *emitted,
                           VarSet *visiting,
                           std::vector<size_t> *binding_indices) const {
    const ScalarBinding &binding = scalar_bindings_[binding_index];
    if (emitted->count(binding.var)) {
      return;
    }
    ICHECK(!visiting->count(binding.var))
        << "InjectSoftwarePipeline: cyclic scalar Bind dependency involving "
        << binding.var;

    visiting->insert(binding.var);
    VarSet deps = FindScalarBindingUses(binding.value);
    for (const ScalarBinding &candidate : scalar_bindings_) {
      if (!deps.count(candidate.var)) {
        continue;
      }
      auto it = scalar_binding_map_.find(candidate.var);
      ICHECK(it != scalar_binding_map_.end());
      AppendScalarBinding(it->second, emitted, visiting, binding_indices);
    }
    visiting->erase(binding.var);

    emitted->insert(binding.var);
    binding_indices->push_back(binding_index);
  }

  std::vector<size_t> RequiredScalarBindings(const Stmt &stmt) const {
    std::vector<size_t> binding_indices;
    if (scalar_bindings_.empty()) {
      return binding_indices;
    }

    VarSet uses = FindScalarBindingUses(stmt);
    if (uses.empty()) {
      return binding_indices;
    }

    VarSet emitted;
    VarSet visiting;
    for (const ScalarBinding &binding : scalar_bindings_) {
      if (!uses.count(binding.var)) {
        continue;
      }
      auto it = scalar_binding_map_.find(binding.var);
      ICHECK(it != scalar_binding_map_.end());
      AppendScalarBinding(it->second, &emitted, &visiting, &binding_indices);
    }
    return binding_indices;
  }

  Stmt RewriteScalarBindingForAccess(size_t binding_index,
                                     const PrimExpr &access_index) {
    const ScalarBinding &binding = scalar_bindings_[binding_index];
    Stmt bind = Bind(binding.var, binding.value, binding.loc);
    bind = PipelineBodyRewriter(memory_, buffer_remap_, pipeline_loop_,
                                max_stage_ != 1, target_)(bind);
    bind = tirx::SubstituteWithDataTypeLegalization(
        bind,
        [values = ffi::Map<Var, PrimExpr>{{pipeline_loop_->var, access_index}}](
            const Var &var) -> ffi::Optional<PrimExpr> {
          return values.Get(var);
        });
    return bind;
  }

  PipelineStage ReplayScalarBindings(PipelineStage block) {
    auto bindings = RequiredScalarBindings(block->body);
    if (bindings.empty())
      return block;
    Array<Stmt> sequence;
    for (size_t index : bindings) {
      const auto &binding = scalar_bindings_[index];
      sequence.push_back(Bind(binding.var, binding.value, binding.loc));
    }
    for (const Stmt &statement : FlattenTopLevelSeq(block->body))
      sequence.push_back(statement);
    return RemakeStage(block, SeqStmt(sequence), memory_, target_);
  }

  /*!
   * \brief Analyze accesses to the buffers in the software pipeline.
   *
   * This method check the 'define' and 'use' stage of the buffers in the
   * software pipeline, which can be used to compute the number of versions
   * needed to maintain after rewriting.
   */
  std::unordered_map<TensorVar, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual>
  GetBufferAccessInfo() {
    std::unordered_map<TensorVar, BufferAccessInfo, ObjectPtrHash,
                       ObjectPtrEqual>
        infos;
    for (const auto &pair : pipeline_info_) {
      const PipelineStage &block = pair.first;
      int stage = pair.second.stage;
      max_stage_ = std::max(max_stage_, stage);

      for (const TensorRegion &write : block->writes) {
        if (!infos.count(RegionTensor(write))) {
          infos.emplace(RegionTensor(write), BufferAccessInfo{});
        }
        auto &info = infos.at(RegionTensor(write));
        if (info.def == -1) {
          info.def = stage;
        } else {
          info.def = std::min(info.def, stage);
        }
      }

      for (const TensorRegion &read : block->reads) {
        if (!infos.count(RegionTensor(read))) {
          infos.emplace(RegionTensor(read), BufferAccessInfo{});
        }
        auto &info = infos.at(RegionTensor(read));
        info.use = std::max(info.use, stage);
      }
    }
    return infos;
  }

  /*!
   * \brief Compute the number of versions need to maintain for buffer accessed
   * in the software pipeline.
   *
   * This method applies liveness analysis to the target buffer to compute the
   * number of versions need to maintain during the software pipeline.
   * Annotation `attr::double_buffer_scope` is handled here which provides a way
   * to override the result of the analysis. Additional double buffering in the
   * software pipeline can be useful to eliminate synchronizations in GPU
   * devices.
   *
   * \param buffer The target buffer
   * \param buffer_info The access information of the target buffer.
   * \return The number of versions required for the target buffer.
   */
  int ComputeBufferVersions(const TensorVar &buffer,
                            const BufferAccessInfo &buffer_info) {
    if (buffer_info.def == -1) {
      // Keep the original number of versions as buffers defined outside the
      // software pipeline should not be mutated.
      return 1;
    }

    // `use - def + 1` is a upper bound of the needed versions
    // We optimize a few case where the number of versions can be smaller than
    // the upper bound
    int num_versions = buffer_info.use - buffer_info.def + 1;
    if (num_versions >= 2) {
      // A special case when `use - def + 1 == 2`. Double buffering is only
      // needed in this case when these exists a reader block_i and a writer
      // block_j such that order(block_i) < order(block_j) and stage(block_i) <
      // stage(block_j) and the access regions of block_i and block_j overlap.
      bool need_multi_version = false;
      for (const auto &pair1 : pipeline_info_) {
        const PipelineStage &writer_block = pair1.first;
        const auto &writer_info = pair1.second;

        auto it1 = std::find_if(
            writer_block->writes.begin(), writer_block->writes.end(),
            [&](const TensorRegion &buffer_region) {
              return RegionTensor(buffer_region).same_as(buffer);
            });
        if (it1 == writer_block->writes.end()) {
          continue;
        }

        for (const auto &pair2 : pipeline_info_) {
          const PipelineStage &reader_block = pair2.first;
          const auto &reader_info = pair2.second;
          auto it2 = std::find_if(
              reader_block->reads.begin(), reader_block->reads.end(),
              [&](const TensorRegion &buffer_region) {
                return RegionTensor(buffer_region).same_as(buffer);
              });
          if (it2 == reader_block->reads.end()) {
            continue;
          }
          if (writer_info.order < reader_info.order &&
              writer_info.stage < reader_info.stage &&
              PipelineRegionsMayConflict(*it1, *it2, thread_domains_)) {
            need_multi_version = true;
            break;
          }
        }
      }
      if (!need_multi_version) {
        num_versions--;
      }
    }
    return num_versions;
  }

  /*!
   * \brief Rewrite buffer allocation to keep multiple versions of original
   * buffer for pipelined accesses. \param buffer The buffer to be resized.
   * \param num_versions The number of versions to keep.
   * \return The resized buffer.
   */
  TensorVar RewriteAllocBuffer(const TensorVar &buffer, int num_versions) {
    return ExpandPipelineTensor(buffer, num_versions);
  }

  struct AsyncStateGlobal {
    TensorSet dst_buffers;
    BufferCommitGroupMap buffer_to_commit_group;
    int commit_group_count{0};
    Optional<PrimExpr> producer_head{PrimExpr(-1)};

    bool writes(const TensorVar &buffer) const {
      return dst_buffers.count(buffer) > 0;
    }
  };

  struct AsyncStateLocal {
    struct PendingWait {
      int insert_before{-1};
      Optional<PrimExpr> wait_count;

      bool valid() const { return wait_count.has_value(); }
    };

    TensorSet seen;
    Optional<PrimExpr> producer_head;
    Optional<PrimExpr> predicate;
    std::vector<std::vector<size_t>> commit_groups;
    std::map<int, PendingWait> pending_waits;
    std::unordered_map<int, int> annotated_group_to_commit_group;
    bool consumed{false};
  };

  struct RewrittenStmtInfo {
    int stage;
    PrimExpr predicate;
    Array<TensorRegion> reads;
    Array<TensorRegion> writes;
    PrimExpr access_index;
    bool is_async;
    Stmt stmt;
  };

  struct FinalStmtInfo {
    int stage;
    PrimExpr access_index;
    PrimExpr predicate;
    Stmt stmt;
  };

  enum class AsyncSyncStmtKind { kOther, kCommit, kWaitStatic, kWaitDynamic };

  struct ClassifiedAsyncSyncStmt {
    AsyncSyncStmtKind kind{AsyncSyncStmtKind::kOther};
    int wait_n{0};
  };

  struct AsyncSyncSummary {
    int commit{0};
    int wait{0};
  };

  enum class HeadAsyncSyncKind {
    kNone,
    kCommit,
    kWaitStatic,
    kWaitDynamic,
    kBlocked,
  };

  struct HeadAsyncSyncInfo {
    HeadAsyncSyncKind kind{HeadAsyncSyncKind::kNone};
    int wait_n{0};

    bool IsBoundary() const {
      return kind == HeadAsyncSyncKind::kCommit ||
             kind == HeadAsyncSyncKind::kWaitDynamic ||
             kind == HeadAsyncSyncKind::kBlocked;
    }
  };

  enum class HeadSeqMode {
    kSingletonOnly,
    kTakeFirstElement,
  };

  struct DeterministicNoWaitCommitEffect {
    bool deterministic{true};
    bool has_wait{false};
    int commit_groups{0};

    static DeterministicNoWaitCommitEffect Unknown() {
      DeterministicNoWaitCommitEffect effect;
      effect.deterministic = false;
      return effect;
    }

    static DeterministicNoWaitCommitEffect Wait() {
      DeterministicNoWaitCommitEffect effect;
      effect.has_wait = true;
      return effect;
    }
  };

  // Analyze a stmt for one specific question used by wait relaxation:
  // can we prove that it contributes a deterministic number of commit groups
  // without crossing a wait boundary? The analyzer exposes the effect as
  // structured state instead of overloading std::optional<int> with both
  // "unknown" and "has wait" meanings.
  class DeterministicNoWaitCommitAnalyzer {
  public:
    explicit DeterministicNoWaitCommitAnalyzer(const PipelineRewriter *rewriter)
        : rewriter_(rewriter) {}

    DeterministicNoWaitCommitEffect Analyze(const Stmt &stmt) const {
      if (stmt.as<BindObj>()) {
        return DeterministicNoWaitCommitEffect{};
      }
      if (const auto *attr = stmt.as<RegionStmtObj>()) {
        return AnalyzeAttr(attr);
      }
      if (const auto *seq = stmt.as<SeqStmtObj>()) {
        DeterministicNoWaitCommitEffect effect;
        for (const Stmt &s : seq->seq) {
          effect = Combine(effect, Analyze(s));
          if (!effect.deterministic) {
            return effect;
          }
        }
        return effect;
      }

      if (const auto *for_node = stmt.as<ForObj>()) {
        return AnalyzeFor(for_node);
      }
      if (stmt.as<IfObj>()) {
        return DeterministicNoWaitCommitEffect::Unknown();
      }
      if (rewriter_->ContainsAsyncSyncScopes(stmt)) {
        return DeterministicNoWaitCommitEffect::Unknown();
      }
      return {};
    }

  private:
    DeterministicNoWaitCommitEffect
    AnalyzeAttr(const RegionStmtObj *attr) const {
      if (PipelineRewriter::IsAsyncWaitQueueScope(attr) ||
          PipelineRewriter::IsAsyncWaitInflightCount(attr)) {
        return DeterministicNoWaitCommitEffect::Wait();
      }
      if (PipelineRewriter::IsAsyncCommitQueueScope(attr)) {
        auto effect = Analyze(attr->body);
        if (!effect.deterministic) {
          return effect;
        }
        ++effect.commit_groups;
        return effect;
      }
      if (!TransparentScope(attr))
        return DeterministicNoWaitCommitEffect::Unknown();
      return Analyze(attr->body);
    }

    DeterministicNoWaitCommitEffect AnalyzeFor(const ForObj *for_node) const {
      auto extent_imm = TryConstInt(LoopExtent(for_node));
      if (!extent_imm || *extent_imm < 0) {
        return DeterministicNoWaitCommitEffect::Unknown();
      }
      auto effect = Analyze(for_node->body);
      if (!effect.deterministic) {
        return effect;
      }
      effect.commit_groups *= static_cast<int>(*extent_imm);
      return effect;
    }

    static DeterministicNoWaitCommitEffect
    Combine(const DeterministicNoWaitCommitEffect &lhs,
            const DeterministicNoWaitCommitEffect &rhs) {
      if (!lhs.deterministic || !rhs.deterministic) {
        return DeterministicNoWaitCommitEffect::Unknown();
      }
      DeterministicNoWaitCommitEffect effect;
      effect.has_wait = lhs.has_wait || rhs.has_wait;
      effect.commit_groups = lhs.commit_groups + rhs.commit_groups;
      return effect;
    }

    const PipelineRewriter *rewriter_;
  };

  static bool IsAsyncCommitQueueScope(const RegionStmtObj *attr) {
    return attr && ScopeKey(attr) == "async_commit_queue_scope";
  }

  static bool IsAsyncWaitQueueScope(const RegionStmtObj *attr) {
    return attr && ScopeKey(attr) == "async_wait_queue_scope";
  }

  static bool IsAsyncWaitInflightCount(const RegionStmtObj *attr) {
    return attr && ScopeKey(attr) == "async_wait_inflight_count";
  }

  static int
  PipelinedRetainGroups(const Optional<int64_t> &pipeline_num_stages) {
    int retain = 1;
    if (pipeline_num_stages) {
      retain = std::max(0, static_cast<int>(pipeline_num_stages.value()) - 1);
    }
    return retain;
  }

  Array<Stmt> FlattenTopLevelSeq(const Stmt &stmt) const {
    if (const auto *seq = stmt.as<SeqStmtObj>()) {
      return seq->seq;
    }
    return {stmt};
  }

  std::optional<int>
  TryGetStaticAsyncWaitCount(const RegionStmtObj *attr) const {
    if (!IsAsyncWaitQueueScope(attr)) {
      return std::nullopt;
    }
    const auto *inner = SingleRegion(attr->body);
    if (!IsAsyncWaitInflightCount(inner)) {
      return std::nullopt;
    }
    auto imm = TryConstInt(ScopeValue(inner));
    if (!imm) {
      return std::nullopt;
    }
    return static_cast<int>(*imm);
  }

  Stmt MakeStaticAsyncWaitStmtLike(const RegionStmtObj *attr,
                                   int new_wait_n) const {
    const auto *inner = SingleRegion(attr->body);
    if (!IsAsyncWaitInflightCount(inner))
      return GetRef<RegionStmt>(attr);
    PrimExpr new_wait = MakeConst(ScopeValue(inner).ty(), new_wait_n);
    Stmt body = ScopeStmt(ScopeNode(inner), ScopeKey(inner), new_wait,
                          inner->body, inner->loc);
    return WithScopeBody(attr, body);
  }

  class HeadAsyncSyncAnalyzer
      : public StmtFunctor<HeadAsyncSyncInfo(const Stmt &)> {
  public:
    static HeadAsyncSyncInfo Analyze(const PipelineRewriter *rewriter,
                                     const Stmt &stmt, HeadSeqMode seq_mode) {
      HeadAsyncSyncAnalyzer analyzer(rewriter, seq_mode);
      return analyzer(stmt);
    }

    HeadAsyncSyncAnalyzer(const PipelineRewriter *rewriter,
                          HeadSeqMode seq_mode)
        : rewriter_(rewriter), seq_mode_(seq_mode) {}

    HeadAsyncSyncInfo Dispatch_(const RegionStmtObj *op) final {
      if (!TransparentScope(op))
        return {HeadAsyncSyncKind::kBlocked, 0};
      if (IsAsyncWaitQueueScope(op)) {
        if (auto wait_n = rewriter_->TryGetStaticAsyncWaitCount(op)) {
          return {HeadAsyncSyncKind::kWaitStatic, *wait_n};
        }
        return {HeadAsyncSyncKind::kWaitDynamic, 0};
      }
      if (IsAsyncCommitQueueScope(op)) {
        return {HeadAsyncSyncKind::kCommit, 0};
      }
      if (IsAsyncWaitInflightCount(op)) {
        return {HeadAsyncSyncKind::kBlocked, 0};
      }
      return Dispatch(op->body);
    }

    HeadAsyncSyncInfo Dispatch_(const SeqStmtObj *op) final {
      if (op->seq.empty()) {
        return {};
      }
      if (seq_mode_ == HeadSeqMode::kSingletonOnly && op->seq.size() != 1) {
        return {HeadAsyncSyncKind::kBlocked, 0};
      }
      return Dispatch(op->seq[0]);
    }

    HeadAsyncSyncInfo DispatchDefault_(const Object *) final { return {}; }

  private:
    const PipelineRewriter *rewriter_;
    HeadSeqMode seq_mode_;
  };

  HeadAsyncSyncInfo AnalyzeHeadAsyncSync(const Stmt &stmt,
                                         HeadSeqMode seq_mode) const {
    return HeadAsyncSyncAnalyzer::Analyze(this, stmt, seq_mode);
  }

  ClassifiedAsyncSyncStmt ClassifySimpleAsyncSyncStmt(const Stmt &stmt) const {
    HeadAsyncSyncInfo info =
        AnalyzeHeadAsyncSync(stmt, HeadSeqMode::kSingletonOnly);
    switch (info.kind) {
    case HeadAsyncSyncKind::kCommit:
      return {AsyncSyncStmtKind::kCommit, 0};
    case HeadAsyncSyncKind::kWaitStatic:
      return {AsyncSyncStmtKind::kWaitStatic, info.wait_n};
    case HeadAsyncSyncKind::kWaitDynamic:
      return {AsyncSyncStmtKind::kWaitDynamic, 0};
    default:
      return {};
    }
  }

  bool ContainsAsyncSyncScopes(const Stmt &stmt) const {
    bool found = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        stmt, [&](const ObjectRef &obj) -> ffi::Expected<ffi::WalkResult> {
          if (found) {
            return ffi::WalkResult::Advance();
          }
          if (const auto *attr = obj.as<RegionStmtObj>()) {
            if (IsAsyncCommitQueueScope(attr) || IsAsyncWaitQueueScope(attr)) {
              found = true;
            }
          }
          return ffi::WalkResult::Advance();
        });
    return found;
  }

  bool ContainsAsyncCommitScopes(const Stmt &stmt) const {
    bool found = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        stmt, [&](const ObjectRef &obj) -> ffi::Expected<ffi::WalkResult> {
          if (found) {
            return ffi::WalkResult::Advance();
          }
          if (const auto *attr = obj.as<RegionStmtObj>()) {
            if (IsAsyncCommitQueueScope(attr)) {
              found = true;
            }
          }
          return ffi::WalkResult::Advance();
        });
    return found;
  }

  AsyncSyncSummary SummarizeAsyncSyncScopes(const Stmt &stmt) const {
    AsyncSyncSummary summary;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        stmt, [&](const RegionStmtObj *attr) -> ffi::Expected<ffi::WalkResult> {
          if (IsAsyncCommitQueueScope(attr)) {
            ++summary.commit;
          } else if (IsAsyncWaitQueueScope(attr)) {
            ++summary.wait;
          }

          return ffi::WalkResult::Advance();
        });
    return summary;
  }

  std::optional<int>
  TryGetDeterministicNoWaitCommitGroups(const Stmt &stmt) const {
    auto effect = DeterministicNoWaitCommitAnalyzer(this).Analyze(stmt);
    if (!effect.deterministic || effect.has_wait) {
      return std::nullopt;
    }
    return effect.commit_groups;
  }

  int GuaranteedNewGroupsBeforeNextWait(const Array<Stmt> &body,
                                        int start_idx) const {
    int guaranteed_groups = 0;
    for (int i = start_idx, n = static_cast<int>(body.size()); i < n; ++i) {
      AsyncSyncSummary summary = SummarizeAsyncSyncScopes(body[i]);
      if (summary.wait > 0) {
        break;
      }
      if (summary.commit == 0) {
        continue;
      }
      if (auto commits = TryGetDeterministicNoWaitCommitGroups(body[i])) {
        guaranteed_groups += *commits;
        continue;
      }
      break;
    }
    return guaranteed_groups;
  }

  class WaitStaticInSimpleWrapperRewriter
      : public StmtFunctor<Optional<Stmt>(const Stmt &)> {
  public:
    static Optional<Stmt> Rewrite(const PipelineRewriter *rewriter,
                                  const Stmt &stmt, int new_wait_n) {
      if (rewriter->ClassifySimpleAsyncSyncStmt(stmt).kind !=
          AsyncSyncStmtKind::kWaitStatic) {
        return Optional<Stmt>();
      }
      WaitStaticInSimpleWrapperRewriter rewrite(rewriter, new_wait_n);
      return rewrite(stmt);
    }

    WaitStaticInSimpleWrapperRewriter(const PipelineRewriter *rewriter,
                                      int new_wait_n)
        : rewriter_(rewriter), new_wait_n_(new_wait_n) {}

    Optional<Stmt> Dispatch_(const RegionStmtObj *op) final {
      if (!TransparentScope(op))
        return std::nullopt;
      if (IsAsyncWaitQueueScope(op)) {
        return rewriter_->MakeStaticAsyncWaitStmtLike(op, new_wait_n_);
      }
      Optional<Stmt> body = Dispatch(op->body);
      if (!body.has_value()) {
        return Optional<Stmt>();
      }
      return WithScopeBody(op, body.value());
    }

    Optional<Stmt> Dispatch_(const SeqStmtObj *op) final {
      if (op->seq.size() != 1) {
        return Optional<Stmt>();
      }
      Optional<Stmt> inner = Dispatch(op->seq[0]);
      if (!inner.has_value()) {
        return Optional<Stmt>();
      }
      return SeqStmt({inner.value()});
    }

    Optional<Stmt> DispatchDefault_(const Object *) final {
      return Optional<Stmt>();
    }

  private:
    const PipelineRewriter *rewriter_;
    int new_wait_n_;
  };

  Optional<Stmt> RewriteWaitStaticInSimpleWrapper(const Stmt &stmt,
                                                  int new_wait_n) const {
    return WaitStaticInSimpleWrapperRewriter::Rewrite(this, stmt, new_wait_n);
  }

  std::optional<int> TryGetHeadStaticWaitCount(const Stmt &stmt) const {
    HeadAsyncSyncInfo info =
        AnalyzeHeadAsyncSync(stmt, HeadSeqMode::kTakeFirstElement);
    if (info.kind == HeadAsyncSyncKind::kWaitStatic) {
      return info.wait_n;
    }
    return std::nullopt;
  }

  class FirstStaticWaitCounter
      : public StmtFunctor<std::optional<int>(const Stmt &)> {
  public:
    static std::optional<int> Find(const PipelineRewriter *rewriter,
                                   const Stmt &stmt) {
      FirstStaticWaitCounter finder(rewriter);
      return finder(stmt);
    }

    explicit FirstStaticWaitCounter(const PipelineRewriter *rewriter)
        : rewriter_(rewriter) {}

    std::optional<int> Dispatch_(const RegionStmtObj *op) final {
      if (!TransparentScope(op))
        return std::nullopt;
      HeadAsyncSyncInfo info = rewriter_->AnalyzeHeadAsyncSync(
          GetRef<Stmt>(op), HeadSeqMode::kTakeFirstElement);
      if (info.kind == HeadAsyncSyncKind::kWaitStatic) {
        return info.wait_n;
      }
      if (info.IsBoundary()) {
        return std::nullopt;
      }
      return Dispatch(op->body);
    }

    std::optional<int> Dispatch_(const SeqStmtObj *op) final {
      for (const Stmt &elem : op->seq) {
        HeadAsyncSyncInfo info = rewriter_->AnalyzeHeadAsyncSync(
            elem, HeadSeqMode::kTakeFirstElement);
        if (info.kind == HeadAsyncSyncKind::kWaitStatic) {
          return info.wait_n;
        }
        if (info.IsBoundary() || rewriter_->ContainsAsyncSyncScopes(elem)) {
          return std::nullopt;
        }
      }
      return std::nullopt;
    }

    std::optional<int> DispatchDefault_(const Object *) final {
      return std::nullopt;
    }

  private:
    const PipelineRewriter *rewriter_;
  };

  std::optional<int> TryGetFirstStaticWaitCount(const Stmt &stmt) const {
    return FirstStaticWaitCounter::Find(this, stmt);
  }

  class HeadStaticWaitInWrapperRewriter
      : public StmtFunctor<Optional<Stmt>(const Stmt &)> {
  public:
    static Optional<Stmt> Rewrite(const PipelineRewriter *rewriter,
                                  const Stmt &stmt, int new_wait_n) {
      HeadStaticWaitInWrapperRewriter rewrite(rewriter, new_wait_n);
      return rewrite(stmt);
    }

    HeadStaticWaitInWrapperRewriter(const PipelineRewriter *rewriter,
                                    int new_wait_n)
        : rewriter_(rewriter), new_wait_n_(new_wait_n) {}

    Optional<Stmt> Dispatch_(const RegionStmtObj *op) final {
      if (!TransparentScope(op))
        return std::nullopt;
      if (IsAsyncWaitQueueScope(op)) {
        return rewriter_->MakeStaticAsyncWaitStmtLike(op, new_wait_n_);
      }
      Optional<Stmt> body = Dispatch(op->body);
      if (!body.has_value()) {
        return Optional<Stmt>();
      }
      return WithScopeBody(op, body.value());
    }

    Optional<Stmt> Dispatch_(const SeqStmtObj *op) final {
      if (op->seq.empty()) {
        return Optional<Stmt>();
      }
      Optional<Stmt> first = Dispatch(op->seq[0]);
      if (!first.has_value()) {
        return Optional<Stmt>();
      }
      Array<Stmt> new_seq = op->seq;
      new_seq.Set(0, first.value());
      return SeqStmt(new_seq);
    }

    Optional<Stmt> DispatchDefault_(const Object *) final {
      return Optional<Stmt>();
    }

  private:
    const PipelineRewriter *rewriter_;
    int new_wait_n_;
  };

  Optional<Stmt> RewriteHeadStaticWaitInWrapper(const Stmt &stmt,
                                                int new_wait_n) const {
    return HeadStaticWaitInWrapperRewriter::Rewrite(this, stmt, new_wait_n);
  }

  class FirstStaticWaitInWrapperRewriter
      : public StmtFunctor<Optional<Stmt>(const Stmt &)> {
  public:
    static Optional<Stmt> Rewrite(const PipelineRewriter *rewriter,
                                  const Stmt &stmt, int new_wait_n) {
      FirstStaticWaitInWrapperRewriter rewrite(rewriter, new_wait_n);
      return rewrite(stmt);
    }

    FirstStaticWaitInWrapperRewriter(const PipelineRewriter *rewriter,
                                     int new_wait_n)
        : rewriter_(rewriter), new_wait_n_(new_wait_n) {}

    Optional<Stmt> Dispatch_(const RegionStmtObj *op) final {
      if (!TransparentScope(op))
        return std::nullopt;
      if (IsAsyncWaitQueueScope(op)) {
        return rewriter_->MakeStaticAsyncWaitStmtLike(op, new_wait_n_);
      }
      if (IsAsyncCommitQueueScope(op) || IsAsyncWaitInflightCount(op)) {
        return Optional<Stmt>();
      }
      Optional<Stmt> body = Dispatch(op->body);
      if (!body.has_value()) {
        return Optional<Stmt>();
      }
      return WithScopeBody(op, body.value());
    }

    Optional<Stmt> Dispatch_(const SeqStmtObj *op) final {
      Array<Stmt> new_seq = op->seq;
      for (int i = 0, n = static_cast<int>(new_seq.size()); i < n; ++i) {
        Optional<Stmt> updated = Dispatch(new_seq[i]);
        if (updated.has_value()) {
          new_seq.Set(i, updated.value());
          return SeqStmt(new_seq);
        }
        if (rewriter_->ContainsAsyncSyncScopes(new_seq[i])) {
          return Optional<Stmt>();
        }
      }
      return Optional<Stmt>();
    }

    Optional<Stmt> DispatchDefault_(const Object *) final {
      return Optional<Stmt>();
    }

  private:
    const PipelineRewriter *rewriter_;
    int new_wait_n_;
  };

  Optional<Stmt> RewriteFirstStaticWaitInWrapper(const Stmt &stmt,
                                                 int new_wait_n) const {
    return FirstStaticWaitInWrapperRewriter::Rewrite(this, stmt, new_wait_n);
  }

  Stmt MaybeRelaxLoopWaits(const For &loop, int pre_outstanding_lb) const {
    int retain = PipelinedRetainGroups(GetPipelineNumStages(loop.get()));
    if (retain <= 0 || !loop.defined()) {
      return loop;
    }
    const auto *seq = loop->body.as<SeqStmtObj>();
    if (!seq || seq->seq.empty()) {
      return loop;
    }

    Array<Stmt> body = seq->seq;
    bool changed = false;
    int outstanding_lb = std::max(0, pre_outstanding_lb);
    int groups_since_wait_lb = 0;
    bool seen_wait_boundary = false;

    for (int i = 0, n = static_cast<int>(body.size()); i < n; ++i) {
      ClassifiedAsyncSyncStmt cls = ClassifySimpleAsyncSyncStmt(body[i]);
      if (cls.kind == AsyncSyncStmtKind::kCommit) {
        ++outstanding_lb;
        ++groups_since_wait_lb;
        continue;
      }
      if (cls.kind == AsyncSyncStmtKind::kWaitDynamic) {
        seen_wait_boundary = true;
        outstanding_lb = 0;
        groups_since_wait_lb = 0;
        continue;
      }
      if (cls.kind == AsyncSyncStmtKind::kWaitStatic) {
        int effective_wait_n = cls.wait_n;
        if (cls.wait_n == 0) {
          int groups_after_wait_lb =
              GuaranteedNewGroupsBeforeNextWait(body, i + 1);
          int per_sync_groups = groups_since_wait_lb;
          bool uses_head_fallback =
              (per_sync_groups == 0 && !seen_wait_boundary);
          if (uses_head_fallback) {
            per_sync_groups = 1;
          }
          int candidate_wait_n = std::max(0, retain * per_sync_groups);
          bool enough_pre_outstanding =
              !uses_head_fallback || outstanding_lb >= (candidate_wait_n + 1);
          if (candidate_wait_n > 0 && enough_pre_outstanding &&
              (!uses_head_fallback || groups_after_wait_lb > 0)) {
            Optional<Stmt> rewritten_wait =
                RewriteWaitStaticInSimpleWrapper(body[i], candidate_wait_n);
            if (rewritten_wait.has_value()) {
              body.Set(i, rewritten_wait.value());
              changed = true;
              effective_wait_n = candidate_wait_n;
            }
          }
        }
        seen_wait_boundary = true;
        outstanding_lb = std::min(outstanding_lb, effective_wait_n);
        groups_since_wait_lb = 0;
        continue;
      }

      AsyncSyncSummary summary = SummarizeAsyncSyncScopes(body[i]);
      if (summary.wait == 0) {
        if (auto commits = TryGetDeterministicNoWaitCommitGroups(body[i])) {
          outstanding_lb += *commits;
          groups_since_wait_lb += *commits;
          continue;
        }
      }
      if (summary.wait > 0) {
        seen_wait_boundary = true;
      }
      outstanding_lb = 0;
      groups_since_wait_lb = 0;
    }

    if (!changed) {
      return loop;
    }
    For new_loop = loop;
    new_loop.CopyOnWrite()->body = SeqStmt(body);
    return new_loop;
  }

  class LoopWaitsInSimpleWrapperRelaxer
      : public StmtFunctor<Optional<Stmt>(const Stmt &)> {
  public:
    static Optional<Stmt> Rewrite(const PipelineRewriter *rewriter,
                                  const Stmt &stmt, int pre_outstanding_lb) {
      LoopWaitsInSimpleWrapperRelaxer relaxer(rewriter, pre_outstanding_lb);
      return relaxer(stmt);
    }

    LoopWaitsInSimpleWrapperRelaxer(const PipelineRewriter *rewriter,
                                    int pre_outstanding_lb)
        : rewriter_(rewriter), pre_outstanding_lb_(pre_outstanding_lb) {}

    Optional<Stmt> Dispatch_(const ForObj *op) final {
      For loop = GetRef<For>(op);
      Stmt relaxed = rewriter_->MaybeRelaxLoopWaits(loop, pre_outstanding_lb_);
      if (relaxed.same_as(loop)) {
        return Optional<Stmt>();
      }
      return relaxed;
    }

    Optional<Stmt> Dispatch_(const RegionStmtObj *op) final {
      if (!TransparentScope(op))
        return std::nullopt;
      Optional<Stmt> body = Dispatch(op->body);
      if (!body.has_value()) {
        return Optional<Stmt>();
      }
      return WithScopeBody(op, body.value());
    }

    Optional<Stmt> Dispatch_(const SeqStmtObj *op) final {
      if (op->seq.size() != 1) {
        return Optional<Stmt>();
      }
      Optional<Stmt> inner = Dispatch(op->seq[0]);
      if (!inner.has_value()) {
        return Optional<Stmt>();
      }
      return SeqStmt({inner.value()});
    }

    Optional<Stmt> DispatchDefault_(const Object *) final {
      return Optional<Stmt>();
    }

  private:
    const PipelineRewriter *rewriter_;
    int pre_outstanding_lb_;
  };

  Optional<Stmt> RelaxLoopWaitsInSimpleWrapper(const Stmt &stmt,
                                               int pre_outstanding_lb) const {
    return LoopWaitsInSimpleWrapperRelaxer::Rewrite(this, stmt,
                                                    pre_outstanding_lb);
  }

  class AsyncPipelineLoopWaitRelaxer : public LayoutMapMutator {
  public:
    explicit AsyncPipelineLoopWaitRelaxer(const PipelineRewriter *rewriter)
        : rewriter_(rewriter) {}

    UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
      Array<Stmt> visited;
      visited.reserve(op->seq.size());
      for (const Stmt &stmt : op->seq) {
        visited.push_back(Mutate(stmt, mode).ValueOrUnchanged(stmt));
      }

      int outstanding_lb = 0;
      for (int i = 0, n = static_cast<int>(visited.size()); i < n; ++i) {
        Stmt current = visited[i];
        Optional<Stmt> relaxed =
            rewriter_->RelaxLoopWaitsInSimpleWrapper(current, outstanding_lb);
        if (relaxed.has_value()) {
          current = relaxed.value();
          visited.Set(i, current);
        }
        ClassifiedAsyncSyncStmt cls =
            rewriter_->ClassifySimpleAsyncSyncStmt(current);
        if (cls.kind == AsyncSyncStmtKind::kCommit) {
          ++outstanding_lb;
          continue;
        }
        if (cls.kind == AsyncSyncStmtKind::kWaitStatic) {
          outstanding_lb = std::min(outstanding_lb, cls.wait_n);
          continue;
        }
        if (cls.kind == AsyncSyncStmtKind::kWaitDynamic) {
          outstanding_lb = 0;
          continue;
        }
        AsyncSyncSummary summary = rewriter_->SummarizeAsyncSyncScopes(current);
        if (summary.wait == 0) {
          if (auto commits =
                  rewriter_->TryGetDeterministicNoWaitCommitGroups(current)) {
            outstanding_lb += *commits;
            continue;
          }
        }
        if (summary.wait > 0) {
          outstanding_lb = 0;
        }
      }

      if (visited.empty()) {
        return Evaluate(0);
      }
      if (visited.size() == 1) {
        return visited[0];
      }
      return SeqStmt(visited);
    }

  private:
    const PipelineRewriter *rewriter_;
  };

  Array<Stmt> RelaxTrailingConsumerWaits(Array<Stmt> seq, int retain) const {
    if (retain <= 0 || seq.size() <= 1) {
      return seq;
    }
    std::vector<int> suffix_wait_indices;
    for (int i = static_cast<int>(seq.size()) - 1; i >= 0; --i) {
      if (ContainsAsyncCommitScopes(seq[i])) {
        break;
      }
      auto first_wait = TryGetFirstStaticWaitCount(seq[i]);
      if (!first_wait.has_value() || *first_wait != 0) {
        break;
      }
      suffix_wait_indices.push_back(i);
    }
    if (suffix_wait_indices.size() <= 1) {
      return seq;
    }
    for (size_t pos = 1; pos < suffix_wait_indices.size(); ++pos) {
      int idx = suffix_wait_indices[pos];
      // Tail consumers drain the final committed groups with no new commits in
      // between. Relax them progressively from the end so the suffix becomes
      // ..., wait<2>, wait<1>, wait<0> instead of rewriting every drain wait to
      // the same retain count.
      int new_wait_n = std::min(retain, static_cast<int>(pos));
      Optional<Stmt> rewritten =
          RewriteFirstStaticWaitInWrapper(seq[idx], new_wait_n);
      if (rewritten.has_value()) {
        seq.Set(idx, rewritten.value());
      }
    }
    return seq;
  }

  void PopulateWaitCounts(const std::vector<RewrittenStmtInfo> &new_stmts,
                          sym::AnalyzerObj *ana_normalized,
                          const BufferCommitGroupMap &buffer_to_commit_group,
                          std::map<int, AsyncStateLocal> *async_states_local) {
    std::vector<int> stmt_to_commit_group(new_stmts.size(), -1);
    std::map<int, std::vector<int>> commit_group_last_stmt;
    for (const auto &[stage_id, state] : *async_states_local) {
      auto &last_stmt = commit_group_last_stmt[stage_id];
      last_stmt.assign(state.commit_groups.size(), -1);
      for (size_t group_id = 0; group_id < state.commit_groups.size();
           ++group_id) {
        for (size_t stmt_idx : state.commit_groups[group_id]) {
          ICHECK_LT(stmt_idx, new_stmts.size());
          ICHECK_EQ(stmt_to_commit_group[stmt_idx], -1);
          stmt_to_commit_group[stmt_idx] = static_cast<int>(group_id);
          last_stmt[group_id] =
              std::max(last_stmt[group_id], static_cast<int>(stmt_idx));
        }
      }
    }

    std::map<int, int> last_committed_group;
    auto record_pending_wait = [&](AsyncStateLocal *state, int commit_group_id,
                                   int insert_before, PrimExpr wait_count) {
      auto &pending_wait = state->pending_waits[commit_group_id];
      if (!pending_wait.valid()) {
        pending_wait = {insert_before, wait_count};
      } else if (analyzer_->CanProve(wait_count <
                                     pending_wait.wait_count.value())) {
        pending_wait = {pending_wait.insert_before, wait_count};
      }
    };

    for (size_t i = 0; i < new_stmts.size(); ++i) {
      if (new_stmts[i].is_async) {
        auto &local_state = (*async_states_local)[new_stmts[i].stage];
        for (const TensorRegion &write_region : new_stmts[i].writes) {
          local_state.seen.insert(RegionTensor(write_region));
        }
        int commit_group_id = stmt_to_commit_group[i];
        if (commit_group_id >= 0) {
          const auto &last_stmt = commit_group_last_stmt.at(new_stmts[i].stage);
          ICHECK_LT(commit_group_id, static_cast<int>(last_stmt.size()));
          if (last_stmt[commit_group_id] == static_cast<int>(i)) {
            last_committed_group[new_stmts[i].stage] = commit_group_id;
          }
        }
        continue;
      }

      int producer_stage_idx = -1;
      for (const TensorRegion &read_region : new_stmts[i].reads) {
        for (const auto &kv : async_states_) {
          if (kv.first <= new_stmts[i].stage &&
              kv.second.writes(RegionTensor(read_region))) {
            ICHECK(producer_stage_idx == -1 || producer_stage_idx == kv.first)
                << "A dependency on multiple async stages is not supported";
            producer_stage_idx = kv.first;
          }
        }
      }

      if (producer_stage_idx == -1) {
        continue;
      }

      auto &dep_local_state = (*async_states_local)[producer_stage_idx];
      int num_commit_group = dep_local_state.commit_groups.size();

      if (num_commit_group == 0) {
        ICHECK(!dep_local_state.producer_head);
        const auto &global_state = async_states_[producer_stage_idx];
        PrimExpr tail_start = analyzer_->Simplify(
            pipeline_loop_->begin + LoopExtent(pipeline_loop_.get()) -
            PrimExpr(max_stage_));
        bool is_tail_consumer =
            ana_normalized->CanProve(new_stmts[i].access_index >= tail_start);
        if (is_tail_consumer && global_state.commit_group_count > 0) {
          int latest_group_id = global_state.commit_group_count - 1;
          PrimExpr latest_producer_head = analyzer_->Simplify(
              pipeline_loop_->begin + LoopExtent(pipeline_loop_.get()) -
              PrimExpr(1));
          std::vector<bool> need_wait_count(global_state.commit_group_count,
                                            true);
          bool handled = false;
          for (const TensorRegion &read_region : new_stmts[i].reads) {
            if (!global_state.writes(RegionTensor(read_region))) {
              continue;
            }
            auto it = global_state.buffer_to_commit_group.find(
                RegionTensor(read_region));
            if (it == global_state.buffer_to_commit_group.end()) {
              handled = false;
              break;
            }
            int commit_group_id = it->second;
            ICHECK_GE(commit_group_id, 0);
            ICHECK_LT(commit_group_id, global_state.commit_group_count);
            if (!need_wait_count[commit_group_id]) {
              continue;
            }
            PrimExpr wait_count = analyzer_->Simplify(
                (latest_producer_head - new_stmts[i].access_index) *
                    global_state.commit_group_count +
                (latest_group_id - commit_group_id));
            if (!ana_normalized->CanProve(wait_count >= 0)) {
              wait_count = PrimExpr(0);
            }
            record_pending_wait(&dep_local_state, commit_group_id,
                                static_cast<int>(i), wait_count);
            need_wait_count[commit_group_id] = false;
            handled = true;
          }
          if (handled) {
            continue;
          }
        }

        PrimExpr wait_count = PrimExpr(0);
        Optional<PrimExpr> producer_head =
            async_states_[producer_stage_idx].producer_head;
        if (producer_head &&
            ana_normalized->CanProve(producer_head.value() >= 0)) {
          wait_count = analyzer_->Simplify(producer_head.value() -
                                           new_stmts[i].access_index);
        }
        record_pending_wait(&dep_local_state, -1, static_cast<int>(i),
                            wait_count);
        continue;
      }

      ICHECK(dep_local_state.producer_head);
      int latest_group_id = -1;
      Optional<PrimExpr> latest_producer_head;
      if (auto it = last_committed_group.find(producer_stage_idx);
          it != last_committed_group.end()) {
        latest_group_id = it->second;
        latest_producer_head = dep_local_state.producer_head.value();
      } else {
        latest_group_id = num_commit_group - 1;
        latest_producer_head = dep_local_state.producer_head.value() - 1;
      }

      std::vector<bool> need_wait_count(num_commit_group, true);
      for (const TensorRegion &read_region : new_stmts[i].reads) {
        if (!async_states_[producer_stage_idx].writes(
                RegionTensor(read_region))) {
          continue;
        }
        auto commit_group_id =
            buffer_to_commit_group.at(RegionTensor(read_region));
        ICHECK_GE(commit_group_id, 0);
        ICHECK_LT(commit_group_id, num_commit_group);
        if (!need_wait_count[commit_group_id]) {
          continue;
        }

        PrimExpr wait_count = PrimExpr(0);
        if (latest_producer_head &&
            ana_normalized->CanProve(latest_producer_head.value() >= 0)) {
          wait_count = analyzer_->Simplify(
              (latest_producer_head.value() - new_stmts[i].access_index) *
                  num_commit_group +
              (latest_group_id - commit_group_id));
          if (!ana_normalized->CanProve(wait_count >= 0)) {
            wait_count = PrimExpr(0);
          }
        }

        record_pending_wait(&dep_local_state, commit_group_id,
                            static_cast<int>(i), wait_count);
        need_wait_count[commit_group_id] = false;
      }
    }
  }

  std::vector<FinalStmtInfo> CompletePipelineLoopStatements(
      const std::vector<RewrittenStmtInfo> &stmts,
      const std::map<int, AsyncStateLocal> &async_states_local,
      sym::AnalyzerObj *ana_normalized) const {
    std::vector<FinalStmtInfo> new_stmts;
    new_stmts.reserve(stmts.size());
    for (const auto &stmt : stmts) {
      new_stmts.push_back(
          {stmt.stage, stmt.access_index, stmt.predicate, stmt.stmt});
    }

    std::vector<int> commit_group_tags(new_stmts.size(), -1);
    std::unordered_map<int, int> commit_group_tag_to_stage;
    int next_commit_group_tag = 0;
    std::map<int, std::map<int, PrimExpr>> waits_before_stmt;
    auto make_wait_stmt = [](int stage_id, PrimExpr wait_count, Stmt body) {
      auto zero = MakeZero(PrimType::Int(32));
      return ScopeStmt(
          zero, "async_wait_queue_scope", int64_t(stage_id),
          ScopeStmt(zero, "async_wait_inflight_count", wait_count, body));
    };
    auto merge_wait_before_stmt = [&](int insert_before, int stage_id,
                                      PrimExpr wait_count) {
      auto &waits_at_stmt = waits_before_stmt[insert_before];
      auto it = waits_at_stmt.find(stage_id);
      if (it == waits_at_stmt.end()) {
        waits_at_stmt.emplace(stage_id, ana_normalized->Simplify(wait_count));
      } else if (ana_normalized->CanProve(wait_count < it->second)) {
        it->second = ana_normalized->Simplify(wait_count);
      }
    };

    for (const auto &[stage_id, state] : async_states_local) {
      if (!state.commit_groups.empty()) {
        for (const auto &group_stmt_indices : state.commit_groups) {
          int commit_group_tag = next_commit_group_tag++;
          commit_group_tag_to_stage.emplace(commit_group_tag, stage_id);
          for (size_t stmt_idx : group_stmt_indices) {
            ICHECK(stmt_idx < new_stmts.size());
            commit_group_tags[stmt_idx] = commit_group_tag;
          }
        }
      }

      for (const auto &[commit_group_id, pending_wait] : state.pending_waits) {
        if (!pending_wait.valid()) {
          continue;
        }
        PrimExpr wait_count =
            ana_normalized->Simplify(pending_wait.wait_count.value());
        if (state.predicate &&
            !ana_normalized->CanProve(state.predicate.value())) {
          PrimExpr predicate =
              ana_normalized->Simplify(state.predicate.value());
          if (IsZero(predicate)) {
            continue;
          }
          merge_wait_before_stmt(pending_wait.insert_before, stage_id,
                                 wait_count);
          continue;
        }

        merge_wait_before_stmt(pending_wait.insert_before, stage_id,
                               wait_count);
      }
    }

    std::vector<FinalStmtInfo> result;
    for (size_t i = 0; i < new_stmts.size();) {
      if (auto it = waits_before_stmt.find(i); it != waits_before_stmt.end()) {
        for (const auto &[stage_id, wait_count] : it->second) {
          Stmt wait_stmt = make_wait_stmt(stage_id, wait_count, Evaluate(0));
          if (auto state_it = async_states_local.find(stage_id);
              state_it != async_states_local.end() &&
              state_it->second.predicate &&
              !ana_normalized->CanProve(state_it->second.predicate.value())) {
            PrimExpr predicate =
                ana_normalized->Simplify(state_it->second.predicate.value());
            if (IsZero(predicate)) {
              continue;
            }
            wait_stmt = If(predicate, SeqStmt(wait_stmt), SeqStmt(Evaluate(0)));
          }
          result.push_back({new_stmts[i].stage, new_stmts[i].access_index,
                            new_stmts[i].predicate, wait_stmt});
        }
      }

      if (commit_group_tags[i] == -1) {
        result.push_back(new_stmts[i]);
        ++i;
        continue;
      }

      int commit_group_tag = commit_group_tags[i];
      int stage_id = commit_group_tag_to_stage.at(commit_group_tag);
      Array<Stmt> group_stmts;
      PrimExpr access_index = new_stmts[i].access_index;
      PrimExpr predicate = new_stmts[i].predicate;
      for (; i < new_stmts.size() && commit_group_tags[i] == commit_group_tag;
           ++i) {
        group_stmts.push_back(new_stmts[i].stmt);
      }
      Stmt group_body =
          group_stmts.size() == 1 ? group_stmts[0] : SeqStmt(group_stmts);
      Stmt commit_queue_scope =
          ScopeStmt(MakeZero(PrimType::Int(32)), "async_commit_queue_scope",
                    int64_t(stage_id), group_body);
      if (!IsOne(predicate) && !ana_normalized->CanProve(predicate)) {
        PrimExpr simplified_predicate = ana_normalized->Simplify(predicate);
        if (!IsZero(simplified_predicate)) {
          commit_queue_scope =
              If(simplified_predicate, SeqStmt(commit_queue_scope),
                 SeqStmt(Evaluate(0)));
        }
      }
      result.push_back({stage_id, access_index, predicate, commit_queue_scope});
    }
    return result;
  }

  /*!
   * \brief Emit the pipeline loop in the given range.
   * \param start The start of the range
   * \param end The end of the range
   * \param unroll_loop Whether the loop should be unrolled.
   * \return The result loop.
   */
  Stmt EmitImpl(const PrimExpr &start, const PrimExpr &end, bool unroll_loop,
                bool need_bound_check) {
    PrimExpr new_loop_var{ffi::UnsafeInit{}};
    PrimExpr extent = end - start;
    Optional<int64_t> pipeline_num_stages =
        GetPipelineNumStages(pipeline_loop_.get());
    // Written against the original loop var; the per-block Substitute below
    // specializes it into each op's logical iteration parity.
    PrimExpr mbar_phase = analyzer_->Simplify(
        floormod(pipeline_loop_->var - pipeline_loop_->begin,
                 MakeConst(pipeline_loop_->var.ty(), 2)));
    auto make_nop = []() { return Evaluate(IntImm::Int32(0)); };

    if (unroll_loop) {
      if (auto extent_imm = TryConstInt(extent)) {
        if (*extent_imm > 1) {
          Array<Stmt> expanded;
          expanded.reserve(static_cast<size_t>(*extent_imm));
          for (int64_t iter = 0; iter < *extent_imm; ++iter) {
            PrimExpr unit_start =
                analyzer_->Simplify(start + IntImm(extent.ty(), iter));
            PrimExpr unit_end =
                analyzer_->Simplify(start + IntImm(extent.ty(), iter + 1));
            Stmt unit_stmt =
                EmitImpl(unit_start, unit_end, false, need_bound_check);
            expanded.push_back(unit_stmt);
          }
          Stmt result = expanded.size() == 1 ? expanded[0] : SeqStmt(expanded);
          return result;
        }
      }
    }

    bool is_unit_loop = analyzer_->CanProveEqual(extent, 1);
    if (is_unit_loop) {
      new_loop_var = start; // use constants as the loop var for unit loops
    } else {
      new_loop_var =
          pipeline_loop_->var.CopyWithSuffix("").as_or_throw<PrimVar>();
      // Bind the iteration domain [start, end) to strengthen analyzer facts.
      BindAnalyzer(analyzer_, new_loop_var.as_or_throw<PrimVar>(),
                   Range::FromMinExtent(start, end - start));
    }
    // Keep the bound constraints active for all analysis below.
    // Only meaningful when the loop var is symbolic (non-unit loop).
    std::unique_ptr<ScopedAnalyzerConstraints> ctx_lb_guard;
    std::unique_ptr<ScopedAnalyzerConstraints> ctx_ub_guard;
    if (!is_unit_loop) {
      PrimVar loop_iter = new_loop_var.as_or_throw<PrimVar>();
      ctx_lb_guard.reset(
          new ScopedAnalyzerConstraints(analyzer_.get(), loop_iter >= start));
      ctx_ub_guard.reset(
          new ScopedAnalyzerConstraints(analyzer_.get(), loop_iter < end));
    }

    sym::Analyzer ana_normalized;
    if (!is_unit_loop) {
      BindAnalyzer(ana_normalized, new_loop_var.as_or_throw<PrimVar>(),
                   Range::FromMinExtent(pipeline_loop_->begin, extent));
    }

    std::vector<RewrittenStmtInfo> new_stmts;
    std::map<int, AsyncStateLocal> async_states_local;
    BufferCommitGroupMap buffer_to_commit_group;

    for (const PipelineStage &block : ordered_stmts_) {
      const auto &pipeline_anno = pipeline_info_.at(block);
      int stage = pipeline_anno.stage;
      PrimExpr inbound = IntImm::Bool(true);
      PrimExpr skewed_loop_var = new_loop_var - stage;
      if (need_bound_check)
        inbound = And(pipeline_loop_->begin <= skewed_loop_var,
                      (skewed_loop_var < pipeline_loop_->begin +
                                             LoopExtent(pipeline_loop_.get())));

      bool scheduled_bind = block->body.as<BindObj>() != nullptr;
      PipelineStage replayed_block = ReplayScalarBindings(block);
      PipelineStage annotated_block =
          RemakeStage(replayed_block,
                      AnnotateTileOpMbarPhase(replayed_block->body, mbar_phase),
                      memory_, target_);
      PipelineStage new_block =
          PipelineBodyRewriter(memory_, buffer_remap_, pipeline_loop_,
                               max_stage_ != 1, target_)(annotated_block);

      PrimExpr delta = start - pipeline_loop_->begin;
      PrimExpr normalized_access_index =
          is_unit_loop ? skewed_loop_var : skewed_loop_var + delta;

      normalized_access_index = analyzer_->Simplify(normalized_access_index);

      // Adjust the block predicate and the body according to the final loop
      // bound
      //  [pipeline_loop_->begin, extent).
      if (!is_unit_loop) {
        PrimVar loop_iter = new_loop_var.as_or_throw<PrimVar>();
        inbound = tirx::SubstituteWithDataTypeLegalization(
            inbound,
            [values = ffi::Map<Var, PrimExpr>{{loop_iter, loop_iter + delta}}](
                const Var &var) -> ffi::Optional<PrimExpr> {
              return values.Get(var);
            });
      }
      inbound = ana_normalized->Simplify(inbound);
      if (IsZero(inbound)) {
        continue;
      }
      new_block = RemakeStage(
          new_block,
          tl::SubstituteWithDataTypeLegalization(
              new_block->body,
              [values = ffi::Map<Var, PrimExpr>{{pipeline_loop_->var,
                                                 normalized_access_index}}](
                  const Var &var) -> ffi::Optional<PrimExpr> {
                return values.Get(var);
              }),
          memory_, target_);

      Stmt rewritten_stmt = ReifyStage(new_block, inbound);
      if (scheduled_bind) {
        // Replay can prepend Bind statements. Keep the whole chain in the
        // consumer's lexical scope so ConvertSSA sees each iteration's value.
        Array<Stmt> bindings;
        for (const Stmt &stmt : FlattenTopLevelSeq(new_block->body)) {
          Bind bind = stmt.as_or_throw<Bind>();
          PrimExpr value = bind->value.as_or_throw<PrimExpr>();
          if (!IsOne(inbound)) {
            value = if_then_else(inbound, value, MakeZero(value.ty()));
          }
          bindings.push_back(Bind(bind->var, value, bind->loc));
        }
        rewritten_stmt = MakePipelineBody(bindings);
      }

      bool is_async = pipeline_anno.async;
      if (is_async) {
        auto &local_state = async_states_local[stage];
        int commit_group_id = -1;
        if (pipeline_anno.async_group_id >= 0) {
          auto it = local_state.annotated_group_to_commit_group.find(
              pipeline_anno.async_group_id);
          if (it == local_state.annotated_group_to_commit_group.end()) {
            commit_group_id = local_state.commit_groups.size();
            local_state.commit_groups.push_back({new_stmts.size()});
            local_state.annotated_group_to_commit_group.emplace(
                pipeline_anno.async_group_id, commit_group_id);
          } else {
            commit_group_id = it->second;
            local_state.commit_groups[commit_group_id].push_back(
                new_stmts.size());
          }
        } else if (local_state.commit_groups.empty() || local_state.consumed) {
          commit_group_id = local_state.commit_groups.size();
          local_state.commit_groups.push_back({new_stmts.size()});
        } else {
          commit_group_id = local_state.commit_groups.size() - 1;
          local_state.commit_groups.back().push_back(new_stmts.size());
        }

        for (const TensorRegion &write_region : new_block->writes) {
          auto &global_state = async_states_[stage];
          global_state.dst_buffers.insert(RegionTensor(write_region));
          global_state.buffer_to_commit_group[RegionTensor(write_region)] =
              commit_group_id;
          global_state.commit_group_count =
              std::max(global_state.commit_group_count,
                       static_cast<int>(local_state.commit_groups.size()));
          buffer_to_commit_group[RegionTensor(write_region)] = commit_group_id;
        }
        async_states_[stage].producer_head = normalized_access_index;
        local_state.producer_head = normalized_access_index;
        if (!local_state.predicate ||
            ana_normalized->CanProve(local_state.predicate.value())) {
          local_state.predicate = inbound;
        } else {
          local_state.predicate =
              ana_normalized->Simplify(local_state.predicate.value() & inbound);
        }
        rewritten_stmt = AnnotateSimtProducer(rewritten_stmt, target_);
      }
      new_stmts.push_back({stage, inbound, new_block->reads, new_block->writes,
                           normalized_access_index, is_async, rewritten_stmt});

      for (const TensorRegion &read_region : new_block->reads) {
        for (const auto &kv : async_states_) {
          if (kv.first <= stage &&
              kv.second.writes(RegionTensor(read_region))) {
            async_states_local[kv.first].consumed = true;
          }
        }
      }
    }

    PopulateWaitCounts(new_stmts, ana_normalized.get(), buffer_to_commit_group,
                       &async_states_local);
    std::vector<FinalStmtInfo> final_stmts = CompletePipelineLoopStatements(
        new_stmts, async_states_local, ana_normalized.get());

    Array<Stmt> stmts;
    for (const auto &stmt_info : final_stmts) {
      stmts.push_back(stmt_info.stmt);
    }

    Stmt new_loop{ffi::UnsafeInit{}};

    if (stmts.empty()) {
      return make_nop();
    }

    if (stmts.size() == 1) {
      new_loop = stmts[0];
    } else {
      new_loop = SeqStmt(stmts);
    }

    if (!is_unit_loop) {
      Map<String, Any> preserved_annotations;
      for (const auto &kv : pipeline_loop_->annotations) {
        const String &key = kv.first;
        if (kv.first != "software_pipeline_stage" &&
            kv.first != "software_pipeline_order" &&
            kv.first != "software_pipeline_async_stages" &&
            kv.first != kPipelineAsyncProducers &&
            kv.first != kPipelineAsyncProducerGroups &&
            kv.first != kPipelineTmaCopies &&
            kv.first != kPipelineReplayableScalarBinds &&
            kv.first != "num_stages") {
          preserved_annotations.Set(key, kv.second);
        }
      }
      if (pipeline_num_stages &&
          preserved_annotations.find("tl_pipelined_num_stages") ==
              preserved_annotations.end()) {
        preserved_annotations.Set("tl_pipelined_num_stages",
                                  pipeline_num_stages.value());
      }
      new_loop = For(
          new_loop_var.as_or_throw<PrimVar>(), pipeline_loop_->begin,
          pipeline_loop_->begin + extent, IntImm(pipeline_loop_->var.ty(), 1),
          unroll_loop ? ForKind::kUnrolled : pipeline_loop_->kind,
          SeqStmt(new_loop), preserved_annotations, pipeline_loop_->loc);
    }
    Stmt result = new_loop;

    return result;
  }

  sym::Analyzer analyzer_;
  const PipelineMemoryAnalysis *memory_;
  Array<TensorVar> pipeline_allocs_;
  Array<TensorVar> local_allocs_;
  For pipeline_loop_;
  PipelineInfo pipeline_info_;
  Array<PipelineStage> scalar_binding_blocks_;
  int max_stage_ = -1;
  Map<TensorVar, TensorVar> buffer_remap_;
  Optional<Target> target_;
  Map<Var, sym::IntSet> thread_domains_;
  Array<PipelineStage> ordered_stmts_;
  std::vector<ScalarBinding> scalar_bindings_;
  ScalarBindingMap scalar_binding_map_;
  std::map<int, AsyncStateGlobal> async_states_;
};

PipelineRewriteResult
RewritePipeline(const PipelineMemoryAnalysis *memory,
                const Array<TensorVar> &pipeline_allocs,
                const Array<TensorVar> &local_allocs, const For &pipeline_loop,
                const PipelineInfo &pipeline_info,
                const Array<PipelineStage> &scalar_binding_blocks,
                Optional<Target> target, Map<Var, sym::IntSet> thread_domains) {
  PipelineRewriter rewriter(memory, pipeline_allocs, local_allocs,
                            pipeline_loop, pipeline_info, scalar_binding_blocks,
                            std::move(target), std::move(thread_domains));
  PipelineRewriteResult result;
  result.pipeline = rewriter.BuildPipeline();
  result.buffer_remap = rewriter.GetBufferRemap();
  return result;
}

/*!
 * \brief Reconcile pipeline buffer views with their final allocation.
 * Preserve loop-local version indices and use slot zero for unexpanded
 * accesses. Update buffer regions and annotated layouts to match the
 * allocation.
 */
class PipelineInjector : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate;
  using LayoutMapMutator::Mutate_;

  explicit PipelineInjector(const tirx::Function &function)
      : memory_(function),
        target_(function->GetAttr<Target>(tvm::attr::kTarget)),
        global_symbol_(function->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
    thread_domains_ = CollectPipelineThreadDomains(function->body);
  }

  static Stmt Inject(const tirx::Function &function) {
    auto injector = make_object<PipelineInjector>(function);
    return injector->Mutate(function->body).ValueOrUnchanged(function->body);
  }

private:
  /*!
   * \brief Check the pipeline satisfies the following conditions:
   * 1. No conflicting order: The order of each statement should be unique.
   * 2. Reordering of statements doesn't break buffer access dependencies.
   * Specifically, for dependency (e.g. read-after-write) from statement A to
   * statement B, it requires: case 1: stage(A) < stage(B) case 2: stage(A) ==
   * stage(B) and order(A) < order(B)
   */
  void ValidatePipelineBody(const PipelineInfo &pipeline_info,
                            const Array<PipelineStage> &original_order) {
    std::unordered_set<int> used_orders;
    for (const PipelineStage &block : original_order) {
      const auto &stmt_info = pipeline_info.at(block);
      int order = stmt_info.order;
      ICHECK(!used_orders.count(order))
          << "ValueError: Two statements in the software pipeline cannot have "
             "the same order"
          << SpanHintSuffix({block->body->loc, block->loc});
      used_orders.insert(order);
    }

    std::unordered_map<PipelineStage, Array<PipelineStage>, ObjectPtrHash,
                       ObjectPtrEqual>
        dep_src2dst;
    BuildDependencyGraph(original_order, &dep_src2dst, nullptr, thread_domains_,
                         &memory_);

    for (const auto &pair : dep_src2dst) {
      const PipelineStage &src = pair.first;
      const auto &src_info = pipeline_info.at(src);
      const Array<PipelineStage> &dsts = pair.second;
      for (const PipelineStage &dst : dsts) {
        const auto &dst_info = pipeline_info.at(dst);
        ICHECK_LE(src_info.stage, dst_info.stage)
            << "ValueError: statement " << dst << " in stage " << dst_info.stage
            << " cannot depends on statement " << src << " in a later stage "
            << src_info.stage
            << SpanHintSuffix({dst->body->loc, src->body->loc});
        if (src_info.stage == dst_info.stage) {
          ICHECK_LT(src_info.order, dst_info.order)
              << "ValueError: two statements with buffer "
                 "access dependency in the same stage of the "
                 "software pipeline cannot be reordered"
              << SpanHintSuffix({dst->body->loc, src->body->loc});
        }
      }
    }
  }

  void ValidateScheduledBindDependencies(
      const PipelineInfo &pipeline_info,
      const Array<PipelineStage> &scheduled_order) {
    std::unordered_map<Var, PipelineStage, ObjectPtrHash, ObjectPtrEqual>
        bind_producers;
    for (const PipelineStage &block : scheduled_order) {
      if (const auto *bind = block->body.as<BindObj>()) {
        bind_producers.emplace(bind->var, block);
      }
    }
    if (bind_producers.empty()) {
      return;
    }

    for (const PipelineStage &consumer : scheduled_order) {
      Array<Var> undefined_vars = UndefinedVars(consumer->body, Array<Var>{});
      for (const Var &var : undefined_vars) {
        auto it = bind_producers.find(var);
        if (it == bind_producers.end() || it->second.same_as(consumer)) {
          continue;
        }

        const PipelineAnnotation &producer_info = pipeline_info.at(it->second);
        const PipelineAnnotation &consumer_info = pipeline_info.at(consumer);
        ICHECK_EQ(producer_info.stage, consumer_info.stage)
            << "ValueError: scheduled scalar Bind '" << var
            << "' is used from a different pipeline stage. Scalar Bind "
               "statements that cannot be replayed must be scheduled in the "
               "same stage as their consumers."
            << SpanHintSuffix(consumer->body->loc);
        ICHECK_LT(producer_info.order, consumer_info.order)
            << "ValueError: scheduled scalar Bind '" << var
            << "' must be ordered before every consumer in the same pipeline "
               "stage."
            << SpanHintSuffix(consumer->body->loc);
      }
    }
  }

  bool HasOverlappableStages(const PipelineInfo &pipeline_info) const {
    std::optional<int> first_stage;
    for (const auto &pair : pipeline_info) {
      int stage = pair.second.stage;
      if (!first_stage.has_value()) {
        first_stage = stage;
      } else if (stage != first_stage.value()) {
        return true;
      }
    }
    return false;
  }

  struct PipelineScheduleUnit {
    PipelineStage block;
    Array<TensorVar> nested_local_allocs;
  };

  struct PipelineSchedule {
    Array<PipelineStage> original_order;
    Array<TensorVar> nested_local_allocs;
  };

  PipelineScheduleUnit MakePipelineScheduleUnit(const Stmt &stmt) {
    PipelineScheduleUnit unit;
    unit.block = MakeBlock(stmt, &memory_, target_);
    return unit;
  }

  PipelineSchedule BuildPipelineSchedule(const Array<Stmt> &stmts) {
    PipelineSchedule schedule;
    for (const Stmt &stmt : stmts) {
      PipelineScheduleUnit unit = MakePipelineScheduleUnit(stmt);
      schedule.original_order.push_back(unit.block);
      schedule.nested_local_allocs.insert(schedule.nested_local_allocs.end(),
                                          unit.nested_local_allocs.begin(),
                                          unit.nested_local_allocs.end());
    }
    return schedule;
  }

  Array<Stmt> StripPipelineDeclarationStmts(const Array<Stmt> &pipeline_body,
                                            Array<TensorVar> *local_allocs,
                                            Array<TensorVar> *flat_allocs) {
    Array<Stmt> stages;
    for (const Stmt &statement : pipeline_body) {
      if (IsPipelineDeclarationStmt(statement)) {
        auto tensor =
            statement.as_or_throw<Bind>()->var.as_or_throw<TensorVar>();
        allocated_buffers_.insert(tensor);
        local_allocs->push_back(tensor);
        flat_allocs->push_back(tensor);
      } else {
        stages.push_back(statement);
      }
    }
    TVM_FFI_CHECK(!stages.empty(), ValueError)
        << "The body of the software pipeline has no executable stages";
    return stages;
  }

  Map<String, Any>
  StripPipelineAnnotations(const Map<String, Any> &annotations) const {
    Map<String, Any> preserved_annotations;
    for (const auto &kv : annotations) {
      const String &key = kv.first;
      if (key != "software_pipeline_stage" &&
          key != "software_pipeline_order" &&
          key != "software_pipeline_async_stages" &&
          key != kPipelineAsyncProducers &&
          key != kPipelineAsyncProducerGroups && key != kPipelineTmaCopies &&
          key != kPipelineReplayableScalarBinds && key != "num_stages" &&
          key != "tl_pipelined_num_stages") {
        preserved_annotations.Set(key, kv.second);
      }
    }
    return preserved_annotations;
  }

  void CollectBufferReplacements(const TensorVar &buffer,
                                 const TensorVar &allocation,
                                 Map<TensorVar, TensorVar> *replacements) {
    if (auto expansions = buffer_expansions_.Get(buffer)) {
      // Include the original buffer for single-version and non-pipelined uses.
      replacements->Set(buffer, allocation);
      for (const TensorVar &expanded : expansions.value()) {
        if (!expanded.same_as(allocation)) {
          replacements->Set(expanded, allocation);
        }
      }
      buffer_expansions_.erase(buffer);
    }
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    Array<Stmt> statements;
    TensorSet local;
    Map<TensorVar, TensorVar> replacements;
    for (const Stmt &statement : op->seq) {
      Stmt rewritten = Mutate(statement, mode).ValueOrUnchanged(statement);
      statements.push_back(rewritten);
      if (IsPipelineDeclarationStmt(statement)) {
        auto tensor =
            statement.as_or_throw<Bind>()->var.as_or_throw<TensorVar>();
        if (!allocated_buffers_.count(tensor))
          local.insert(tensor);
        allocated_buffers_.insert(tensor);
      }
    }
    for (const TensorVar &tensor : local) {
      if (auto pending = pending_buffer_remap_.Get(tensor)) {
        CollectBufferReplacements(tensor, pending.value(), &replacements);
        replacements.Set(tensor, pending.value());
        pending_buffer_remap_.erase(tensor);
      }
      allocated_buffers_.erase(tensor);
    }
    Stmt result = SeqStmt(statements, op->loc);
    if (!replacements.empty()) {
      auto reconcile = make_object<PipelineTensorRewriter>(replacements);
      result = reconcile->Rewrite(result);
    }
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = LayoutMapMutator::Mutate_(op, mode)
                       .ValueOrUnchanged(GetRef<For>(op))
                       .as_or_throw<For>();
    if (!HasPipelineAnnotation(op)) {
      return for_node;
    }
    // Step 2: Find the body and buffer allocations of the pipeline.
    Stmt pipeline_body_root = for_node->body;
    Array<TensorVar> pipeline_allocs;
    Array<TensorVar>
        block_local_allocs; // flat allocations inside pipeline body
    Array<TensorVar> flat_local_allocs;

    Array<Stmt> pipeline_body_stmts = NormalizePipelineBody(pipeline_body_root);

    // PipelinePlanning emits stage/order annotations only for executable
    // pipeline statements. Flat TIRX keeps loop-local AllocBuffer/DeclBuffer as
    // standalone statements in the loop body, so strip them from the stage
    // stream before blockizing and consuming annotations. The declarations are
    // still registered as local allocations so RewritePipeline can
    // multi-version and reattach them.
    pipeline_body_stmts = StripPipelineDeclarationStmts(
        pipeline_body_stmts, &block_local_allocs, &flat_local_allocs);

    PipelineInfo pipeline_info;
    PipelineSchedule schedule = BuildPipelineSchedule(pipeline_body_stmts);
    Array<PipelineStage> original_order = schedule.original_order;

    // Collect all buffers that are actually used in the pipeline loop body.
    // This includes buffers allocated in outer blocks (like logits_smem) that
    // are used inside the pipeline loop.
    pipeline_allocs =
        CollectUsedPipelineBuffers(MakePipelineBody(pipeline_body_stmts),
                                   &memory_, allocated_buffers_, target_);

    Optional<Array<int64_t>> replayable_bind_mask;
    if (auto replayable_bind_anno =
            op->annotations.Get(kPipelineReplayableScalarBinds)) {
      auto mask = IntegerArray(replayable_bind_anno.value());
      if (mask.size() == original_order.size()) {
        bool valid_mask = true;
        for (size_t i = 0; i < original_order.size(); ++i) {
          if (mask[i] != 0 &&
              original_order[i]->body.as<BindObj>() == nullptr) {
            valid_mask = false;
            break;
          }
        }
        if (valid_mask) {
          replayable_bind_mask = std::move(mask);
        }
      }
    }
    TensorSet pipeline_write_buffers =
        CollectPipelineWriteBuffers(original_order);
    Array<PipelineStage> scalar_binding_blocks;
    Array<PipelineStage> scheduled_order;
    std::vector<char> is_replayable_bind;
    is_replayable_bind.reserve(original_order.size());
    for (size_t i = 0; i < original_order.size(); ++i) {
      const PipelineStage &block = original_order[i];
      const bool semantically_replayable =
          IsReplayableScalarBindBlock(block, pipeline_write_buffers, &memory_);
      bool replayable = semantically_replayable;
      if (replayable_bind_mask.has_value()) {
        replayable = replayable_bind_mask.value()[i] != 0;
        if (replayable && !semantically_replayable) {
          const auto *bind = block->body.as<BindObj>();
          ICHECK(bind != nullptr);
          LOG(FATAL) << "tirx::Function " << global_symbol_
                     << " marks scalar Bind '" << bind->var
                     << "' as replayable via " << kPipelineReplayableScalarBinds
                     << ", but the Bind has an unsupported type, has side "
                        "effects, or reads a buffer written by the pipeline "
                        "and cannot be replayed safely";
        }
      }
      is_replayable_bind.push_back(replayable ? 1 : 0);
      if (replayable) {
        scalar_binding_blocks.push_back(block);
      } else {
        scheduled_order.push_back(block);
      }
    }
    ICHECK(!scheduled_order.empty())
        << "ValueError: The body of the software pipeline has no schedulable "
           "statements after removing replayable scalar Bind statements";

    auto pipeline_stages =
        IntegerArray(op->annotations.at("software_pipeline_stage"));
    auto pipeline_orders =
        IntegerArray(op->annotations.at("software_pipeline_order"));
    ICHECK_EQ(pipeline_stages.size(), pipeline_orders.size())
        << "tirx::Function " << global_symbol_
        << " has software_pipeline_stage annotation " << pipeline_stages
        << " and software_pipeline_order annotation " << pipeline_orders
        << " with different sizes";

    bool annotations_include_replayable_binds = false;
    if (pipeline_stages.size() == scheduled_order.size()) {
      annotations_include_replayable_binds = false;
    } else if (pipeline_stages.size() == original_order.size()) {
      annotations_include_replayable_binds = true;
    } else {
      LOG(FATAL) << "tirx::Function " << global_symbol_
                 << " has schedulable pipeline order "
                 << scheduled_order.Map(
                        [](const auto &block) { return block->name_hint; })
                 << " and original order "
                 << original_order.Map(
                        [](const auto &block) { return block->name_hint; })
                 << ", but pipeline annotation is " << pipeline_stages
                 << " with different size";
    }

    std::vector<size_t> scheduled_annotation_indices;
    scheduled_annotation_indices.reserve(scheduled_order.size());
    if (annotations_include_replayable_binds) {
      size_t scheduled_index = 0;
      for (size_t i = 0; i < original_order.size(); ++i) {
        if (is_replayable_bind[i]) {
          continue;
        }
        ICHECK(scheduled_index < scheduled_order.size());
        ICHECK(scheduled_order[scheduled_index].same_as(original_order[i]));
        scheduled_annotation_indices.push_back(i);
        ++scheduled_index;
      }
    } else {
      for (size_t i = 0; i < scheduled_order.size(); ++i) {
        scheduled_annotation_indices.push_back(i);
      }
    }

    auto expected_annotation_size = annotations_include_replayable_binds
                                        ? original_order.size()
                                        : scheduled_order.size();

    std::unordered_set<int> pipeline_async_stages;
    if (auto async_annot =
            op->annotations.Get("software_pipeline_async_stages")) {
      for (int64_t stage : IntegerArray(async_annot.value())) {
        pipeline_async_stages.insert(static_cast<int>(stage));
      }
    }
    Optional<Array<int64_t>> pipeline_async_producers;
    if (auto async_producers_anno =
            op->annotations.Get(kPipelineAsyncProducers)) {
      auto async_flags = IntegerArray(async_producers_anno.value());
      ICHECK_EQ(async_flags.size(), expected_annotation_size)
          << "tirx::Function " << global_symbol_ << " has schedulable order "
          << scheduled_order.Map(
                 [](const auto &block) { return block->name_hint; })
          << ", but async producer annotation is " << async_flags
          << " with different size";
      pipeline_async_producers = async_flags;
    }
    Optional<Array<int64_t>> pipeline_async_producer_groups;
    if (auto async_groups_anno =
            op->annotations.Get(kPipelineAsyncProducerGroups)) {
      auto async_group_ids = IntegerArray(async_groups_anno.value());
      ICHECK_EQ(async_group_ids.size(), expected_annotation_size)
          << "tirx::Function " << global_symbol_ << " has schedulable order "
          << scheduled_order.Map(
                 [](const auto &block) { return block->name_hint; })
          << ", but async producer group annotation is " << async_group_ids
          << " with different size";
      pipeline_async_producer_groups = async_group_ids;
    }

    for (size_t i = 0; i < scheduled_order.size(); i++) {
      size_t annotation_index = scheduled_annotation_indices[i];
      int stage = static_cast<int>(pipeline_stages[annotation_index]);
      bool is_async_candidate =
          pipeline_async_producers
              ? pipeline_async_producers.value()[annotation_index] != 0
              : (pipeline_async_stages.count(stage) > 0);
      // Stages that already carry pipeline async control attrs keep that
      // ownership; the injector only annotates plain producer stages.
      bool is_async = is_async_candidate && !ContainsPipelineAsyncControlAttrs(
                                                scheduled_order[i]->body);
      PipelineAnnotation stage_order{
          stage,
          /*order=*/
          static_cast<int>(pipeline_orders[annotation_index]),
          /*async=*/is_async,
          /*async_group_id=*/
          pipeline_async_producer_groups
              ? static_cast<int>(
                    pipeline_async_producer_groups.value()[annotation_index])
              : -1};
      pipeline_info.emplace(scheduled_order[i], stage_order);
    }

    if (annotations_include_replayable_binds) {
      for (const PipelineStage &binding_block : scalar_binding_blocks) {
        const auto *bind = binding_block->body.as<BindObj>();
        ICHECK(bind != nullptr);
        bool seen_consumer = false;
        bool multiple_consumers = false;
        PipelineAnnotation first_consumer;
        for (const PipelineStage &consumer : scheduled_order) {
          Array<Var> undefined_vars =
              UndefinedVars(consumer->body, Array<Var>{});
          bool uses_binding = false;
          for (const Var &var : undefined_vars) {
            if (var.same_as(bind->var)) {
              uses_binding = true;
              break;
            }
          }
          if (!uses_binding) {
            continue;
          }
          const PipelineAnnotation &anno = pipeline_info.at(consumer);
          if (!seen_consumer) {
            first_consumer = anno;
            seen_consumer = true;
          } else if (first_consumer.stage != anno.stage ||
                     first_consumer.order != anno.order) {
            multiple_consumers = true;
            break;
          }
        }
        if (multiple_consumers) {
          LOG(WARNING)
              << "Scalar Bind '" << bind->var
              << "' is used by multiple pipeline stages; its annotation is "
                 "ignored and the bind is replayed at each use.";
        }
      }
    }

    ValidateScheduledBindDependencies(pipeline_info, scheduled_order);
    ValidatePipelineBody(pipeline_info, scheduled_order);

    if (!HasOverlappableStages(pipeline_info)) {
      for (const auto &buffer : flat_local_allocs) {
        allocated_buffers_.erase(buffer);
      }
      return For(for_node->var, for_node->begin, for_node->end, for_node->step,
                 for_node->kind, for_node->body,
                 StripPipelineAnnotations(for_node->annotations),
                 for_node->loc);
    }

    // Step 3.5: Pipeline-level TMA barrier management.
    // When TMA copies are present (without warp specialization), rewrite
    // them to use tl.tileop.tma_copy with shared pipeline barriers and insert
    // mbarrier_wait_parity before the first consumer stage.
    // Creates pipeline_mbar[pipeline_depth] at final size so LowerTileOp
    // uses the provided barrier instead of allocating separate per-copy ones.
    Optional<TensorVar> pipeline_barrier_buf;
    {
      int max_stage = 0;
      for (const auto &pair : pipeline_info) {
        max_stage = std::max(max_stage, pair.second.stage);
      }
      // Use the actual pipeline depth (number of buffer copies) for barrier
      // sizing, not the SW pipeline stage count (max_stage + 1).
      // Even for pipeline_depth=1 we create a shared barrier so that
      // LowerTileOp uses it instead of allocating separate per-copy barriers.
      Optional<int64_t> pipelined_num_stages = GetPipelineNumStages(op);
      int pipeline_depth = pipelined_num_stages.has_value()
                               ? static_cast<int>(pipelined_num_stages.value())
                               : max_stage + 1;
      // Clamp to at least 1 so we always allocate at least one barrier slot.
      pipeline_depth = std::max(pipeline_depth, 1);
      if (max_stage > 0) {
        if (auto tma_copies_anno = op->annotations.Get(kPipelineTmaCopies)) {
          auto raw_tma_copies = IntegerArray(tma_copies_anno.value());
          Array<int64_t> tma_copies;
          if (raw_tma_copies.size() == scheduled_order.size()) {
            tma_copies = raw_tma_copies;
          } else if (raw_tma_copies.size() == original_order.size()) {
            for (size_t annotation_index : scheduled_annotation_indices) {
              tma_copies.push_back(raw_tma_copies[annotation_index]);
            }
          }
          if (tma_copies.size() == scheduled_order.size()) {
            bool has_tma_copy =
                std::any_of(tma_copies.begin(), tma_copies.end(),
                            [](int64_t tc) { return tc != 0; });
            if (has_tma_copy) {
              pipeline_barrier_buf = RewritePipelineTmaBarriers(
                  scheduled_order, pipeline_info, tma_copies, &memory_,
                  allocated_buffers_, block_local_allocs, for_node->var,
                  for_node->begin, pipeline_depth, target_);
            }
          }
        }
      }
    }

    // Step 4: Rewrite the pipeline body.
    // local_allocs contains buffers allocated in the pipeline block itself.
    // pipeline_allocs contains all buffers that need multi-versioning,
    // including buffers from outer blocks.
    // Step 4.5: Expand all barrier buffers for pipelining.
    // This handles both ISP-created pipeline_mbar AND user-written
    // T.alloc_barrier, so that no late standalone barrier-only fixup is needed.
    // Must run BEFORE local_allocs is copied from block_local_allocs.
    {
      Optional<int64_t> pipelined_ns = GetPipelineNumStages(op);
      int barrier_depth = 1;
      if (pipelined_ns.has_value()) {
        barrier_depth = static_cast<int>(pipelined_ns.value());
      } else if (op->annotations.count("num_stages")) {
        barrier_depth = static_cast<int>(
            GetIntegerAnnotation(op->annotations.Get("num_stages").value()));
      }
      Map<TensorVar, TensorVar> barrier_remap = ExpandPipelineBarriers(
          scheduled_order, pipeline_info, &memory_, allocated_buffers_,
          block_local_allocs, pipeline_allocs, for_node->var, for_node->begin,
          barrier_depth, target_);
      // Register expanded barriers for outer block alloc_buffers update.
      for (const auto &[old_buf, new_buf] : barrier_remap) {
        pending_buffer_remap_.Set(old_buf, new_buf);
      }
    }

    Array<TensorVar> local_allocs = block_local_allocs;
    local_allocs.insert(local_allocs.end(),
                        schedule.nested_local_allocs.begin(),
                        schedule.nested_local_allocs.end());

    PipelineRewriteResult rewrite_result = RewritePipeline(
        &memory_, pipeline_allocs, local_allocs, for_node, pipeline_info,
        scalar_binding_blocks, target_, thread_domains_);
    Stmt pipeline = rewrite_result.pipeline;

    // Initializers are explicit operands at their allocation scope. Existing
    // entries are resized by reconciliation; the injector-owned barrier starts
    // with one arrival per expanded slot.
    if (pipeline_barrier_buf) {
      TensorVar barrier = pending_buffer_remap_.Get(*pipeline_barrier_buf)
                              .value_or(*pipeline_barrier_buf);
      auto slots = TryConstInt(barrier.shape()[0]);
      TVM_FFI_CHECK(slots && *slots > 0, ValueError);
      Array<PrimExpr> counts;
      for (int64_t i = 0; i < *slots; ++i)
        counts.push_back(IntImm::Int32(1));
      Stmt annotation = MakeAnnotation(Map<String, Any>{
          {"barrier_init", Map<Var, Array<PrimExpr>>{{barrier, counts}}}});
      // The allocated barrier must be in scope before metadata uses it.
      auto insert = [&](const SeqStmt &body) {
        auto seq = body->seq;
        size_t pos = 0;
        while (pos < seq.size() && IsPipelineDeclarationStmt(seq[pos]))
          ++pos;
        seq.insert(seq.begin() + pos, annotation);
        return SeqStmt(seq, body->loc);
      };
      if (const auto *scope = pipeline.as<RegionStmtObj>())
        pipeline = WithScopeBody(scope, insert(scope->body));
      else
        pipeline = insert(SeqStmt(pipeline));
    }

    // Reserve enough space for every pipeline using the allocation. Keep all
    // loop-local views until the allocation scope can reconcile them without
    // changing the version indices generated by each pipeline.
    for (const auto &[buffer, expanded] : rewrite_result.buffer_remap) {
      Array<TensorVar> expansions =
          buffer_expansions_.Get(buffer).value_or(Array<TensorVar>{});
      expansions.push_back(expanded);
      buffer_expansions_.Set(buffer, expansions);

      TensorVar allocation =
          pending_buffer_remap_.Get(buffer).value_or(expanded);
      if (expanded.shape()[0].as_or_throw<IntImm>()->value >
          allocation.shape()[0].as_or_throw<IntImm>()->value) {
        allocation = expanded;
      }
      pending_buffer_remap_.Set(buffer, allocation);
    }
    pipeline = LowerAsyncCommitWaitAttrs(pipeline);

    return pipeline;
  }

  bool HasPipelineAnnotation(const ForObj *op) const {
    auto it1 = op->annotations.find("software_pipeline_stage");
    auto it2 = op->annotations.find("software_pipeline_order");
    bool has_stage = it1 != op->annotations.end();
    bool has_order = it2 != op->annotations.end();
    if (has_stage && has_order) {
      return true;
    }
    if (has_stage) {
      LOG(FATAL)
          << "ValueError: Stage of the software pipeline is not defined.";
    }
    if (has_order) {
      LOG(FATAL)
          << "ValueError: Order of the software pipeline is not defined.";
    }
    return false;
  }

  PipelineMemoryAnalysis memory_;
  std::unordered_set<TensorVar, ObjectPtrHash, ObjectPtrEqual>
      allocated_buffers_;
  Map<TensorVar, TensorVar> pending_buffer_remap_;
  // All loop-local expanded views, keyed by their original allocation.
  Map<TensorVar, Array<TensorVar>> buffer_expansions_;
  Optional<Target> target_;
  Optional<String> global_symbol_;
  Map<Var, sym::IntSet> thread_domains_;
  // Track whether any pipeline was actually injected in the current
  // subtree.  Used to avoid unnecessary reads/writes recalculation
  // on blocks whose descendants were not modified.
  bool subtree_modified_ = false;
};

Stmt InjectPipeline(const tirx::Function &func) {
  return PipelineInjector::Inject(func);
}

} // namespace software_pipeline

/*!
 * \brief Transform annotated loops into pipelined one that parallelize
 * producers and consumers. \return The IR transform pass.
 */
tvm::transform::Pass InjectSoftwarePipeline() {
  auto pass_func = [](tirx::Function function, IRModule module,
                      tvm::transform::PassContext context) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    Stmt body = software_pipeline::InjectPipeline(function);
    function.CopyOnWrite()->body = SeqStmt(body);
    return function;
  };
  // Generated prologues and epilogues duplicate lexical definitions. The
  // native SSA conversion refreshes them and their type/shape uses together.
  return tvm::transform::Sequential(
      {tirx::transform::CreateFunctionPass(pass_func, 0,
                                           "tl.InjectSoftwarePipelineBody"),
       tirx::transform::ConvertSSA()},
      "tl.InjectSoftwarePipeline");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.InjectSoftwarePipeline",
                        InjectSoftwarePipeline);
}

} // namespace tl
} // namespace tvm
