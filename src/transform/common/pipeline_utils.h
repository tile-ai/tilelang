/*!
 * \file pipeline_utils.h
 * \brief Native pipeline statement, stage and thread-domain utilities.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

#include "ir/analyzer.h"
#include "transform/common/native_layout_utils.h"
#include <tvm/sym/int_set.h>

namespace tvm::tl {
using namespace tirx;

static constexpr const char *kPipelineTmaCopies =
    "software_pipeline_tma_copies";
static constexpr const char *kPipelineAsyncProducers =
    "software_pipeline_async_producers";
static constexpr const char *kPipelineAsyncProducerGroups =
    "software_pipeline_async_producer_groups";
static constexpr const char *kPipelineReplayableScalarBinds =
    "software_pipeline_replayable_scalar_binds";

inline bool IsPipelineDeclarationStmt(const Stmt &stmt) {
  if (const auto *bind = stmt.as<BindObj>()) {
    if (const auto *call = bind->value.as<CallObj>())
      return call->op.same_as(alloc_tensor_op());
  }
  return false;
}

inline ffi::Array<Stmt> NormalizePipelineBody(const Stmt &body) {
  if (const auto *seq = body.as<SeqStmtObj>())
    return seq->seq;
  return {body};
}
inline Stmt MakePipelineBody(const ffi::Array<Stmt> &statements) {
  return SeqStmt(statements);
}

inline ffi::Map<Var, sym::IntSet>
CollectPipelineThreadDomains(const Stmt &body) {
  ffi::Map<Var, sym::IntSet> domains;
  VisitNativePostOrder(body, [&](const ffi::ObjectRef &node) {
    const auto *region = node.as<RegionStmtObj>();
    if (!region || !region->op.same_as(launch_thread_op()))
      return;
    auto var = region->body_params[0].as_or_throw<PrimVar>();
    auto extent = region->args[1].as_or_throw<PrimExpr>();
    auto domain = sym::IntSet::FromRange(
        Range::FromMinExtent(IntImm(extent.ty(), 0), extent));
    if (auto previous = domains.Get(var))
      domain = sym::Union({previous.value(), domain});
    domains.Set(var, domain);
  });
  return domains;
}

// Distinct tensor identities may alias through decl_tensor. A caller without
// the definition graph must keep that pair conservative, never infer disjoint
// storage from distinct Var identities alone.
inline bool
PipelineRegionsMayConflict(const TensorRegion &lhs, const TensorRegion &rhs,
                           const ffi::Map<Var, sym::IntSet> &thread_domains) {
  if (!lhs->source.same_as(rhs->source) ||
      lhs->region.size() != rhs->region.size())
    return true;
  auto tensor = RegionTensor(lhs);
  auto analyzer = CreateTileLangAnalyzer();
  auto disjoint = [&](const sym::IntSet &a, const sym::IntSet &b) {
    if (a.IsNothing() || b.IsNothing())
      return true;
    return a.HasLowerBound() && a.HasUpperBound() && b.HasLowerBound() &&
           b.HasUpperBound() && TryConstInt(analyzer->Simplify(a.min())) &&
           TryConstInt(analyzer->Simplify(a.max())) &&
           TryConstInt(analyzer->Simplify(b.min())) &&
           TryConstInt(analyzer->Simplify(b.max())) &&
           sym::Intersect({a, b}).IsNothing();
  };
  if (tensor.ty()->layout.has_value()) {
    auto physical = [&](const TensorRegion &region) {
      auto domains = thread_domains;
      ffi::Array<PrimExpr> indices;
      auto bounds = sym::EvalSet(region->region, thread_domains);
      for (size_t i = 0; i < region->region.size(); ++i) {
        PrimVar index("region_index", region->region[i]->begin.ty());
        domains.Set(index, bounds[i]);
        indices.push_back(index);
      }
      return sym::EvalSet(TensorElementOffset(tensor, indices), domains);
    };
    // A distributed layout has more than one address axis, and cannot be
    // interpreted as a scalar address interval by this helper.
    auto zeros = tensor.shape().Map(
        [](const PrimExpr &x) -> PrimExpr { return IntImm(x.ty(), 0); });
    if (tensor.ty()->ElemOffset(zeros).size() != 1)
      return true;
    return !disjoint(physical(lhs), physical(rhs));
  }
  auto left = sym::EvalSet(lhs->region, thread_domains);
  auto right = sym::EvalSet(rhs->region, thread_domains);
  for (size_t i = 0; i < left.size(); ++i)
    if (disjoint(left[i], right[i]))
      return false;
  return true;
}

inline ffi::Optional<int64_t> GetPipelineNumStages(const ForObj *loop) {
  auto annotations = ReadLoopAnnotations(loop);
  for (const char *key : {"num_stages", "tl_pipelined_num_stages"}) {
    if (auto value = annotations.Get(key)) {
      if (auto integer = value->as<int64_t>())
        return *integer;
      if (auto expr = value->as<PrimExpr>())
        return TryConstInt(*expr);
    }
  }
  if (auto value = annotations.Get("software_pipeline_stage")) {
    int64_t max_stage = -1;
    for (const auto &stage : value->cast<ffi::Array<ffi::Any>>()) {
      auto integer = stage.as<int64_t>();
      if (!integer)
        integer = TryConstInt(stage.cast<PrimExpr>());
      if (!integer)
        return std::nullopt;
      max_stage = std::max(max_stage, *integer);
    }
    if (max_stage >= 0)
      return max_stage + 1;
  }
  return std::nullopt;
}

inline Range ComputeThreadBounds(const ffi::Optional<PrimVar> &thread,
                                 const sym::Analyzer &analyzer) {
  if (thread.has_value() && analyzer->const_int_bound.IsBound(*thread)) {
    auto bounds = analyzer->const_int_bound(*thread);
    return Range::FromMinExtent(
        IntImm(thread.value().ty(), bounds->min_value),
        IntImm(thread.value().ty(), bounds->max_value - bounds->min_value + 1));
  }
  return Range::FromMinExtent(0, 1);
}

} // namespace tvm::tl
#endif // TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_
