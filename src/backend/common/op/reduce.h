/*!
 * \file tl/backend/common/op/reduce.h
 * \brief Shared tl.reduce AllReduce lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_REDUCE_H_
#define TVM_TL_BACKEND_COMMON_OP_REDUCE_H_

#include "backend/common/op/reduction_ownership.h"
#include "backend/common/op/reduction_semantics.h"
#include "backend/common/target_utils.h"
#include "ir/analyzer.h"
#include "op/reduce.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include "layout/layout.h"
#include "layout/utils.h"
#include "op/utils.h"
#include "transform/loop_partition.h"

#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/math.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;
using namespace prim;
using namespace ffi;

// Scratch tensors have physical, per-thread shapes. Their explicit Bind is
// emitted at the enclosing operation's lexical allocation scope.
inline TensorVar MakeReductionTensor(Array<PrimExpr> shape, PrimType dtype,
                                     String name, runtime::MemoryScope scope) {
  return TensorVar(std::move(name), TensorType(std::move(shape), dtype, scope));
}

inline Bind AllocateReductionTensor(const TensorVar &tensor) {
  return Bind(tensor, Call(tensor.ty(), alloc_tensor_op(), {}, {},
                           {tensor.ty()}, tensor->loc));
}

inline Array<PrimExpr> ReductionPhysicalIndices(PrimExpr offset,
                                                const TensorVar &tensor) {
  Array<PrimExpr> indices(tensor.shape().size(), MakeConst(offset.ty(), 0));
  for (size_t i = tensor.shape().size(); i > 0; --i) {
    PrimExpr extent = cast(tensor.shape()[i - 1], offset.ty());
    indices.Set(i - 1, floormod(offset, extent));
    offset = floordiv(offset, extent);
  }
  return indices;
}

namespace reduce {

inline bool NeedsPaddingGuard(const Fragment &layout,
                              sym::AnalyzerObj *analyzer) {
  PrimExpr logical_points = cast(layout->ReplicateExtent(), PrimType::Int(64));
  PrimExpr physical_points = cast(layout->ThreadExtent(), PrimType::Int(64));
  for (const PrimExpr &extent : layout->InputShape())
    logical_points = logical_points * cast(extent, PrimType::Int(64));
  for (const PrimExpr &extent : layout->OutputShape())
    physical_points = physical_points * cast(extent, PrimType::Int(64));
  return !analyzer->CanProveEqual(logical_points, physical_points);
}

// Offset and ragged fragments do not cover their rectangular physical shape.
// NoCheck can supply an inverse candidate, but it must recover every original
// logical element and replica before it can select destination writes.
inline Layout DestinationInverse(const Fragment &layout,
                                 sym::AnalyzerObj *context) {
  const bool guarded = NeedsPaddingGuard(layout, context);
  auto [inverse, level] = layout->InverseWithLevel(guarded);
  if (level == sym::IterMapLevel::Bijective)
    return inverse;
  auto analyzer = context->Clone();
  Array<PrimExpr> logical;
  PrimExpr nonempty = IntImm::Bool(true);
  for (const PrimExpr &extent : layout->InputShape()) {
    PrimVar index("reduce_inverse_axis_" + std::to_string(logical.size()),
                  extent.ty());
    BindAnalyzer(analyzer, index,
                 Range::FromMinExtent(IntImm(index.ty(), 0), extent));
    logical.push_back(index);
    nonempty = nonempty && (extent > IntImm(extent.ty(), 0));
  }
  PrimExpr replicas = layout->ReplicateExtent();
  PrimVar replica("reduce_inverse_replica", replicas.ty());
  BindAnalyzer(analyzer, replica,
               Range::FromMinExtent(IntImm(replica.ty(), 0), replicas));
  nonempty = nonempty && (replicas > IntImm(replicas.ty(), 0));
  ScopedAnalyzerConstraints constraints(analyzer, nonempty);
  auto physical = layout->Forward(logical);
  physical.push_back(layout->ForwardThread(logical, replica));
  auto recovered = inverse->Forward(physical);
  logical.push_back(replica);
  TVM_FFI_CHECK_EQ(recovered.size(), logical.size(), ValueError)
      << "Reduction destination inverse must recover all axes and replicas";
  for (size_t i = 0; i < logical.size(); ++i) {
    TVM_FFI_CHECK(analyzer->CanProveEqual(recovered[i], logical[i]), ValueError)
        << "Reduction destination inverse does not recover coordinate " << i;
  }
  return inverse;
}

inline PrimExpr DestinationThreadPredicate(const Fragment &layout,
                                           const Array<PrimExpr> &logical,
                                           const Array<PrimExpr> &physical,
                                           PrimExpr thread_index,
                                           sym::AnalyzerObj *analyzer) {
  if (auto range = layout->ThreadRange())
    thread_index = thread_index - range->begin;
  auto physical_and_thread = physical;
  physical_and_thread.push_back(thread_index);
  auto recovered =
      DestinationInverse(layout, analyzer)->Forward(physical_and_thread);
  TVM_FFI_CHECK_EQ(recovered.size(), logical.size() + 1, ValueError)
      << "Reduction destination inverse has the wrong number of axes";
  PrimExpr replica = recovered.back();
  PrimExpr predicate = (replica >= IntImm(replica.ty(), 0)) &&
                       (replica < layout->ReplicateExtent()) &&
                       (thread_index >= IntImm(thread_index.ty(), 0));
  if (auto range = layout->ThreadRange())
    predicate = predicate && (thread_index < range->extent);
  for (size_t i = 0; i < logical.size(); ++i) {
    predicate = predicate && (recovered[i] == logical[i]) &&
                (recovered[i] >= IntImm(recovered[i].ty(), 0)) &&
                (recovered[i] < layout->InputShape()[i]);
  }
  // Bounds alone do not reject holes in a padded inverse. Reconstruct the
  // owner so an unused physical point cannot write a neighboring element.
  predicate =
      predicate && (layout->ForwardThread(logical, replica) == thread_index);
  return analyzer->Simplify(predicate);
}

inline void ValidateAllReduceThreadRange(const Fragment &layout,
                                         const Range &range,
                                         sym::AnalyzerObj *context);

/*!
 * \brief Resolve a proven participating thread range for named AllReduce.
 *
 * The result is derived from the reduce layout's forward thread map, which is
 * also the source of the guard later emitted by PartitionLoop. Const-int bounds
 * provide the minimum and maximum participating thread IDs.
 */
inline Range ResolveAllReduceThreadRange(const Fragment &red_layout,
                                         const Range &thread_bounds,
                                         const Target &target) {
  const auto block_min = TryConstInt(thread_bounds->begin);
  const auto block_extent = TryConstInt(thread_bounds->extent);
  const auto replicate = TryConstInt(red_layout->ReplicateExtent());
  if (block_min == std::nullopt || block_extent == std::nullopt ||
      replicate == std::nullopt) {
    LOG(FATAL) << "tl.reduce: cannot resolve the scalar AllReduce barrier: "
                  "the CTA thread bounds or reduce layout replicate extent "
                  "are not compile-time constants.";
  }
  ICHECK_GT(*block_extent, 0)
      << "tl.reduce: CTA thread extent must be positive";
  ICHECK_GT(*replicate, 0)
      << "tl.reduce: reduce layout replicate extent must be positive";

  auto analyzer = CreateTileLangAnalyzer();
  for (size_t i = 0; i < red_layout->InputShape().size(); ++i) {
    PrimVar placeholder = InputPlaceholder(i);
    BindAnalyzer(analyzer, placeholder,
                 Range::FromMinExtent(MakeConst(placeholder.ty(), 0),
                                      red_layout->InputShape()[i]));
  }
  PrimVar replicate_var = ReplicationPlaceholder();
  BindAnalyzer(analyzer, replicate_var,
               Range::FromMinExtent(MakeConst(replicate_var.ty(), 0),
                                    red_layout->ReplicateExtent()));

  // PartitionLoop feeds (threadIdx.x - ThreadRange.min) into the inverse
  // layout. Convert the forward map back to the corresponding absolute CTA
  // thread ID before computing its image.
  PrimExpr thread_expr = red_layout->GetForwardThread();
  if (red_layout->ThreadRange().has_value()) {
    thread_expr =
        analyzer->Simplify(thread_expr + red_layout->ThreadRange()->begin);
  }
  const sym::ConstIntBound bound = analyzer->const_int_bound(thread_expr);
  if (bound->min_value == sym::ConstIntBound::kNegInf ||
      bound->max_value == sym::ConstIntBound::kPosInf) {
    LOG(FATAL) << "tl.reduce: cannot determine the scalar AllReduce "
                  "participating thread range.";
  }

  const int64_t base = bound->min_value;
  const int64_t end = bound->max_value;
  // Bounds only propose an interval. The shared validation below must prove
  // that every physical slot occupies every thread in it before count can be
  // used by either a scalar or a batch named barrier.
  const int64_t count = end - base + 1;
  ICHECK_GE(base, *block_min)
      << "tl.reduce: scalar AllReduce participating thread range starts "
         "before the CTA thread bounds";
  ICHECK_LT(end, *block_min + *block_extent)
      << "tl.reduce: scalar AllReduce participating thread range ends after "
         "the CTA thread bounds";

  int64_t warp_size = 32;
  if (auto warp_size_attr = target->GetAttr<int64_t>("thread_warp_size")) {
    warp_size = warp_size_attr.value();
  }
  ICHECK_EQ(base % warp_size, 0)
      << "tl.reduce: partial scalar AllReduce requires a warp-aligned "
         "participating thread range, got base "
      << base;
  ICHECK_EQ(count % warp_size, 0)
      << "tl.reduce: partial scalar AllReduce requires a warp-aligned thread "
         "range, got "
      << count << " threads";

  Range range =
      Range::FromMinExtent(MakeConst(thread_expr.ty(), base),
                           MakeConst(thread_bounds->extent.ty(), count));
  ValidateAllReduceThreadRange(red_layout, range, analyzer.get());
  return range;
}

// Every scalar or batch named barrier must have this same set of active
// threads for each physical register slot. Prove it is a dense rectangle,
// not merely a bounding interval of a gapped owner map. Logical round-trip
// establishes injectivity; bounds and equal finite cardinalities then
// establish coverage of the whole rectangle.
inline void ValidateAllReduceThreadRange(const Fragment &layout,
                                         const Range &range,
                                         sym::AnalyzerObj *context) {
  auto analyzer = context->Clone();
  PrimExpr logical_points = cast(layout->ReplicateExtent(), PrimType::Int(64));
  PrimExpr physical_points = cast(range->extent, PrimType::Int(64));
  Array<PrimExpr> logical;
  for (const PrimExpr &extent : layout->InputShape()) {
    logical_points = logical_points * cast(extent, PrimType::Int(64));
    PrimVar index("reduce_owner_axis_" + std::to_string(logical.size()),
                  extent.ty());
    BindAnalyzer(analyzer, index,
                 Range::FromMinExtent(IntImm(index.ty(), 0), extent));
    logical.push_back(index);
  }
  for (const PrimExpr &extent : layout->OutputShape())
    physical_points = physical_points * cast(extent, PrimType::Int(64));
  TVM_FFI_CHECK(analyzer->CanProveEqual(logical_points, physical_points),
                ValueError)
      << "tl.reduce named AllReduce requires dense physical slots over one "
         "contiguous "
         "thread range";
  (void)DestinationInverse(layout, context);
  PrimVar replica("reduce_owner_replica", layout->ReplicateExtent().ty());
  BindAnalyzer(
      analyzer, replica,
      Range::FromMinExtent(IntImm(replica.ty(), 0), layout->ReplicateExtent()));
  auto physical = layout->Forward(logical);
  for (size_t i = 0; i < physical.size(); ++i) {
    TVM_FFI_CHECK(
        analyzer->CanProve(physical[i] >= 0) &&
            analyzer->CanProve(physical[i] < layout->OutputShape()[i]),
        ValueError)
        << "tl.reduce physical slot is outside its allocation";
  }
  PrimExpr thread = layout->ForwardThread(logical, replica);
  TVM_FFI_CHECK(analyzer->CanProve(thread >= 0), ValueError)
      << "tl.reduce relative thread owner must be nonnegative";
  if (auto origin = layout->ThreadRange()) {
    TVM_FFI_CHECK(analyzer->CanProve(thread < origin->extent), ValueError)
        << "tl.reduce relative thread owner exceeds its declared thread extent";
    thread = thread + origin->begin;
  }
  TVM_FFI_CHECK(analyzer->CanProve(thread >= range->begin) &&
                    analyzer->CanProve(thread < range->begin + range->extent),
                ValueError)
      << "tl.reduce owner is outside its participating thread range";
}

inline int64_t SignedMin(int bits) {
  if (bits >= 64) {
    return std::numeric_limits<int64_t>::min();
  }
  return -(static_cast<int64_t>(1) << (bits - 1));
}

inline int64_t SignedMax(int bits) {
  if (bits >= 64) {
    return std::numeric_limits<int64_t>::max();
  }
  return (static_cast<int64_t>(1) << (bits - 1)) - 1;
}

inline uint64_t UnsignedMax(int bits) {
  if (bits >= 64) {
    return std::numeric_limits<uint64_t>::max();
  }
  return (static_cast<uint64_t>(1) << bits) - 1;
}

inline int GetPreferredVectorizedSize(PrimType dt,
                                      bool supports_fp32x2 = false) {
  if (dt.MatchesElementType(kDLBfloat, 16) ||
      dt.MatchesElementType(kDLFloat, 16) ||
      (supports_fp32x2 && dt.MatchesCode(kDLFloat) && dt.bits() == 32))
    return 2;
  return 1;
}

inline void CheckAllReduceWidth(int reducing_threads, int scale,
                                const char *op_name) {
  ICHECK_GT(reducing_threads, 0)
      << op_name << ": AllReduce threads must be positive, got "
      << reducing_threads;
  ICHECK_GT(scale, 0) << op_name << ": AllReduce scale must be positive, got "
                      << scale;
  ICHECK_EQ(reducing_threads % scale, 0)
      << op_name << ": AllReduce threads (" << reducing_threads
      << ") must be divisible by scale (" << scale << ")";
  int logical_width = reducing_threads / scale;
  ICHECK((logical_width & (logical_width - 1)) == 0)
      << op_name << ": XOR-butterfly AllReduce requires logical_width "
      << "(threads / scale) to be a positive power of two, got "
      << logical_width << " (threads=" << reducing_threads
      << ", scale=" << scale << ")";
  // The butterfly pairs thread t with t ^ (scale * k).  That only moves the
  // reduce coordinate when the non-reduced part of the thread index occupies
  // the bits below `scale`, i.e. when `scale` is itself a power of two.
  // Otherwise the pairs straddle reduce groups (and replicas) and the result
  // is silently wrong or reads outside the workspace.
  ICHECK((scale & (scale - 1)) == 0)
      << op_name << ": XOR-butterfly AllReduce requires scale (the thread "
      << "stride between consecutive reduce participants) to be a power of "
      << "two, got " << scale << " (threads=" << reducing_threads
      << "). Pad the non-reduced fragment extent to a power of two.";
}

inline std::optional<std::string> MakeCodegenReducer(const ReduceOpNode &op,
                                                     int vsize = 1) {
  const bool use_nan_op =
      op.nan_propagate && (op.dst.dtype().MatchesElementType(kDLFloat, 16) ||
                           op.dst.dtype().MatchesElementType(kDLBfloat, 16));

  auto base = [&]() -> std::string {
    if (op.type->IsSum() || op.type->IsAbsSum())
      return "tl::SumOp";
    if (op.type->IsMax())
      return use_nan_op ? "tl::MaxOpNan" : "tl::MaxOp";
    if (op.type->IsMin())
      return use_nan_op ? "tl::MinOpNan" : "tl::MinOp";
    if (op.type->IsAbsMax())
      return use_nan_op ? "tl::MaxOpNan" : "tl::MaxOp";
    if (op.type->IsBitAnd())
      return "tl::BitAndOp";
    if (op.type->IsBitOr())
      return "tl::BitOrOp";
    if (op.type->IsBitXor())
      return "tl::BitXorOp";
    LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
    return "";
  }();

  if (vsize <= 1)
    return base;

  if (!(op.type->IsSum() || op.type->IsAbsSum() || op.type->IsMax() ||
        op.type->IsMin() || op.type->IsAbsMax())) {
    return std::nullopt;
  }

  if (vsize == 2) {
    if (op.dst.dtype().MatchesCode(kDLFloat) && op.dst.dtype().bits() == 32)
      return base + "_f32x2";
    if (op.dst.dtype().MatchesElementType(kDLBfloat, 16))
      return base + "_bf16x2";
    if (op.dst.dtype().MatchesElementType(kDLFloat, 16))
      return base + "_fp16x2";
  }
  return std::nullopt;
}

inline bool CanUsePackedRamp(const PrimExpr &index, const PrimVar &var,
                             int vsize, sym::AnalyzerObj *analyzer) {
  ICHECK_GT(vsize, 1);

  PrimExpr vector_size = MakeConst(var.ty(), vsize);
  PrimExpr packed_var = var * vector_size;
  PrimExpr ramp_base =
      analyzer->Simplify(SubstituteLayoutVars(index, {{var, packed_var}}));

  PrimExpr index_mod = FloorMod(ramp_base, MakeConst(index.ty(), vsize));
  if (!analyzer->CanProveEqual(index_mod, MakeConst(index.ty(), 0))) {
    return false;
  }

  for (int lane = 1; lane < vsize; ++lane) {
    PrimExpr lane_value = MakeConst(var.ty(), lane);
    PrimExpr lane_index = analyzer->Simplify(
        SubstituteLayoutVars(index, {{var, packed_var + lane_value}}));
    PrimExpr expected =
        analyzer->Simplify(ramp_base + MakeConst(index.ty(), lane));
    if (!analyzer->CanProveEqual(lane_index, expected)) {
      return false;
    }
  }

  return true;
}

// A reduce is lowered in two phases: each thread first reduces the values it
// owns locally, then the thread-level reducer combines the splits encoded in
// the source fragment's thread expression.  This plan is the shared ownership
// contract consumed by both phases.
struct ReduceOwnershipPlan {
  Array<PrimExpr> local_src_indices;
  Array<LayoutIterVar> local_reduce_vars;
  std::vector<ThreadReduceStep> thread_steps;
};

inline int64_t
ThreadOwnedReduceFactor(const std::vector<ThreadReduceStep> &steps) {
  int64_t factor = 1;
  for (const auto &step : steps) {
    ICHECK_LE(factor, std::numeric_limits<int64_t>::max() / step.extent)
        << "Reduce thread-owned factor overflow: factor=" << factor
        << ", extent=" << step.extent;
    factor *= step.extent;
  }
  return factor;
}

inline std::vector<int64_t>
CandidateThreadOwnedFactors(int64_t thread_owned_factor,
                            const PrimExpr &local_extent,
                            sym::AnalyzerObj *analyzer) {
  std::vector<int64_t> factors;
  for (int64_t factor = 2; factor <= thread_owned_factor / factor; ++factor) {
    if (thread_owned_factor % factor != 0) {
      continue;
    }
    factors.push_back(factor);
    if (factor != thread_owned_factor / factor) {
      factors.push_back(thread_owned_factor / factor);
    }
  }
  if (thread_owned_factor > 1) {
    factors.push_back(thread_owned_factor);
  }

  std::sort(factors.begin(), factors.end(),
            [](int64_t lhs, int64_t rhs) { return lhs > rhs; });
  factors.erase(std::unique(factors.begin(), factors.end()), factors.end());

  std::vector<int64_t> divisible_factors;
  for (int64_t factor : factors) {
    if (analyzer->CanProveEqual(
            FloorMod(local_extent, IntImm(PrimType::Int(32), factor)), 0)) {
      divisible_factors.push_back(factor);
    }
  }
  return divisible_factors;
}

inline std::optional<std::pair<PrimExpr, LayoutIterVar>>
TryRemoveThreadOwnedFactor(const PrimExpr &expr, const LayoutIterVar &iter_var,
                           int64_t factor, sym::AnalyzerObj *analyzer) {
  PrimExpr factor_expr = IntImm(PrimType::Int(32), factor);
  PrimVar old_var = iter_var->var;
  PrimExpr old_extent = analyzer->Simplify(iter_var->dom->extent);
  if (!analyzer->CanProveEqual(FloorMod(old_extent, factor_expr), 0)) {
    return std::nullopt;
  }

  BindAnalyzer(analyzer, old_var, Range(0, old_extent),
               /*allow_override=*/true);
  PrimExpr masked = FloorDiv(old_var, factor_expr) * factor_expr;
  PrimExpr simplified_expr = analyzer->Simplify(expr);
  PrimExpr masked_expr = analyzer->Simplify(
      SubstituteLayoutVars(simplified_expr, {{old_var, masked}}));
  if (!analyzer->CanProveEqual(masked_expr, simplified_expr)) {
    return std::nullopt;
  }

  PrimExpr new_extent = analyzer->Simplify(FloorDiv(old_extent, factor_expr));
  PrimVar new_var(old_var->name, old_var.ty());
  PrimExpr new_expr = analyzer->Simplify(SubstituteLayoutVars(
      simplified_expr, {{old_var, new_var * factor_expr}}));
  LayoutIterVar new_iter_var = LayoutIterVar(Range(0, new_extent), new_var);
  BindAnalyzer(analyzer, new_var, Range(0, new_extent),
               /*allow_override=*/true);
  return std::make_pair(new_expr, new_iter_var);
}

inline void CheckThreadOwnedStepsProjectable(
    const PrimExpr &index_expr, const PrimVar &reduce_var,
    const std::vector<ThreadReduceStep> &steps, sym::AnalyzerObj *analyzer) {
  if (steps.empty()) {
    return;
  }

  // Zero out exactly the reduce-var segments owned by thread splits:
  // each split covers floormod(floordiv(rv, lower_factor), extent).  A local
  // segment (e.g. rv % 2 under thread split lower_factor=2) must survive the
  // projection, so masking the whole low range [0, prod(extent)) is too
  // coarse and rejects valid packed layouts.
  PrimExpr projected_reduce_var = reduce_var;
  for (const auto &step : steps) {
    PrimExpr lower = MakeConst(reduce_var.ty(), step.lower_factor);
    PrimExpr extent = MakeConst(reduce_var.ty(), step.extent);
    projected_reduce_var =
        projected_reduce_var -
        FloorMod(FloorDiv(reduce_var, lower), extent) * lower;
  }

  PrimExpr simplified_index = analyzer->Simplify(index_expr);
  PrimExpr projected_index = analyzer->Simplify(SubstituteLayoutVars(
      simplified_index,
      {{reduce_var, analyzer->Simplify(projected_reduce_var)}}));

  ICHECK(analyzer->CanProveEqual(projected_index, simplified_index))
      << "ReduceOp cannot lower a layout where a source index depends on a "
         "thread-owned reduce segment: src_index="
      << simplified_index << ", projected_src_index=" << projected_index
      << ", reduce_var=" << reduce_var
      << ", projected_reduce_var=" << projected_reduce_var;
}

inline std::pair<PrimExpr, LayoutIterVar>
BuildLocalReduceIterator(const PrimExpr &index_expr,
                         const Array<LayoutIterVar> &input_iters,
                         const PrimVar &reduce_var,
                         const std::vector<ThreadReduceStep> &thread_steps,
                         sym::AnalyzerObj *analyzer) {
  auto [expr, iter_var] =
      CompressIterator(index_expr, input_iters, reduce_var, analyzer);
  auto proof_analyzer = CreateTileLangAnalyzer();
  for (const auto &iv : input_iters) {
    BindAnalyzer(proof_analyzer, iv->var, iv->dom, /*allow_override=*/true);
  }
  CheckThreadOwnedStepsProjectable(index_expr, reduce_var, thread_steps,
                                   proof_analyzer.get());

  int64_t remaining_thread_owned_factor = ThreadOwnedReduceFactor(thread_steps);
  PrimExpr cur_expr = expr;
  LayoutIterVar cur_iter_var = iter_var;
  while (remaining_thread_owned_factor > 1) {
    PrimExpr cur_extent = proof_analyzer->Simplify(cur_iter_var->dom->extent);
    auto candidates = CandidateThreadOwnedFactors(
        remaining_thread_owned_factor, cur_extent, proof_analyzer.get());
    bool removed = false;
    for (int64_t factor : candidates) {
      auto updated = TryRemoveThreadOwnedFactor(cur_expr, cur_iter_var, factor,
                                                proof_analyzer.get());
      if (updated.has_value()) {
        std::tie(cur_expr, cur_iter_var) = updated.value();
        remaining_thread_owned_factor /= factor;
        removed = true;
        break;
      }
    }
    if (!removed) {
      break;
    }
  }
  return {analyzer->Simplify(cur_expr), cur_iter_var};
}

inline ReduceOwnershipPlan
MakeReduceOwnershipPlan(const Array<PrimExpr> &src_indices,
                        const PrimExpr &src_thread,
                        const Array<LayoutIterVar> &src_vars,
                        const PrimVar &reduce_var, sym::AnalyzerObj *analyzer) {
  // Use src_thread as the single source of truth for thread-owned reduce
  // splits.  These steps are later used verbatim to emit scalar or batched
  // AllReduce, so the local loop must not enumerate the same split again.
  auto thread_iter_sum =
      NormalizeLayoutIterSum(src_thread, ToVMap(src_vars), analyzer);
  auto thread_steps = CollectThreadReduceSteps(thread_iter_sum, reduce_var);

  // A surviving fused mark may still hide dependence on the reduction axis.
  // Prove that the collected segments cover all thread ownership; otherwise
  // treating an empty list as a purely local reduction would duplicate values.
  auto ownership_analyzer = analyzer->Clone();
  for (const auto &var : src_vars)
    BindAnalyzer(ownership_analyzer, var->var, var->dom, true);
  PrimExpr expected_thread = SubstituteLayoutVars(
      src_thread, {{reduce_var, MakeConst(reduce_var.ty(), 0)}});
  for (const auto &step : thread_steps)
    expected_thread =
        expected_thread +
        FloorMod(
            FloorDiv(reduce_var, MakeConst(reduce_var.ty(), step.lower_factor)),
            MakeConst(reduce_var.ty(), step.extent)) *
            MakeConst(reduce_var.ty(), step.scale);
  ICHECK(ownership_analyzer->CanProveEqual(src_thread, expected_thread))
      << "ReduceOp cannot separate the thread-owned reduction digits: "
      << src_thread << ", collected=" << expected_thread;

  // Build the per-thread source indexing plan.  A thread-owned factor is
  // removed from the compressed local iterator only after proving that
  // projecting out that factor leaves the physical source index unchanged.
  // This keeps ownership from src_thread, while using src_indices as a safety
  // check against dropping a factor that still selects different local values.
  Array<PrimExpr> local_src_indices;
  Array<LayoutIterVar> local_reduce_vars;
  for (const auto &src_index : src_indices) {
    auto [expr, var] = BuildLocalReduceIterator(src_index, src_vars, reduce_var,
                                                thread_steps, analyzer);
    local_src_indices.push_back(expr);
    local_reduce_vars.push_back(var);
  }

  return ReduceOwnershipPlan{local_src_indices, local_reduce_vars,
                             thread_steps};
}

inline PrimExpr MakeUpdate(const ReduceOpNode &op, PrimExpr dst_val,
                           PrimExpr src_val) {
  const bool use_nan_op =
      op.nan_propagate && (dst_val.ty().MatchesElementType(kDLFloat, 16) ||
                           dst_val.ty().MatchesElementType(kDLBfloat, 16));
  if (op.type->IsSum() || op.type->IsAbsSum()) {
    return dst_val + src_val;
  } else if (op.type->IsBitAnd()) {
    return op.clear ? src_val : bitwise_and(dst_val, src_val);
  } else if (op.type->IsBitOr()) {
    return bitwise_or(dst_val, src_val);
  } else if (op.type->IsBitXor()) {
    return bitwise_xor(dst_val, src_val);
  } else if (op.type->IsMax() || op.type->IsAbsMax()) {
    return use_nan_op
               ? Call(dst_val.ty(), Op::Get("tl.max_nan"), {dst_val, src_val})
                     .as_or_throw<PrimExpr>()
               : PrimExpr(Max(dst_val, src_val));
  } else if (op.type->IsMin()) {
    return use_nan_op
               ? Call(dst_val.ty(), Op::Get("tl.min_nan"), {dst_val, src_val})
                     .as_or_throw<PrimExpr>()
               : PrimExpr(Min(dst_val, src_val));
  }
  LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
  TVM_FFI_THROW(ValueError) << "Unsupported reduction expression";
}

} // namespace reduce

template <typename Impl> struct ReduceLowerer {
  static void CheckAllReduceWidth(int reducing_threads, int scale,
                                  const char *op_name, Target) {
    reduce::CheckAllReduceWidth(reducing_threads, scale, op_name);
  }

  static Stmt LowerLocal(const ReduceOpNode &op, const TensorVar &src_buffer,
                         const TensorVar &dst_buffer,
                         const LowerArgs &lower_args) {
    ICHECK_EQ(op.batch, 1)
        << "ReduceOp: local reduction does not support batch";
    ICHECK(op.src.dtype() == op.dst.dtype())
        << "ReduceOp: local packed reduction currently requires matching src "
           "and dst dtypes";

    int src_dim = static_cast<int>(op.src.shape().size());
    int dst_dim = static_cast<int>(op.dst.shape().size());
    ICHECK(op.dim >= 0 && op.dim < src_dim);
    ICHECK(dst_dim == src_dim - 1 || dst_dim == src_dim)
        << "ReduceOp: local reduction dimension mismatch";

    Array<PrimVar> dst_vars;
    Array<PrimExpr> dst_indices;
    for (int i = 0; i < dst_dim; ++i) {
      PrimVar var("i" + std::to_string(i), op.dst.shape()[i].ty());
      dst_vars.push_back(var);
      dst_indices.push_back(var);
    }

    auto make_src_indices = [&](PrimExpr reduce_index) {
      Array<PrimExpr> indices;
      for (int i = 0; i < src_dim; ++i) {
        if (i == op.dim) {
          indices.push_back(reduce_index);
        } else if (dst_dim == src_dim) {
          indices.push_back(dst_vars[i]);
        } else {
          indices.push_back(dst_vars[i < op.dim ? i : i - 1]);
        }
      }
      return indices;
    };

    int vsize = Impl::GetPreferredVectorizedSize(op, lower_args.target);
    const auto reduce_extent = TryConstInt(op.src.shape()[op.dim]);
    bool can_pack = op.clear && vsize == 2 && reduce_extent &&
                    *reduce_extent >= vsize && *reduce_extent % vsize == 0 &&
                    reduce::MakeCodegenReducer(op, vsize).has_value();

    Array<Stmt> stmts;
    TensorVar packed_buffer{ffi::UnsafeInit{}};
    if (can_pack) {
      packed_buffer = MakeReductionTensor(
          dst_buffer.shape(), dst_buffer.dtype().WithLanes(vsize),
          dst_buffer->name + "_pack", src_buffer.scope());
      stmts.push_back(TensorStore(packed_buffer, dst_indices,
                                  reduce::MakeInitValue(op, vsize)));

      PrimVar rv("rv", PrimType::Int(32));
      PrimExpr base = rv * vsize;
      PrimExpr src_value{ffi::UnsafeInit{}};
      if (op.dim == src_dim - 1) {
        Array<PrimExpr> src_indices =
            make_src_indices(Ramp(base, IntImm(PrimType::Int(32), 1), vsize));
        TensorLoad src_load(src_buffer, src_indices);
        src_value = src_load;
      } else {
        PrimExpr value0 = TensorLoad(src_buffer, make_src_indices(base));
        PrimExpr value1 = TensorLoad(
            src_buffer, make_src_indices(base + IntImm(PrimType::Int(32), 1)));
        src_value = Shuffle({value0, value1}, {0, 1});
      }
      Stmt reduce_body = TensorStore(
          packed_buffer, dst_indices,
          reduce::MakeReduce(op, vsize, TensorLoad(packed_buffer, dst_indices),
                             src_value));
      stmts.push_back(For(rv, IntImm(rv.ty(), 0),
                          IntImm(PrimType::Int(32), *reduce_extent / vsize),
                          IntImm(rv.ty(), 1), ForKind::kUnrolled,
                          SeqStmt(reduce_body)));

      PrimExpr packed = TensorLoad(packed_buffer, dst_indices);
      PrimExpr result =
          reduce::MakeReduce(op, 1, Shuffle::ExtractElement(packed, 0),
                             Shuffle::ExtractElement(packed, 1));
      stmts.push_back(TensorStore(dst_buffer, dst_indices, result));
    } else {
      if (op.clear) {
        stmts.push_back(
            TensorStore(dst_buffer, dst_indices, reduce::MakeInitValue(op)));
      }
      PrimVar rv("rv", PrimType::Int(32));
      Stmt reduce_body = TensorStore(
          dst_buffer, dst_indices,
          reduce::MakeReduce(op, 1, TensorLoad(dst_buffer, dst_indices),
                             TensorLoad(src_buffer, make_src_indices(rv))));
      stmts.push_back(For(rv, IntImm(rv.ty(), 0), op.src.shape()[op.dim],
                          IntImm(rv.ty(), 1), ForKind::kUnrolled,
                          SeqStmt(reduce_body)));
    }

    Stmt body = SeqStmt(stmts);
    for (int i = dst_dim - 1; i >= 0; --i) {
      body =
          For(dst_vars[i], IntImm(dst_vars[i].ty(), 0), op.dst.shape()[i],
              IntImm(dst_vars[i].ty(), 1), ForKind::kUnrolled, SeqStmt(body));
    }
    if (can_pack) {
      body = SeqStmt({AllocateReductionTensor(packed_buffer), body});
    }
    return body;
  }

  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &lower_args,
                    sym::AnalyzerObj *analyzer) {
    if (op.nan_propagate &&
        (op.dst.dtype().MatchesElementType(kDLFloat, 16) ||
         op.dst.dtype().MatchesElementType(kDLBfloat, 16)) &&
        !Impl::SupportsFp16Bf16NanReduce(lower_args.target)) {
      LOG(FATAL) << "ReduceOp: nan_propagate=True for fp16/bf16 "
                    "max/min/absmax is only supported on CUDA targets "
                    "(requires __hmax_nan/__hmin_nan intrinsics). Target was: "
                 << lower_args.target->str();
    }
    auto get_buffer = [&](const TensorVar &buffer) {
      auto it = lower_args.buffer_remap.find(buffer);
      return it == lower_args.buffer_remap.end() ? buffer : (*it).second;
    };

    if (IsLocalBuffer(op.src) && IsLocalBuffer(op.dst, /*allow_var*/ true)) {
      return LowerLocal(op, get_buffer(op.src), get_buffer(op.dst), lower_args);
    }

    if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
      auto src_buffer = get_buffer(op.src);
      auto dst_buffer = get_buffer(op.dst);
      auto src_layout = lower_args.layout_map[op.src].as<Fragment>().value();
      auto dst_layout = lower_args.layout_map[op.dst].as<Fragment>().value();
      auto red_layout = ComputeReducerLayout(src_layout, op.dim);
      // Lower in the canonical reduced shape while preserving the destination's
      // physical register indices, replication, and participating thread range.
      dst_layout = (dst_layout->Reshape(red_layout->InputShape(), analyzer))
                       .template as_or_throw<Fragment>();
      auto src_dim = src_layout->InputDim();
      auto dst_dim = dst_layout->InputDim();

      auto is_1d_reduce = src_dim == dst_dim && dst_dim == 1;

      if (is_1d_reduce) {
        ICHECK(IsOne(dst_layout->OutputShape().back()))
            << "Reduce for scalar not implemented.";
      } else {
        ICHECK_EQ(src_dim, dst_dim + 1) << "Reduce dimension mismatch.";
      }

      Array<LayoutIterVar> dst_vars;
      for (size_t i = 0; i < dst_dim; ++i) {
        PrimVar var = PrimVar(std::string{char('i' + i)},
                              dst_layout->InputShape()[i].ty());
        dst_vars.push_back(
            LayoutIterVar(Range(0, dst_layout->InputShape()[i]), var));
      }

      Array<LayoutIterVar> src_vars;
      if (!is_1d_reduce) {
        src_vars = dst_vars;
      }
      Range reduce_dom(0, src_layout->InputShape()[op.dim]);
      LayoutIterVar reduce_iv(reduce_dom,
                              PrimVar("rv", reduce_dom->extent.ty()));
      src_vars.insert(src_vars.begin() + op.dim, reduce_iv);

      auto src_indices = src_layout->Forward(
          src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
      auto dst_indices = dst_layout->Forward(
          dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
      auto red_indices = red_layout->Forward(
          dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
      auto src_thread = src_layout->ForwardThread(
          src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }), {});

      auto reduce_plan = reduce::MakeReduceOwnershipPlan(
          src_indices, src_thread, src_vars, src_vars[op.dim]->var, analyzer);

      Array<Stmt> stmts;

      auto require_init = op.clear;
      if (op.type->IsSum() || op.type->IsAbsSum() || op.type->IsBitAnd() ||
          op.type->IsBitOr() || op.type->IsBitXor()) {
        require_init = true;
      }

      auto clear_buffer = dst_buffer;
      auto need_duplicate = false;
      auto need_update = false;
      if ((op.type->IsSum() || op.type->IsAbsSum()) && !op.clear) {
        need_duplicate = true;
        need_update = true;
      } else if (op.type->IsBitAnd() && !op.clear) {
        need_duplicate = true;
        need_update = true;
      } else if ((op.type->IsBitOr() || op.type->IsBitXor()) && !op.clear) {
        need_duplicate = true;
        need_update = true;
      } else if ((op.type->IsMax() || op.type->IsMin() ||
                  op.type->IsAbsMax()) &&
                 !op.clear) {
        need_duplicate = true;
        need_update = true;
      }

      if (!analyzer->CanProve(dst_layout->ReplicateExtent() ==
                              red_layout->ReplicateExtent())) {
        need_duplicate = true;
      }
      ICHECK(!analyzer->CanProve(dst_layout->ReplicateExtent() >
                                 red_layout->ReplicateExtent()))
          << "Inconsistent layouts between src and dst in ReduceOp: "
          << "dst_layout=" << dst_layout << "red_layout=" << red_layout;

      if (need_duplicate) {
        clear_buffer = MakeReductionTensor(
            red_layout->OutputShape(), dst_buffer.dtype(),
            dst_buffer->name + "_clear", dst_buffer.scope());
      }

      Array<PrimExpr> src_indice_compressed = reduce_plan.local_src_indices;
      Array<LayoutIterVar> src_var_compressed = reduce_plan.local_reduce_vars;

      bool can_pack = false;
      bool need_pack_buffer = false;
      TensorVar clear_buffer_packed{ffi::UnsafeInit{}};
      Array<TensorVar> batch_pack_buffers;
      {
        int vsize = Impl::GetPreferredVectorizedSize(op, lower_args.target);
        if (vsize > 1 && !src_var_compressed.empty()) {
          auto *ext = src_var_compressed.back()->dom->extent.as<IntImmObj>();
          if (ext && ext->value >= vsize && ext->value % vsize == 0 &&
              reduce::MakeCodegenReducer(op, vsize).has_value() &&
              reduce::CanUsePackedRamp(src_indice_compressed.back(),
                                       src_var_compressed.back()->var, vsize,
                                       analyzer)) {
            can_pack = true;
            PrimType vec_dtype = clear_buffer.dtype().WithLanes(vsize);
            clear_buffer_packed = MakeReductionTensor(
                red_layout->OutputShape(), vec_dtype,
                clear_buffer->name + "_pack", clear_buffer.scope());
            need_pack_buffer = true;

            Array<Stmt> local_body;

            if (require_init ||
                (need_duplicate && (op.type->IsMax() || op.type->IsMin() ||
                                    op.type->IsAbsMax()))) {
              local_body.push_back(
                  TensorStore(clear_buffer_packed, red_indices,
                              reduce::MakeInitValue(op, vsize)));
            }

            const auto ext_int =
                TryConstInt(src_var_compressed.back()->dom->extent);
            int64_t inner_extent = *ext_int;
            PrimExpr halved_extent =
                IntImm(PrimType::Int(32), inner_extent / vsize);

            LayoutIterVar inner_var = src_var_compressed.back();

            PrimExpr ramp_base = SubstituteLayoutVars(
                src_indice_compressed.back(),
                {{inner_var->var,
                  inner_var->var * IntImm(PrimType::Int(32), 2)}});
            src_indice_compressed.Set(
                src_indice_compressed.size() - 1,
                Ramp(ramp_base, IntImm(PrimType::Int(32), 1), vsize));

            auto src_load = TensorLoad(src_buffer, src_indice_compressed);

            Stmt reduce_local = TensorStore(
                clear_buffer_packed, red_indices,
                reduce::MakeReduce(op, vsize,
                                   TensorLoad(clear_buffer_packed, red_indices),
                                   src_load));

            reduce_local = For(inner_var->var, IntImm(inner_var->var.ty(), 0),
                               halved_extent, IntImm(inner_var->var.ty(), 1),
                               ForKind::kUnrolled, SeqStmt(reduce_local));

            for (int i = static_cast<int>(src_layout->OutputDim()) - 2; i >= 0;
                 --i) {
              reduce_local = For(src_var_compressed[i]->var,
                                 IntImm(src_var_compressed[i]->var.ty(), 0),
                                 src_var_compressed[i]->dom->extent,
                                 IntImm(src_var_compressed[i]->var.ty(), 1),
                                 ForKind::kUnrolled, SeqStmt(reduce_local));
            }
            local_body.push_back(reduce_local);

            auto acc_vec = TensorLoad(clear_buffer_packed, red_indices);
            auto lane0 = Shuffle::ExtractElement(acc_vec, 0);
            auto lane1 = Shuffle::ExtractElement(acc_vec, 1);
            auto scalar_result = reduce::MakeReduce(op, 1, lane0, lane1);
            local_body.push_back(
                TensorStore(clear_buffer, red_indices, scalar_result));

            stmts.push_back(SeqStmt(local_body));
          }
        }
      }

      if (!can_pack) {
        if (require_init ||
            (need_duplicate &&
             (op.type->IsMax() || op.type->IsMin() || op.type->IsAbsMax()))) {
          stmts.push_back(TensorStore(clear_buffer, red_indices,
                                      reduce::MakeInitValue(op)));
        }

        Stmt reduce_local = TensorStore(
            clear_buffer, red_indices,
            reduce::MakeReduce(op, 1, TensorLoad(clear_buffer, red_indices),
                               TensorLoad(src_buffer, src_indice_compressed)));

        for (int i = static_cast<int>(src_layout->OutputDim()) - 1; i >= 0;
             --i) {
          reduce_local = For(src_var_compressed[i]->var,
                             IntImm(src_var_compressed[i]->var.ty(), 0),
                             src_var_compressed[i]->dom->extent,
                             IntImm(src_var_compressed[i]->var.ty(), 1),
                             ForKind::kUnrolled, SeqStmt(reduce_local));
        }
        stmts.push_back(reduce_local);
      }

      const int batch = op.batch;
      if (batch > 1) {
        int64_t N_total = 1;
        for (const auto &s : clear_buffer.shape()) {
          const auto p = TryConstInt(s);
          ICHECK(p != std::nullopt)
              << "ReduceOp: batch > 1 requires compile-time "
                 "constant output shape";
          N_total *= *p;
        }
        ICHECK_LE(batch, N_total)
            << "ReduceOp: batch=" << batch
            << " exceeds per-thread output element count N=" << N_total;
        ICHECK_EQ(N_total % batch, 0) << "ReduceOp: batch=" << batch
                                      << " must evenly divide N=" << N_total;
      }

      bool use_batch = batch > 1;
      const bool red_requires_padding_guard =
          reduce::NeedsPaddingGuard(red_layout, analyzer);

      auto make_dst_loop = [&](Stmt body,
                               const Array<LayoutIterVar> &vars) -> Stmt {
        for (int i = static_cast<int>(vars.size()) - 1; i >= 0; --i) {
          body = For(vars[i]->var, IntImm(vars[i]->var.ty(), 0),
                     vars[i]->dom->extent, IntImm(vars[i]->var.ty(), 1),
                     ForKind::kParallel, SeqStmt(body));
        }
        body = PartitionLoop((body).template as_or_throw<For>(),
                             lower_args.thread_index, analyzer, red_layout,
                             red_requires_padding_guard);
        body = PragmaUnrollLoop((body).template as_or_throw<For>());
        return body;
      };

      auto make_fresh_dst_vars = [&](const std::string &suffix)
          -> std::tuple<Array<LayoutIterVar>, Array<PrimExpr>,
                        Array<PrimExpr>> {
        Array<LayoutIterVar> vars;
        for (size_t i = 0; i < dst_dim; ++i) {
          PrimVar v(std::string{char('i' + i)} + suffix,
                    dst_layout->InputShape()[i].ty());
          vars.push_back(
              LayoutIterVar(Range(0, dst_layout->InputShape()[i]), v));
        }
        auto d_idx = dst_layout->Forward(
            vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
        auto r_idx = red_layout->Forward(
            vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
        return {vars, d_idx, r_idx};
      };

      if (use_batch) {
        Optional<Range> batch_thread_range;
        if (TargetSupportsNamedBarrier(lower_args.target) &&
            std::any_of(reduce_plan.thread_steps.begin(),
                        reduce_plan.thread_steps.end(), [](const auto &step) {
                          return step.ReducingThreads() > 32;
                        })) {
          batch_thread_range = reduce::ResolveAllReduceThreadRange(
              red_layout, lower_args.thread_bounds, lower_args.target);
        }
        Stmt pre_body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
        pre_body = make_dst_loop(pre_body, dst_vars);

        Array<Stmt> phases;
        phases.push_back(pre_body);

        for (const auto &thread_step : reduce_plan.thread_steps) {
          int reducing_threads = thread_step.ReducingThreads();
          Impl::CheckAllReduceWidth(reducing_threads, thread_step.scale,
                                    "tl.reduce", lower_args.target);
          int block_threads =
              static_cast<int>(*TryConstInt(lower_args.thread_bounds->extent));
          auto thread_offset = lower_args.thread_bounds->begin;
          PrimExpr all_threads = lower_args.thread_bounds->extent;
          if (batch_thread_range) {
            thread_offset = batch_thread_range->begin;
            all_threads = batch_thread_range->extent;
          }
          Array<Stmt> batch_phases;

          int vsize = Impl::GetPreferredVectorizedSize(op, lower_args.target);
          bool can_batch_pack =
              vsize > 1 && batch >= vsize && batch % vsize == 0 &&
              reduce::MakeCodegenReducer(op, vsize).has_value();
          int eff_batch = can_batch_pack ? (batch / vsize) : batch;
          std::string reducer =
              reduce::MakeCodegenReducer(op, can_batch_pack ? vsize : 1)
                  .value();
          std::string allreduce = Impl::MakeBatchAllReduce(
              reducer, reducing_threads, thread_step.scale, thread_offset,
              all_threads, eff_batch, block_threads, lower_args.target);

          PrimType ws_dtype = can_batch_pack
                                  ? clear_buffer.dtype().WithLanes(vsize)
                                  : clear_buffer.dtype();
          Expr workspace{ffi::UnsafeInit{}};
          bool need_workspace = Impl::AllReduceNeedsWorkspace(
              reducing_threads, thread_step.scale, lower_args.target);
          if (need_workspace) {
            int ws_size = block_threads * eff_batch;
            workspace = lower_args.add_workspace(ws_size, ws_dtype);
          }

          int64_t N_total = 1;
          for (const auto &s : clear_buffer.shape()) {
            N_total *= *TryConstInt(s);
          }
          int num_chunks = static_cast<int>(N_total / batch);

          int buf_ndim = static_cast<int>(clear_buffer.shape().size());
          std::vector<int64_t> buf_shape_vals;
          for (const auto &s : clear_buffer.shape()) {
            buf_shape_vals.push_back(*TryConstInt(s));
          }
          std::vector<int64_t> buf_strides(buf_ndim, 1);
          for (int d = buf_ndim - 2; d >= 0; d--) {
            buf_strides[d] = buf_strides[d + 1] * buf_shape_vals[d + 1];
          }

          if (can_batch_pack) {
            int packed_batch = batch / vsize;

            TensorVar pack_buf = MakeReductionTensor(
                {IntImm(PrimType::Int(32), packed_batch)},
                clear_buffer.dtype().WithLanes(vsize),
                clear_buffer->name + "_pack", clear_buffer.scope());

            batch_pack_buffers.push_back(pack_buf);

            for (int chunk = 0; chunk < num_chunks; chunk++) {
              int64_t flat_offset = static_cast<int64_t>(chunk) * batch;

              PrimVar pack_j("pack_j", PrimType::Int(32));
              PrimExpr base = IntImm(PrimType::Int(32), flat_offset);
              PrimExpr scaled = pack_j * vsize;

              Array<PrimExpr> idx_a;
              Array<PrimExpr> idx_b;
              PrimExpr fa = base + scaled;
              PrimExpr fb = base + scaled + IntImm(PrimType::Int(32), 1);
              for (int d = 0; d < buf_ndim; d++) {
                idx_a.push_back(FloorMod(
                    FloorDiv(fa, IntImm(PrimType::Int(32), buf_strides[d])),
                    IntImm(PrimType::Int(32), buf_shape_vals[d])));
                idx_b.push_back(FloorMod(
                    FloorDiv(fb, IntImm(PrimType::Int(32), buf_strides[d])),
                    IntImm(PrimType::Int(32), buf_shape_vals[d])));
              }
              auto a_load = TensorLoad(clear_buffer, idx_a);
              auto b_load = TensorLoad(clear_buffer, idx_b);
              Stmt pack_body = TensorStore(pack_buf, {pack_j},
                                           Shuffle({a_load, b_load}, {0, 1}));
              Stmt pack_loop = For(pack_j, IntImm(pack_j.ty(), 0), packed_batch,
                                   IntImm(pack_j.ty(), 1), ForKind::kUnrolled,
                                   SeqStmt(pack_body));
              batch_phases.push_back(pack_loop);

              Expr packed_ptr = MakeTensorAccessPtr(
                  TensorLoad(pack_buf, {IntImm(PrimType::Int(32), 0)}),
                  IntImm(PrimType::Int(32), packed_batch), 3);
              Array<Expr> args = {StringImm(allreduce), packed_ptr};
              if (need_workspace) {
                args.push_back(workspace);
              }
              batch_phases.push_back(
                  Evaluate(Call(PrimType::Void(), call_extern_op(), args)));

              PrimVar unpack_j("unpack_j", PrimType::Int(32));
              PrimExpr ubase = IntImm(PrimType::Int(32), flat_offset);
              PrimExpr uscaled = unpack_j * vsize;
              Array<PrimExpr> uidx_a;
              Array<PrimExpr> uidx_b;
              PrimExpr ufa = ubase + uscaled;
              PrimExpr ufb = ubase + uscaled + IntImm(PrimType::Int(32), 1);
              for (int d = 0; d < buf_ndim; d++) {
                uidx_a.push_back(FloorMod(
                    FloorDiv(ufa, IntImm(PrimType::Int(32), buf_strides[d])),
                    IntImm(PrimType::Int(32), buf_shape_vals[d])));
                uidx_b.push_back(FloorMod(
                    FloorDiv(ufb, IntImm(PrimType::Int(32), buf_strides[d])),
                    IntImm(PrimType::Int(32), buf_shape_vals[d])));
              }
              auto packed_val = TensorLoad(pack_buf, {unpack_j});
              Stmt unpack_body = SeqStmt({
                  TensorStore(clear_buffer, uidx_a,
                              Shuffle::ExtractElement(packed_val, 0)),
                  TensorStore(clear_buffer, uidx_b,
                              Shuffle::ExtractElement(packed_val, 1)),
              });
              Stmt unpack_loop = For(unpack_j, IntImm(unpack_j.ty(), 0),
                                     packed_batch, IntImm(unpack_j.ty(), 1),
                                     ForKind::kUnrolled, SeqStmt(unpack_body));
              batch_phases.push_back(unpack_loop);
            }
          } else {
            for (int chunk = 0; chunk < num_chunks; chunk++) {
              int64_t flat_offset = static_cast<int64_t>(chunk) * batch;
              Array<PrimExpr> chunk_indices;
              for (int d = 0; d < buf_ndim; d++) {
                int64_t idx =
                    (flat_offset / buf_strides[d]) % buf_shape_vals[d];
                chunk_indices.push_back(IntImm(PrimType::Int(32), idx));
              }
              Expr ptr =
                  MakeTensorAccessPtr(TensorLoad(clear_buffer, chunk_indices),
                                      IntImm(PrimType::Int(32), batch), 3);

              Array<Expr> args = {StringImm(allreduce), ptr};
              if (need_workspace) {
                args.push_back(workspace);
              }
              batch_phases.push_back(
                  Evaluate(Call(PrimType::Void(), call_extern_op(), args)));
            }
          }
          Stmt batch_body = SeqStmt(batch_phases);
          if (batch_thread_range) {
            PrimExpr active =
                (lower_args.thread_index >= batch_thread_range->begin) &&
                (lower_args.thread_index <
                 batch_thread_range->begin + batch_thread_range->extent);
            active = analyzer->Simplify(active);
            if (!analyzer->CanProve(active))
              batch_body = If(active, SeqStmt(batch_body));
          }
          phases.push_back(batch_body);
        }

        if (need_duplicate) {
          auto [post_vars, post_dst_idx, post_red_idx] =
              make_fresh_dst_vars("_p");

          PrimExpr predicate = reduce::DestinationThreadPredicate(
              dst_layout,
              post_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }),
              post_dst_idx, lower_args.thread_index, analyzer);

          PrimExpr update =
              need_update
                  ? reduce::MakeUpdate(op, TensorLoad(dst_buffer, post_dst_idx),
                                       TensorLoad(clear_buffer, post_red_idx))
                  : TensorLoad(clear_buffer, post_red_idx);
          auto store = TensorStore(dst_buffer, post_dst_idx, update);
          Stmt post_body{ffi::UnsafeInit{}};
          if (analyzer->CanProve(predicate)) {
            post_body = store;
          } else {
            post_body = If(predicate, SeqStmt(store));
          }
          phases.push_back(make_dst_loop(post_body, post_vars));
        }

        Stmt body = phases.size() > 1 ? SeqStmt(phases) : phases[0];
        if (need_duplicate) {
          body = SeqStmt({AllocateReductionTensor(clear_buffer), body});
        }
        if (need_pack_buffer) {
          body = SeqStmt({AllocateReductionTensor(clear_buffer_packed), body});
        }
        for (auto it = batch_pack_buffers.rbegin();
             it != batch_pack_buffers.rend(); ++it)
          body = SeqStmt({AllocateReductionTensor(*it), body});
        return body;
      }

      for (const auto &thread_step : reduce_plan.thread_steps) {
        int reducing_threads = thread_step.ReducingThreads();
        Impl::CheckAllReduceWidth(reducing_threads, thread_step.scale,
                                  "tl.reduce", lower_args.target);
        auto thread_offset = lower_args.thread_bounds->begin;
        PrimExpr all_threads = lower_args.thread_bounds->extent;
        if (reducing_threads > 32 &&
            TargetSupportsNamedBarrier(lower_args.target)) {
          Range thread_range = reduce::ResolveAllReduceThreadRange(
              red_layout, lower_args.thread_bounds, lower_args.target);
          thread_offset = thread_range->begin;
          all_threads = thread_range->extent;
        }
        std::string allreduce = Impl::MakeScalarAllReduce(
            reduce::MakeCodegenReducer(op).value(), reducing_threads,
            thread_step.scale, thread_offset, all_threads, lower_args.target);
        Array<Expr> thread_reduce_args = {
            StringImm(allreduce), TensorLoad(clear_buffer, red_indices)};
        if (Impl::AllReduceNeedsWorkspace(reducing_threads, thread_step.scale,
                                          lower_args.target)) {
          int workspace_size =
              static_cast<int>(*TryConstInt(lower_args.thread_bounds->extent));
          Expr workspace =
              lower_args.add_workspace(workspace_size, clear_buffer.dtype());
          thread_reduce_args.push_back(workspace);
        }
        auto call =
            Call(clear_buffer.dtype(), call_extern_op(), thread_reduce_args);
        stmts.push_back(TensorStore(clear_buffer, red_indices,
                                    call.as_or_throw<PrimExpr>()));
      }

      if (need_duplicate) {
        PrimExpr predicate = reduce::DestinationThreadPredicate(
            dst_layout,
            dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }),
            dst_indices, lower_args.thread_index, analyzer);
        PrimExpr update =
            need_update
                ? reduce::MakeUpdate(op, TensorLoad(dst_buffer, dst_indices),
                                     TensorLoad(clear_buffer, red_indices))
                : TensorLoad(clear_buffer, red_indices);
        auto store = TensorStore(dst_buffer, dst_indices, update);
        if (analyzer->CanProve(predicate)) {
          stmts.push_back(store);
        } else {
          stmts.push_back(If(predicate, SeqStmt(store)));
        }
      }

      auto body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
      for (int i = static_cast<int>(dst_layout->InputDim()) - 1; i >= 0; --i) {
        body = For(dst_vars[i]->var, IntImm(dst_vars[i]->var.ty(), 0),
                   dst_vars[i]->dom->extent, IntImm(dst_vars[i]->var.ty(), 1),
                   ForKind::kParallel, SeqStmt(body));
      }

      if (dst_layout->InputDim() > 0) {
        body = PartitionLoop((body).template as_or_throw<For>(),
                             lower_args.thread_index, analyzer, red_layout,
                             red_requires_padding_guard);
        body = PragmaUnrollLoop((body).template as_or_throw<For>());
      } else {
        auto guard =
            (lower_args.thread_index == lower_args.thread_bounds->begin);
        body = If(guard, SeqStmt(body));
      }

      if (need_duplicate) {
        body = SeqStmt({AllocateReductionTensor(clear_buffer), body});
      }
      if (need_pack_buffer) {
        body = SeqStmt({AllocateReductionTensor(clear_buffer_packed), body});
      }
      return body;
    }

    LOG(FATAL) << "Reduce for buffers in scope (" << op.src.scope() << ", "
               << op.dst.scope() << ") is not implemented.";
    TVM_FFI_THROW(ValueError) << "Unsupported lowering scope";
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_REDUCE_H_
