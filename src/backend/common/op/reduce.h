/*!
 * \file tl/backend/common/op/reduce.h
 * \brief Shared tl.reduce AllReduce lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_REDUCE_H_
#define TVM_TL_BACKEND_COMMON_OP_REDUCE_H_

#include "backend/common/target_utils.h"
#include "op/reduce.h"
#include "reduce_value.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include "cuda/op/builtin.h"
#include "layout/layout.h"
#include "layout/utils.h"
#include "op/utils.h"
#include "transform/loop_partition.h"

#include "reduce_layout.h"
#include <tvm/sym/analyzer.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/op/abi.h>
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
using namespace ffi;

namespace reduce {

/*!
 * \brief Resolve the participating thread range of a scalar AllReduce.
 *
 * The result is derived from the reduce layout's forward thread map, which is
 * also the source of the guard later emitted by PartitionLoop. Const-int bounds
 * provide the minimum and maximum participating thread IDs.
 */
inline Range ResolveAllReduceThreadRange(const Fragment &red_layout,
                                         const Range &thread_bounds,
                                         const Target &target) {
  auto block_min = GetConstInt(thread_bounds->begin);
  auto block_extent = GetConstInt(thread_bounds->extent);
  auto replicate = GetConstInt(red_layout->ReplicateExtent());
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

  sym::Analyzer analyzer;
  for (size_t i = 0; i < red_layout->InputShape().size(); ++i) {
    PrimVar placeholder = InputPlaceholder(i);
    analyzer->Bind(placeholder,
                   Range::FromMinExtent(MakeConst(placeholder.ty(), 0),
                                        red_layout->InputShape()[i]));
  }
  PrimVar replicate_var = ReplicationPlaceholder();
  analyzer->Bind(replicate_var,
                 Range::FromMinExtent(MakeConst(replicate_var.ty(), 0),
                                      red_layout->ReplicateExtent()));

  // PartitionLoop feeds (threadIdx.x - ThreadRange.min) into the inverse
  // layout. Convert the forward map back to the corresponding absolute CTA
  // thread ID before computing its image.
  PrimExpr thread_expr = red_layout->GetForwardThread();
  if (red_layout->ThreadRange().has_value()) {
    thread_expr = analyzer->Simplify(thread_expr +
                                     red_layout->ThreadRange().value()->begin);
  }
  const sym::ConstIntBound bound = analyzer->const_int_bound(thread_expr);
  if (bound->min_value == sym::ConstIntBound::kNegInf ||
      bound->max_value == sym::ConstIntBound::kPosInf) {
    LOG(FATAL) << "tl.reduce: cannot determine the scalar AllReduce "
                  "participating thread range.";
  }

  const int64_t base = bound->min_value;
  const int64_t end = bound->max_value;
  // TODO: Consider restoring CountSatisfyingValues when the Z3 prover is
  // stable enough to reliably compute the exact participating thread image.
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

  return Range::FromMinExtent(MakeConst(thread_expr.ty(), base),
                              MakeConst(thread_bounds->extent.ty(), count));
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
  int shift = 0;
  ICHECK(IsPowerOfTwoInt(PrimExpr(logical_width), &shift))
      << op_name << ": XOR-butterfly AllReduce requires logical_width "
      << "(threads / scale) to be a positive power of two, got "
      << logical_width << " (threads=" << reducing_threads
      << ", scale=" << scale << ")";
  // The butterfly pairs thread t with t ^ (scale * k).  That only moves the
  // reduce coordinate when the non-reduced part of the thread index occupies
  // the bits below `scale`, i.e. when `scale` is itself a power of two.
  // Otherwise the pairs straddle reduce groups (and replicas) and the result
  // is silently wrong or reads outside the workspace.
  ICHECK(IsPowerOfTwoInt(PrimExpr(scale), &shift))
      << op_name << ": XOR-butterfly AllReduce requires scale (the thread "
      << "stride between consecutive reduce participants) to be a power of "
      << "two, got " << scale << " (threads=" << reducing_threads
      << "). Pad the non-reduced fragment extent to a power of two.";
}

inline bool CanUsePackedRamp(const PrimExpr &index, const PrimVar &var,
                             int vsize, sym::Analyzer analyzer) {
  ICHECK_GT(vsize, 1);

  PrimExpr vector_size = MakeConst(var.ty(), vsize);
  PrimExpr packed_var = var * vector_size;
  PrimExpr ramp_base =
      analyzer->Simplify(Substitute(index, {{var, packed_var}}));

  PrimExpr index_mod = FloorMod(ramp_base, MakeConst(index.ty(), vsize));
  if (!analyzer->CanProveEqual(index_mod, MakeConst(index.ty(), 0))) {
    return false;
  }

  for (int lane = 1; lane < vsize; ++lane) {
    PrimExpr lane_value = MakeConst(var.ty(), lane);
    PrimExpr lane_index =
        analyzer->Simplify(Substitute(index, {{var, packed_var + lane_value}}));
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
  Array<LayoutAxis> local_reduce_vars;
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
                            sym::Analyzer analyzer) {
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
            FloorMod(local_extent, MakeConst(local_extent.ty(), factor)), 0)) {
      divisible_factors.push_back(factor);
    }
  }
  return divisible_factors;
}

inline std::optional<std::pair<PrimExpr, LayoutAxis>>
TryRemoveThreadOwnedFactor(const PrimExpr &expr, const LayoutAxis &iter_var,
                           int64_t factor, sym::Analyzer analyzer) {
  PrimExpr factor_expr = MakeConst(iter_var->var.ty(), factor);
  PrimVar old_var = iter_var->var;
  PrimExpr old_extent = analyzer->Simplify(iter_var->dom->extent);
  if (!analyzer->CanProveEqual(FloorMod(old_extent, factor_expr), 0)) {
    return std::nullopt;
  }

  analyzer->Bind(old_var, Range(0, old_extent), /*allow_override=*/true);
  PrimExpr masked = FloorDiv(old_var, factor_expr) * factor_expr;
  PrimExpr simplified_expr = analyzer->Simplify(expr);
  PrimExpr masked_expr =
      analyzer->Simplify(Substitute(simplified_expr, {{old_var, masked}}));
  if (!analyzer->CanProveEqual(masked_expr, simplified_expr)) {
    return std::nullopt;
  }

  PrimExpr new_extent = analyzer->Simplify(FloorDiv(old_extent, factor_expr));
  PrimVar new_var(old_var->name, old_var.ty());
  PrimExpr new_expr = analyzer->Simplify(
      Substitute(simplified_expr, {{old_var, new_var * factor_expr}}));
  LayoutAxis new_iter_var = LayoutAxis(Range(0, new_extent), new_var);
  analyzer->Bind(new_var, Range(0, new_extent), /*allow_override=*/true);
  return std::make_pair(new_expr, new_iter_var);
}

inline void CheckThreadOwnedStepsProjectable(
    const PrimExpr &index_expr, const PrimVar &reduce_var,
    const std::vector<ThreadReduceStep> &steps, sym::Analyzer analyzer) {
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
  PrimExpr projected_index = analyzer->Simplify(
      Substitute(simplified_index,
                 {{reduce_var, analyzer->Simplify(projected_reduce_var)}}));

  ICHECK(analyzer->CanProve(projected_index == simplified_index,
                            sym::ProofStrength::kSymbolicBound))
      << "ReduceOp cannot lower a layout where a source index depends on a "
         "thread-owned reduce segment: src_index="
      << simplified_index << ", projected_src_index=" << projected_index
      << ", reduce_var=" << reduce_var
      << ", projected_reduce_var=" << projected_reduce_var;
}

inline std::pair<PrimExpr, LayoutAxis> BuildLocalReduceIterator(
    const PrimExpr &index_expr, const Array<LayoutAxis> &input_iters,
    const PrimVar &reduce_var,
    const std::vector<ThreadReduceStep> &thread_steps, sym::Analyzer analyzer) {
  auto [expr, iter_var] =
      CompressIterator(index_expr, input_iters, reduce_var, analyzer);
  sym::Analyzer proof_analyzer;
  for (const auto &iv : input_iters) {
    proof_analyzer->Bind(iv->var, iv->dom, /*allow_override=*/true);
  }
  CheckThreadOwnedStepsProjectable(index_expr, reduce_var, thread_steps,
                                   proof_analyzer);

  int64_t remaining_thread_owned_factor = ThreadOwnedReduceFactor(thread_steps);
  PrimExpr cur_expr = expr;
  LayoutAxis cur_iter_var = iter_var;
  while (remaining_thread_owned_factor > 1) {
    PrimExpr cur_extent = proof_analyzer->Simplify(cur_iter_var->dom->extent);
    auto candidates = CandidateThreadOwnedFactors(remaining_thread_owned_factor,
                                                  cur_extent, proof_analyzer);
    bool removed = false;
    for (int64_t factor : candidates) {
      auto updated = TryRemoveThreadOwnedFactor(cur_expr, cur_iter_var, factor,
                                                proof_analyzer);
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
                        const Array<LayoutAxis> &src_vars,
                        const PrimVar &reduce_var, sym::Analyzer analyzer) {
  // Use src_thread as the single source of truth for thread-owned reduce
  // splits.  These steps are later used verbatim to emit scalar or batched
  // AllReduce, so the local loop must not enumerate the same split again.
  auto thread_steps = CollectThreadReduceSteps(src_thread, ToVMap(src_vars),
                                               reduce_var, analyzer);

  // Build the per-thread source indexing plan.  A thread-owned factor is
  // removed from the compressed local iterator only after proving that
  // projecting out that factor leaves the physical source index unchanged.
  // This keeps ownership from src_thread, while using src_indices as a safety
  // check against dropping a factor that still selects different local values.
  Array<PrimExpr> local_src_indices;
  Array<LayoutAxis> local_reduce_vars;
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
    return use_nan_op ? Call(dst_val.ty(), tl::max_nan(), {dst_val, src_val})
                            .as_or_throw<PrimExpr>()
                      : PrimExpr(Max(dst_val, src_val));
  } else if (op.type->IsMin()) {
    return use_nan_op ? Call(dst_val.ty(), tl::min_nan(), {dst_val, src_val})
                            .as_or_throw<PrimExpr>()
                      : PrimExpr(Min(dst_val, src_val));
  }
  LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
  TVM_FFI_THROW(ValueError) << "Unsupported reduction type";
  TVM_FFI_UNREACHABLE();
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

    const auto &source_ranges = op.srcRegion_->region;
    const auto &destination_ranges = op.dstRegion_->region;
    int src_dim = static_cast<int>(source_ranges.size());
    int dst_dim = static_cast<int>(destination_ranges.size());
    ICHECK(op.dim >= 0 && op.dim < src_dim);
    ICHECK(dst_dim == src_dim - 1 || dst_dim == src_dim)
        << "ReduceOp: local reduction dimension mismatch";

    Array<PrimVar> dst_vars;
    Array<PrimExpr> dst_indices;
    Array<PrimExpr> packed_indices;
    Array<PrimExpr> packed_shape;
    auto remap_indices = [&](const TensorVar &tensor, Array<PrimExpr> indices) {
      if (lower_args.buffer_remap.count(tensor) &&
          lower_args.layout_map.count(tensor)) {
        return lower_args.layout_map[tensor]->Forward(indices);
      }
      return indices;
    };
    for (int axis = 0; axis < dst_dim; ++axis) {
      PrimVar var("reduce_output_" + std::to_string(axis),
                  destination_ranges[axis]->extent.ty());
      dst_vars.push_back(var);
      dst_indices.push_back(destination_ranges[axis]->begin + var);
      packed_indices.push_back(var);
      packed_shape.push_back(destination_ranges[axis]->extent);
    }
    dst_indices = remap_indices(op.dst, dst_indices);

    auto make_src_indices = [&](PrimExpr reduce_index) {
      Array<PrimExpr> indices;
      for (int axis = 0; axis < src_dim; ++axis) {
        PrimExpr coordinate = reduce_index;
        if (axis == op.dim) {
          coordinate = reduce_index;
        } else if (dst_dim == src_dim) {
          coordinate = dst_vars[axis];
        } else {
          coordinate = dst_vars[axis < op.dim ? axis : axis - 1];
        }
        indices.push_back(source_ranges[axis]->begin + coordinate);
      }
      return remap_indices(op.src, indices);
    };

    int vsize = Impl::GetPreferredVectorizedSize(op, lower_args.target);
    auto reduce_extent = GetConstInt(source_ranges[op.dim]->extent);
    bool can_pack = op.clear && vsize == 2 && reduce_extent &&
                    *reduce_extent >= vsize && *reduce_extent % vsize == 0 &&
                    reduce::MakeCodegenReducer(op, vsize).has_value();

    Array<Stmt> stmts;
    TensorVar packed_buffer{ffi::UnsafeInit{}};
    if (can_pack) {
      packed_buffer = TensorVar(dst_buffer->name + "_pack",
                                TensorType(packed_shape,
                                           dst_buffer.dtype().WithLanes(vsize),
                                           src_buffer.scope()));
      stmts.push_back(TensorStore(packed_buffer, packed_indices,
                                  reduce::MakeInitValue(op, vsize)));

      PrimVar rv("rv");
      PrimExpr base = rv * vsize;
      PrimExpr value0 = TensorLoad(src_buffer, make_src_indices(base));
      PrimExpr value1 =
          TensorLoad(src_buffer, make_src_indices(base + PrimExpr(1)));
      PrimExpr src_value = Shuffle({value0, value1}, {0, 1});
      Stmt reduce_body = TensorStore(
          packed_buffer, packed_indices,
          reduce::MakeReduce(
              op, vsize, TensorLoad(packed_buffer, packed_indices), src_value));
      stmts.push_back(For(rv, MakeConst((rv).ty(), 0),
                          MakeConst(rv.ty(), *reduce_extent / vsize),
                          MakeConst((rv).ty(), 1), ForKind::kUnrolled,
                          SeqStmt(reduce_body)));

      PrimExpr packed = TensorLoad(packed_buffer, packed_indices);
      PrimExpr result =
          reduce::MakeReduce(op, 1, Shuffle::ExtractElement(packed, 0),
                             Shuffle::ExtractElement(packed, 1));
      stmts.push_back(TensorStore(dst_buffer, dst_indices, result));
    } else {
      if (op.clear) {
        stmts.push_back(
            TensorStore(dst_buffer, dst_indices, reduce::MakeInitValue(op)));
      }
      PrimVar rv("rv");
      Stmt reduce_body = TensorStore(
          dst_buffer, dst_indices,
          reduce::MakeReduce(op, 1, TensorLoad(dst_buffer, dst_indices),
                             TensorLoad(src_buffer, make_src_indices(rv))));
      stmts.push_back(For(
          rv, MakeConst((rv).ty(), 0), source_ranges[op.dim]->extent,
          MakeConst((rv).ty(), 1), ForKind::kUnrolled, SeqStmt(reduce_body)));
    }

    Stmt body = SeqStmt(stmts);
    for (int axis = dst_dim - 1; axis >= 0; --axis) {
      body = For(dst_vars[axis], MakeConst(dst_vars[axis].ty(), 0),
                 destination_ranges[axis]->extent,
                 MakeConst(dst_vars[axis].ty(), 1), ForKind::kUnrolled,
                 SeqStmt(body));
    }
    if (can_pack) {
      body = SeqStmt(
          {Bind(packed_buffer, Call((packed_buffer).ty(), alloc_tensor_op(), {},
                                    std::nullopt, {(packed_buffer).ty()})),
           body});
    }
    return body;
  }

  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &lower_args,
                    sym::Analyzer analyzer) {
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
                       .as_or_throw<Fragment>();
      auto src_dim = src_layout->InputDim();
      auto dst_dim = dst_layout->InputDim();

      auto is_1d_reduce = src_dim == dst_dim && dst_dim == 1;

      if (is_1d_reduce) {
        ICHECK(IsOne(dst_layout->OutputShape().back()))
            << "Reduce for scalar not implemented.";
      } else {
        ICHECK_EQ(src_dim, dst_dim + 1) << "Reduce dimension mismatch.";
      }

      Array<LayoutAxis> dst_vars;
      for (size_t i = 0; i < dst_dim; ++i) {
        PrimVar var = PrimVar(std::string{char('i' + i)});
        dst_vars.push_back(
            LayoutAxis(Range(0, dst_layout->InputShape()[i]), var));
      }

      Array<LayoutAxis> src_vars;
      if (!is_1d_reduce) {
        src_vars = dst_vars;
      }
      Range reduce_dom(0, src_layout->InputShape()[op.dim]);
      LayoutAxis reduce_iv(reduce_dom, PrimVar("rv"));
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
        clear_buffer =
            TensorVar(dst_buffer->name + "_clear",
                      TensorType(red_layout->OutputShape(), dst_buffer.dtype(),
                                 dst_buffer.scope()));
      }

      Array<PrimExpr> src_indice_compressed = reduce_plan.local_src_indices;
      Array<LayoutAxis> src_var_compressed = reduce_plan.local_reduce_vars;

      bool can_pack = false;
      bool need_pack_buffer = false;
      bool need_batch_pack_buffer = false;
      TensorVar clear_buffer_packed{ffi::UnsafeInit{}};
      TensorVar clear_batch_pack_buffer{ffi::UnsafeInit{}};
      {
        int vsize = Impl::GetPreferredVectorizedSize(op, lower_args.target);
        if (vsize > 1 && !src_var_compressed.empty()) {
          auto *ext = src_var_compressed.back()->dom->extent.as<IntImmNode>();
          if (ext && ext->value >= vsize && ext->value % vsize == 0 &&
              reduce::MakeCodegenReducer(op, vsize).has_value() &&
              reduce::CanUsePackedRamp(src_indice_compressed.back(),
                                       src_var_compressed.back()->var, vsize,
                                       analyzer)) {
            can_pack = true;
            PrimType vec_dtype = clear_buffer.dtype().WithLanes(vsize);
            clear_buffer_packed =
                TensorVar(clear_buffer->name + "_pack",
                          TensorType(red_layout->OutputShape(), vec_dtype,
                                     clear_buffer.scope()));
            need_pack_buffer = true;

            Array<Stmt> local_body;

            if (require_init ||
                (need_duplicate && (op.type->IsMax() || op.type->IsMin() ||
                                    op.type->IsAbsMax()))) {
              local_body.push_back(
                  TensorStore(clear_buffer_packed, red_indices,
                              reduce::MakeInitValue(op, vsize)));
            }

            auto ext_int = GetConstInt(src_var_compressed.back()->dom->extent);
            int64_t inner_extent = *ext_int;
            PrimExpr halved_extent = MakeConst(
                src_var_compressed.back()->var.ty(), inner_extent / vsize);

            LayoutAxis inner_var = src_var_compressed.back();

            PrimExpr ramp_base =
                Substitute(src_indice_compressed.back(),
                           {{inner_var->var, inner_var->var * PrimExpr(2)}});
            src_indice_compressed.Set(
                src_indice_compressed.size() - 1,
                Ramp(ramp_base, IntImm(PrimType::Int(32), 1), vsize));

            auto src_load = TensorLoad(src_buffer, src_indice_compressed);

            Stmt reduce_local = TensorStore(
                clear_buffer_packed, red_indices,
                reduce::MakeReduce(op, vsize,
                                   TensorLoad(clear_buffer_packed, red_indices),
                                   src_load));

            reduce_local =
                For(inner_var->var, MakeConst((inner_var->var).ty(), 0),
                    halved_extent, MakeConst((inner_var->var).ty(), 1),
                    ForKind::kUnrolled, SeqStmt(reduce_local));

            for (int i = static_cast<int>(src_layout->OutputDim()) - 2; i >= 0;
                 --i) {
              reduce_local =
                  For(src_var_compressed[i]->var,
                      MakeConst((src_var_compressed[i]->var).ty(), 0),
                      src_var_compressed[i]->dom->extent,
                      MakeConst((src_var_compressed[i]->var).ty(), 1),
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
                             MakeConst((src_var_compressed[i]->var).ty(), 0),
                             src_var_compressed[i]->dom->extent,
                             MakeConst((src_var_compressed[i]->var).ty(), 1),
                             ForKind::kUnrolled, SeqStmt(reduce_local));
        }
        stmts.push_back(reduce_local);
      }

      const int batch = op.batch;
      if (batch > 1) {
        int64_t N_total = 1;
        for (const auto &s : clear_buffer.shape()) {
          auto p = GetConstInt(s);
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

      auto make_dst_loop = [&](Stmt body,
                               const Array<LayoutAxis> &vars) -> Stmt {
        for (int i = static_cast<int>(vars.size()) - 1; i >= 0; --i) {
          body = For(vars[i]->var, MakeConst((vars[i]->var).ty(), 0),
                     vars[i]->dom->extent, MakeConst((vars[i]->var).ty(), 1),
                     ForKind::kParallel, SeqStmt(body));
        }
        body = PartitionLoop((body).as_or_throw<For>(), lower_args.thread_index,
                             analyzer, red_layout);
        body = PragmaUnrollLoop((body).as_or_throw<For>());
        return body;
      };

      auto make_fresh_dst_vars = [&](const std::string &suffix)
          -> std::tuple<Array<LayoutAxis>, Array<PrimExpr>, Array<PrimExpr>> {
        Array<LayoutAxis> vars;
        for (size_t i = 0; i < dst_dim; ++i) {
          PrimVar v(std::string{char('i' + i)} + suffix);
          vars.push_back(LayoutAxis(Range(0, dst_layout->InputShape()[i]), v));
        }
        auto d_idx = dst_layout->Forward(
            vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
        auto r_idx = red_layout->Forward(
            vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
        return {vars, d_idx, r_idx};
      };

      if (use_batch) {
        Stmt pre_body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
        pre_body = make_dst_loop(pre_body, dst_vars);

        Array<Stmt> phases;
        phases.push_back(pre_body);

        for (const auto &thread_step : reduce_plan.thread_steps) {
          int reducing_threads = thread_step.ReducingThreads();
          Impl::CheckAllReduceWidth(reducing_threads, thread_step.scale,
                                    "tl.reduce", lower_args.target);
          int block_threads =
              static_cast<int>(*GetConstInt(lower_args.thread_bounds->extent));
          auto thread_offset = lower_args.thread_bounds->begin;

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
              lower_args.thread_bounds->extent, eff_batch, block_threads,
              lower_args.target);

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
            N_total *= *GetConstInt(s);
          }
          int num_chunks = static_cast<int>(N_total / batch);

          int buf_ndim = static_cast<int>(clear_buffer.shape().size());
          std::vector<int64_t> buf_shape_vals;
          for (const auto &s : clear_buffer.shape()) {
            buf_shape_vals.push_back(*GetConstInt(s));
          }
          std::vector<int64_t> buf_strides(buf_ndim, 1);
          for (int d = buf_ndim - 2; d >= 0; d--) {
            buf_strides[d] = buf_strides[d + 1] * buf_shape_vals[d + 1];
          }

          if (can_batch_pack) {
            int packed_batch = batch / vsize;

            TensorVar pack_buf =
                TensorVar(clear_buffer->name + "_pack",
                          TensorType({PrimExpr(packed_batch)},
                                     clear_buffer.dtype().WithLanes(vsize),
                                     clear_buffer.scope()));

            need_batch_pack_buffer = true;
            clear_batch_pack_buffer = pack_buf;

            for (int chunk = 0; chunk < num_chunks; chunk++) {
              int64_t flat_offset = static_cast<int64_t>(chunk) * batch;

              PrimVar pack_j("pack_j");
              PrimExpr base = MakeConst(PrimType::Int(64), flat_offset);
              PrimExpr scaled = pack_j * vsize;

              Array<PrimExpr> idx_a;
              Array<PrimExpr> idx_b;
              PrimExpr fa = base + scaled;
              PrimExpr fb = base + scaled + PrimExpr(1);
              for (int d = 0; d < buf_ndim; d++) {
                idx_a.push_back(
                    FloorMod(FloorDiv(fa, MakeConst(fa.ty(), buf_strides[d])),
                             MakeConst(fa.ty(), buf_shape_vals[d])));
                idx_b.push_back(
                    FloorMod(FloorDiv(fb, MakeConst(fb.ty(), buf_strides[d])),
                             MakeConst(fb.ty(), buf_shape_vals[d])));
              }
              auto a_load = TensorLoad(clear_buffer, idx_a);
              auto b_load = TensorLoad(clear_buffer, idx_b);
              Stmt pack_body = TensorStore(pack_buf, {pack_j},
                                           Shuffle({a_load, b_load}, {0, 1}));
              Stmt pack_loop = For(pack_j, MakeConst((pack_j).ty(), 0),
                                   packed_batch, MakeConst((pack_j).ty(), 1),
                                   ForKind::kUnrolled, SeqStmt(pack_body));
              phases.push_back(pack_loop);

              Expr packed_ptr = Call(std::nullopt, address_of_op(),
                                     {TensorLoad(pack_buf, {PrimExpr(0)})});
              Array<Expr> args = {StringImm(allreduce), packed_ptr};
              if (need_workspace) {
                args.push_back(workspace);
              }
              phases.push_back(Evaluate(
                  Call(TupleType(Array<Type>{}), call_extern_op(), args)));

              PrimVar unpack_j("unpack_j");
              PrimExpr ubase = MakeConst(PrimType::Int(64), flat_offset);
              PrimExpr uscaled = unpack_j * vsize;
              Array<PrimExpr> uidx_a;
              Array<PrimExpr> uidx_b;
              PrimExpr ufa = ubase + uscaled;
              PrimExpr ufb = ubase + uscaled + PrimExpr(1);
              for (int d = 0; d < buf_ndim; d++) {
                uidx_a.push_back(
                    FloorMod(FloorDiv(ufa, MakeConst(ufa.ty(), buf_strides[d])),
                             MakeConst(ufa.ty(), buf_shape_vals[d])));
                uidx_b.push_back(
                    FloorMod(FloorDiv(ufb, MakeConst(ufb.ty(), buf_strides[d])),
                             MakeConst(ufb.ty(), buf_shape_vals[d])));
              }
              auto packed_val = TensorLoad(pack_buf, {unpack_j});
              Stmt unpack_body = SeqStmt({
                  TensorStore(clear_buffer, uidx_a,
                              Shuffle::ExtractElement(packed_val, 0)),
                  TensorStore(clear_buffer, uidx_b,
                              Shuffle::ExtractElement(packed_val, 1)),
              });
              Stmt unpack_loop =
                  For(unpack_j, MakeConst((unpack_j).ty(), 0), packed_batch,
                      MakeConst((unpack_j).ty(), 1), ForKind::kUnrolled,
                      SeqStmt(unpack_body));
              phases.push_back(unpack_loop);
            }
          } else {
            for (int chunk = 0; chunk < num_chunks; chunk++) {
              int64_t flat_offset = static_cast<int64_t>(chunk) * batch;
              Array<PrimExpr> chunk_indices;
              for (int d = 0; d < buf_ndim; d++) {
                int64_t idx =
                    (flat_offset / buf_strides[d]) % buf_shape_vals[d];
                chunk_indices.push_back(MakeConst(PrimType::Int(64), idx));
              }
              Expr ptr = Call(std::nullopt, address_of_op(),
                              {TensorLoad(clear_buffer, chunk_indices)});

              Array<Expr> args = {StringImm(allreduce), ptr};
              if (need_workspace) {
                args.push_back(workspace);
              }
              phases.push_back(Evaluate(
                  Call(TupleType(Array<Type>{}), call_extern_op(), args)));
            }
          }
        }

        if (need_duplicate) {
          auto [post_vars, post_dst_idx, post_red_idx] =
              make_fresh_dst_vars("_p");

          PrimExpr predicate = MakeConst(PrimType::Bool(), true);
          {
            // The fragment inverse expects a zero-based logical thread
            // coordinate within the destination layout's thread range, so
            // normalize the absolute thread index against that range.
            PrimExpr local_thread_index = lower_args.thread_index;
            if (dst_layout->ThreadRange().has_value()) {
              local_thread_index =
                  local_thread_index - dst_layout->ThreadRange().value()->begin;
            }
            auto dst_th = post_dst_idx;
            dst_th.push_back(local_thread_index);
            auto inv = dst_layout->Inverse()->Forward(dst_th);
            inv.pop_back();
            for (int i = 0; i < static_cast<int>(dst_layout->InputDim()); i++) {
              predicate = predicate && (inv[i] == post_vars[i]->var);
            }
            predicate = analyzer->Simplify(predicate);
          }

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
          body = SeqStmt({Bind(clear_buffer,
                               Call((clear_buffer).ty(), alloc_tensor_op(), {},
                                    std::nullopt, {(clear_buffer).ty()})),
                          body});
        }
        if (need_pack_buffer) {
          body = SeqStmt(
              {Bind(clear_buffer_packed,
                    Call((clear_buffer_packed).ty(), alloc_tensor_op(), {},
                         std::nullopt, {(clear_buffer_packed).ty()})),
               body});
        }
        if (need_batch_pack_buffer) {
          body = SeqStmt(
              {Bind(clear_batch_pack_buffer,
                    Call((clear_batch_pack_buffer).ty(), alloc_tensor_op(), {},
                         std::nullopt, {(clear_batch_pack_buffer).ty()})),
               body});
        }
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
              static_cast<int>(*GetConstInt(lower_args.thread_bounds->extent));
          Expr workspace =
              lower_args.add_workspace(workspace_size, clear_buffer.dtype());
          thread_reduce_args.push_back(workspace);
        }
        auto call =
            Call(clear_buffer.dtype(), call_extern_op(), thread_reduce_args)
                .as_or_throw<PrimExpr>();
        stmts.push_back(TensorStore(clear_buffer, red_indices, call));
      }

      PrimExpr predicate = MakeConst(PrimType::Bool(), true);
      {
        // The fragment inverse expects a zero-based logical thread coordinate
        // within the destination layout's thread range, so normalize the
        // absolute thread index against that range.
        PrimExpr local_thread_index = lower_args.thread_index;
        if (dst_layout->ThreadRange().has_value()) {
          local_thread_index =
              local_thread_index - dst_layout->ThreadRange().value()->begin;
        }
        auto dst_th_indices = dst_indices;
        dst_th_indices.push_back(local_thread_index);
        auto inv = dst_layout->Inverse()->Forward(dst_th_indices);
        inv.pop_back();
        for (int i = 0; i < static_cast<int>(dst_layout->InputDim()); i++) {
          predicate = predicate && (inv[i] == dst_vars[i]->var);
        }
        predicate = analyzer->Simplify(predicate);
      }
      if (need_duplicate) {
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
        body =
            For(dst_vars[i]->var, MakeConst((dst_vars[i]->var).ty(), 0),
                dst_vars[i]->dom->extent, MakeConst((dst_vars[i]->var).ty(), 1),
                ForKind::kParallel, SeqStmt(body));
      }

      if (dst_layout->InputDim() > 0) {
        body = PartitionLoop((body).as_or_throw<For>(), lower_args.thread_index,
                             analyzer, red_layout);
        body = PragmaUnrollLoop((body).as_or_throw<For>());
      } else {
        auto guard =
            (lower_args.thread_index == lower_args.thread_bounds->begin);
        body = If(guard, SeqStmt(body));
      }

      if (need_duplicate) {
        body = SeqStmt(
            {Bind(clear_buffer, Call((clear_buffer).ty(), alloc_tensor_op(), {},
                                     std::nullopt, {(clear_buffer).ty()})),
             body});
      }
      if (need_pack_buffer) {
        body =
            SeqStmt({Bind(clear_buffer_packed,
                          Call((clear_buffer_packed).ty(), alloc_tensor_op(),
                               {}, std::nullopt, {(clear_buffer_packed).ty()})),
                     body});
      }
      return body;
    }

    LOG(FATAL) << "Reduce for buffers in scope (" << op.src.scope() << ", "
               << op.dst.scope() << ") is not implemented.";
    TVM_FFI_THROW(ValueError) << "Unsupported reduction scope";
    TVM_FFI_UNREACHABLE();
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_REDUCE_H_
