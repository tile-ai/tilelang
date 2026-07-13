/*!
 * \file tl/backend/common/op/reduce.h
 * \brief Shared tl.reduce AllReduce lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_REDUCE_H_
#define TVM_TL_BACKEND_COMMON_OP_REDUCE_H_

#include "op/reduce.h"
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>

#include "layout/layout.h"
#include "layout/utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "tir/transforms/ir_utils.h"
#include "transform/loop_partition.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
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

inline Array<PrimExpr> InputPlaceholders(size_t n) {
  Array<PrimExpr> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(InputPlaceholder(i));
  }
  return result;
}

inline Fragment ComputeReducerLayout(const Fragment &src_layout, int dim) {
  PrimExpr src_rep_extent = src_layout->ReplicateExtent();
  PrimExpr indice_rep_extent = src_layout->InputShape()[dim];
  PrimExpr reducer_rep_extent = indice_rep_extent * src_rep_extent;

  auto fwd = InputPlaceholders(src_layout->InputDim() - 1);
  fwd.insert(fwd.begin() + dim,
             FloorMod(ReplicationPlaceholder(), indice_rep_extent));

  auto thd = src_layout->ForwardThread(
      fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));

  auto reducer_shape = src_layout->InputShape();
  reducer_shape.erase(reducer_shape.begin() + dim);
  if (reducer_shape.empty()) {
    reducer_shape.push_back(1);
  }

  return Fragment(reducer_shape, {}, thd, reducer_rep_extent, std::nullopt)
      ->CondenseReplicateVar()
      ->BindThreadRange(src_layout->ThreadRange());
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

inline int GetPreferedVectorizedSize(DataType dt,
                                     bool supports_fp32x2 = false) {
  if (dt.is_bfloat16() || dt.is_float16() ||
      (supports_fp32x2 && dt.is_float() && dt.bits() == 32))
    return 2;
  return 1;
}

inline PrimExpr MakeInitValue(const ReduceOpNode &op, int vsize = 1) {
  auto dst_dtype = op.dst->dtype;
  auto is_int = dst_dtype.is_int();
  bool is_uint = dst_dtype.is_uint();
  auto bits = dst_dtype.bits();

  PrimExpr scalar;
  if (op.type->IsSum() || op.type->IsAbsSum()) {
    scalar = make_zero(op.dst->dtype);
  } else if (op.type->IsMax()) {
    if (is_int) {
      scalar = make_const(op.dst->dtype, SignedMin(bits));
    } else if (is_uint) {
      scalar = make_const(op.dst->dtype, 0);
    } else {
      scalar = make_const(op.dst->dtype, -INFINITY);
    }
  } else if (op.type->IsMin()) {
    if (is_int) {
      scalar = make_const(op.dst->dtype, SignedMax(bits));
    } else if (is_uint) {
      scalar = make_const(op.dst->dtype, UnsignedMax(bits));
    } else {
      scalar = make_const(op.dst->dtype, INFINITY);
    }
  } else if (op.type->IsAbsMax()) {
    scalar = make_const(op.dst->dtype, 0);
  } else if (op.type->IsBitAnd()) {
    if (is_int) {
      scalar = make_const(op.dst->dtype, -1);
    } else if (is_uint) {
      scalar = make_const(op.dst->dtype, UnsignedMax(bits));
    } else {
      scalar = make_const(op.dst->dtype, -INFINITY);
    }
  } else if (op.type->IsBitOr() || op.type->IsBitXor()) {
    scalar = make_zero(op.dst->dtype);
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
    scalar = PrimExpr();
  }

  if (vsize <= 1)
    return scalar;
  return Broadcast(scalar, vsize);
}

inline PrimExpr MakeReduce(const ReduceOpNode &op, int vsize,
                           const PrimExpr &acc, const PrimExpr &b) {
  if (vsize != 1 && vsize != 2) {
    LOG(FATAL) << "Unsupported reduce vector size: " << vsize;
    return PrimExpr();
  }

  PrimExpr rhs = b;
  if (acc->dtype != rhs->dtype) {
    rhs = Cast(acc->dtype, rhs);
  }

  const bool use_nan_op = op.nan_propagate && (acc.dtype().is_float16() ||
                                               acc.dtype().is_bfloat16());

  if (vsize == 1) {
    if (op.type->IsSum()) {
      return acc + rhs;
    } else if (op.type->IsAbsSum()) {
      return acc + Max(rhs, -rhs);
    } else if (op.type->IsMax()) {
      return use_nan_op ? Call(acc.dtype(), tl::max_nan(), {acc, rhs})
                        : PrimExpr(Max(acc, rhs));
    } else if (op.type->IsMin()) {
      return use_nan_op ? Call(acc.dtype(), tl::min_nan(), {acc, rhs})
                        : PrimExpr(Min(acc, rhs));
    } else if (op.type->IsAbsMax()) {
      auto abs_rhs = Max(rhs, -rhs);
      return use_nan_op ? Call(acc.dtype(), tl::max_nan(), {acc, abs_rhs})
                        : PrimExpr(Max(acc, abs_rhs));
    } else if (op.type->IsBitAnd()) {
      return acc & rhs;
    } else if (op.type->IsBitOr()) {
      return acc | rhs;
    } else if (op.type->IsBitXor()) {
      return acc ^ rhs;
    }
    LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
    return PrimExpr();
  }

  if (op.type->IsSum()) {
    return Call(acc.dtype(), tl::add2(), {acc, rhs});
  } else if (op.type->IsAbsSum()) {
    return Call(acc.dtype(), tl::add2(),
                {acc, Call(acc.dtype(), tl::abs2(), {rhs})});
  } else if (op.type->IsMax()) {
    return Call(acc.dtype(), use_nan_op ? tl::max2_nan() : tl::max2(),
                {acc, rhs});
  } else if (op.type->IsMin()) {
    return Call(acc.dtype(), use_nan_op ? tl::min2_nan() : tl::min2(),
                {acc, rhs});
  } else if (op.type->IsAbsMax()) {
    return Call(acc.dtype(), use_nan_op ? tl::max2_nan() : tl::max2(),
                {acc, Call(acc.dtype(), tl::abs2(), {rhs})});
  }
  LOG(FATAL) << "Unsupported packed reduce type: " << op.type->type;
  return PrimExpr();
}

inline std::optional<std::string> MakeCodegenReducer(const ReduceOpNode &op,
                                                     int vsize = 1) {
  const bool use_nan_op = op.nan_propagate && (op.dst->dtype.is_float16() ||
                                               op.dst->dtype.is_bfloat16());

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
    if (op.dst->dtype.is_float() && op.dst->dtype.bits() == 32)
      return base + "_f32x2";
    if (op.dst->dtype.is_bfloat16())
      return base + "_bf16x2";
    if (op.dst->dtype.is_float16())
      return base + "_fp16x2";
  }
  return std::nullopt;
}

inline bool CanUsePackedRamp(const PrimExpr &index, const Var &var, int vsize,
                             arith::Analyzer *analyzer) {
  ICHECK_GT(vsize, 1);

  PrimExpr vector_size = make_const(var.dtype(), vsize);
  PrimExpr packed_var = var * vector_size;
  PrimExpr ramp_base =
      analyzer->Simplify(Substitute(index, {{var, packed_var}}));

  PrimExpr index_mod = FloorMod(ramp_base, make_const(index.dtype(), vsize));
  if (!analyzer->CanProveEqual(index_mod, make_zero(index.dtype()))) {
    return false;
  }

  for (int lane = 1; lane < vsize; ++lane) {
    PrimExpr lane_value = make_const(var.dtype(), lane);
    PrimExpr lane_index =
        analyzer->Simplify(Substitute(index, {{var, packed_var + lane_value}}));
    PrimExpr expected =
        analyzer->Simplify(ramp_base + make_const(index.dtype(), lane));
    if (!analyzer->CanProveEqual(lane_index, expected)) {
      return false;
    }
  }

  return true;
}

struct ThreadReduceStep {
  int extent;
  int scale;

  int ReducingThreads() const {
    ICHECK_LE(extent, std::numeric_limits<int>::max() / scale)
        << "Reduce thread count overflow: extent=" << extent
        << ", scale=" << scale;
    return extent * scale;
  }
};

// A reduce is lowered in two phases: each thread first reduces the values it
// owns locally, then the thread-level reducer combines the splits encoded in
// the source fragment's thread expression.  This plan is the shared ownership
// contract consumed by both phases.
struct ReduceOwnershipPlan {
  Array<PrimExpr> local_src_indices;
  Array<IterVar> local_reduce_vars;
  std::vector<ThreadReduceStep> thread_steps;
};

inline std::vector<ThreadReduceStep>
CollectThreadReduceSteps(const arith::IterSumExpr &thread_iter_sum,
                         const Var &reduce_var) {
  std::vector<ThreadReduceStep> steps;
  for (const auto &iter_split : thread_iter_sum->args) {
    auto mark = iter_split->source->source.as<Var>();
    if (!mark || !mark.value().same_as(reduce_var)) {
      continue;
    }

    auto scale = as_const_int(iter_split->scale);
    auto extent = as_const_int(iter_split->extent);
    ICHECK(scale != nullptr && extent != nullptr);
    if (*extent == 1) {
      continue;
    }
    ICHECK_LE(*scale, std::numeric_limits<int>::max());
    ICHECK_LE(*extent, std::numeric_limits<int>::max());
    steps.push_back(
        ThreadReduceStep{static_cast<int>(*extent), static_cast<int>(*scale)});
  }
  return steps;
}

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
                            arith::Analyzer *analyzer) {
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
    if (analyzer->CanProveEqual(FloorMod(local_extent, Integer(factor)), 0)) {
      divisible_factors.push_back(factor);
    }
  }
  return divisible_factors;
}

inline std::optional<std::pair<PrimExpr, IterVar>>
TryRemoveThreadOwnedFactor(const PrimExpr &expr, const IterVar &iter_var,
                           int64_t factor, arith::Analyzer *analyzer) {
  PrimExpr factor_expr = Integer(factor);
  Var old_var = iter_var->var;
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
  Var new_var(old_var->name_hint, old_var->type_annotation);
  PrimExpr new_expr = analyzer->Simplify(
      Substitute(simplified_expr, {{old_var, new_var * factor_expr}}));
  IterVar new_iter_var =
      IterVar(Range(0, new_extent), new_var, IterVarType::kDataPar);
  analyzer->Bind(new_var, Range(0, new_extent), /*allow_override=*/true);
  return std::make_pair(new_expr, new_iter_var);
}

inline void CheckThreadOwnedFactorProjectable(const PrimExpr &index_expr,
                                              const Var &reduce_var,
                                              int64_t thread_owned_factor,
                                              arith::Analyzer *analyzer) {
  if (thread_owned_factor <= 1) {
    return;
  }

  PrimExpr factor = make_const(reduce_var.dtype(), thread_owned_factor);
  PrimExpr masked_reduce_var = FloorDiv(reduce_var, factor) * factor;
  PrimExpr simplified_index = analyzer->Simplify(index_expr);
  PrimExpr projected_index = analyzer->Simplify(
      Substitute(simplified_index, {{reduce_var, masked_reduce_var}}));

  ICHECK(analyzer->CanProveEqual(projected_index, simplified_index))
      << "ReduceOp cannot lower a layout where a source index depends on a "
         "thread-owned reduce factor: src_index="
      << simplified_index << ", projected_src_index=" << projected_index
      << ", reduce_var=" << reduce_var
      << ", thread_owned_factor=" << thread_owned_factor;
}

inline std::pair<PrimExpr, IterVar>
BuildLocalReduceIterator(const PrimExpr &index_expr,
                         const Array<IterVar> &input_iters,
                         const Var &reduce_var, int64_t thread_owned_factor,
                         arith::Analyzer *analyzer) {
  auto [expr, iter_var] =
      CompressIterator(index_expr, input_iters, reduce_var, analyzer);
  arith::Analyzer proof_analyzer;
  for (const auto &iv : input_iters) {
    proof_analyzer.Bind(iv->var, iv->dom, /*allow_override=*/true);
  }
  CheckThreadOwnedFactorProjectable(index_expr, reduce_var, thread_owned_factor,
                                    &proof_analyzer);

  int64_t remaining_thread_owned_factor = thread_owned_factor;
  PrimExpr cur_expr = expr;
  IterVar cur_iter_var = iter_var;
  while (remaining_thread_owned_factor > 1) {
    PrimExpr cur_extent = proof_analyzer.Simplify(cur_iter_var->dom->extent);
    auto candidates = CandidateThreadOwnedFactors(remaining_thread_owned_factor,
                                                  cur_extent, &proof_analyzer);
    bool removed = false;
    for (int64_t factor : candidates) {
      auto updated = TryRemoveThreadOwnedFactor(cur_expr, cur_iter_var, factor,
                                                &proof_analyzer);
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
                        const Array<IterVar> &src_vars, const Var &reduce_var,
                        arith::Analyzer *analyzer) {
  // Use src_thread as the single source of truth for thread-owned reduce
  // splits.  These steps are later used verbatim to emit scalar or batched
  // AllReduce, so the local loop must not enumerate the same split again.
  auto thread_iter_sum =
      arith::NormalizeToIterSum(src_thread, ToVMap(src_vars), analyzer);
  auto thread_steps = CollectThreadReduceSteps(thread_iter_sum, reduce_var);
  int64_t thread_owned_factor = ThreadOwnedReduceFactor(thread_steps);

  // Build the per-thread source indexing plan.  A thread-owned factor is
  // removed from the compressed local iterator only after proving that
  // projecting out that factor leaves the physical source index unchanged.
  // This keeps ownership from src_thread, while using src_indices as a safety
  // check against dropping a factor that still selects different local values.
  Array<PrimExpr> local_src_indices;
  Array<IterVar> local_reduce_vars;
  for (const auto &src_index : src_indices) {
    auto [expr, var] = BuildLocalReduceIterator(src_index, src_vars, reduce_var,
                                                thread_owned_factor, analyzer);
    local_src_indices.push_back(expr);
    local_reduce_vars.push_back(var);
  }

  return ReduceOwnershipPlan{local_src_indices, local_reduce_vars,
                             thread_steps};
}

inline PrimExpr MakeUpdate(const ReduceOpNode &op, PrimExpr dst_val,
                           PrimExpr src_val) {
  if (op.type->IsSum() || op.type->IsAbsSum()) {
    return dst_val + src_val;
  } else if (op.type->IsBitAnd()) {
    return op.clear ? src_val : bitwise_and(dst_val, src_val);
  } else if (op.type->IsBitOr()) {
    return bitwise_or(dst_val, src_val);
  } else if (op.type->IsBitXor()) {
    return bitwise_xor(dst_val, src_val);
  } else if (op.type->IsMax() || op.type->IsAbsMax()) {
    return Max(dst_val, src_val);
  } else if (op.type->IsMin()) {
    return Min(dst_val, src_val);
  }
  LOG(FATAL) << "Unsupported reduce type: " << op.type->type;
  return PrimExpr();
}

} // namespace reduce

template <typename Impl> struct ReduceLowerer {
  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    if (op.nan_propagate &&
        (op.dst->dtype.is_float16() || op.dst->dtype.is_bfloat16()) &&
        !Impl::SupportsFp16Bf16NanReduce(lower_args.target)) {
      LOG(FATAL) << "ReduceOp: nan_propagate=True for fp16/bf16 "
                    "max/min/absmax is only supported on CUDA targets "
                    "(requires __hmax_nan/__hmin_nan intrinsics). Target was: "
                 << lower_args.target->str();
    }
    auto get_buffer = [&](const Buffer &buf) {
      if (lower_args.buffer_remap.count(buf)) {
        return lower_args.buffer_remap[buf];
      }
      return buf;
    };

    if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
      auto src_buffer = get_buffer(op.src);
      auto dst_buffer = get_buffer(op.dst);
      auto src_layout = lower_args.layout_map[op.src].as<Fragment>().value();
      auto dst_layout = lower_args.layout_map[op.dst].as<Fragment>().value();
      auto red_layout = reduce::ComputeReducerLayout(src_layout, op.dim);
      auto src_dim = src_layout->InputDim();
      auto dst_dim = dst_layout->InputDim();

      auto is_1d_reduce = src_dim == dst_dim && dst_dim == 1;

      if (is_1d_reduce) {
        ICHECK(is_one(dst_layout->OutputShape().back()))
            << "Reduce for scalar not implemented.";
      } else {
        ICHECK_EQ(src_dim, dst_dim + 1) << "Reduce dimension mismatch.";
      }

      Array<IterVar> dst_vars;
      for (size_t i = 0; i < dst_dim; ++i) {
        Var var = Var(std::string{char('i' + i)});
        dst_vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), var,
                                   IterVarType::kDataPar));
      }

      Array<IterVar> src_vars;
      if (!is_1d_reduce) {
        src_vars = dst_vars;
      }
      Range reduce_dom(0, src_layout->InputShape()[op.dim]);
      IterVar reduce_iv(reduce_dom, Var("rv"), IterVarType::kDataPar);
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
        clear_buffer = decl_buffer(red_layout->OutputShape(), dst_buffer->dtype,
                                   dst_buffer->name + "_clear",
                                   GetPtrStorageScope(dst_buffer->data));
      }

      Array<PrimExpr> src_indice_compressed = reduce_plan.local_src_indices;
      Array<IterVar> src_var_compressed = reduce_plan.local_reduce_vars;

      bool can_pack = false;
      bool need_pack_buffer = false;
      bool need_batch_pack_buffer = false;
      Buffer clear_buffer_packed;
      Buffer clear_batch_pack_buffer;
      {
        int vsize = Impl::GetPreferedVectorizedSize(clear_buffer->dtype,
                                                    lower_args.target);
        if (vsize > 1 && !src_var_compressed.empty()) {
          auto *ext = src_var_compressed.back()->dom->extent.as<IntImmNode>();
          if (ext && ext->value >= vsize && ext->value % vsize == 0 &&
              reduce::MakeCodegenReducer(op, vsize).has_value() &&
              reduce::CanUsePackedRamp(src_indice_compressed.back(),
                                       src_var_compressed.back()->var, vsize,
                                       analyzer)) {
            can_pack = true;
            DataType vec_dtype = clear_buffer->dtype.with_lanes(vsize);
            clear_buffer_packed =
                decl_buffer(red_layout->OutputShape(), vec_dtype,
                            clear_buffer->name + "_pack",
                            GetPtrStorageScope(clear_buffer->data));
            need_pack_buffer = true;

            Array<Stmt> local_body;

            if (require_init ||
                (need_duplicate && (op.type->IsMax() || op.type->IsMin() ||
                                    op.type->IsAbsMax()))) {
              local_body.push_back(BufferStore(clear_buffer_packed,
                                               reduce::MakeInitValue(op, vsize),
                                               red_indices));
            }

            const auto *ext_int =
                as_const_int(src_var_compressed.back()->dom->extent);
            int64_t inner_extent = *ext_int;
            PrimExpr halved_extent = Integer(inner_extent / vsize);

            IterVar inner_var = src_var_compressed.back();

            PrimExpr ramp_base =
                Substitute(src_indice_compressed.back(),
                           {{inner_var->var, inner_var->var * Integer(2)}});
            src_indice_compressed.Set(
                src_indice_compressed.size() - 1,
                Ramp(ramp_base, IntImm(DataType::Int(32), 1), vsize));

            auto src_load = BufferLoad(src_buffer, src_indice_compressed);
            auto *src_writer = src_load.CopyOnWrite();
            src_writer->dtype = src_buffer->dtype.with_lanes(vsize);

            Stmt reduce_local = BufferStore(
                clear_buffer_packed,
                reduce::MakeReduce(op, vsize,
                                   BufferLoad(clear_buffer_packed, red_indices),
                                   src_load),
                red_indices);

            reduce_local =
                For(inner_var->var, 0, halved_extent, ForKind::kUnrolled,
                    reduce_local, std::nullopt,
                    {{tirx::attr::pragma_unroll_explicit, Bool(false)}});

            for (int i = static_cast<int>(src_layout->OutputDim()) - 2; i >= 0;
                 --i) {
              reduce_local =
                  For(src_var_compressed[i]->var, 0,
                      src_var_compressed[i]->dom->extent, ForKind::kUnrolled,
                      reduce_local, std::nullopt,
                      {{tirx::attr::pragma_unroll_explicit, Bool(false)}});
            }
            local_body.push_back(reduce_local);

            auto acc_vec = BufferLoad(clear_buffer_packed, red_indices);
            auto lane0 = Shuffle::ExtractElement(acc_vec, 0);
            auto lane1 = Shuffle::ExtractElement(acc_vec, 1);
            auto scalar_result = reduce::MakeReduce(op, 1, lane0, lane1);
            local_body.push_back(
                BufferStore(clear_buffer, scalar_result, red_indices));

            stmts.push_back(SeqStmt(local_body));
          }
        }
      }

      if (!can_pack) {
        if (require_init ||
            (need_duplicate &&
             (op.type->IsMax() || op.type->IsMin() || op.type->IsAbsMax()))) {
          stmts.push_back(BufferStore(clear_buffer, reduce::MakeInitValue(op),
                                      red_indices));
        }

        Stmt reduce_local = BufferStore(
            clear_buffer,
            reduce::MakeReduce(op, 1, BufferLoad(clear_buffer, red_indices),
                               BufferLoad(src_buffer, src_indice_compressed)),
            red_indices);

        for (int i = static_cast<int>(src_layout->OutputDim()) - 1; i >= 0;
             --i) {
          reduce_local = For(
              src_var_compressed[i]->var, 0, src_var_compressed[i]->dom->extent,
              ForKind::kUnrolled, reduce_local, std::nullopt,
              {{tirx::attr::pragma_unroll_explicit, Bool(false)}});
        }
        stmts.push_back(reduce_local);
      }

      const int batch = op.batch;
      if (batch > 1) {
        int64_t N_total = 1;
        for (const auto &s : clear_buffer->shape) {
          const int64_t *p = as_const_int(s);
          ICHECK(p != nullptr) << "ReduceOp: batch > 1 requires compile-time "
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

      auto make_dst_loop = [&](Stmt body, const Array<IterVar> &vars) -> Stmt {
        for (int i = static_cast<int>(vars.size()) - 1; i >= 0; --i) {
          body = For(vars[i]->var, 0, vars[i]->dom->extent, ForKind::kParallel,
                     body);
        }
        body = PartitionLoop(Downcast<For>(body), lower_args.thread_var,
                             analyzer, red_layout);
        body = PragmaUnrollLoop(Downcast<For>(body));
        return body;
      };

      auto make_fresh_dst_vars = [&](const std::string &suffix)
          -> std::tuple<Array<IterVar>, Array<PrimExpr>, Array<PrimExpr>> {
        Array<IterVar> vars;
        for (size_t i = 0; i < dst_dim; ++i) {
          Var v(std::string{char('i' + i)} + suffix);
          vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), v,
                                 IterVarType::kDataPar));
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
          auto thread_offset = lower_args.thread_bounds->min;

          int vsize = Impl::GetPreferedVectorizedSize(clear_buffer->dtype,
                                                      lower_args.target);
          bool can_batch_pack =
              vsize > 1 && batch >= vsize && batch % vsize == 0 &&
              reduce::MakeCodegenReducer(op, vsize).has_value();
          int eff_batch = can_batch_pack ? (batch / vsize) : batch;
          std::string reducer =
              reduce::MakeCodegenReducer(op, can_batch_pack ? vsize : 1)
                  .value();
          std::string allreduce = Impl::MakeBatchAllReduce(
              reducer, reducing_threads, thread_step.scale, thread_offset,
              lower_args.thread_bounds->extent, eff_batch, reducing_threads,
              lower_args.target);

          DataType ws_dtype = can_batch_pack
                                  ? clear_buffer->dtype.with_lanes(vsize)
                                  : clear_buffer->dtype;
          PrimExpr workspace;
          bool need_workspace = reducing_threads > 32;
          if (need_workspace) {
            int ws_size = reducing_threads * eff_batch;
            workspace = lower_args.add_workspace(ws_size, ws_dtype);
          }

          int64_t N_total = 1;
          for (const auto &s : clear_buffer->shape) {
            N_total *= *as_const_int(s);
          }
          int num_chunks = static_cast<int>(N_total / batch);

          int buf_ndim = static_cast<int>(clear_buffer->shape.size());
          std::vector<int64_t> buf_shape_vals;
          for (const auto &s : clear_buffer->shape) {
            buf_shape_vals.push_back(*as_const_int(s));
          }
          std::vector<int64_t> buf_strides(buf_ndim, 1);
          for (int d = buf_ndim - 2; d >= 0; d--) {
            buf_strides[d] = buf_strides[d + 1] * buf_shape_vals[d + 1];
          }

          if (can_batch_pack) {
            int packed_batch = batch / vsize;

            Buffer pack_buf = decl_buffer(
                {Integer(packed_batch)}, clear_buffer->dtype.with_lanes(vsize),
                clear_buffer->name + "_pack",
                GetPtrStorageScope(clear_buffer->data));

            need_batch_pack_buffer = true;
            clear_batch_pack_buffer = pack_buf;

            for (int chunk = 0; chunk < num_chunks; chunk++) {
              int64_t flat_offset = static_cast<int64_t>(chunk) * batch;

              Var pack_j("pack_j");
              PrimExpr base = Integer(flat_offset);
              PrimExpr scaled = pack_j * vsize;

              Array<PrimExpr> idx_a;
              Array<PrimExpr> idx_b;
              PrimExpr fa = base + scaled;
              PrimExpr fb = base + scaled + Integer(1);
              for (int d = 0; d < buf_ndim; d++) {
                idx_a.push_back(FloorMod(FloorDiv(fa, Integer(buf_strides[d])),
                                         Integer(buf_shape_vals[d])));
                idx_b.push_back(FloorMod(FloorDiv(fb, Integer(buf_strides[d])),
                                         Integer(buf_shape_vals[d])));
              }
              auto a_load = BufferLoad(clear_buffer, idx_a);
              auto b_load = BufferLoad(clear_buffer, idx_b);
              Stmt pack_body = BufferStore(
                  pack_buf, Shuffle({a_load, b_load}, {0, 1}), {pack_j});
              Stmt pack_loop =
                  For(pack_j, 0, packed_batch, ForKind::kUnrolled, pack_body);
              phases.push_back(pack_loop);

              PrimExpr packed_ptr =
                  Call(DataType::Handle(), builtin::address_of(),
                       {BufferLoad(pack_buf, {Integer(0)})});
              Array<PrimExpr> args = {StringImm(allreduce), packed_ptr};
              if (need_workspace) {
                args.push_back(workspace);
              }
              phases.push_back(Evaluate(
                  Call(DataType::Handle(), builtin::call_extern(), args)));

              Var unpack_j("unpack_j");
              PrimExpr ubase = Integer(flat_offset);
              PrimExpr uscaled = unpack_j * vsize;
              Array<PrimExpr> uidx_a;
              Array<PrimExpr> uidx_b;
              PrimExpr ufa = ubase + uscaled;
              PrimExpr ufb = ubase + uscaled + Integer(1);
              for (int d = 0; d < buf_ndim; d++) {
                uidx_a.push_back(
                    FloorMod(FloorDiv(ufa, Integer(buf_strides[d])),
                             Integer(buf_shape_vals[d])));
                uidx_b.push_back(
                    FloorMod(FloorDiv(ufb, Integer(buf_strides[d])),
                             Integer(buf_shape_vals[d])));
              }
              auto packed_val = BufferLoad(pack_buf, {unpack_j});
              Stmt unpack_body = SeqStmt({
                  BufferStore(clear_buffer,
                              Shuffle::ExtractElement(packed_val, 0), uidx_a),
                  BufferStore(clear_buffer,
                              Shuffle::ExtractElement(packed_val, 1), uidx_b),
              });
              Stmt unpack_loop = For(unpack_j, 0, packed_batch,
                                     ForKind::kUnrolled, unpack_body);
              phases.push_back(unpack_loop);
            }
          } else {
            for (int chunk = 0; chunk < num_chunks; chunk++) {
              int64_t flat_offset = static_cast<int64_t>(chunk) * batch;
              Array<PrimExpr> chunk_indices;
              for (int d = 0; d < buf_ndim; d++) {
                int64_t idx =
                    (flat_offset / buf_strides[d]) % buf_shape_vals[d];
                chunk_indices.push_back(Integer(idx));
              }
              PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                                  {BufferLoad(clear_buffer, chunk_indices)});

              Array<PrimExpr> args = {StringImm(allreduce), ptr};
              if (need_workspace) {
                args.push_back(workspace);
              }
              phases.push_back(Evaluate(
                  Call(DataType::Handle(), builtin::call_extern(), args)));
            }
          }
        }

        if (need_duplicate) {
          auto [post_vars, post_dst_idx, post_red_idx] =
              make_fresh_dst_vars("_p");

          PrimExpr predicate = Bool(true);
          {
            auto dst_th = post_dst_idx;
            dst_th.push_back(lower_args.thread_var);
            auto inv = dst_layout->Inverse()->Forward(dst_th);
            inv.pop_back();
            for (int i = 0; i < static_cast<int>(dst_layout->InputDim()); i++) {
              predicate = predicate && (inv[i] == post_vars[i]->var);
            }
            predicate = analyzer->Simplify(predicate);
          }

          PrimExpr update =
              need_update
                  ? reduce::MakeUpdate(op, BufferLoad(dst_buffer, post_dst_idx),
                                       BufferLoad(clear_buffer, post_red_idx))
                  : BufferLoad(clear_buffer, post_red_idx);
          auto store = BufferStore(dst_buffer, update, post_dst_idx);
          Stmt post_body;
          if (analyzer->CanProve(predicate)) {
            post_body = store;
          } else {
            post_body = IfThenElse(predicate, store);
          }
          phases.push_back(make_dst_loop(post_body, post_vars));
        }

        Stmt body = phases.size() > 1 ? SeqStmt(phases) : phases[0];
        if (need_duplicate) {
          body = SeqStmt({AllocBuffer(clear_buffer), body});
        }
        if (need_pack_buffer) {
          body = SeqStmt({AllocBuffer(clear_buffer_packed), body});
        }
        if (need_batch_pack_buffer) {
          body = SeqStmt({AllocBuffer(clear_batch_pack_buffer), body});
        }
        return body;
      }

      for (const auto &thread_step : reduce_plan.thread_steps) {
        int reducing_threads = thread_step.ReducingThreads();
        auto thread_offset = lower_args.thread_bounds->min;
        std::string allreduce = Impl::MakeScalarAllReduce(
            reduce::MakeCodegenReducer(op).value(), reducing_threads,
            thread_step.scale, thread_offset, lower_args.thread_bounds->extent,
            lower_args.target);
        Array<PrimExpr> thread_reduce_args = {
            StringImm(allreduce), BufferLoad(clear_buffer, red_indices)};
        if (reducing_threads > 32) {
          int workspace_size =
              static_cast<int>(*as_const_int(lower_args.thread_bounds->extent));
          PrimExpr workspace =
              lower_args.add_workspace(workspace_size, clear_buffer->dtype);
          thread_reduce_args.push_back(workspace);
        }
        auto call = Call(clear_buffer->dtype, builtin::call_extern(),
                         thread_reduce_args);
        stmts.push_back(BufferStore(clear_buffer, call, red_indices));
      }

      PrimExpr predicate = Bool(true);
      {
        auto dst_th_indices = dst_indices;
        dst_th_indices.push_back(lower_args.thread_var);
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
                ? reduce::MakeUpdate(op, BufferLoad(dst_buffer, dst_indices),
                                     BufferLoad(clear_buffer, red_indices))
                : BufferLoad(clear_buffer, red_indices);
        auto store = BufferStore(dst_buffer, update, dst_indices);
        if (analyzer->CanProve(predicate)) {
          stmts.push_back(store);
        } else {
          stmts.push_back(IfThenElse(predicate, store));
        }
      }

      auto body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
      for (int i = static_cast<int>(dst_layout->InputDim()) - 1; i >= 0; --i) {
        body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent,
                   ForKind::kParallel, body);
      }

      if (dst_layout->InputDim() > 0) {
        body = PartitionLoop(Downcast<For>(body), lower_args.thread_var,
                             analyzer, red_layout);
        body = PragmaUnrollLoop(Downcast<For>(body));
      } else {
        auto guard = (lower_args.thread_var == lower_args.thread_bounds->min);
        body = IfThenElse(guard, body);
      }

      if (need_duplicate) {
        body = SeqStmt({AllocBuffer(clear_buffer), body});
      }
      if (need_pack_buffer) {
        body = SeqStmt({AllocBuffer(clear_buffer_packed), body});
      }
      return body;
    }

    LOG(FATAL) << "Reduce for buffers in scope (" << op.src.scope() << ", "
               << op.dst.scope() << ") is not implemented.";
    return Stmt();
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_REDUCE_H_
