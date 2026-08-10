/*!
 * \file tl/op/reduce_plan.h
 * \brief Target-independent reduction layout and thread-group planning helpers.
 */

#ifndef TVM_TL_OP_REDUCE_PLAN_H_
#define TVM_TL_OP_REDUCE_PLAN_H_

#include "layout/layout.h"
#include "support/check.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tirx/builtin.h>

#include <limits>
#include <optional>
#include <vector>

namespace tvm {
namespace tl {
namespace reduction {

using namespace tirx;

inline Array<PrimExpr> InputPlaceholders(size_t count) {
  Array<PrimExpr> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    result.push_back(InputPlaceholder(i));
  }
  return result;
}

/*!
 * \brief Project one logical reduction dimension out of a Fragment layout.
 *
 * The removed logical coordinate is folded into the result replica coordinate,
 * preserving one physical result copy for every independently executable
 * reduction group.
 */
inline Fragment ComputeReducerLayout(const Fragment &source_layout,
                                     int reduction_dim) {
  ICHECK_GE(reduction_dim, 0);
  ICHECK_LT(reduction_dim, static_cast<int>(source_layout->InputDim()));
  PrimExpr source_replica_extent = source_layout->ReplicateExtent();
  PrimExpr reduction_extent = source_layout->InputShape()[reduction_dim];
  PrimExpr result_replica_extent = reduction_extent * source_replica_extent;

  Array<PrimExpr> forward = InputPlaceholders(source_layout->InputDim() - 1);
  forward.insert(forward.begin() + reduction_dim,
                 FloorMod(ReplicationPlaceholder(), reduction_extent));

  PrimExpr thread = source_layout->ForwardThread(
      forward, FloorDiv(ReplicationPlaceholder(), reduction_extent));

  Array<PrimExpr> result_shape = source_layout->InputShape();
  result_shape.erase(result_shape.begin() + reduction_dim);
  if (result_shape.empty()) {
    result_shape.push_back(1);
  }

  return Fragment(result_shape, {}, thread, result_replica_extent, std::nullopt)
      ->CondenseReplicateVar()
      ->BindThreadRange(source_layout->ThreadRange());
}

struct ThreadReduceStep {
  int extent;
  int scale;
  // Position of this split inside the reduce var: the split covers
  // floormod(floordiv(rv, lower_factor), extent).
  int64_t lower_factor;

  int ReducingThreads() const {
    ICHECK_LE(extent, std::numeric_limits<int>::max() / scale)
        << "Reduce thread count overflow: extent=" << extent
        << ", scale=" << scale;
    return extent * scale;
  }

  bool operator==(const ThreadReduceStep &other) const {
    return extent == other.extent && scale == other.scale &&
           lower_factor == other.lower_factor;
  }
};

/*!
 * \brief Try to extract the physical thread splits owned by a reduction axis.
 *
 * This helper is used by speculative planners, so unsupported symbolic splits
 * return std::nullopt instead of turning a legal program into a compiler error.
 */
inline std::optional<std::vector<ThreadReduceStep>>
TryCollectThreadReduceSteps(const arith::IterSumExpr &thread_iter_sum,
                            const Var &reduce_var) {
  std::vector<ThreadReduceStep> steps;
  for (const arith::IterSplitExpr &iter_split : thread_iter_sum->args) {
    Optional<Var> mark = iter_split->source->source.as<Var>();
    if (!mark.defined() || !mark.value().same_as(reduce_var)) {
      continue;
    }

    const int64_t *scale = as_const_int(iter_split->scale);
    const int64_t *extent = as_const_int(iter_split->extent);
    const int64_t *lower_factor = as_const_int(iter_split->lower_factor);
    if (scale == nullptr || extent == nullptr || lower_factor == nullptr ||
        *scale <= 0 || *extent <= 0 ||
        *scale > std::numeric_limits<int>::max() ||
        *extent > std::numeric_limits<int>::max()) {
      return std::nullopt;
    }
    if (*extent == 1) {
      continue;
    }
    steps.push_back(ThreadReduceStep{static_cast<int>(*extent),
                                     static_cast<int>(*scale), *lower_factor});
  }
  return steps;
}

inline std::vector<ThreadReduceStep>
CollectThreadReduceSteps(const arith::IterSumExpr &thread_iter_sum,
                         const Var &reduce_var) {
  std::optional<std::vector<ThreadReduceStep>> steps =
      TryCollectThreadReduceSteps(thread_iter_sum, reduce_var);
  ICHECK(steps.has_value())
      << "Reduction thread mapping must use compile-time constant splits";
  return std::move(steps.value());
}

} // namespace reduction
} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_REDUCE_PLAN_H_
