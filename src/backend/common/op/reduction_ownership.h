/*!
 * \file tl/backend/common/op/reduction_ownership.h
 * \brief Thread ownership algebra shared by reducer planning and GPU lowering.
 */
#ifndef TVM_TL_BACKEND_COMMON_OP_REDUCTION_OWNERSHIP_H_
#define TVM_TL_BACKEND_COMMON_OP_REDUCTION_OWNERSHIP_H_
#include "layout/expr_utils.h"
#include "layout/layout.h"
#include "support/check.h"
#include <limits>
#include <tvm/sym/iter_affine_map.h>
#include <vector>

namespace tvm::tl::backend::reduce {
using ffi::Array;
inline Array<PrimExpr> InputPlaceholders(size_t n) {
  Array<PrimExpr> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(InputPlaceholder(i));
  }
  return result;
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
};

inline std::vector<ThreadReduceStep>
CollectThreadReduceSteps(const sym::IterSumExpr &thread_iter_sum,
                         const Var &reduce_var) {
  std::vector<ThreadReduceStep> steps;
  for (const auto &iter_split : thread_iter_sum->args) {
    auto mark = iter_split->source->source.as<Var>();
    if (!mark || !mark.value().same_as(reduce_var)) {
      continue;
    }

    auto scale = TryConstInt(iter_split->scale);
    auto extent = TryConstInt(iter_split->extent);
    auto lower_factor = TryConstInt(iter_split->lower_factor);
    ICHECK(scale.has_value() && extent.has_value() && lower_factor.has_value());
    if (*extent == 1) {
      continue;
    }
    ICHECK_LE(*scale, std::numeric_limits<int>::max());
    ICHECK_LE(*extent, std::numeric_limits<int>::max());
    steps.push_back(ThreadReduceStep{static_cast<int>(*extent),
                                     static_cast<int>(*scale), *lower_factor});
  }
  return steps;
}

} // namespace tvm::tl::backend::reduce
#endif // TVM_TL_BACKEND_COMMON_OP_REDUCTION_OWNERSHIP_H_
