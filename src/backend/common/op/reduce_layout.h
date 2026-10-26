#include "compat/analyzer.h"
#ifndef TILELANG_BACKEND_COMMON_OP_REDUCE_LAYOUT_H_
#define TILELANG_BACKEND_COMMON_OP_REDUCE_LAYOUT_H_

#include "layout/layout.h"

#include <limits>
#include <vector>

namespace tvm::tl::backend::reduce {

inline ffi::Array<PrimExpr> InputPlaceholders(size_t count) {
  ffi::Array<PrimExpr> result;
  result.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    result.push_back(InputPlaceholder(index));
  }
  return result;
}

struct ThreadReduceStep {
  int extent;
  int scale;
  int64_t lower_factor;

  int ReducingThreads() const {
    ICHECK_GT(scale, 0);
    ICHECK_LE(extent, std::numeric_limits<int>::max() / scale)
        << "Reduce thread count overflow: extent=" << extent
        << ", scale=" << scale;
    return extent * scale;
  }
};

inline std::vector<ThreadReduceStep>
CollectThreadReduceSteps(const sym::IterSumExpr &thread_iter_sum,
                         const PrimVar &reduce_var) {
  std::vector<ThreadReduceStep> steps;
  for (const auto &iter_split : thread_iter_sum->args) {
    auto mark = iter_split->source->source.as<PrimVar>();
    if (!mark || !mark.value().same_as(reduce_var)) {
      continue;
    }
    auto scale = GetConstInt(iter_split->scale);
    auto extent = GetConstInt(iter_split->extent);
    auto lower_factor = GetConstInt(iter_split->lower_factor);
    ICHECK(scale && extent && lower_factor);
    if (*extent == 1) {
      continue;
    }
    ICHECK_GT(*scale, 0);
    ICHECK_GT(*extent, 0);
    ICHECK_LE(*scale, std::numeric_limits<int>::max());
    ICHECK_LE(*extent, std::numeric_limits<int>::max());
    steps.push_back(
        {static_cast<int>(*extent), static_cast<int>(*scale), *lower_factor});
  }
  return steps;
}

inline std::vector<ThreadReduceStep>
CollectThreadReduceSteps(const PrimExpr &thread,
                         const ffi::Map<PrimVar, Range> &domains,
                         const PrimVar &reduce_var, sym::Analyzer analyzer) {
  ffi::Map<PrimVar, PrimExpr> other_origins;
  for (const auto &[variable, domain] : domains) {
    tl::arith::Bind(analyzer, variable, domain, true);
    if (!variable.same_as(reduce_var))
      other_origins.Set(variable, domain->begin);
  }
  PrimExpr baseline =
      Substitute(thread, {{reduce_var, domains[reduce_var]->begin}});
  PrimExpr contribution = tl::arith::Simplify(
      analyzer, Substitute(thread - baseline, other_origins));
  TVM_FFI_CHECK(tl::arith::CanProve(analyzer, thread == baseline + contribution,
                                    sym::ProofStrength::kSymbolicBound),
                ValueError)
      << "Reduction thread ownership is not separable for " << reduce_var
      << " in " << thread;
  return CollectThreadReduceSteps(
      sym::NormalizeToIterSum(contribution, {{reduce_var, domains[reduce_var]}},
                              analyzer),
      reduce_var);
}

} // namespace tvm::tl::backend::reduce

#endif
