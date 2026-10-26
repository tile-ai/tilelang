#include "compat/analyzer.h"
#ifndef TILELANG_COMPAT_THREAD_H_
#define TILELANG_COMPAT_THREAD_H_
#include "op/prim.h"

#include "compat/expr.h"
#include <tvm/sym/analyzer.h>

namespace tvm::tl {

inline Range ComputeThreadBounds(const ffi::Optional<PrimVar> &thread,
                                 const sym::Analyzer &analyzer) {
  if (!thread.has_value())
    return Range(PrimExpr(0), PrimExpr(1));
  TVM_FFI_CHECK(analyzer->const_int_bound.IsBound(thread.value()), ValueError)
      << "Thread index must have an enclosing launch extent";
  auto bounds = tl::arith::ConstIntBoundOf(analyzer, thread.value());
  TVM_FFI_CHECK(bounds->min_value >= 0 && bounds->max_value < 1024, ValueError)
      << "Thread bounds must be finite and within a 1024-thread block";
  return Range(tl::MakeConst(thread.value().ty(), bounds->min_value),
               tl::MakeConst(thread.value().ty(), bounds->max_value + 1));
}

} // namespace tvm::tl

#endif
