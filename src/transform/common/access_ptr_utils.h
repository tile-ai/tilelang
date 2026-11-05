/*!
 * \file access_ptr_utils.h
 * \brief Traverse address indices without introducing a memory read.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_ACCESS_PTR_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_ACCESS_PTR_UTILS_H_

#include <tvm/ir/expr.h>

namespace tvm::tl::detail {

template <typename VisitExprFn>
TensorLoad VisitAccessPtrBase(const Expr &expr, VisitExprFn &&visit_expr) {
  const auto *node = expr.as<TensorLoadObj>();
  TVM_FFI_CHECK(node, TypeError)
      << "tl.access_ptr base must be a native TensorLoad, but got " << expr;
  TensorLoad load = ffi::GetRef<TensorLoad>(node);
  ffi::Array<PrimExpr> indices;
  bool changed = false;
  for (const PrimExpr &index : load->indices) {
    PrimExpr replacement = visit_expr(index);
    changed |= !replacement.same_as(index);
    indices.push_back(replacement);
  }
  if (!changed)
    return load;
  // Predication belongs to the containing native If. Preserve the stored type
  // until an index-width-changing caller deliberately reinfers this load.
  return TensorLoad(load->source, std::move(indices), load.ty(), load->loc);
}

} // namespace tvm::tl::detail

#endif // TVM_TL_TRANSFORM_COMMON_ACCESS_PTR_UTILS_H_
