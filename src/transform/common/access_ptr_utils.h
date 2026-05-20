/*!
 * \file access_ptr_utils.h
 * \brief Shared utilities for tl.access_ptr lowering helpers.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_ACCESS_PTR_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_ACCESS_PTR_UTILS_H_

#include <tvm/tirx/expr.h>

namespace tvm {
namespace tl {

namespace detail {

template <typename VisitExprFn>
tirx::BufferLoad VisitAccessPtrBase(const tvm::PrimExpr &expr,
                                    VisitExprFn &&visit_expr) {
  const auto *base_load_node = expr.as<tirx::BufferLoadNode>();
  ICHECK(base_load_node) << "tl.access_ptr base must be BufferLoad, but got "
                         << expr;
  tirx::BufferLoad base_load =
      tvm::ffi::GetRef<tirx::BufferLoad>(base_load_node);

  tvm::ffi::Array<tvm::PrimExpr> indices;
  bool changed = false;
  for (const tvm::PrimExpr &index : base_load->indices) {
    tvm::PrimExpr new_index = visit_expr(index);
    changed = changed || !new_index.same_as(index);
    indices.push_back(new_index);
  }

  tvm::ffi::Optional<tvm::PrimExpr> predicate = base_load->predicate;
  if (predicate.defined()) {
    tvm::PrimExpr new_predicate = visit_expr(predicate.value());
    changed = changed || !new_predicate.same_as(predicate.value());
    predicate = new_predicate;
  }

  if (!changed) {
    return base_load;
  }
  return tirx::BufferLoad(base_load->buffer, indices, predicate,
                          base_load->span);
}

} // namespace detail

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_ACCESS_PTR_UTILS_H_
