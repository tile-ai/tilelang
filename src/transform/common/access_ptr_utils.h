/*!
 * \file access_ptr_utils.h
 * \brief Shared utilities for tl.access_ptr lowering helpers.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_ACCESS_PTR_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_ACCESS_PTR_UTILS_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace tl {

namespace detail {

template <typename VisitExprFn>
BufferLoad VisitAccessPtrBase(const PrimExpr &expr, VisitExprFn &&visit_expr) {
  const auto *base_load_node = expr.as<BufferLoadNode>();
  ICHECK(base_load_node) << "tl.access_ptr base must be BufferLoad, but got "
                         << expr;
  BufferLoad base_load = tvm::ffi::GetRef<BufferLoad>(base_load_node);

  Array<PrimExpr> indices;
  bool changed = false;
  for (const PrimExpr &index : base_load->indices) {
    PrimExpr new_index = visit_expr(index);
    changed = changed || !new_index.same_as(index);
    indices.push_back(new_index);
  }

  Optional<PrimExpr> predicate = base_load->predicate;
  if (predicate.defined()) {
    PrimExpr new_predicate = visit_expr(predicate.value());
    changed = changed || !new_predicate.same_as(predicate.value());
    predicate = new_predicate;
  }

  if (!changed) {
    return base_load;
  }
  return BufferLoad(base_load->buffer, indices, predicate, base_load->span);
}

} // namespace detail

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_ACCESS_PTR_UTILS_H_
