/*!
 * \file bind_utils.h
 * \brief Shared helpers for scalar Bind analysis.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_BIND_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_BIND_UTILS_H_

#include <tvm/ir/type.h>
#include <tvm/tirx/stmt.h>

#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tirx;

using BufferSet =
    std::unordered_set<Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

inline bool IsReplayableScalarBind(const Stmt &stmt,
                                   const ffi::Array<BufferRegion> &reads,
                                   const BufferSet &write_buffers) {
  const auto *bind = stmt.as<BindNode>();
  if (bind == nullptr) {
    return false;
  }
  if (!bind->var.dtype().is_scalar() || bind->var.dtype().is_handle() ||
      bind->var->type_annotation.as<PointerTypeNode>()) {
    return false;
  }
  if (!bind->value.dtype().is_scalar() || bind->value.dtype().is_handle()) {
    return false;
  }
  for (const BufferRegion &read : reads) {
    if (write_buffers.count(read->buffer)) {
      return false;
    }
  }
  return true;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_BIND_UTILS_H_
