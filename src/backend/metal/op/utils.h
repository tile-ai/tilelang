/*!
 * \file tl/backend/metal/op/utils.h
 * \brief Metal-specific operator helpers.
 */

#ifndef TVM_TL_BACKEND_METAL_OP_UTILS_H_
#define TVM_TL_BACKEND_METAL_OP_UTILS_H_

#include "op/utils.h"

namespace tvm {
namespace tl {
namespace metal {

inline bool IsSIMDGroupBuffer(const tir::Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "metal.simdgroup";
}

inline bool IsRegisterBuffer(const tir::Buffer &buffer) {
  return IsFragmentBuffer(buffer) || IsSIMDGroupBuffer(buffer);
}

} // namespace metal
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_METAL_OP_UTILS_H_
