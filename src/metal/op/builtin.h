/*!
 * \file tl/metal/op/builtin.h
 * \brief Metal-specific TileLang intrinsic Ops.
 */

#ifndef TVM_TL_METAL_OP_BUILTIN_H_
#define TVM_TL_METAL_OP_BUILTIN_H_

#include "op/builtin.h"

namespace tvm {
namespace tl {

// Metal cooperative tensor operations.

TVM_DLL const Op &cooperative_tensor_fill();
TVM_DLL const Op &cooperative_tensor_load();
TVM_DLL const Op &cooperative_tensor_store();
TVM_DLL const Op &cooperative_tensor_multiply_accumulate();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_METAL_OP_BUILTIN_H_
