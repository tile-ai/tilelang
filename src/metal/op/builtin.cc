/*!
 * \file tl/metal/op/builtin.cc
 * \brief Registration of Metal-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {

using namespace tirx;

TIR_DEFINE_TL_BUILTIN(cooperative_tensor_fill, 5, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(cooperative_tensor_load, 11, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(cooperative_tensor_store, 11, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(cooperative_tensor_multiply_accumulate, 13, kOpaque,
                      false);

} // namespace tl
} // namespace tvm

#undef TIR_DEFINE_TL_BUILTIN
