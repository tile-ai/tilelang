/*!
 * \file tl/ascend/target_utils.h
 * \brief Ascend target attribute helpers.
 */

#ifndef TVM_TL_ASCEND_TARGET_UTILS_H_
#define TVM_TL_ASCEND_TARGET_UTILS_H_

#include <tvm/ir/type.h>
#include <tvm/target/target.h>

#include "support/base.h"

namespace tvm {
namespace tl {

TILELANG_DLL bool TargetIsAscend(Target target);
TILELANG_DLL bool IsAscendVectorizableFP8(PrimType dtype);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TARGET_UTILS_H_
