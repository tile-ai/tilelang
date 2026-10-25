/*!
 * \file tl/rocm/target_utils.h
 * \brief ROCm target attribute helpers.
 */

#ifndef TVM_TL_ROCM_TARGET_UTILS_H_
#define TVM_TL_ROCM_TARGET_UTILS_H_

#include <tvm/target/target.h>

#include "support/base.h"

namespace tvm {
namespace tl {

TILELANG_DLL bool TargetIsRocm(Target target);
TILELANG_DLL bool TargetIsCDNA(Target target);
TILELANG_DLL bool TargetIsRDNA(Target target);
TILELANG_DLL bool TargetIsGfx950(Target target);

TILELANG_DLL bool TargetRocmHasAsyncCopy(Target target);
TILELANG_DLL int TargetRocmGetWarpSize(Target target);
TILELANG_DLL int TargetGetRDNAGeneration(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ROCM_TARGET_UTILS_H_
