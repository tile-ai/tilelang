/*!
 * \file tl/rocm/target_utils.h
 * \brief ROCm target attribute helpers.
 */

#ifndef TVM_TL_ROCM_TARGET_UTILS_H_
#define TVM_TL_ROCM_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsRocm(Target target);
bool TargetIsCDNA(Target target);
bool TargetIsRDNA(Target target);
bool TargetIsGfx950(Target target);

bool TargetRocmHasAsyncCopy(Target target);
int TargetRocmGetWarpSize(Target target);
int TargetGetRDNAGeneration(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ROCM_TARGET_UTILS_H_
