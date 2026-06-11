/*!
 * \file tl/metal/target_utils.h
 * \brief Metal target attribute helpers.
 */

#ifndef TVM_TL_METAL_TARGET_UTILS_H_
#define TVM_TL_METAL_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsMetal(Target target);
int TargetMetalGetWarpSize(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_METAL_TARGET_UTILS_H_
