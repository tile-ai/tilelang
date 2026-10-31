/*!
 * \file tl/cpu/target_utils.h
 * \brief CPU target attribute helpers.
 */

#ifndef TVM_TL_CPU_TARGET_UTILS_H_
#define TVM_TL_CPU_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsCPU(Target target);

/** Return true when the target explicitly opts in to Intel AMX codegen. */
bool TargetHasAMX(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_CPU_TARGET_UTILS_H_
