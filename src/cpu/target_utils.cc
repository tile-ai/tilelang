/*!
 * \file tl/cpu/target_utils.cc
 * \brief CPU target attribute helpers.
 */

#include "cpu/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

#include "dlpack/dlpack.h"

namespace tvm {
namespace tl {

bool TargetIsCPU(Target target) {
  return target->GetTargetDeviceType() == kDLCPU;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TargetIsCPU",
                        [](Target target) { return TargetIsCPU(target); });
}

} // namespace tl
} // namespace tvm
