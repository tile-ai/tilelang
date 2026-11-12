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

bool TargetHasAMX(Target target) {
  if (!TargetIsCPU(target))
    return false;
  for (const auto &key : target->keys) {
    if (key == "amx")
      return true;
  }
  return false;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TargetIsCPU",
                        [](Target target) { return TargetIsCPU(target); });
  refl::GlobalDef().def("tl.TargetHasAMX",
                        [](Target target) { return TargetHasAMX(target); });
}

} // namespace tl
} // namespace tvm
