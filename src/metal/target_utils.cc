/*!
 * \file tl/metal/target_utils.cc
 * \brief Metal target attribute helpers.
 */

#include "metal/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

#include "dlpack/dlpack.h"

namespace tvm {
namespace tl {

bool TargetIsMetal(Target target) {
  return target->GetTargetDeviceType() == kDLMetal;
}

int TargetMetalGetWarpSize(Target target) {
  (void)target;
  return 32;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsMetal",
           [](Target target) { return TargetIsMetal(target); })
      .def("tl.TargetMetalGetWarpSize",
           [](Target target) { return TargetMetalGetWarpSize(target); });
}

} // namespace tl
} // namespace tvm
