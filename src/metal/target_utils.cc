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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TargetIsMetal",
                        [](Target target) { return TargetIsMetal(target); });
}

} // namespace tl
} // namespace tvm
