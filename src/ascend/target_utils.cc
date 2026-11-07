/*!
 * \file tl/ascend/target_utils.cc
 * \brief Ascend target attribute helpers.
 */

#include "ascend/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

using ffi::String;

bool TargetIsAscend(Target target) {
  return target.defined() && target->kind->name == "ascend";
}

bool TargetIsPTO(Target target) {
  // PTO shares the "ascend" target kind with AscendC; "pto" in target keys
  // selects the PTO codegen (see tilelang/ascend/target.py). Keep this the
  // single C++ decision point so LayoutInference, Verify/Vectorize passes
  // and any later consumer cannot drift apart on the key spelling.
  if (!TargetIsAscend(target)) {
    return false;
  }
  for (const String &key : target->keys) {
    if (key == "pto") {
      return true;
    }
  }
  return false;
}

bool IsAscendVectorizableFP8(DataType dtype) {
  // NOTE: E8M0 is a special type of FP8 which is not handled here.
  // We only handle FP8 types which can be represented with
  // __asc_fp8_interpretation_t here.
  return dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
         dtype.is_float8_e5m2();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TargetIsAscend",
                        [](Target target) { return TargetIsAscend(target); });
}

} // namespace tl
} // namespace tvm
