/*!
 * \file tl/ascend/target_utils.cc
 * \brief Ascend target attribute helpers.
 */

#include "ascend/target_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include <string>

#include "op/builtin.h"

namespace {

bool HasCompileFlagToken(const std::string &entry, const char *flag) {
  const std::string needle(flag);
  std::size_t pos = 0;
  while (pos < entry.size()) {
    pos = entry.find_first_not_of(" \t\r\n", pos);
    if (pos == std::string::npos)
      return false;
    std::size_t end = entry.find_first_of(" \t\r\n", pos);
    std::size_t len = (end == std::string::npos ? entry.size() : end) - pos;
    if (entry.compare(pos, len, needle) == 0)
      return true;
    if (end == std::string::npos)
      return false;
    pos = end;
  }
  return false;
}

} // namespace

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

int64_t AscendReleasedUBReserveBytes() {
  constexpr const char *kDisableVFStackReservedUB =
      "--cce-disable-vf-stack-reserved-ubuf";
  constexpr const char *kDisableAscReservedUB =
      "--cce-disable-asc-reserved-ubuf";
  auto flags = tvm::transform::PassContext::Current()
                   ->GetConfig<ffi::Array<ffi::String>>(kDeviceCompileFlags);
  if (!flags.has_value())
    return 0;
  bool release_vf_stack = false;
  bool release_api = false;
  for (const ffi::String &entry : flags.value()) {
    std::string text = static_cast<std::string>(entry);
    release_vf_stack = release_vf_stack ||
                       HasCompileFlagToken(text, kDisableVFStackReservedUB);
    release_api =
        release_api || HasCompileFlagToken(text, kDisableAscReservedUB);
  }
  int64_t released = 0;
  if (release_vf_stack)
    released += kAscendVFStackReserveBytes;
  if (release_api)
    released += kAscendApiReserveBytes;
  return released;
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
