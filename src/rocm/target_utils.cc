/*!
 * \file tl/rocm/target_utils.cc
 * \brief ROCm target attribute helpers.
 */

#include "rocm/target_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>

#include <string>

#include "dlpack/dlpack.h"

namespace tvm {
namespace tl {

bool TargetIsRocm(Target target) {
  return target->GetTargetDeviceType() == kDLROCM;
}

bool TargetIsCDNA(Target target) {
  if (!TargetIsRocm(target))
    return false;
  if (target->attrs.count("mcpu")) {
    std::string mcpu = Downcast<ffi::String>(target->attrs.at("mcpu"));
    // if mcpu start with "gfx9", it is CDNA
    return mcpu.find("gfx9") == 0;
  }
  return false;
}

bool TargetIsRDNA(Target target) {
  if (!TargetIsRocm(target))
    return false;
  if (target->attrs.count("mcpu")) {
    std::string mcpu = Downcast<ffi::String>(target->attrs.at("mcpu"));
    // gfx11xx, gfx12xx are RDNA architectures
    return mcpu.find("gfx11") == 0 || mcpu.find("gfx12") == 0;
  }
  return false;
}

bool TargetIsGfx950(Target target) {
  if (!TargetIsRocm(target))
    return false;
  if (target->attrs.count("mcpu")) {
    std::string mcpu = Downcast<ffi::String>(target->attrs.at("mcpu"));
    return mcpu.find("gfx950") != std::string::npos;
  }
  return false;
}

bool TargetRocmHasAsyncCopy(Target target) {
  if (!TargetIsCDNA(target))
    return false;
  if (target->attrs.count("mcpu")) {
    std::string mcpu = Downcast<ffi::String>(target->attrs.at("mcpu"));
    if (mcpu.rfind("gfx9", 0) == 0) {
      int gfx_version = std::stoi(mcpu.substr(3, 2));
      return gfx_version >= 94;
    }
  }
  return false;
}

int TargetRocmGetWarpSize(Target target) {
  if (TargetIsCDNA(target)) {
    return 64;
  }
  return 32;
}

int TargetGetRDNAGeneration(Target target) {
  if (!TargetIsRDNA(target))
    return 0;
  if (target->attrs.count("mcpu")) {
    std::string mcpu = Downcast<ffi::String>(target->attrs.at("mcpu"));
    if (mcpu.rfind("gfx11", 0) == 0)
      return 11;
    if (mcpu.rfind("gfx12", 0) == 0)
      return 12;
  }
  return 0;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsRocm",
           [](Target target) { return TargetIsRocm(target); })
      .def("tl.TargetIsCDNA",
           [](Target target) { return TargetIsCDNA(target); })
      .def("tl.TargetIsRDNA",
           [](Target target) { return TargetIsRDNA(target); })
      .def("tl.TargetIsGfx950",
           [](Target target) { return TargetIsGfx950(target); })
      .def("tl.TargetRocmGetWarpSize",
           [](Target target) { return TargetRocmGetWarpSize(target); })
      .def("tl.TargetGetRDNAGeneration",
           [](Target target) { return TargetGetRDNAGeneration(target); });
}

} // namespace tl
} // namespace tvm
