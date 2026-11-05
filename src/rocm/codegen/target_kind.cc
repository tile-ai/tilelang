/*!
 * \file rocm/codegen/target_kind.cc
 * \brief TileLang's HIP source target, distinct from LLVM AMDGPU codegen.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/target/target_kind.h>

namespace tvm::backend::rocm {
// TileLang sources are linked into the same compiler as the fixed TIRx base.
// Reuse its ROCm device/triple canonicalization without exporting another ABI.
ffi::Map<ffi::String, ffi::Any>
UpdateROCmAttrs(ffi::Map<ffi::String, ffi::Any> target);
} // namespace tvm::backend::rocm

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace tvm;
  TargetKindDef("hip")
      .set_default_device_type(kDLROCM)
      .def_option<ffi::String>("mcpu")
      .def_option<ffi::String>("mtriple")
      .def_option<ffi::Array<ffi::String>>("mattr")
      // Old HIP options had no effective resource defaults. Architecture
      // normalization and explicit user options supply these values.
      .def_option<int64_t>("max_num_threads")
      .def_option<int64_t>("max_threads_per_block")
      .def_option<int64_t>("max_shared_memory_per_block")
      .def_option<int64_t>("thread_warp_size")
      .set_default_keys({"hip", "gpu"})
      .set_target_canonicalizer(backend::rocm::UpdateROCmAttrs);
}
