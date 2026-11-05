/*!
 * \file cuda/target_options.cc
 * \brief CUDA target options consumed by TileLang compilation callbacks.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/target/target_kind.h>

namespace tvm::tl {

TVM_FFI_STATIC_INIT_BLOCK() {
  // CUDA's existing canonicalizer and architecture/resource options stay in
  // TIRx. TileLang consumes this ordered code list when choosing NVCC gencode,
  // cubin/fatbin output, and the corresponding persistent binary-cache key.
  TargetKindDef("cuda").def_option<ffi::Array<ffi::String>>("code");
}

} // namespace tvm::tl
