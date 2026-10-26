#include "transform/tirx_passes.h"
#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl::transform {

tvm::transform::Pass LowerIntrin() {
  return tvm::transform::Sequential({tirx_passes::LowerIntrin()},
                                    "tl.LowerIntrin");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerIntrin", LowerIntrin);
}

} // namespace tvm::tl::transform
