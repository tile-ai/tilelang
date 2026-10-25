#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl::transform {

tvm::transform::Pass LowerIntrin() {
  return tvm::transform::Sequential({tirx::transform::LowerIntrin()},
                                    "tl.LowerIntrin");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerIntrin", LowerIntrin);
}

} // namespace tvm::tl::transform
