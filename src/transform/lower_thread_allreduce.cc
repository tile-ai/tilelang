#include <tvm/tirx/transform.h>

namespace tvm::tl {
tvm::transform::Pass LowerThreadAllreduce() {
  return tirx::transform::LowerThreadAllreduce();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerThreadAllreduce",
                                   LowerThreadAllreduce);
}
} // namespace tvm::tl
