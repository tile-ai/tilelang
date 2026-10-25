#include "codegen_c_host.h"

#include "native_codegen.h"

namespace tvm::tl {
ffi::Module BuildTileLangCHost(IRModule module, Target target) {
  return BuildNativeCodegen(std::move(module), std::move(target),
                            "tilelang_callback_c_host_postproc");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("target.build.tilelang_c_host",
                                   BuildTileLangCHost);
}
} // namespace tvm::tl
