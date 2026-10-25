#include "codegen_c.h"

#include <tvm/ffi/reflection/registry.h>

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("target.build.tilelang_c",
                                        tvm::codegen::BuildTileLangC);
}
