#ifndef TILELANG_BACKEND_COMMON_CODEGEN_NATIVE_CODEGEN_H_
#define TILELANG_BACKEND_COMMON_CODEGEN_NATIVE_CODEGEN_H_

#include <tvm/codegen/codegen.h>

namespace tvm::tl {
IRModule PrepareNativeCodegen(IRModule module, const Target &target);
ffi::Module BuildNativeCodegen(IRModule module, Target target,
                               const ffi::String &postproc = "",
                               bool source_only = false);
} // namespace tvm::tl

#endif
