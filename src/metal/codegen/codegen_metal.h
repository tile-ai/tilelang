#ifndef TILELANG_METAL_CODEGEN_CODEGEN_METAL_H_
#define TILELANG_METAL_CODEGEN_CODEGEN_METAL_H_

#include <tvm/codegen/codegen.h>

namespace tvm::codegen {
ffi::Module BuildTileLangMetal(IRModule module, Target target);
ffi::Module BuildTileLangMetalWithoutCompile(IRModule module, Target target);
} // namespace tvm::codegen

#endif
