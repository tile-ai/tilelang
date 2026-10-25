#ifndef TILELANG_CUDA_CODEGEN_CODEGEN_CUDA_H_
#define TILELANG_CUDA_CODEGEN_CODEGEN_CUDA_H_

#include <tvm/codegen/codegen.h>

namespace tvm::codegen {
ffi::Module BuildTileLangCUDA(IRModule module, Target target, bool source_only);
}

#endif
