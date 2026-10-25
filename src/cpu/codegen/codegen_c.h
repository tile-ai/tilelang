#ifndef TILELANG_CPU_CODEGEN_CODEGEN_C_H_
#define TILELANG_CPU_CODEGEN_CODEGEN_C_H_

#include <tvm/codegen/codegen.h>

namespace tvm::codegen {
ffi::Module BuildTileLangC(IRModule module, Target target);
}

#endif
