#ifndef TILELANG_BACKEND_COMMON_CODEGEN_CODEGEN_C_HOST_H_
#define TILELANG_BACKEND_COMMON_CODEGEN_CODEGEN_C_HOST_H_

#include <tvm/codegen/codegen.h>

namespace tvm::tl {
ffi::Module BuildTileLangCHost(IRModule module, Target target);
}

#endif
