#include "codegen_c.h"

#include "backend/common/codegen/native_codegen.h"

namespace tvm::codegen {
ffi::Module BuildTileLangC(IRModule module, Target target) {
  return tl::BuildNativeCodegen(std::move(module), std::move(target));
}
} // namespace tvm::codegen
