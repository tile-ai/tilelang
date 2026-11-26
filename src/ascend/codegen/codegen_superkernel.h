#ifndef TVM_TL_ASCEND_CODEGEN_CODEGEN_SUPERKERNEL_H_
#define TVM_TL_ASCEND_CODEGEN_CODEGEN_SUPERKERNEL_H_

#include <tvm/ir/module.h>

namespace tvm {
namespace codegen {

class CodeGenTileLangAscend;

// Append optional SK entries after ordinary Ascend kernel code generation.
// The module's ordinary functions and launch metadata are not modified.
void EmitSuperKernel(const IRModule &mod, CodeGenTileLangAscend *codegen);

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_ASCEND_CODEGEN_CODEGEN_SUPERKERNEL_H_
