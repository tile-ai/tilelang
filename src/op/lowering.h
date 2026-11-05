/*!
 * \file tl/op/lowering.h
 * \brief Close and inline a native tile-operator helper without losing
 * captures.
 */
#ifndef TVM_TL_OP_LOWERING_H_
#define TVM_TL_OP_LOWERING_H_

#include <tvm/ir/stmt.h>

namespace tvm::tl {

// The callback returns (tirx.Function, Array<Expr> capture_args). Function
// parameters close the helper; each argument is evaluated in its caller.
// Inlining renews local definitions and preserves their lexical allocation
// scope.
TVM_TIRX_DLL Stmt InlineTileOpFunction(const ffi::Any &lowered,
                                       const ffi::String &name);

} // namespace tvm::tl
#endif // TVM_TL_OP_LOWERING_H_
