/*!
 * \file tl/op/logical.cc
 * \brief Logical operations.
 *
 */

#include <tvm/ffi/function.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../support/ffi_aliases.h"

namespace tvm {
namespace tl {
using namespace tir;

PrimExpr any_of_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  CHECK(call != nullptr);
  const Array<PrimExpr> &arg = call->args;
  ICHECK_EQ(arg.size(), 2);
  PrimExpr buffer_address = arg[0];
  PrimExpr elems = arg[1];
  return tir::Call(DataType::Bool(), tir::builtin::call_extern(),
                   {StringImm("tl::Any"), buffer_address, elems});
}

PrimExpr all_of_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  CHECK(call != nullptr);
  const Array<PrimExpr> &arg = call->args;
  ICHECK_EQ(arg.size(), 2);
  PrimExpr buffer_address = arg[0];
  PrimExpr elems = arg[1];
  return tir::Call(DataType::Bool(), tir::builtin::call_extern(),
                   {StringImm("tl::All"), buffer_address, elems});
}

TVM_REGISTER_OP("tl.any_of")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "any_of")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", any_of_op);

TVM_REGISTER_OP("tl.all_of")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "all_of")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", all_of_op);

} // namespace tl
} // namespace tvm
