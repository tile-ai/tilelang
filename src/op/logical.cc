/*!
 * \file tl/op/logical.cc
 * \brief Logical operations.
 *
 */

#include "support/check.h"
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tirx;

PrimExpr any_of_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  const ffi::Array<PrimExpr> &arg = call->args;
  ICHECK_EQ(arg.size(), 2);
  PrimExpr buffer_address = arg[0];
  PrimExpr elems = arg[1];
  return tirx::Call(DataType::Bool(), tirx::builtin::call_extern(),
                    {StringImm("tl::Any"), buffer_address, elems});
}

PrimExpr all_of_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  const ffi::Array<PrimExpr> &arg = call->args;
  ICHECK_EQ(arg.size(), 2);
  PrimExpr buffer_address = arg[0];
  PrimExpr elems = arg[1];
  return tirx::Call(DataType::Bool(), tirx::builtin::call_extern(),
                    {StringImm("tl::All"), buffer_address, elems});
}

TVM_REGISTER_OP("tl.any_of")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "any_of")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", any_of_op)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", any_of_op);

TVM_REGISTER_OP("tl.all_of")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "all_of")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", all_of_op)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", all_of_op);

} // namespace tl
} // namespace tvm
