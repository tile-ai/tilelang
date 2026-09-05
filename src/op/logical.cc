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
  ICHECK_EQ(arg.size(), 3);
  PrimExpr buffer_address = arg[0];
  PrimExpr elems = arg[1];
  const auto scope_imm = arg[2].as<StringImmNode>();
  ICHECK(scope_imm != nullptr);
  const auto scope = scope_imm->value;
  ffi::String fn_name;
  if (scope == "warp") {
    fn_name = "tl::AnyWarp";
  } else if (scope == "thread" || scope == "auto") {
    // By default auto uses tl::Any, if it can prove no warp divergence it'll
    // use tl::AnyWarp instead
    fn_name = "tl::Any";
  } else {
    ICHECK(false) << "Invalid scope: " << scope;
  }
  return tirx::Call(DataType::Bool(), tirx::builtin::call_extern(),
                    {StringImm(fn_name), buffer_address, elems});
}

PrimExpr all_of_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  const ffi::Array<PrimExpr> &arg = call->args;
  ICHECK_EQ(arg.size(), 3);
  PrimExpr buffer_address = arg[0];
  PrimExpr elems = arg[1];
  const auto scope_imm = arg[2].as<StringImmNode>();
  ICHECK(scope_imm != nullptr);
  const auto scope = scope_imm->value;
  ffi::String fn_name;
  if (scope == "warp") {
    fn_name = "tl::AllWarp";
  } else if (scope == "thread" || scope == "auto") {
    // By default auto uses tl::Any, if it can prove no warp divergence it'll
    // use tl::AnyWarp instead
    fn_name = "tl::All";
  } else {
    ICHECK(false) << "Invalid scope: " << scope;
  }
  return tirx::Call(DataType::Bool(), tirx::builtin::call_extern(),
                    {StringImm(fn_name), buffer_address, elems});
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
