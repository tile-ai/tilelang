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

namespace {

PrimExpr LowerLogicalReduction(PrimExpr args, ffi::String thread_helper,
                               ffi::String warp_helper) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  const ffi::Array<PrimExpr> &arg = call->args;
  ICHECK(arg.size() == 2 || arg.size() == 3);
  PrimExpr buffer_address = arg[0];
  PrimExpr elems = arg[1];

  ffi::String helper = thread_helper;
  if (arg.size() == 3) {
    const auto *scope = arg[2].as<StringImmNode>();
    ICHECK(scope != nullptr);
    if (scope->value == "warp") {
      helper = warp_helper;
    } else {
      ICHECK(scope->value == "thread" || scope->value == "auto")
          << "Invalid internal logical reduction scope: " << scope->value;
    }
  }

  return tirx::Call(DataType::Bool(), tirx::builtin::call_extern(),
                    {StringImm(helper), buffer_address, elems});
}

} // namespace

PrimExpr any_of_op(PrimExpr args) {
  return LowerLogicalReduction(args, "tl::Any", "tl::AnyWarp");
}

PrimExpr all_of_op(PrimExpr args) {
  return LowerLogicalReduction(args, "tl::All", "tl::AllWarp");
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
