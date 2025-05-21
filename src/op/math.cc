// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/math.cc
 * \brief Math operations.
 *
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tir;

PrimExpr pow_of_int_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  CHECK(call != nullptr);
  const Array<PrimExpr> &arg = call->args;
  ICHECK_EQ(arg.size(), 2);
  PrimExpr base = arg[0];
  PrimExpr exp = arg[1];
  String pow_of_int_name =
      "tl::pow_of_int<" + std::to_string(exp.as<IntImmNode>()->value) + ">";
  return tir::Call(base.dtype(), tir::builtin::call_extern(),
                   {StringImm(pow_of_int_name), base});
}

TVM_REGISTER_OP("tl.pow_of_int")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "pow_of_int")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", pow_of_int_op);

} // namespace tl
} // namespace tvm
