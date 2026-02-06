/*!
 * \file tl/op/math.cc
 * \brief Math operations.
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
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", pow_of_int_op)
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", pow_of_int_op);

PrimExpr infinity_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  CHECK(call != nullptr);
  const DataType &dtype = call->dtype;
  ICHECK_EQ(dtype.lanes(), 1);

  // NOTE(wt): Codegen for PrintConst:Inf will handle this based on dtype
  if (dtype.is_float()) {
    if (dtype.bits() == 64 || dtype.bits() == 32 || dtype.bits() == 16) {
      return FloatImm(dtype, std::numeric_limits<float>::infinity(),
                      call->span);
    }
  } else if (dtype.is_bfloat16()) {
    return FloatImm(dtype, std::numeric_limits<float>::infinity(), call->span);
  }
  LOG(FATAL) << "Cannot decide infinity for type " << dtype;
  throw; // Unreachable, keeps compiler happy
}

TVM_REGISTER_OP("tl.infinity")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "infinity")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", infinity_op)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", infinity_op);

} // namespace tl
} // namespace tvm
