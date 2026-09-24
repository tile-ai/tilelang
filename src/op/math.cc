/*!
 * \file tl/op/math.cc
 * \brief Math operations.
 *
 */

#include "builtin.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tirx;

// Backends without a device helper expand clamp after vectorization using
// ordinary TIR operations.
PrimExpr LowerClamp(PrimExpr expr) {
  Call call = Downcast<Call>(expr);
  ICHECK_EQ(call->args.size(), 3);
  DataType dtype = call->dtype;
  Var x("clamp_x", dtype), lo("clamp_lo", dtype), hi("clamp_hi", dtype);
  for (const PrimExpr &arg : call->args) {
    ICHECK_EQ(arg.dtype(), dtype)
        << "tl.clamp operands must have matching types";
  }
  ffi::Array<PrimExpr> results;
  for (int lane = 0; lane < dtype.lanes(); ++lane) {
    auto extract = [&](const Var &var) -> PrimExpr {
      return dtype.is_scalar() ? PrimExpr(var)
                               : Shuffle::ExtractElement(var, lane);
    };
    PrimExpr lane_x = extract(x), lane_lo = extract(lo), lane_hi = extract(hi);
    PrimExpr result = min(max(lane_x, lane_lo), lane_hi);
    // Keep isnan opaque to the arithmetic simplifier. Low-precision formats
    // need an fp32 predicate; float64 retains its native precision.
    for (const PrimExpr &value : {lane_hi, lane_lo, lane_x}) {
      PrimExpr check = dtype.element_of() == DataType::Float(64)
                           ? value
                           : cast(DataType::Float(32), value);
      result = Select(isnan(check), value, result);
    }
    results.push_back(result);
  }
  PrimExpr result = dtype.is_scalar() ? results[0] : Shuffle::Concat(results);
  // Binding all three arguments preserves single evaluation of side effects.
  return Let(x, call->args[0],
             Let(lo, call->args[1], Let(hi, call->args[2], result)));
}

TVM_REGISTER_OP("tl.clamp")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic", LowerClamp)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", LowerClamp)
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", LowerClamp)
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", LowerClamp);

PrimExpr pow_of_int_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  const ffi::Array<PrimExpr> &arg = call->args;
  ICHECK_EQ(arg.size(), 2);
  PrimExpr base = arg[0];
  PrimExpr exp = arg[1];
  ffi::String pow_of_int_name =
      "tl::pow_of_int<" + std::to_string(exp.as<IntImmNode>()->value) + ">";
  return tirx::Call(base.dtype(), tirx::builtin::call_extern(),
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
  ICHECK(call != nullptr);
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
  } else if (dtype.is_tfloat32()) {
    return FloatImm(dtype, std::numeric_limits<float>::infinity(), call->span);
  } else if (dtype.is_float8_e5m2()) {
    // e5m2 is the only fp8 format with a representable inf; the rest keep
    // the fatal below.
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

PrimExpr round_ties_away_from_zero_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 1);
  const DataType &dtype = call->dtype;
  if (dtype.is_int() || dtype.is_uint() || dtype.is_bool()) {
    return call->args[0];
  }
  return tirx::Call(dtype, tirx::builtin::call_pure_extern(),
                    {StringImm("tl::RoundTiesAwayFromZero"), call->args[0]},
                    call->annotations);
}

TVM_REGISTER_OP("tl.round_ties_away_from_zero")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true)
    .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                  "round_ties_away_from_zero")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               round_ties_away_from_zero_op)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               round_ties_away_from_zero_op);

} // namespace tl
} // namespace tvm
