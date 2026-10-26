/*!
 * \file tl/op/math.cc
 * \brief Math operations.
 *
 */

#include "builtin.h"
#include "builtin_registry.h"
#include "compat/expr.h"
#include "op/prim.h"
#include "support/check.h"
#include <tvm/tirx/op/abi.h>

#include <limits>

namespace tvm::tl {

PrimExpr LowerClamp(PrimExpr expression) {
  const CallNode *call = expression.as<CallNode>();
  ICHECK(call);
  ICHECK_EQ(call->args.size(), 3);
  PrimType type = expression.ty();
  ICHECK(!type.IsScalableVector()) << "tl.clamp requires a fixed-width vector";
  PrimVar value_var("clamp_value", type);
  PrimVar lower_var("clamp_lower", type);
  PrimVar upper_var("clamp_upper", type);
  for (const Expr &argument : call->args) {
    ICHECK(argument.as_or_throw<PrimExpr>().ty() == type)
        << "tl.clamp operands must have matching types";
  }
  ffi::Array<PrimExpr> results;
  for (int lane = 0; lane < type.lanes(); ++lane) {
    auto extract = [&](const PrimVar &variable) -> PrimExpr {
      return type.IsScalar() ? PrimExpr(variable)
                             : Shuffle::ExtractElement(variable, lane);
    };
    PrimExpr value = extract(value_var);
    PrimExpr lower = extract(lower_var);
    PrimExpr upper = extract(upper_var);
    PrimExpr result = min(max(value, lower), upper);
    if (type.bits() < 16 && !type.MatchesCode(kDLInt, kDLUInt, kDLBool)) {
      PrimType compute_type = PrimType::Float(32);
      result = tl::CastValue(min(max(tl::CastValue(value, compute_type),
                                     tl::CastValue(lower, compute_type)),
                                 tl::CastValue(upper, compute_type)),
                             value.ty());
    }
    for (const PrimExpr &operand : {upper, lower, value}) {
      PrimExpr check = type.MatchesElementType(kDLFloat, 64)
                           ? operand
                           : tl::CastValue(operand, PrimType::Float(32));
      result = Select(isnan(check), operand, result);
    }
    results.push_back(result);
  }
  PrimExpr result = type.IsScalar() ? results[0] : Shuffle::Concat(results);
  return Let(
      value_var, call->args[0].as_or_throw<PrimExpr>(),
      Let(lower_var, call->args[1].as_or_throw<PrimExpr>(),
          Let(upper_var, call->args[2].as_or_throw<PrimExpr>(), result)));
}

PrimExpr LowerPowOfInt(PrimExpr expression) {
  const CallNode *call = expression.as<CallNode>();
  ICHECK(call);
  ICHECK_EQ(call->args.size(), 2);
  PrimExpr base = call->args[0].as_or_throw<PrimExpr>();
  auto exponent = GetConstInt(call->args[1].as_or_throw<PrimExpr>());
  TVM_FFI_CHECK(exponent.has_value(), ValueError)
      << "tl.pow_of_int requires an int64 constant exponent";
  ffi::String symbol =
      "tl::pow_of_int<" + std::to_string(exponent.value()) + ">";
  return Call(base.ty(), tirx::call_pure_extern_op(), {StringImm(symbol), base},
              std::nullopt, {}, call->loc)
      .as_or_throw<PrimExpr>();
}

PrimExpr LowerInfinity(PrimExpr expression) {
  const CallNode *call = expression.as<CallNode>();
  ICHECK(call);
  PrimType type = expression.ty();
  ICHECK_EQ(type.lanes(), 1);
  if ((type.MatchesCode(kDLFloat) &&
       (type.bits() == 64 || type.bits() == 32 || type.bits() == 16)) ||
      type.MatchesCode(kDLBfloat, kDLFloat8_e5m2)) {
    return FloatImm(expression.ty(), std::numeric_limits<double>::infinity(),
                    call->loc);
  }
  TVM_FFI_THROW(TypeError) << "Cannot represent infinity in " << type;
}

PrimExpr LowerRoundTiesAwayFromZero(PrimExpr expression) {
  const CallNode *call = expression.as<CallNode>();
  ICHECK(call);
  ICHECK_EQ(call->args.size(), 1);
  PrimType type = expression.ty();
  if (type.MatchesCode(kDLInt, kDLUInt, kDLBool)) {
    return call->args[0].as_or_throw<PrimExpr>();
  }
  PrimType compute_type =
      type.bits() < 16 ? PrimType::Float(32, type.lanes()) : type;
  PrimExpr value =
      tl::CastValue(call->args[0].as_or_throw<PrimExpr>(), compute_type);
  PrimExpr result = Call(compute_type, tirx::call_pure_extern_op(),
                         {StringImm("tl::RoundTiesAwayFromZero"), value},
                         std::nullopt, {}, call->loc)
                        .as_or_throw<PrimExpr>();
  return tl::CastValue(result, type);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char *backend :
       {"ascend", "llvm", "hip", "rocm", "metal", "webgpu", "c", "cuda"}) {
    OpDef("tl.clamp")
        .set_attr<tirx::FLowerIntrinsic>(
            std::string(backend) + ".FLowerIntrinsic", LowerClamp);
  }
  RegisterBuiltin<2>("tl.pow_of_int", CallEffectKind::kPure);
  RegisterBuiltin<1>("tl.infinity", CallEffectKind::kPure);
  RegisterBuiltin<1>("tl.round_ties_away_from_zero", CallEffectKind::kPure,
                     true);
  for (const char *backend : {"cuda", "hip", "rocm"}) {
    OpDef("tl.pow_of_int")
        .set_attr<tirx::FLowerIntrinsic>(
            std::string(backend) + ".FLowerIntrinsic", LowerPowOfInt);
    OpDef("tl.round_ties_away_from_zero")
        .set_attr<tirx::FLowerIntrinsic>(std::string(backend) +
                                             ".FLowerIntrinsic",
                                         LowerRoundTiesAwayFromZero);
  }
  for (const char *backend : {"cuda", "hip", "rocm", "default"}) {
    OpDef("tl.infinity")
        .set_attr<tirx::FLowerIntrinsic>(
            std::string(backend) + ".FLowerIntrinsic", LowerInfinity);
  }
}

} // namespace tvm::tl
