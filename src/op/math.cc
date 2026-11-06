/*!
 * \file tl/op/math.cc
 * \brief Native scalar/vector math contracts and backend intrinsic lowering.
 */
#include "builtin.h"
#include "ir/annotations.h"
#include "ir/dtype.h"
#include "support/check.h"

#include <tvm/ffi/dtype.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>

#include <limits>

namespace tvm::tl {
using namespace tirx;
using namespace prim;

namespace {
void ValidateMathResult(const CallObj *call, PrimType result) {
  TVM_FFI_CHECK(!result.IsVoid(), TypeError);
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  if (!call->ty_args.empty())
    TVM_FFI_CHECK_EQ(call->ty_args[0].as_or_throw<PrimType>(), result,
                     TypeError)
        << "Math result type disagrees with its type argument";
  TVM_FFI_CHECK_EQ(call->ty.as_or_throw<PrimType>(), result, TypeError)
      << "Math result type disagrees with its operand";
}

Type InferPower(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 2U, ValueError)
      << "tl.pow_of_int expects two runtime operands";
  const auto result = args[0].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(!result.IsVoid(), TypeError);
  return result;
}

void ValidatePower(const CallObj *call) {
  ValidateMathResult(call, InferPower(call).as_or_throw<PrimType>());
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK(args[1].as_or_throw<IntImm>()->value.as<int64_t>().has_value(),
                ValueError)
      << "tl.pow_of_int exponent must be a representable integer constant";
}

Type InferRound(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError)
      << "tl.round_ties_away_from_zero expects one runtime operand";
  const auto result = args[0].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(!result.IsVoid(), TypeError);
  return result;
}

void ValidateRound(const CallObj *call) {
  ValidateMathResult(call, InferRound(call).as_or_throw<PrimType>());
}

Type InferInfinity(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError)
      << "tl.infinity expects one runtime operand";
  const PrimType dtype(
      ffi::StringToDLDataType(args[0].as_or_throw<StringImm>()->value));
  const bool supported =
      (dtype.MatchesCode(kDLFloat) &&
       (dtype.bits() == 16 || dtype.bits() == 32 || dtype.bits() == 64)) ||
      dtype.MatchesElementType(kDLBfloat, 16) ||
      dtype.MatchesCode(kDLFloat8_e5m2) || IsTensorFloat32(dtype);
  TVM_FFI_CHECK(dtype.IsScalar() && supported, TypeError)
      << "tl.infinity requires a supported scalar floating result type";
  return dtype;
}

void ValidateInfinity(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError)
      << "tl.infinity expects one runtime operand";
  args[0].as_or_throw<StringImm>();
  ValidateMathResult(call, InferInfinity(call).as_or_throw<PrimType>());
}
} // namespace

// Bind each operand once before lane expansion, including side-effecting calls.
PrimExpr LowerClamp(PrimExpr expression) {
  const Call call = expression.as_or_throw<Call>();
  const auto arguments = DecodeCallOperands(call.get());
  ICHECK_EQ(arguments.size(), 3U);
  const PrimType dtype = expression.ty();
  PrimVar x("clamp_x", dtype), lo("clamp_lo", dtype), hi("clamp_hi", dtype);
  for (const Expr &argument : arguments) {
    TVM_FFI_CHECK(argument.as_or_throw<PrimExpr>().ty() == dtype, TypeError)
        << "tl.clamp operands must have matching types";
  }
  ffi::Array<PrimExpr> results;
  for (int lane = 0; lane < dtype.lanes(); ++lane) {
    auto extract = [&](const PrimVar &var) -> PrimExpr {
      return dtype.IsScalar() ? PrimExpr(var)
                              : Shuffle::ExtractElement(var, lane);
    };
    PrimExpr lane_x = extract(x), lane_lo = extract(lo), lane_hi = extract(hi);
    PrimExpr result = min(max(lane_x, lane_lo), lane_hi);
    if (!dtype.MatchesCode(kDLInt, kDLUInt, kDLBool)) {
      // Preserve the source rule's x, lo, hi NaN priority. Conversion to fp32
      // only computes a predicate; the selected value keeps its original type.
      for (const PrimExpr &value : {lane_hi, lane_lo, lane_x}) {
        PrimExpr check = dtype.WithLanes(1) == PrimType::Float(64)
                             ? value
                             : cast(value, PrimType::Float(32));
        result = Select(isnan(check), value, result);
      }
    }
    results.push_back(result);
  }
  PrimExpr result = dtype.IsScalar() ? results[0] : Shuffle::Concat(results);
  return Let(x, arguments[0].as_or_throw<PrimExpr>(),
             Let(lo, arguments[1].as_or_throw<PrimExpr>(),
                 Let(hi, arguments[2].as_or_throw<PrimExpr>(), result)),
             call->loc);
}

PrimExpr LowerPowOfInt(PrimExpr expression) {
  const Call call = expression.as_or_throw<Call>();
  const auto arguments = DecodeCallOperands(call.get());
  ICHECK_EQ(arguments.size(), 2U);
  const auto exponent = arguments[1].as_or_throw<IntImm>()->value.as<int64_t>();
  TVM_FFI_CHECK(exponent.has_value(), ValueError)
      << "tl.pow_of_int exponent must be a representable integer constant";
  PrimExpr base = arguments[0].as_or_throw<PrimExpr>();
  ffi::String symbol = "tl::pow_of_int<" + std::to_string(*exponent) + ">";
  return Call(expression.ty(), call_pure_extern_op(), {StringImm(symbol), base},
              {}, {}, call->loc)
      .as_or_throw<PrimExpr>();
}

PrimExpr LowerInfinity(PrimExpr expression) {
  const Call call = expression.as_or_throw<Call>();
  const PrimType dtype = expression.ty();
  TVM_FFI_CHECK(dtype.IsScalar(), ValueError)
      << "tl.infinity requires a scalar result";
  const bool supported =
      (dtype.MatchesCode(kDLFloat) &&
       (dtype.bits() == 16 || dtype.bits() == 32 || dtype.bits() == 64)) ||
      dtype.MatchesElementType(kDLBfloat, 16) ||
      dtype.MatchesCode(kDLFloat8_e5m2) || IsTensorFloat32(dtype);
  TVM_FFI_CHECK(supported, ValueError)
      << "Cannot decide infinity for type " << dtype;
  return FloatImm(dtype, std::numeric_limits<double>::infinity(), call->loc);
}

PrimExpr LowerRoundTiesAwayFromZero(PrimExpr expression) {
  const Call call = expression.as_or_throw<Call>();
  const auto arguments = DecodeCallOperands(call.get());
  ICHECK_EQ(arguments.size(), 1U);
  if (expression.ty().MatchesCode(kDLInt, kDLUInt, kDLBool))
    return arguments[0].as_or_throw<PrimExpr>();
  return Call(expression.ty(), call_pure_extern_op(),
              {StringImm("tl::RoundTiesAwayFromZero"), arguments[0]}, {}, {},
              call->loc)
      .as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef clamp("tl.clamp");
  for (const char *target : {"ascend", "llvm", "hip", "metal", "webgpu"})
    clamp.set_attr<FLowerIntrinsic>(std::string(target) + ".FLowerIntrinsic",
                                    LowerClamp);

  OpDef power("tl.pow_of_int");
  power
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidatePower>())
      .signature(sig::arg<PrimExpr>("base"), sig::arg<IntImm>("exponent"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_native_fn_attr<&InferPower>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kVectorizable, false);
  power.set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", LowerPowOfInt)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", LowerPowOfInt);

  OpDef infinity("tl.infinity");
  infinity
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateInfinity>())
      .signature(sig::arg<StringImm>("dtype"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_native_fn_attr<&InferInfinity>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kVectorizable, false);
  infinity.set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", LowerInfinity)
      .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", LowerInfinity)
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", LowerInfinity);

  OpDef round("tl.round_ties_away_from_zero");
  round
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateRound>())
      .signature(sig::arg<PrimExpr>("value"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_native_fn_attr<&InferRound>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kVectorizable, true);
  round
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                                 LowerRoundTiesAwayFromZero)
      .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                                 LowerRoundTiesAwayFromZero);
}
} // namespace tvm::tl
