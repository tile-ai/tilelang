#include "prim.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>

#include <limits>

namespace tvm::tl {
namespace {

PrimExpr CastIntegerConstant(const IntImmNode *constant, PrimType type) {
  if (type.MatchesCode(kDLBool) || type.MatchesElementType(kDLInt, 1))
    return IntImm(type, constant->value != 0 ? 1 : 0, constant->loc);
  if (type.MatchesCode(kDLInt, kDLUInt)) {
    TVM_FFI_CHECK_GT(type.bits(), 0, ValueError);
    ffi::BigInt modulus = ffi::BigInt(1) << type.bits();
    ffi::BigInt converted = constant->value & (modulus - 1);
    if (type.MatchesCode(kDLInt) && converted >= (modulus >> 1))
      converted -= modulus;
    return IntImm(type, converted, constant->loc);
  }
  return tl::MakeConst(type, constant->value, constant->loc);
}

PrimExpr FloatingOperand(PrimExpr value, Location loc) {
  if (value.ty().MatchesCode(kDLFloat) && value.ty().bits() >= 32)
    return value;
  return tl::CastValue(value, PrimType::Float(32, value.ty().lanes()), loc);
}

Type FiniteType(const CallNode *call) {
  return PrimType::Bool(call->args[0].as_or_throw<PrimExpr>().ty().lanes());
}

PrimExpr LowerIsFinite(PrimExpr expression) {
  auto call = expression.as_or_throw<Call>();
  PrimExpr value =
      FloatingOperand(call->args[0].as_or_throw<PrimExpr>(), call->loc);
  return !tl::IsInf(value, call->loc) && !prim::isnan(value, call->loc);
}

PrimExpr LowerAbs(PrimExpr expression) {
  auto call = expression.as_or_throw<Call>();
  PrimExpr value =
      FloatingOperand(call->args[0].as_or_throw<PrimExpr>(), call->loc);
  PrimExpr result =
      Call(value.ty(), tirx::call_pure_extern_op(),
           {StringImm(value.ty().bits() == 64 ? "fabs" : "fabsf"), value},
           call->attrs, {}, call->loc)
          .as_or_throw<PrimExpr>();
  return tl::CastValue(result, expression.ty(), call->loc);
}

} // namespace

PrimExpr CastValue(PrimExpr value, PrimType type, Location loc) {
  if (value.ty() == type)
    return value;
  TVM_FFI_CHECK(!value.ty().IsVoid(), TypeError)
      << "Cannot cast an expression with the void sentinel type";
  if (type.IsScalar()) {
    if (tl::IsCustomDType(type) || tl::IsCustomDType(value.ty()))
      return prim::Cast(type, value, loc);
    if (auto integer = value.as<IntImmNode>())
      return CastIntegerConstant(integer, type);
    if (auto floating = value.as<FloatImmNode>())
      return tl::MakeConst(type, floating->value, floating->loc);
    return prim::Cast(type, value, loc);
  }
  PrimType element = type.WithLanes(1);
  if (value.ty().IsScalar()) {
    PrimExpr scalar = tl::CastValue(value, element, loc);
    if (type.IsScalableVector()) {
      PrimExpr lanes = prim::Mul(Call(PrimType::Int(32), prim::vscale_op(), {})
                                     .as_or_throw<PrimExpr>(),
                                 type.VScaleFactor());
      return prim::Broadcast(scalar, lanes, loc);
    }
    return prim::Broadcast(scalar, type.lanes(), loc);
  }
  TVM_FFI_CHECK(value.ty().lanes() == type.lanes(), TypeError)
      << "Cast operands must have matching vector lanes";
  if (auto broadcast = value.as<prim::BroadcastNode>())
    return prim::Broadcast(tl::CastValue(broadcast->value, element, loc),
                           broadcast->lanes, loc);
  return prim::Cast(type, value, loc);
}

PrimExpr Infinity(PrimType type, Location loc) {
  TVM_FFI_CHECK(type.MatchesCode(kDLFloat, kDLBfloat, kDLFloat8_e5m2),
                TypeError)
      << "Cannot represent infinity in " << type;
  return tl::MakeConst(type, std::numeric_limits<double>::infinity(), loc);
}

PrimExpr IsInf(PrimExpr value, Location loc) {
  if (value.ty().MatchesCode(kDLInt, kDLUInt, kDLBool))
    return tl::MakeConst(PrimType::Bool(value.ty().lanes()), false, loc);
  value = FloatingOperand(value, loc);
  PrimExpr infinity = tl::Infinity(value.ty(), loc);
  return value == infinity || value == -infinity;
}

PrimExpr IsFinite(PrimExpr value, Location loc) {
  if (value.ty().MatchesCode(kDLInt, kDLUInt, kDLBool))
    return tl::MakeConst(PrimType::Bool(value.ty().lanes()), true, loc);
  return Call(PrimType::Bool(value.ty().lanes()), Op::Get("tl.isfinite"),
              {value}, {}, {}, loc)
      .as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.isfinite")
      .signature(sig::arg<PrimExpr>("value", "The input value."))
      .set_native_fn_attr<&FiniteType>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kVectorizable, true)
      .set_attr<tirx::FLowerIntrinsic>("tl.default.FLowerIntrinsic",
                                       LowerIsFinite);
  OpDef("prim.fabs")
      .set_attr<tirx::FLowerIntrinsic>("tl.default.FLowerIntrinsic", LowerAbs);
  ffi::reflection::GlobalDef()
      .def("tl.ir.Cast", CastValue)
      .def("tl.ir.Infinity", Infinity)
      .def("tl.ir.IsInf", IsInf)
      .def("tl.ir.IsFinite", IsFinite);
}

} // namespace tvm::tl
