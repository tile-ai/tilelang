/*!
 * \file intrin_rule_cuda.cc
 * \brief TileLang CUDA extensions to native intrinsic lowering.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/attr.h>

#include "backend/common/codegen/intrin_rule.h"

namespace tvm::tl {
namespace {

struct CUDAMath {
  std::string operator()(const PrimType &type, std::string name) const {
    if (type.MatchesCode(kDLFloat)) {
      switch (type.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16:
        if (name == "fabs")
          return "__habs";
        if (name == "round")
          return "hrint";
        return "h" + name;
      default:
        return "";
      }
    }
    if (type.MatchesElementType(kDLBfloat, 16)) {
      if (name == "fabs")
        return "__habs";
      if (name == "round")
        return "hrint";
      return "h" + name;
    }
    if (type.MatchesCode(kDLInt, kDLUInt)) {
      if (type.bits() == 32)
        return "__" + name;
      if (type.bits() == 64)
        return "__" + name + "ll";
    }
    return "";
  }
};

PrimExpr DispatchCUDAPopcount(const PrimExpr &expr) {
  const auto *call = expr.as<CallObj>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 1);
  PrimExpr operand = call->args[0].as_or_throw<PrimExpr>();
  PrimType type = operand.ty();
  const int bits = type.bits();
  if (!type.MatchesCode(kDLInt, kDLUInt) ||
      (bits != 8 && bits != 16 && bits != 32 && bits != 64))
    return expr;
  // Preserve the original width before widening: int8(-1) has eight set bits.
  operand = prim::Cast(PrimType::UInt(bits, type.lanes()), operand);
  if (bits < 32)
    operand = prim::Cast(PrimType::UInt(32, type.lanes()), operand);
  auto count =
      Call(PrimType::Int(32, type.lanes()), tirx::call_pure_extern_op(),
           {StringImm(bits == 64 ? "__popcll" : "__popc"), operand},
           call->attrs, {}, call->loc)
          .as_or_throw<PrimExpr>();
  return prim::Cast(expr.ty(), count);
}

PrimExpr DispatchCUDAIsFinite(const PrimExpr &expr) {
  const auto *call = expr.as<CallObj>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 1);
  PrimType type = call->args[0].as_or_throw<PrimExpr>().ty();
  if (type.MatchesCode(kDLFloat) && (type.bits() == 32 || type.bits() == 64))
    return intrin::CallPureExtern(call, "isfinite", call->args);
  return expr;
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  // Native Op attributes have no numeric priority. A TileLang-specific column
  // makes precedence explicit in tl.LowerIntrin and independent of TU order.
  constexpr const char *key = "tl.cuda.FLowerIntrinsic";
  OpDef("prim.popcount")
      .set_attr<tirx::FLowerIntrinsic>(key, DispatchCUDAPopcount);
  for (const char *name :
       {"rsqrt", "asin", "acos", "asinh", "acosh", "atanh", "log1p", "atan2",
        "hypot", "copysign", "ldexp", "nextafter"}) {
    OpDef(ffi::String("prim.") + name)
        .set_attr<tirx::FLowerIntrinsic>(key,
                                         intrin::DispatchPureExtern<CUDAMath>);
  }
  OpDef("tl.isfinite")
      .set_attr<tirx::FLowerIntrinsic>(key, DispatchCUDAIsFinite);
}

} // namespace tvm::tl
