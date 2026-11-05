/*!
 * \file op/isfinite.cc
 * \brief Preserve TileLang's finite predicate until backend intrinsic lowering.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/attr.h>

namespace tvm::tl {
namespace {

Type InferIsFiniteType(const CallObj *call) {
  PrimType type = call->args[0].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(type.MatchesCode(kDLFloat, kDLBfloat), TypeError)
      << "tl.isfinite requires a floating-point value, got " << type;
  return PrimType::Bool(type.lanes());
}

PrimExpr LowerIsFinite(const PrimExpr &expr) {
  const auto *call = expr.as<CallObj>();
  return prim::isfinite(call->args[0].as_or_throw<PrimExpr>(), call->loc);
}

PrimExpr LowerCIsFinite(const PrimExpr &expr) {
  const auto *call = expr.as<CallObj>();
  PrimType type = call->args[0].as_or_throw<PrimExpr>().ty();
  if (type.MatchesCode(kDLFloat) && (type.bits() == 32 || type.bits() == 64)) {
    return Call(expr.ty(), Op::Get("tl.c.isfinite"), call->args, call->attrs,
                {}, call->loc)
        .as_or_throw<PrimExpr>();
  }
  return LowerIsFinite(expr);
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  // Native global-symbol ops use the header's declaration. A call_pure_extern
  // would emit a conflicting C prototype for the C++ <math.h> overload set.
  OpDef("tl.c.isfinite")
      .signature(sig::arg<PrimExpr>("value"))
      .set_native_fn_attr<&InferIsFiniteType>(op_attr::kInferType)
      .set_attr(op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kGlobalSymbol, ffi::String("isfinite"));
  OpDef("tl.isfinite")
      .signature(sig::arg<PrimExpr>("value"))
      .set_native_fn_attr<&InferIsFiniteType>(op_attr::kInferType)
      .set_attr(op_attr::kCallEffectKind, CallEffectKind::kPure)
      // This entry belongs to our new op. It also allows the generic native
      // C compiler to consume the op without TileLang's additional C rules.
      .set_attr<tirx::FLowerIntrinsic>("c.FLowerIntrinsic", LowerCIsFinite)
      .set_attr<tirx::FLowerIntrinsic>("default.FLowerIntrinsic",
                                       LowerIsFinite);
}

} // namespace tvm::tl
