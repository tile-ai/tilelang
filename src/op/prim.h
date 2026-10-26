#ifndef TILELANG_OP_PRIM_H_
#define TILELANG_OP_PRIM_H_

#include <tvm/ir/prim/op.h>

#include "dtype.h"

namespace tvm::tl {

PrimExpr CastValue(PrimExpr value, PrimType type, Location loc = UnknownLoc());
PrimExpr Infinity(PrimType type, Location loc = UnknownLoc());
PrimExpr IsInf(PrimExpr value, Location loc = UnknownLoc());
PrimExpr IsFinite(PrimExpr value, Location loc = UnknownLoc());

template <typename Value>
PrimExpr MakeConst(PrimType type, Value value, Location loc = UnknownLoc()) {
  if (!tl::IsCustomDType(type))
    return prim::MakeConst(type, value, loc);
  PrimExpr scalar =
      FloatImm(type.WithLanes(1), static_cast<double>(value), loc);
  if (type.IsScalar())
    return scalar;
  if (type.IsScalableVector()) {
    PrimExpr lanes = prim::Mul(
        Call(PrimType::Int(32), prim::vscale_op(), {}).as_or_throw<PrimExpr>(),
        type.VScaleFactor());
    return prim::Broadcast(scalar, lanes, loc);
  }
  return prim::Broadcast(scalar, type.lanes(), loc);
}

} // namespace tvm::tl

#endif
