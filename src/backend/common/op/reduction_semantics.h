/*!
 * \file tl/backend/common/op/reduction_semantics.h
 * \brief Shared scalar/packed reduction algebra, independent of GPU lowering.
 */
#ifndef TVM_TL_BACKEND_COMMON_OP_REDUCTION_SEMANTICS_H_
#define TVM_TL_BACKEND_COMMON_OP_REDUCTION_SEMANTICS_H_

#include "op/builtin.h"
#include "op/reduce.h"
#include <cmath>

namespace tvm::tl::backend::reduce {
using namespace prim;

inline PrimExpr MakeInitValue(const ReduceOpNode &op, int vsize = 1) {
  const PrimType dtype = op.dst.dtype();
  const bool is_int = dtype.MatchesCode(kDLInt);
  const bool is_uint = dtype.MatchesCode(kDLUInt, kDLBool);
  const int bits = dtype.MatchesCode(kDLBool) ? 1 : dtype.bits();
  PrimExpr scalar = MakeConst(dtype, 0);
  if (op.type->IsMax()) {
    if (is_int)
      scalar = IntImm(dtype, -(ffi::BigInt(1) << (bits - 1)));
    else if (!is_uint)
      scalar = MakeConst(dtype, -INFINITY);
  } else if (op.type->IsMin()) {
    if (is_int)
      scalar = IntImm(dtype, (ffi::BigInt(1) << (bits - 1)) - 1);
    else if (is_uint)
      scalar = IntImm(dtype, (ffi::BigInt(1) << bits) - 1);
    else
      scalar = MakeConst(dtype, INFINITY);
  } else if (op.type->IsBitAnd()) {
    ICHECK(is_int || is_uint) << "Bitwise reduction needs an integer dtype";
    scalar = is_int ? IntImm(dtype, -1)
                    : IntImm(dtype, (ffi::BigInt(1) << bits) - 1);
  } else {
    ICHECK(op.type->IsSum() || op.type->IsAbsSum() || op.type->IsAbsMax() ||
           op.type->IsBitOr() || op.type->IsBitXor());
  }
  ICHECK_GE(vsize, 1);
  return vsize == 1 ? scalar : PrimExpr(Broadcast(scalar, vsize));
}

inline PrimExpr MakeReduce(const ReduceOpNode &op, int vsize,
                           const PrimExpr &acc, const PrimExpr &value) {
  ICHECK(vsize == 1 || vsize == 2)
      << "Unsupported reduction vector size " << vsize;
  PrimExpr rhs = value.ty() == acc.ty() ? value : cast(value, acc.ty());
  const bool nan =
      op.nan_propagate && (acc.ty().MatchesElementType(kDLFloat, 16) ||
                           acc.ty().MatchesElementType(kDLBfloat, 16));
  auto call = [&](const char *name, ffi::Array<Expr> args) {
    return Call(acc.ty(), Op::Get(name), std::move(args))
        .as_or_throw<PrimExpr>();
  };
  if (vsize == 1) {
    if (op.type->IsAbsSum() || op.type->IsAbsMax()) {
      if (!rhs.ty().MatchesCode(kDLUInt, kDLBool))
        rhs = max(rhs, -rhs);
    }
    if (op.type->IsSum() || op.type->IsAbsSum())
      return acc + rhs;
    if (op.type->IsMax() || op.type->IsAbsMax())
      return nan ? call("tl.max_nan", {acc, rhs}) : max(acc, rhs);
    if (op.type->IsMin())
      return nan ? call("tl.min_nan", {acc, rhs}) : min(acc, rhs);
    if (op.type->IsBitAnd())
      return acc & rhs;
    if (op.type->IsBitOr())
      return acc | rhs;
    if (op.type->IsBitXor())
      return acc ^ rhs;
  } else {
    if (op.type->IsAbsSum() || op.type->IsAbsMax())
      rhs = call("tl.abs2", {rhs});
    if (op.type->IsSum() || op.type->IsAbsSum())
      return call("tl.add2", {acc, rhs});
    if (op.type->IsMax() || op.type->IsAbsMax())
      return call(nan ? "tl.max2_nan" : "tl.max2", {acc, rhs});
    if (op.type->IsMin())
      return call(nan ? "tl.min2_nan" : "tl.min2", {acc, rhs});
  }
  TVM_FFI_THROW(ValueError) << "Unsupported reduction type/vector pair "
                            << op.type->type << "/" << vsize;
}

} // namespace tvm::tl::backend::reduce
#endif // TVM_TL_BACKEND_COMMON_OP_REDUCTION_SEMANTICS_H_
