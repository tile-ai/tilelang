/*!
 * \file int64_promoter.h
 * \brief Widen signed address arithmetic without changing data conversions.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_
#define TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_

#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/tirx/expr.h>

namespace tvm::tl {

// Used only for address expressions. Public scalar parameters, tensor element
// types and explicit narrowing/unsigned/bitwise operations retain their types
// and semantics. In particular int64(int8(x)) is not rewritten to int64(x).
class Int64Promoter {
public:
  PrimExpr operator()(const PrimExpr &expression) const {
    PrimType type = expression.ty();
    if (!type.MatchesCode(kDLInt) || type.bits() >= 64)
      return expression;
    const PrimType wide_type = type.WithBits(64);
    if (const auto *literal = expression.as<IntImmObj>())
      return IntImm(wide_type, literal->value, expression->loc);
    if (expression.as<prim::CastObj>() || expression.as<TensorLoadObj>() ||
        type.bits() < 32)
      return prim::cast(expression, wide_type);
#define TL_WIDEN_INDEX_BINARY(Name)                                            \
  if (const auto *node = expression.as<prim::Name##Obj>())                     \
  return prim::Name((*this)(node->a), (*this)(node->b), node->loc)
    TL_WIDEN_INDEX_BINARY(Add);
    TL_WIDEN_INDEX_BINARY(Sub);
    TL_WIDEN_INDEX_BINARY(Mul);
    TL_WIDEN_INDEX_BINARY(Div);
    TL_WIDEN_INDEX_BINARY(Mod);
    TL_WIDEN_INDEX_BINARY(FloorDiv);
    TL_WIDEN_INDEX_BINARY(FloorMod);
    TL_WIDEN_INDEX_BINARY(Min);
    TL_WIDEN_INDEX_BINARY(Max);
#undef TL_WIDEN_INDEX_BINARY
    if (const auto *ramp = expression.as<prim::RampObj>())
      return prim::Ramp((*this)(ramp->base), (*this)(ramp->stride), ramp->lanes,
                        ramp->loc);
    if (const auto *broadcast = expression.as<prim::BroadcastObj>())
      return prim::Broadcast((*this)(broadcast->value), broadcast->lanes,
                             broadcast->loc);
    if (const auto *select = expression.as<prim::SelectObj>())
      return prim::Select(select->condition, (*this)(select->true_value),
                          (*this)(select->false_value), select->loc);
    if (const auto *call = expression.as<CallObj>();
        call && call->op.same_as(prim::if_then_else_op()))
      return if_then_else(call->args[0].as_or_throw<PrimExpr>(),
                          (*this)(call->args[1].as_or_throw<PrimExpr>()),
                          (*this)(call->args[2].as_or_throw<PrimExpr>()));
    return prim::cast(expression, wide_type);
  }
};

} // namespace tvm::tl
#endif // TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_
