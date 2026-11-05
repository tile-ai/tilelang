/*!
 * \file layout/expr_utils.h
 * \brief Shared IR operations for TileLang's scalar layout algebra.
 */

#ifndef TVM_TL_LAYOUT_EXPR_UTILS_H_
#define TVM_TL_LAYOUT_EXPR_UTILS_H_

#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/stmt_functor.h>

#include <optional>

#include <ir/prim/op_utils.h>
#include <tirx/ir/data_type_rewriter.h>

namespace tvm::tl {

// A bounded layout algorithm may use an int64 literal only after this check.
// Native IntImm also supports values wider than int64; do not truncate them.
inline std::optional<int64_t> TryConstInt(const PrimExpr &expr) {
  if (const auto *value = expr.as<IntImmObj>())
    return value->value.as<int64_t>();
  return std::nullopt;
}

// Reuse native arithmetic legalization, with a TileLang boundary for lazy
// conditionals. The upstream legalizer preserves a Call's old result type even
// when its branches change width; layout coordinates must remain well typed.
class LayoutVarSubstitutor final : public tirx::DataTypeLegalizer {
public:
  template <typename TVar>
  explicit LayoutVarSubstitutor(const ffi::Map<TVar, PrimExpr> &values) {
    for (const auto &[var, replacement] : values)
      VarRemapSet(var, replacement);
  }

  using tirx::DataTypeLegalizer::Mutate;
  using tirx::DataTypeLegalizer::Mutate_;

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op,
                                 ffi::InplaceMode mode) final {
    if (!op->op.same_as(prim::if_then_else_op()))
      return tirx::DataTypeLegalizer::Mutate_(op, mode);

    Call original = ffi::GetRef<Call>(op);
    Call call = tirx::StmtExprMutator::Mutate_(op, ffi::InplaceMode::kDisallow)
                    .ValueOrUnchanged(original)
                    .as_or_throw<Call>();
    TVM_FFI_ICHECK_EQ(call->args.size(), 3);
    PrimExpr condition = call->args[0].as_or_throw<PrimExpr>();
    PrimExpr true_value = call->args[1].as_or_throw<PrimExpr>();
    PrimExpr false_value = call->args[2].as_or_throw<PrimExpr>();
    prim::detail::BinaryOpMatchTypes(true_value, false_value, call->loc);
    if (call.same_as(original) && true_value.same_as(call->args[1]) &&
        false_value.same_as(call->args[2]) && call->ty.same_as(true_value.ty()))
      return ffi::Unchanged();

    // Keep the lazy operator, metadata and source location. A Select would
    // eagerly evaluate both branches and could introduce an invalid access.
    return Call(true_value.ty(), call->op, {condition, true_value, false_value},
                call->attrs, call->ty_args, call->loc);
  }
};

/*!
 * \brief Substitute native scalar coordinate identities in a layout expression.
 *
 * Seed the native variable remap before visiting the original expression, so
 * inserted expressions are not recursively substituted. This preserves repeat
 * (i -> i % extent) and simultaneous axis renaming (i -> j, j -> k).
 * Rebuild arithmetic and lazy conditional types after an index-width change:
 * generic structural mapping would retain stale int32 types when an int64
 * coordinate is inserted.
 */
template <typename TVar>
PrimExpr SubstituteLayoutVars(const PrimExpr &expr,
                              const ffi::Map<TVar, PrimExpr> &values) {
  return ffi::make_object<LayoutVarSubstitutor>(values)
      ->Mutate(expr, ffi::InplaceMode::kDisallow)
      .ValueOrUnchanged(expr);
}

template <typename TVar>
ffi::Array<PrimExpr>
SubstituteLayoutVars(const ffi::Array<PrimExpr> &exprs,
                     const ffi::Map<TVar, PrimExpr> &values) {
  return exprs.Map(
      [&](const PrimExpr &expr) { return SubstituteLayoutVars(expr, values); });
}

// Explicit map type makes brace-initialized substitutions unambiguous.
inline PrimExpr
SubstituteLayoutVars(const PrimExpr &expr,
                     const ffi::Map<PrimVar, PrimExpr> &values) {
  return SubstituteLayoutVars<PrimVar>(expr, values);
}

inline ffi::Array<PrimExpr>
SubstituteLayoutVars(const ffi::Array<PrimExpr> &exprs,
                     const ffi::Map<PrimVar, PrimExpr> &values) {
  return SubstituteLayoutVars<PrimVar>(exprs, values);
}

// Public Python layout mapping has exact arity, unlike LayoutNode::Forward's
// internal leading-axis convention. Fragment callers include the replica axis
// explicitly in variables. Match the former IndexMap mapping's simplification
// policy after applying the shared TileLang coordinate substitution.
inline ffi::Array<PrimExpr>
MapLayoutIndices(const ffi::Array<PrimExpr> &variables,
                 const ffi::Array<PrimExpr> &expressions,
                 const ffi::Array<PrimExpr> &indices) {
  TVM_FFI_ICHECK_EQ(indices.size(), variables.size());
  ffi::Map<PrimVar, PrimExpr> values;
  for (size_t i = 0; i < variables.size(); ++i)
    values.Set(variables[i].as_or_throw<PrimVar>(), indices[i]);
  sym::Analyzer analyzer;
  return SubstituteLayoutVars(expressions, values)
      .Map([&](const PrimExpr &expr) { return analyzer->Simplify(expr); });
}

} // namespace tvm::tl

#endif // TVM_TL_LAYOUT_EXPR_UTILS_H_
