/*!
 * \file tl/op/native_utils.h
 * \brief Small native IR helpers shared by TileLang operator lowering.
 */
#ifndef TVM_TL_OP_NATIVE_UTILS_H_
#define TVM_TL_OP_NATIVE_UTILS_H_
#include "operator.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl {

inline PrimExpr LoopExtent(const ForObj *loop) {
  TVM_FFI_CHECK(prim::IsOne(loop->step), ValueError)
      << "TileLang parallel layout lowering requires a unit-step loop";
  return loop->end - loop->begin;
}

// A native scope body remains SeqStmt. Inspection may look through a singleton
// sequence, but reconstruction must wrap it back into the enclosing scope.
inline Stmt SingleStatement(const Stmt &stmt) {
  const auto *seq = stmt.as<SeqStmtObj>();
  return seq && seq->seq.size() == 1 ? seq->seq[0] : stmt;
}

inline ffi::Map<ffi::String, ffi::Any>
DecodeAttributeRegion(const RegionStmtObj *region) {
  auto values =
      DecodeAnnotations(
          region->args[0].as_or_throw<tvm::Tuple>(),
          region->attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema")
              .value())
          .cast<ffi::Map<ffi::Any, ffi::Any>>();
  ffi::Map<ffi::String, ffi::Any> result;
  for (const auto &[key, value] : values)
    result.Set(key.cast<ffi::String>(), value);
  return result;
}

inline Stmt SubstituteLoopVars(const Stmt &stmt,
                               const ffi::Map<Var, PrimExpr> &values) {
  return tirx::SubstituteWithDataTypeLegalization(
      stmt, [&values](const Var &var) -> ffi::Optional<PrimExpr> {
        return values.Get(var);
      });
}

inline PrimExpr SubstituteLoopVars(const PrimExpr &expr,
                                   const ffi::Map<Var, PrimExpr> &values) {
  return tirx::SubstituteWithDataTypeLegalization(
      expr, [&values](const Var &var) -> ffi::Optional<PrimExpr> {
        return values.Get(var);
      });
}

inline ffi::Array<PrimExpr>
SubstituteLoopVars(const ffi::Array<PrimExpr> &exprs,
                   const ffi::Map<Var, PrimExpr> &values) {
  return exprs.Map(
      [&](const PrimExpr &expr) { return SubstituteLoopVars(expr, values); });
}

template <typename Callback>
void VisitNativePostOrder(ffi::AnyView object, Callback callback) {
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      object,
      [&](const ffi::ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
        callback(node);
        return ffi::WalkResult::Advance();
      });
}

template <typename Callback>
bool UsesAnyVar(ffi::AnyView expr, Callback callback) {
  bool found = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      expr, [&](const VarObj *var) -> ffi::Expected<ffi::WalkResult> {
        if (callback(var)) {
          found = true;
          return ffi::WalkResult::Interrupt(
              ffi::VisitInterrupt(ffi::GetRef<Var>(var)));
        }
        return ffi::WalkResult::Skip();
      });
  return found;
}

} // namespace tvm::tl
#endif // TVM_TL_OP_NATIVE_UTILS_H_
