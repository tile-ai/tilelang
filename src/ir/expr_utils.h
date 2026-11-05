/*!
 * \file tl/ir/expr_utils.h
 * \brief Definition-preserving copies of native primitive expressions.
 */
#ifndef TILELANG_IR_EXPR_UTILS_H_
#define TILELANG_IR_EXPR_UTILS_H_

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/expr.h>

namespace tvm::tl {

// Copy local definitions before inserting an expression at another IR site.
// Free variables remain the caller's original uses; names, types and source
// locations are retained. Seed only original definitions so identities already
// created by structural hooks are not renamed a second time.
inline PrimExpr RenewExpressionDefs(const PrimExpr &expression) {
  ffi::Map<Var, Var> remap;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      expression, [&](const Var &var, TVMFFIDefRegionKind kind) {
        if (kind != kTVMFFIDefRegionKindNone)
          remap.Set(var, var);
        return ffi::WalkResult::Advance();
      });
  if (remap.empty())
    return expression;
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
             expression,
             [remap = std::move(remap)](const Var &var,
                                        TVMFFIDefRegionKind kind) mutable {
               auto mapped = remap.Get(var);
               if (!mapped)
                 return var;
               if (!mapped->same_as(var))
                 return *mapped;
               if (kind == kTVMFFIDefRegionKindNone)
                 return var;
               Var fresh(var->name, var->ty, var->loc);
               remap.Set(var, fresh);
               return fresh;
             })
      .as_or_throw<PrimExpr>();
}

} // namespace tvm::tl

#endif // TILELANG_IR_EXPR_UTILS_H_
