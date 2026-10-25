#ifndef TILELANG_COMPAT_EXPR_H_
#define TILELANG_COMPAT_EXPR_H_

#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <optional>

namespace tvm::tl {

using namespace prim;

inline std::optional<int64_t> GetConstInt(const Expr &expression) {
  if (const auto *constant = expression.as<IntImmNode>()) {
    return constant->value.as<int64_t>();
  }
  return std::nullopt;
}

inline std::optional<int64_t> GetIntAnnotation(const ffi::Any &value) {
  if (auto integer = value.as<int64_t>()) {
    return *integer;
  }
  if (auto boolean = value.as<bool>()) {
    return *boolean ? 1 : 0;
  }
  if (auto expression = value.as<Expr>()) {
    return GetConstInt(expression.value());
  }
  return std::nullopt;
}

inline PrimExpr Substitute(const PrimExpr &expression,
                           const ffi::Map<PrimVar, PrimExpr> &replacements) {
  return tirx::SubstituteWithDataTypeLegalization(
      expression,
      [&replacements](const Var &variable) -> ffi::Optional<PrimExpr> {
        if (auto scalar = variable.as<PrimVar>()) {
          return replacements.Get(scalar.value());
        }
        return std::nullopt;
      });
}

inline ffi::Array<PrimExpr>
Substitute(const ffi::Array<PrimExpr> &expressions,
           const ffi::Map<PrimVar, PrimExpr> &replacements) {
  return expressions.Map([&replacements](const PrimExpr &expression) {
    return Substitute(expression, replacements);
  });
}

inline Stmt Substitute(const Stmt &statement,
                       const ffi::Map<PrimVar, PrimExpr> &replacements) {
  return tirx::SubstituteWithDataTypeLegalization(
      statement,
      [&replacements](const Var &variable) -> ffi::Optional<PrimExpr> {
        if (auto scalar = variable.as<PrimVar>()) {
          return replacements.Get(scalar.value());
        }
        return std::nullopt;
      });
}

} // namespace tvm::tl

#endif
