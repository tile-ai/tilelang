
/*!
 * \file assume.cc
 * \brief Utils on assume statements
 */

#include "assume.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>

namespace tvm {
namespace tl {

using namespace tirx;

std::optional<PrimExpr> GetAssumeExprInEvaluateForm(Stmt stmt) {
  auto eval = stmt.as<EvaluateNode>();
  if (!eval)
    return std::nullopt;
  auto call = eval->value.as<CallNode>();
  if (!call)
    return std::nullopt;
  if (!call->op.same_as(prim::assume_op()))
    return std::nullopt;
  return call->args[0].as_or_throw<PrimExpr>();
}

bool IsAssumeInEvaluateForm(const Stmt &stmt) {
  return GetAssumeExprInEvaluateForm(stmt).has_value();
}

} // namespace tl
} // namespace tvm
