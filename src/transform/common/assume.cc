
/*!
 * \file assume.cc
 * \brief Utils on assume statements
 */

#include "assume.h"
#include <tvm/tirx/builtin.h>
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
  if (!call->op.same_as(builtin::assume()))
    return std::nullopt;
  return call->args[0];
}

bool IsAssumeInEvaluateForm(const Stmt &stmt) {
  return GetAssumeExprInEvaluateForm(stmt).has_value();
}

} // namespace tl
} // namespace tvm
