
/*!
 * \file assume.h
 * \brief Utils on assume statements
 */

#ifndef TVM_TL_TRANSFORM_COMMON_ASSUME_H_
#define TVM_TL_TRANSFORM_COMMON_ASSUME_H_

#include "tvm/tir/stmt.h"
#include <optional>

namespace tvm {
namespace tl {

using namespace tir;

// Get the expression inside an assume statement, if any. Returns nullopt if
// the statement is not an assume statement.
std::optional<PrimExpr> GetAssumeExprInEvaluateForm(Stmt stmt);

// Check if a statement is an assume statement.
bool IsAssumeInEvaluateForm(const Stmt &stmt);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_ASSUME_H_
