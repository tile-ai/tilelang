/*!
 * \file merge_if_stmt.h
 * \brief Merge consecutive If statements with the same condition
 */
#ifndef TVM_TL_TRANSFORM_MERGE_IF_STMT_H_
#define TVM_TL_TRANSFORM_MERGE_IF_STMT_H_

#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

using namespace tir;

// Forward declaration
class MergeIfStmtRewriter;

/*!
 * \brief Apply MergeIfStmt transformation to a PrimFunc
 *
 * This function merges consecutive IfThenElse statements that have the same
 * condition into a single if statement with a SeqStmt body.
 *
 * Example:
 *   if (cond) { stmt1 }
 *   if (cond) { stmt2 }
 *   if (cond) { stmt3 }
 *
 * Becomes:
 *   if (cond) {
 *     stmt1
 *     stmt2
 *     stmt3
 *   }
 *
 * \param f The PrimFunc to transform
 * \return Transformed PrimFunc with merged if statements
 */
PrimFunc MergeIfStmtSubstitute(PrimFunc &f);

/*!
 * \brief Apply MergeIfStmt transformation to a statement
 * \param stmt The statement to transform
 * \return Transformed statement with merged if statements
 */
Stmt ApplyMergeIfStmt(Stmt stmt);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_MERGE_IF_STMT_H_
