/*!
 * \file tl/ir/lexical_scope.h
 * \brief TileLang's once-executed definition and allocation scopes.
 */

#ifndef TVM_TL_IR_LEXICAL_SCOPE_H_
#define TVM_TL_IR_LEXICAL_SCOPE_H_

#include <tvm/ir/op.h>
#include <tvm/ir/stmt.h>

namespace tvm::tl {

/*! \brief A scoped region with no operands, parameters, or results. */
const Op &lexical_scope_op();

/*! \brief Preserve a definition scope, optionally marking an explicit local
 * pool. */
RegionStmt MakeLexicalScope(SeqStmt body, Location loc = UnknownLoc(),
                            bool explicit_alloc_scope = false);

bool IsLexicalScope(const RegionStmtObj *region);
bool IsExplicitAllocationScope(const RegionStmtObj *region);

} // namespace tvm::tl

#endif // TVM_TL_IR_LEXICAL_SCOPE_H_
