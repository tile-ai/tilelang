/*!
 * \file ascend/codegen/vector_index.h
 * \brief Contiguous vector-index analysis for CCE and PTO memory emission.
 */
#ifndef TILELANG_ASCEND_CODEGEN_VECTOR_INDEX_H_
#define TILELANG_ASCEND_CODEGEN_VECTOR_INDEX_H_

#include <tvm/ir/stmt.h>
#include <tvm/sym/analyzer.h>

namespace tvm::codegen::ascend {

// Record literal-valued SSA definitions without substituting memory snapshots.
void BindIndexConstant(const BindObj *bind, const sym::Analyzer &analyzer);

// Recover a Ramp only when every lane of a pure integer expression agrees.
PrimExpr NormalizeVectorIndex(const PrimExpr &index,
                              const sym::Analyzer &analyzer);

} // namespace tvm::codegen::ascend
#endif // TILELANG_ASCEND_CODEGEN_VECTOR_INDEX_H_
