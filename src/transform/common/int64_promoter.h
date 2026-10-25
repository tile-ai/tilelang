/*!
 * \file int64_promoter.h
 * \brief Helper rewriter that promotes sub-64-bit integer expressions to int64.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_
#define TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_

#include <tvm/ir/prim/expr.h>

#include "support/base.h"

namespace tvm::tl {

class Int64Promoter {
public:
  TILELANG_DLL PrimExpr operator()(const PrimExpr &expression) const;
};

} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_
