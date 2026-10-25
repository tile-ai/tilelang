#ifndef TVM_TL_TRANSFORM_MULTI_VERSION_BUFFER_REWRITER_H_
#define TVM_TL_TRANSFORM_MULTI_VERSION_BUFFER_REWRITER_H_

#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

namespace tvm::tl {
Stmt RewriteBufferVersions(
    Stmt statement, ffi::Map<tirx::TensorVar, tirx::TensorVar> replacements,
    ffi::Map<tirx::TensorVar, ffi::Array<PrimExpr>> versions);
}

#endif
