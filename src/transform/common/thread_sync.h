#ifndef TILELANG_TRANSFORM_COMMON_THREAD_SYNC_H_
#define TILELANG_TRANSFORM_COMMON_THREAD_SYNC_H_

#include <tvm/tirx/function.h>

#include "support/base.h"

namespace tvm::tl {

TILELANG_DLL SeqStmt
RewriteThreadSync(const tirx::Function &function, const SeqStmt &body,
                  ffi::Optional<ffi::String> storage_scope = std::nullopt,
                  bool cuda_features = true);

}

#endif
