#pragma once

#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

struct ROCmAsyncCopyInjectResult {
  tvm::tirx::Stmt stmt;
  bool injected_rocm_async_copy{false};
};

/*! \brief Inject ROCm async-copy lowering patterns into a statement.
 *
 * This is the statement-level entrypoint used by other transforms to apply the
 * same rewrite as ROCm async-copy lowering, but scoped to a region (e.g., a
 * lowered parallel loop) rather than the whole PrimFunc. Callers decide whether
 * a region should use async-copy lowering before invoking this helper.
 */
ROCmAsyncCopyInjectResult
InjectROCmAsyncCopy(const tvm::tirx::Stmt &body,
                    bool async_without_async_commit_wait = false);

} // namespace tl
} // namespace tvm
