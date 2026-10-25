#pragma once

#include <tvm/sym/analyzer.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

struct PTXAsyncCopyInjectResult {
  tvm::Stmt stmt;
  bool injected_ptx_async_copy{false};
};

/*! \brief Inject PTX cp.async lowering patterns into a statement.
 *
 * This is the statement-level entrypoint used by other transforms to apply the
 * same rewrite as CUDA PTX async-copy passes, but scoped to a region (e.g.,
 * a lowered parallel loop) rather than the whole PrimFunc. Callers decide
 * whether a region should use async-copy lowering before invoking this helper.
 */
PTXAsyncCopyInjectResult
InjectPTXAsyncCopy(const tvm::Stmt &body,
                   bool async_without_async_commit_wait = false,
                   ffi::Optional<sym::Analyzer> analyzer = std::nullopt);

} // namespace tl
} // namespace tvm
