#pragma once

#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

struct PTXAsyncCopyInjectResult {
  tvm::tirx::Stmt stmt;
  bool injected_ptx_async_copy{false};
};

/*! \brief Inject PTX cp.async lowering patterns into a statement.
 *
 * This is the statement-level entrypoint used by other transforms to apply the
 * same rewrite as the `tl.LowerPTXAsyncCopy` pass, but scoped to a region
 * (e.g., a lowered parallel loop) rather than the whole PrimFunc.
 */
PTXAsyncCopyInjectResult
InjectPTXAsyncCopy(const tvm::tirx::Stmt &body, bool enable_auto_async_copy,
                   bool async_without_async_commit_wait = false);

} // namespace tl
} // namespace tvm
