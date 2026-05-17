#pragma once

#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

struct PTXAsyncCopyInjectResult {
  tvm::tir::Stmt stmt;
  bool injected_ptx_async_copy{false};
};

/*! \brief Inject PTX cp.async lowering patterns into a statement.
 *
 * This is the statement-level entrypoint used by other transforms to apply the
 * same rewrite as the `tl.LowerPTXAsyncCopy` pass, but scoped to a region
 * (e.g., a lowered parallel loop) rather than the whole PrimFunc.
 *
 * `enable_buffer_load_lds` enables the gfx950-specific routing that emits
 * tl::ptx_cp_async_lds for eligible 16-byte non-predicated shared-memory-
 * destined copies whose LDS index is lane-contiguous (no XOR swizzle). The
 * ROCm copy lowering pass passes this flag only when the target is gfx950+.
 */
PTXAsyncCopyInjectResult
InjectPTXAsyncCopy(const tvm::tir::Stmt &body, bool enable_auto_async_copy,
                   bool async_without_async_commit_wait = false,
                   bool enable_buffer_load_lds = false);

} // namespace tl
} // namespace tvm
