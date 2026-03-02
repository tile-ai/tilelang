#pragma once

#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

/*! \brief Inject PTX cp.async lowering patterns into a statement.
 *
 * This is the statement-level entrypoint used by other transforms to apply the
 * same rewrite as the `tl.LowerPTXAsyncCopy` pass, but scoped to a region
 * (e.g., a lowered parallel loop) rather than the whole PrimFunc.
 */
tvm::tir::Stmt InjectPTXAsyncCopy(const tvm::tir::Stmt &body,
                                  bool enable_auto_async_copy);

} // namespace tl
} // namespace tvm
