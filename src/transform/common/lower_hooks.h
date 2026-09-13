/*!
 * \file tl/transform/common/lower_hooks.h
 * \brief Backend hook registries consumed by LowerTileOp.
 *
 * These registries let backend-owned translation units contribute behavior to
 * the shared LowerTileOp pass without the pass depending on any backend
 * header. Registration happens from static initializers in backend TUs that
 * are compiled into every build configuration alongside the code they hook.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_LOWER_HOOKS_H_
#define TVM_TL_TRANSFORM_COMMON_LOWER_HOOKS_H_

#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

/*!
 * \brief Backend hook applied to the result of lowering one parallel loop.
 *
 * \param lowered The statement produced by parallel-loop lowering.
 * \param original The pre-lowering loop; hooks read their driving
 *        annotations from it.
 * \param target The target the loop is lowered for.
 *
 * Hooks must return \p lowered unchanged when they do not apply.
 */
using LoweredParallelLoopHook = tirx::Stmt (*)(tirx::Stmt lowered,
                                               const tirx::ForNode *original,
                                               const Target &target);

void RegisterLoweredParallelLoopHook(LoweredParallelLoopHook hook);
const std::vector<LoweredParallelLoopHook> &LoweredParallelLoopHooks();

/*!
 * \brief Backend hook remapping the node payload of a registered AttrStmt key
 * when LowerTileOp rewrites buffer data vars.
 *
 * \param node The AttrStmt node payload after the generic mutation.
 * \param remap Returns the replacement for a Var (identity when unmapped).
 */
using AttrNodeRemapHook =
    ffi::Any (*)(const ffi::Any &node,
                 const std::function<tirx::Var(const tirx::Var &)> &remap);

void RegisterAttrNodeRemapHook(const std::string &attr_key,
                               AttrNodeRemapHook hook);
/*! \brief Returns nullptr when no hook is registered for \p attr_key. */
AttrNodeRemapHook LookupAttrNodeRemapHook(const std::string &attr_key);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_LOWER_HOOKS_H_
