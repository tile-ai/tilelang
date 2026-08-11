/*!
 * \file tl/transform/reducer/reducer_loop_layout.h
 * \brief Reducer-aware constraints for parallel-loop layout inference.
 */

#ifndef TVM_TL_TRANSFORM_REDUCER_REDUCER_LOOP_LAYOUT_H_
#define TVM_TL_TRANSFORM_REDUCER_REDUCER_LOOP_LAYOUT_H_

#include "op/operator.h"

#include <tvm/tirx/function.h>

namespace tvm {
namespace tl {

/*!
 * \brief Discover optional reducer-driven parallel-loop layout constraints.
 *
 * This analysis runs after an unconstrained baseline layout solve. It uses the
 * complete inferred destination and loop layouts to recognize proven
 * LocalComplete reducers, then returns their destination layouts as concrete
 * constraints for a fresh layout solve. Reducers that require projected or
 * canonical plans are deliberately omitted.
 */
ffi::Map<tirx::For, Fragment> DiscoverReducerLoopLayoutConstraints(
    const tirx::PrimFunc &func, const LayoutMap &inferred_layouts,
    const ffi::Map<tirx::For, Fragment> &inferred_loop_layouts);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_REDUCER_REDUCER_LOOP_LAYOUT_H_
