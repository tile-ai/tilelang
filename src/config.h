/*!
 * \file tl/config.h
 * \brief TileLang configuration utilities.
 */

#ifndef TVM_TL_CONFIG_H_
#define TVM_TL_CONFIG_H_

#include <tvm/ffi/optional.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {
namespace tl_config {

/*!
 * \brief Check if reducer plan decision logging is enabled. When on,
 * ReducerPlanAndMaterialize logs each epoch's chosen physical plan and the
 * narrow-plan rejection reason at INFO level (always DLOG'd otherwise).
 */
inline bool ReducerPlanVerboseEnabled() {
  auto ctxt = tvm::transform::PassContext::Current();
  return ctxt
      ->GetConfig("tl.enable_reducer_plan_verbose", ffi::Optional<Bool>())
      .value_or(Bool(false));
}

/*!
 * \brief Check if free-mode layout attempts are scored by the IO cost model
 *  (bytes x vector-width/coalescing penalty over fragment<->global copies)
 *  instead of by register count alone. Default on; set to false to restore
 *  the legacy register-count-only selection.
 */
inline bool LayoutCostModelEnabled() {
  auto ctxt = tvm::transform::PassContext::Current();
  return ctxt->GetConfig("tl.layout_cost_model", ffi::Optional<Bool>())
      .value_or(Bool(true));
}

/*!
 * \brief Check if vectorize planner verbose output is enabled.
 */
inline bool VectorizePlannerVerboseEnabled() {
  auto ctxt = tvm::transform::PassContext::Current();
  return ctxt
      ->GetConfig("tl.enable_vectorize_planner_verbose", ffi::Optional<Bool>())
      .value_or(Bool(false));
}

/*!
 * \brief Check if 256-bit vectorization is disabled.
 */
inline bool Vectorize256Disabled() {
  auto ctxt = tvm::transform::PassContext::Current();
  return ctxt->GetConfig("tl.disable_vectorize_256", ffi::Optional<Bool>())
      .value_or(Bool(false));
}

} // namespace tl_config
} // namespace tl
} // namespace tvm

#endif // TVM_TL_CONFIG_H_
