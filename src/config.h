/*!
 * \file tl/config.h
 * \brief TileLang configuration utilities.
 */

#ifndef TVM_TL_CONFIG_H_
#define TVM_TL_CONFIG_H_

#include <tvm/ffi/optional.h>
#include <tvm/ir/transform.h>

#include <string>

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
 * \brief The cost model that ranks free-mode layout attempts. Valid
 *  values: "register-count" (default — total fragment register slots) and
 *  "io-aware" (bytes x vector-width/coalescing over fragment<->global
 *  traffic, registers as tiebreak).
 */
inline std::string LayoutCostModelName() {
  auto ctxt = tvm::transform::PassContext::Current();
  return ctxt->GetConfig("tl.layout_cost_model", ffi::Optional<ffi::String>())
      .value_or(ffi::String("register-count"));
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

/*!
 * \brief Check if ``#line`` directive emission from TIR source spans is
 * enabled. When on, C-family codegen maps generated statements back to
 * their Python source lines via ``#line N "file"`` (consumed by the
 * CodeGenCWithLineDirectives builders).
 */
inline bool EmitLineDirectivesEnabled() {
  auto ctxt = tvm::transform::PassContext::Current();
  return ctxt->GetConfig("tl.emit_line_directives", ffi::Optional<Bool>())
      .value_or(Bool(false));
}

} // namespace tl_config
} // namespace tl
} // namespace tvm

#endif // TVM_TL_CONFIG_H_
