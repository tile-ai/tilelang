/*!
 * \file tl/config.h
 * \brief TileLang configuration utilities.
 */

#ifndef TVM_TL_CONFIG_H_
#define TVM_TL_CONFIG_H_

#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {
namespace tl_config {

/*!
 * \brief Check if vectorize planner verbose output is enabled.
 */
inline bool VectorizePlannerVerboseEnabled() {
  auto ctxt = transform::PassContext::Current();
  return ctxt
      ->GetConfig("tl.enable_vectorize_planner_verbose", Optional<Bool>())
      .value_or(Bool(false));
}

/*!
 * \brief Check if 256-bit vectorization is disabled.
 */
inline bool Vectorize256Disabled() {
  auto ctxt = transform::PassContext::Current();
  return ctxt->GetConfig("tl.disable_vectorize_256", Optional<Bool>())
      .value_or(Bool(false));
}

} // namespace tl_config
} // namespace tl
} // namespace tvm

#endif // TVM_TL_CONFIG_H_
