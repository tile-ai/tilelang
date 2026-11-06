/*!
 * \file transform/layout_inference/policy.h
 * \brief Explicit backend region policies for the shared native layout pass.
 */
#ifndef TILELANG_TRANSFORM_LAYOUT_INFERENCE_POLICY_H_
#define TILELANG_TRANSFORM_LAYOUT_INFERENCE_POLICY_H_

#include <tvm/tirx/stmt.h>
#include <tvm/tirx/transform.h>

#include <functional>

namespace tvm::tl {

// Backends may leave a native region to a later specialized lowering stage.
// The default pass traverses all regions and keeps its existing fusion rules.
// Policies inspect real IR; they do not remove bodies, change loop kinds, or
// move dynamic operands into attributes.
struct LayoutInferenceOptions {
  std::function<bool(const RegionStmtObj *)> opaque_region;
  std::function<bool(const ForObj *, const ffi::Array<RegionStmt> &)>
      preserve_parallel_nest;

  bool IsOpaque(const RegionStmtObj *region) const {
    return opaque_region && opaque_region(region);
  }
};

tvm::transform::Pass LayoutInferenceWithPolicy(LayoutInferenceOptions options);

} // namespace tvm::tl
#endif // TILELANG_TRANSFORM_LAYOUT_INFERENCE_POLICY_H_
