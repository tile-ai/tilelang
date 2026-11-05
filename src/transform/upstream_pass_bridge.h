/*!
 * \file upstream_pass_bridge.h
 * \brief TileLang-owned staging over the pinned compiler's internal functions.
 */
#ifndef TVM_TL_TRANSFORM_UPSTREAM_PASS_BRIDGE_H_
#define TVM_TL_TRANSFORM_UPSTREAM_PASS_BRIDGE_H_

#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>

namespace tvm::tl::upstream_pass_bridge {

// Host-side operands retained until final device resources are known. This
// region is consumed by tl.LowerDeviceKernelLaunch before packed ABI/codegen.
const Op &PendingCudaLaunchOp();

// Input: target-bound functions with native device regions. Output: host/device
// functions in SSA, with captures converted to native tensor/pointer
// parameters. Deliberately leaves launch ABI lowering until TileLang finishes
// device passes.
tvm::transform::Pass OutlineDeviceRegions();

// Input: outlined device bodies and final launch/resource metadata. Output:
// native call_ffi_kernel/call_extern calls and corresponding device attributes.
tvm::transform::Pass LowerDeviceKernelLaunch();

// Lower the native reduction operation and repair its definitions without
// constructing a temporary module or hiding the caller's pass context.
tirx::Function LowerThreadAllreduce(tirx::Function function);

} // namespace tvm::tl::upstream_pass_bridge
#endif // TVM_TL_TRANSFORM_UPSTREAM_PASS_BRIDGE_H_
