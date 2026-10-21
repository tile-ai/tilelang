/*!
 * \file preprocess_ir.h
 * \brief Give every schedulable statement a stable "tl.ws_op_id" marker.
 */
#pragma once

#include "./common.h"

namespace tvm {
namespace tl {
namespace cuda {

// Give every schedulable statement a stable "tl.ws_op_id" marker in the
// forms MaterializeWSSchedule consumes; idempotent. Unschedulable
// constructs decline the kernel (warning + nullopt).
ffi::Optional<tirx::Stmt> PreprocessIR(tirx::Stmt body, const Target &target);

} // namespace cuda
} // namespace tl
} // namespace tvm
