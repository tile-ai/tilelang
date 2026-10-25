/*!
 * \file role_based_scheduler.h
 * \brief The "role_based" automatic warp-specialization scheduler.
 */
#pragma once

#include "./common.h"
#include <tvm/tirx/function.h>

namespace tvm {
namespace tl {
namespace cuda {

ffi::Array<Var> ScheduleReadOnlyStorage(const tirx::Function &function);

// Fixed-role heuristic: classify ops by lowering eligibility (Load / MMA /
// Store / Worker), pull warp-private def-use chains into their consumers'
// roles, and pipeline every shared / tensor-memory storage handed across
// roles. See role_based_scheduler.cc for the full model.
ffi::Optional<WSSchedule> RoleBasedSchedule(const RegionStmt &block,
                                            const Stmt &body,
                                            int worker_threads,
                                            const Target &target);

} // namespace cuda
} // namespace tl
} // namespace tvm
