/*!
 * \file role_based_scheduler.h
 * \brief The "role_based" automatic warp-specialization scheduler.
 */
#pragma once

#include "./common.h"

namespace tvm {
namespace tl {

// Fixed-role heuristic: classify ops by lowering eligibility (Load / MMA /
// Store / Worker), pull warp-private def-use chains into their consumers'
// roles, and pipeline every shared / tensor-memory storage handed across
// roles. See role_based_scheduler.cc for the full model.
ffi::Optional<WSSchedule> RoleBasedSchedule(const tirx::SBlock &block,
                                            const tirx::Stmt &body,
                                            int worker_threads,
                                            const Target &target);

} // namespace tl
} // namespace tvm
