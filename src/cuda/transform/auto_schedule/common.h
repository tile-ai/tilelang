/*!
 * \file common.h
 * \brief Shared declarations of the AutoSchedule entrypoint and schedulers.
 *
 * A scheduler receives one eligible kernel — the root block and its body
 * normalized so every schedulable statement carries a "tl.ws_op_id" marker —
 * and returns a typed WSSchedule for MaterializeWSSchedule to lower, or
 * nothing, logging the reason on-site (visible with TVM_LOG_DEBUG=1). It
 * must not rewrite the kernel body.
 */
#pragma once

#include <tvm/ffi/optional.h>
#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>

#include "support/check.h"
#include "transform/common/warp_specialize.h"

namespace tvm {
namespace tl {

using SchedulerFn = ffi::Optional<WSSchedule> (*)(const tirx::SBlock &block,
                                                  const tirx::Stmt &body,
                                                  int worker_threads,
                                                  const Target &target);

// An id value arrives as an ffi String (loop annotations) or as a
// StringImm (call annotations, whose values must be objects).
inline ffi::String ExtractOpId(const ffi::Any &value) {
  if (auto string = value.try_cast<ffi::String>())
    return string.value();
  if (const auto *imm = value.as<tirx::StringImmNode>())
    return imm->value;
  TVM_FFI_THROW(ValueError) << "AutoSchedule op id must be a string";
  return "";
}

} // namespace tl
} // namespace tvm
