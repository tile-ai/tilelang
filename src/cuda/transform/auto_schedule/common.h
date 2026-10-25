/*!
 * \file common.h
 * \brief Shared declarations of the AutoWarpSpecialization entrypoint and
 * schedulers.
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
namespace cuda {

using SchedulerFn = ffi::Optional<WSSchedule> (*)(const RegionStmt &block,
                                                  const Stmt &body,
                                                  int worker_threads,
                                                  const Target &target);

// An id value arrives as an ffi String (loop annotations) or as a
// StringImm (call annotations, whose values must be objects).
inline ffi::String ExtractOpId(const ffi::Any &value) {
  if (auto string = value.try_cast<ffi::String>())
    return string.value();
  if (const auto *imm = value.as<StringImmNode>())
    return imm->value;
  TVM_FFI_THROW(ValueError) << "AutoWarpSpecialization op id must be a string";
  return "";
}

inline ffi::Optional<ffi::String> GetWSOpId(const Stmt &statement) {
  if (const auto *region = statement.as<RegionStmtNode>()) {
    if (auto value = region->attrs->dict.Get(kWSOpIdKey))
      return ExtractOpId(*value);
  }
  if (const auto *loop = statement.as<ForNode>()) {
    if (auto value = loop->annotations.Get(kWSOpIdKey))
      return ExtractOpId(*value);
  }
  const CallNode *call = nullptr;
  if (const auto *binding = statement.as<BindNode>())
    call = binding->value.as<CallNode>();
  if (const auto *evaluation = statement.as<EvaluateNode>())
    call = evaluation->value.as<CallNode>();
  if (call && call->attrs) {
    if (auto attributes = call->attrs.value().as<DictAttrs>())
      if (auto value = attributes.value()->dict.Get(kWSOpIdKey))
        return ExtractOpId(*value);
  }
  return std::nullopt;
}

} // namespace cuda
} // namespace tl
} // namespace tvm
