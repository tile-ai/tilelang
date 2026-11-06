/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*! \file ascend/ir/task.h
 *  \brief Ascend task, schedule-unit and flat scheduled binding semantics.
 */
#ifndef TILELANG_ASCEND_IR_TASK_H_
#define TILELANG_ASCEND_IR_TASK_H_

#include "ascend/ir/core_mask.h"
#include "ir/annotations.h"
#include "support/check.h"
#include <optional>
#include <tvm/tirx/stmt.h>

namespace tvm::tl::ascend {
using namespace tirx;
using namespace prim;
using ScheduleUnitMetadata = ffi::Map<ffi::String, ffi::Any>;
inline constexpr const char *kScheduleUnitStage = "stage";
inline constexpr int kUnscheduledStage = -1;

struct TaskMetadata {
  std::optional<int64_t> latency;
  std::optional<int64_t> ii;
  std::optional<CoreMask> core_mask;
};

namespace task_metadata_detail {

inline ffi::Map<ffi::String, ffi::Any>
CopyTaskMetadata(const ffi::Any &existing) {
  ffi::Map<ffi::String, ffi::Any> result;
  if (const auto *map = existing.as<ffi::MapObj>()) {
    for (const auto &[key, value] : *map)
      result.Set(key.cast<ffi::String>(), value);
  }
  return result;
}

inline IntImm RequireConstantTaskMetadataInt(const ffi::Any &value_any,
                                             const ffi::String &key) {
  PrimExpr value_expr = value_any.cast<PrimExpr>();
  const auto *value = value_expr.as<IntImmObj>();
  ICHECK(value != nullptr) << "T.Task " << key
                           << " must be a constant integer, got " << value_expr;
  TVM_FFI_CHECK(value->value.as<int64_t>().has_value() &&
                    value_expr.ty().IsScalar() &&
                    !value_expr.ty().MatchesCode(kDLBool),
                ValueError)
      << "T.Task " << key << " must be a scalar constant integer";
  return ffi::GetRef<IntImm>(value);
}

} // namespace task_metadata_detail

// Parse recognized fields; other passes retain ownership of unknown fields.
// Legacy scalar payloads are interpreted only by source normalization.
inline TaskMetadata ParseTaskMetadata(const ffi::Any &node) {
  TaskMetadata result;
  const auto *map = node.as<ffi::MapObj>();
  if (map == nullptr)
    return result;
  for (const auto &[key_any, value_any] : *map) {
    ffi::String key = key_any.cast<ffi::String>();
    if (key != "latency" && key != "ii" && key != "core_mask")
      continue;
    const auto value =
        task_metadata_detail::RequireConstantTaskMetadataInt(value_any, key);
    if (key == "latency") {
      ICHECK_GE(value->value, 0) << "T.Task latency must be non-negative";
      result.latency = value->value.as<int64_t>().value();
    } else if (key == "ii") {
      ICHECK_GT(value->value, 0) << "T.Task ii must be positive";
      result.ii = value->value.as<int64_t>().value();
    } else {
      ICHECK_GT(value->value, 0)
          << "T.Task core_mask must select at least one Ascend core";
      ICHECK_LE(value->value, static_cast<int64_t>(kCoreAll))
          << "T.Task core_mask exceeds the supported AIV/AIC mask";
      CoreMask core_mask =
          static_cast<CoreMask>(value->value.as<int64_t>().value());
      ICHECK(IsValidCoreMask(core_mask))
          << "T.Task has invalid core_mask " << value->value;
      result.core_mask = core_mask;
    }
  }
  return result;
}

// Merge the defined fields into the task annotations while preserving
// metadata owned by other compiler passes.
inline ffi::Any MergeTaskMetadata(const TaskMetadata &metadata,
                                  const ffi::Any &existing) {
  ffi::Map<ffi::String, ffi::Any> result =
      task_metadata_detail::CopyTaskMetadata(existing);
  if (metadata.latency.has_value()) {
    ICHECK_GE(metadata.latency.value(), 0);
    result.Set("latency", IntImm(PrimType::Int(64), metadata.latency.value()));
  }
  if (metadata.ii.has_value()) {
    ICHECK_GT(metadata.ii.value(), 0);
    result.Set("ii", IntImm(PrimType::Int(64), metadata.ii.value()));
  }
  if (metadata.core_mask.has_value()) {
    ICHECK_NE(metadata.core_mask.value(), kCoreUnassigned);
    ICHECK(IsValidCoreMask(metadata.core_mask.value()));
    result.Set("core_mask",
               IntImm(PrimType::Int(64), metadata.core_mask.value()));
  }
  return result;
}

// Operator identities are owned by this dialect; consumers compare Op refs.
TVM_TIRX_DLL const Op &task_op();
TVM_TIRX_DLL const Op &per_core_task_op();
TVM_TIRX_DLL const Op &schedule_unit_op();
TVM_TIRX_DLL const Op &stage_op();
TVM_TIRX_DLL const Op &scheduled_bind_op();

// Semantic regions have no body parameters or result variables. Static task
// annotations live in attrs["annotations"]; expression-dependent metadata is
// one explicit Tuple operand described by attrs["annotation_schema"].
TVM_TIRX_DLL bool IsTask(const RegionStmtObj *region);
TVM_TIRX_DLL bool IsPerCoreTask(const RegionStmtObj *region);
TVM_TIRX_DLL bool IsSemanticTask(const RegionStmtObj *region);
TVM_TIRX_DLL bool IsScheduleUnit(const RegionStmtObj *region);
TVM_TIRX_DLL bool IsStageScope(const RegionStmtObj *region);
TVM_TIRX_DLL ScheduleUnitMetadata TaskAnnotations(const RegionStmtObj *region);
TVM_TIRX_DLL RegionStmt MakeTask(const Stmt &body,
                                 const ScheduleUnitMetadata &metadata = {});
TVM_TIRX_DLL RegionStmt
MakePerCoreTask(const Stmt &body, const ScheduleUnitMetadata &metadata = {});
TVM_TIRX_DLL RegionStmt WithTaskAnnotations(RegionStmt task,
                                            const ffi::Any &metadata);
TVM_TIRX_DLL int ReadScheduleStage(const ffi::Any &metadata);
TVM_TIRX_DLL int GetScheduleUnitStage(const RegionStmtObj *region);
TVM_TIRX_DLL int GetScheduleUnitStage(const Stmt &stmt);
TVM_TIRX_DLL int GetStageScopeStage(const RegionStmtObj *region);
TVM_TIRX_DLL RegionStmt MakeScheduleUnit(const Stmt &body, int stage);
TVM_TIRX_DLL RegionStmt WithScheduleUnitStage(RegionStmt unit, int stage);
TVM_TIRX_DLL RegionStmt MakeStageScope(const Stmt &body, int stage);

inline bool IsScheduledBind(const CallObj *call) {
  return call && call->op.same_as(scheduled_bind_op());
}
inline bool IsScheduledBind(const BindObj *bind) {
  return bind && IsScheduledBind(bind->value.as<CallObj>());
}

// The surrounding Bind retains its definition scope. The opaque call evaluates
// exactly one nonvoid value once; wrapping the Bind in a zero-result region
// would hide definitions used by subsequent statements and TensorTypes.
struct ScheduledBinding {
  Expr value;
  ScheduleUnitMetadata task;
  int stage;
};
TVM_TIRX_DLL ScheduledBinding ReadScheduledBinding(const CallObj *call);
TVM_TIRX_DLL Bind MakeScheduledBinding(const Bind &bind, int stage,
                                       const ScheduleUnitMetadata &task = {});
TVM_TIRX_DLL Bind WithScheduledBindingMetadata(
    const Bind &bind, int stage, const ScheduleUnitMetadata &task);
TVM_TIRX_DLL std::optional<int> GetTaskSubcore(const Stmt &stmt);

// Lexical metadata declarations retain their original place in the sequence.
inline bool IsTaskMetadataStatement(const Stmt &stmt) {
  const auto *evaluate = stmt.as<EvaluateObj>();
  const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
  return IsAnnotationCall(call);
}

} // namespace tvm::tl::ascend
#endif // TILELANG_ASCEND_IR_TASK_H_
