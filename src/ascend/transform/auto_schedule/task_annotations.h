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

#pragma once

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/map.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <cstdint>
#include <optional>

#include "../core_mask.h"
#include "support/check.h"
#include "transform/common/native_layout_utils.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;
using namespace prim;

struct TaskCost {
  int64_t latency{0};
  int64_t ii{0};
};

// Metadata carried by T.Task between MaterializeScheduleUnits,
// AssignCore, EstimateLatency, AutoSchedule, PrepareMultiBuffer, ResolveCore,
// InsertSync, MaterializeMultiBuffer, and LowerScheduledTIR. The frontend
// helper controls which fields users may provide; backend consumers share this
// representation for user hints and compiler-produced fields.
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

inline const IntImmObj *
RequireConstantTaskMetadataInt(const ffi::Any &value_any,
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
  return value;
}

} // namespace task_metadata_detail

inline std::optional<int> GetTaskSubcore(const Stmt &stmt) {
  Stmt single = SingleStatement(stmt);
  ffi::Map<ffi::String, ffi::Any> metadata;
  if (const auto *bind = single.as<BindObj>()) {
    const auto *call = bind->value.as<CallObj>();
    if (!call || !call->op.as<OpObj>() ||
        call->op.as<OpObj>()->name != "tl.ascend.scheduled_bind")
      return std::nullopt;
    metadata = task_metadata_detail::CopyTaskMetadata(
        DecodeTileOpArguments(call).annotations.at("task"));
  } else {
    const auto *marker = single.as<RegionStmtObj>();
    if (!marker || marker->op->name != "tl.attr")
      return std::nullopt;
    auto fields = ReadNativeAnnotations(marker);
    auto key = fields.at("key").cast<ffi::String>();
    if (key != "tl.ascend_task" && key != "tl.ascend_per_core_task")
      return std::nullopt;
    metadata = task_metadata_detail::CopyTaskMetadata(fields.at("node"));
  }
  if (auto value = metadata.Get("subcore")) {
    int64_t sid =
        task_metadata_detail::RequireConstantTaskMetadataInt(*value, "subcore")
            ->value.as<int64_t>()
            .value();
    TVM_FFI_CHECK(sid == 0 || sid == 1, ValueError)
        << "T.Task subcore must be 0 or 1";
    return static_cast<int>(sid);
  }
  return std::nullopt;
}

// Parse recognized task metadata from the structured node payload. Legacy
// scalar payloads carry no metadata.
inline TaskMetadata ParseTaskMetadata(const ffi::Any &node) {
  TaskMetadata result;
  const auto *map = node.as<ffi::MapObj>();
  if (map == nullptr)
    return result;
  for (const auto &[key_any, value_any] : *map) {
    ffi::String key = key_any.cast<ffi::String>();
    if (key != "latency" && key != "ii" && key != "core_mask")
      continue;
    const auto *value =
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

// Merge the defined fields into the structured node payload while preserving
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

// Apply any latency/II overrides present on T.Task to an estimated cost.
inline TaskCost ResolveTaskCost(const TaskCost &estimated,
                                const TaskMetadata &metadata) {
  TaskCost result = estimated;
  if (metadata.latency.has_value()) {
    result.latency = metadata.latency.value();
  } else if (result.latency == 0) {
    // Keep unmodeled compiler-estimated tasks schedulable. An explicit zero
    // override remains subject to the latency >= II validation below.
    result.latency = 1;
  }
  if (metadata.ii.has_value())
    result.ii = metadata.ii.value();
  ICHECK_GE(result.latency, 0);
  ICHECK_GT(result.ii, 0);
  ICHECK_GE(result.latency, result.ii)
      << "T.Task latency must be greater than or equal to ii, got latency="
      << result.latency << " and ii=" << result.ii;
  return result;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
