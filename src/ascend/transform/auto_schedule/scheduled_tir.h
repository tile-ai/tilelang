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

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "../buffer_alias.h"
#include "../buffer_version.h"
#include "../core_mask.h"
#include "./ir_structure.h"
#include "ascend/transform/attr.h"
#include "support/check.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {
namespace ascend {

// Short-lived scheduled TIR contract from MaterializeScheduleUnits through
// LowerScheduledTIR. After native scheduling guards, every `tl.schedule_unit`
// contains exactly one T.Task/T.PerCoreTask or one serial/unrolled For whose
// body is another schedule-unit list. Stage lives on the schedule unit; core
// ownership always lives on the task, including compiler-generated sync tasks.
// Buffer version counts live once on each kernel's tilelang_root block under
// tl.buffer_versions_map.
constexpr const char *kUnlimitMemoryScopes = "tl.unlimit_memory_scopes";
constexpr const char *kVectorCount = "vector_count";
constexpr const char *kRootConflictHints = "tl.root_conflict_hints";

inline Stmt WrapScheduledGuards(Stmt body, const GuardList &guards) {
  for (auto it = guards.rbegin(); it != guards.rend(); ++it)
    body = (*it)->Wrap(std::move(body));
  return body;
}

// Kernel-level state that is part of scheduled TIR but cannot live on an
// IRStructure node. The codec preserves this state exactly; each pass owns any
// semantic transition between input and output snapshots.
struct ScheduledTIRMetadata {
  // Kernel root shell carrying fields that are not represented by IRStructure.
  RegionStmt kernel_root{ffi::UnsafeInit{}};
  // Frontend overrides before AutoSchedule; selected versions afterward.
  BufferVersionMap buffer_versions;
  // User-declared physical versions before MaterializeMultiBuffer; all
  // physical versions afterward.
  BufferVersionMap manual_buffer_versions;
  // Frontend-selected physical-index strategy, consumed by PrepareMultiBuffer.
  BufferVersionModeMap buffer_version_modes;
  // Memory scopes whose scheduler capacity limit is disabled.
  ffi::Array<ffi::String> unlimit_memory_scopes;
  // Positive storage-alias proof produced by InsertSync. An explicit presence
  // bit distinguishes a valid empty contract from pre-InsertSync snapshots.
  BufferAliasMap buffer_aliases;
  bool has_buffer_aliases{false};
  // Number of AIV subcores requested by T.MixedKernel. Absent for T.Kernel.
  std::optional<int> num_aiv_subcores;
  // Conflict declarations attached to the kernel's outermost sequence.
  ffi::Array<ffi::Any> root_conflict_hints;
};

// Decoded scheduled TIR. Passes transform this pair and leave serialization to
// the shared codec.
struct ScheduledTIR {
  std::vector<std::shared_ptr<IRStructure>> tree;
  ScheduledTIRMetadata metadata;
};

using ScheduledExtraInfoFactory = std::unique_ptr<IRExtraInfo> (*)();

RegionStmt EncodeScheduledTIR(ScheduledTIR scheduled_tir);

ScheduledTIR
DecodeScheduledTIR(const RegionStmt &root, const ConstrSet &outer_ctx,
                   ScheduledExtraInfoFactory make_extra_info = nullptr);

inline int64_t RequireScheduleUnitInt(const RegionStmtNode *op,
                                      const char *key) {
  std::optional<int64_t> value = GetScheduleUnitInt(op, key);
  ICHECK(value.has_value())
      << "Scheduled node is missing integer metadata `" << key << "`";
  return value.value();
}

std::optional<CoreMask>
GetScheduledCoreMask(const RegionStmtNode *schedule_unit);

CoreMask RequireScheduledCoreMask(const RegionStmtNode *schedule_unit);

} // namespace ascend
} // namespace tl
} // namespace tvm
