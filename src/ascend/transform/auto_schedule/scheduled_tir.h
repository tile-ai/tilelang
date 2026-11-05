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

#include "../buffer_alias.h"
#include "../buffer_version.h"
#include "ir_structure.h"
#include "native_task_ir.h"

#include <memory>
#include <optional>
#include <vector>

namespace tvm::tl::ascend {

inline constexpr const char *kUnlimitMemoryScopes = "tl.unlimit_memory_scopes";
inline constexpr const char *kVectorCount = "vector_count";
inline constexpr const char *kRootConflictHints = "tl.root_conflict_hints";

inline Stmt WrapScheduledGuards(Stmt body, const GuardList &guards) {
  for (auto it = guards.rbegin(); it != guards.rend(); ++it)
    body = (*it)->Wrap(std::move(body));
  return body;
}

// Dynamic metadata remains on native operand-bearing carriers at its lexical
// definition site, including definitions in nested loops and branches. New
// storage entries follow their own definitions; other kernel metadata stays at
// the root tail. Dynamic operands are never hoisted into the root arguments.
struct ScheduledTIRMetadata {
  RegionStmt kernel_root{ffi::UnsafeInit{}};
  ConstrSet outer_ctx;
  BufferVersionMap buffer_versions;
  BufferVersionMap manual_buffer_versions;
  BufferVersionModeMap buffer_version_modes;
  ffi::Array<ffi::String> unlimit_memory_scopes;
  BufferAliasMap buffer_aliases;
  bool has_buffer_aliases{false};
  std::optional<int> num_aiv_subcores;
  // Analysis aggregate only. Existing operand-bearing carriers retain their
  // own lists; clearing the aggregate deliberately consumes all those lists.
  ffi::Array<ffi::Any> root_conflict_hints;
  ffi::Array<ffi::Any> original_root_conflict_hints;
};

struct ScheduledTIR {
  std::vector<std::shared_ptr<IRStructure>> tree;
  ScheduledTIRMetadata metadata;
};

using ScheduledExtraInfoFactory = std::unique_ptr<IRExtraInfo> (*)();

RegionStmt EncodeScheduledTIR(ScheduledTIR scheduled_tir);
ScheduledTIR
DecodeScheduledTIR(const RegionStmt &root, const ConstrSet &outer_ctx,
                   ScheduledExtraInfoFactory make_extra_info = nullptr,
                   const ScheduleUnitMetadata &kernel_annotations = {},
                   NativeParameterAliasContract parameter_alias_contract = {});

std::optional<CoreMask> GetScheduledCoreMask(const Stmt &schedule_unit);
CoreMask RequireScheduledCoreMask(const Stmt &schedule_unit);

} // namespace tvm::tl::ascend
