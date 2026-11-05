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

/*!
 * \file native_task_ir.h
 * \brief Native, temporary task syntax consumed by Ascend scheduled lowering.
 */

#pragma once

#include "../attr.h"
#include "task_annotations.h"
#include "transform/common/native_layout_utils.h"

#include <limits>
#include <optional>
#include <utility>

namespace tvm::tl::ascend {

using ScheduleUnitMetadata = ffi::Map<ffi::String, ffi::Any>;
inline constexpr const char *kScheduleUnitStage = "stage";
inline constexpr int kUnscheduledStage = -1;
inline constexpr const char *kScheduledBind = "tl.ascend.scheduled_bind";

// This is an analysis view of a native region, not an alternate IR node.
// Reconstruction re-encodes every dynamic field into explicit operands.
struct NativeTaskAttribute {
  RegionStmt region;
  ffi::Any node;
  ffi::String key;
  ffi::Any value;

  RegionStmt WithBody(const Stmt &body) const {
    return MakeNativeAttribute(node, key, value, AsSequence(body), region->loc);
  }
  RegionStmt WithNode(ffi::Any replacement) const {
    return MakeNativeAttribute(std::move(replacement), key, value, region->body,
                               region->loc);
  }
  static SeqStmt AsSequence(const Stmt &body) {
    if (auto sequence = body.as<SeqStmt>())
      return sequence.value();
    return SeqStmt(ffi::Array<Stmt>{body}, body->loc);
  }
};

inline std::optional<NativeTaskAttribute>
ReadTaskAttribute(const RegionStmtObj *region) {
  if (region == nullptr || region->op->name != "tl.attr")
    return std::nullopt;
  TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                ValueError)
      << "Ascend task attributes have no body parameters or results";
  auto fields = ReadNativeAnnotations(region);
  return NativeTaskAttribute{ffi::GetRef<RegionStmt>(region), fields.at("node"),
                             fields.at("key").cast<ffi::String>(),
                             fields.at("value")};
}

inline std::optional<NativeTaskAttribute> ReadTaskAttribute(const Stmt &stmt) {
  return ReadTaskAttribute(SingleStatement(stmt).as<RegionStmtObj>());
}

inline bool IsSemanticTaskKey(const ffi::String &key) {
  return key == attr::kAscendTask || key == attr::kAscendPerCoreTask;
}

inline bool IsScheduleGuardAttribute(const ffi::String &key) {
  return key == "tilelang_assume" || key == "tl.assume_requires_runtime_check";
}

inline std::optional<ffi::String> NativeBlockName(const RegionStmtObj *op) {
  auto attribute = ReadTaskAttribute(op);
  if (!attribute || attribute->key != "tl.sblock")
    return std::nullopt;
  return attribute->value.cast<ScheduleUnitMetadata>()
      .at("name")
      .cast<ffi::String>();
}

inline bool IsScheduleUnit(const RegionStmtObj *op) {
  auto attribute = ReadTaskAttribute(op);
  return attribute && attribute->key == attr::kScheduleUnit;
}

inline bool IsScheduledBind(const CallObj *call) {
  return call != nullptr && call->op.as<OpObj>() &&
         call->op.as<OpObj>()->name == kScheduledBind;
}

inline bool IsScheduledBind(const BindObj *bind) {
  return bind != nullptr && IsScheduledBind(bind->value.as<CallObj>());
}

// Flat definitions must remain in their native sequence. In particular, an
// alloc_tensor/decl_tensor Bind can define a TensorType used by the next Bind.
// A zero-result task region would hide that definition. This typed call marks
// its one value evaluation while the surrounding Bind keeps the original Var.
struct NativeScheduledBinding {
  Expr value;
  ScheduleUnitMetadata task;
  int stage;
};

NativeScheduledBinding ReadScheduledBinding(const CallObj *call);
Bind MakeScheduledBinding(const Bind &bind, int stage,
                          const ScheduleUnitMetadata &task = {});
Bind WithScheduledBindingMetadata(const Bind &bind, int stage,
                                  const ScheduleUnitMetadata &task);

inline int ReadScheduleStage(const ffi::Any &metadata) {
  auto map = metadata.cast<ScheduleUnitMetadata>();
  auto field = map.Get(kScheduleUnitStage);
  TVM_FFI_CHECK(field.has_value(), ValueError)
      << "Scheduled node is missing integer metadata `stage`";
  const auto *integer = task_metadata_detail::RequireConstantTaskMetadataInt(
      field.value(), kScheduleUnitStage);
  int64_t stage = integer->value.as<int64_t>().value();
  TVM_FFI_CHECK(stage >= kUnscheduledStage &&
                    stage <= std::numeric_limits<int>::max(),
                ValueError)
      << "Scheduled stage must be -1 or a non-negative int32 value";
  return static_cast<int>(stage);
}

inline int GetScheduleUnitStage(const RegionStmtObj *op) {
  auto attribute = ReadTaskAttribute(op);
  TVM_FFI_CHECK(attribute && attribute->key == attr::kScheduleUnit, ValueError)
      << "Expected a native tl.schedule_unit region";
  return ReadScheduleStage(attribute->node);
}

inline int GetScheduleUnitStage(const Stmt &stmt) {
  if (const auto *bind = stmt.as<BindObj>(); IsScheduledBind(bind))
    return ReadScheduledBinding(bind->value.as<CallObj>()).stage;
  return GetScheduleUnitStage(stmt.as<RegionStmtObj>());
}

inline RegionStmt MakeScheduleUnit(const Stmt &body, int stage) {
  return MakeNativeAttribute(
      ScheduleUnitMetadata{
          {kScheduleUnitStage, IntImm(PrimType::Int(64), stage)}},
      attr::kScheduleUnit, IntImm(PrimType::Int(32), 1),
      NativeTaskAttribute::AsSequence(body), body->loc);
}

inline RegionStmt MakeTask(const Stmt &body,
                           const ScheduleUnitMetadata &metadata = {}) {
  ParseTaskMetadata(metadata);
  return MakeNativeAttribute(metadata, attr::kAscendTask,
                             IntImm(PrimType::Int(32), 1),
                             NativeTaskAttribute::AsSequence(body), body->loc);
}

// Metadata statements are lexical directives, not hardware tasks. Keep their
// exact position in the sequence so all referenced definitions dominate them.
inline bool IsTaskMetadataStatement(const Stmt &stmt) {
  const auto *evaluate = stmt.as<EvaluateObj>();
  const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
  return call && call->op.same_as(Op::Get("tl.annotation"));
}

} // namespace tvm::tl::ascend
