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

/*! \file ascend/transform/source_scopes.h
 *  \brief Interpretation of source annotations before their lowering boundary.
 */
#ifndef TILELANG_ASCEND_TRANSFORM_SOURCE_SCOPES_H_
#define TILELANG_ASCEND_TRANSFORM_SOURCE_SCOPES_H_

#include "ascend/ir/block.h"
#include "ascend/ir/task.h"
#include "ir/annotations.h"
#include "ir/stmt_functor.h"
#include <optional>

namespace tvm::tl::ascend {

// Only tl.attr source syntax uses this view. Task/schedule consumers use the
// formal task dialect and mutate native RegionStmt fields directly.
struct SourceAttribute {
  RegionStmt region;
  ffi::Any node;
  ffi::String key;
  ffi::Any value;

  RegionStmt WithBody(const Stmt &body) const {
    RegionStmt result = region;
    result.CopyOnWrite()->body = SeqStmt(body);
    return result;
  }
};

inline std::optional<SourceAttribute>
ReadSourceAttribute(const RegionStmtObj *region) {
  if (!region || !region->op.same_as(Op::Get("tl.attr")))
    return std::nullopt;
  TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                ValueError)
      << "Ascend source attributes have no body parameters or results";
  auto fields = ReadAnnotations(region);
  return SourceAttribute{ffi::GetRef<RegionStmt>(region), fields.at("node"),
                         fields.at("key").cast<ffi::String>(),
                         fields.at("value")};
}
inline std::optional<SourceAttribute> ReadSourceAttribute(const Stmt &stmt) {
  return ReadSourceAttribute(SingleStatement(stmt).as<RegionStmtObj>());
}
inline bool IsScheduleGuardAttribute(const ffi::String &key) {
  return key == "tilelang_assume" || key == "tl.assume_requires_runtime_check";
}
// Normalize legacy task attribute spellings once, before schedule analysis.
TVM_TIRX_DLL SeqStmt NormalizeTaskScopes(const SeqStmt &body);

} // namespace tvm::tl::ascend
#endif // TILELANG_ASCEND_TRANSFORM_SOURCE_SCOPES_H_
