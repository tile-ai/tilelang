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

#include "source_scopes.h"
#include "attr.h"
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl::ascend {
namespace {
class TaskSourceNormalizer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    auto mutated = StmtExprMutator::Mutate_(op, mode);
    const bool children_changed = !mutated.IsUnchanged();
    Stmt original = ffi::GetRef<Stmt>(op);
    RegionStmt region =
        std::move(mutated).ValueOrUnchanged(original).as_or_throw<RegionStmt>();
    auto source = ReadSourceAttribute(region.get());
    if (!source) {
      if (children_changed)
        return region;
      return ffi::Unchanged();
    }
    RegionStmt result = region;
    if (source->key == attr::kAscendTask)
      result = MakeTask(region->body,
                        task_metadata_detail::CopyTaskMetadata(source->node));
    else if (source->key == attr::kAscendPerCoreTask)
      result = MakePerCoreTask(
          region->body, task_metadata_detail::CopyTaskMetadata(source->node));
    else if (source->key == attr::kScheduleUnit)
      result = MakeScheduleUnit(region->body, ReadScheduleStage(source->node));
    else if (source->key == attr::kAscendStage) {
      int stage =
          ReadScheduleStage(ScheduleUnitMetadata{{"stage", source->node}});
      result = MakeStageScope(region->body, stage);
    }
    if (result.same_as(region)) {
      if (children_changed)
        return region;
      return ffi::Unchanged();
    }
    result.CopyOnWrite()->loc = op->loc;
    return result;
  }
};
} // namespace

SeqStmt NormalizeTaskScopes(const SeqStmt &body) {
  auto normalizer = ffi::make_object<TaskSourceNormalizer>();
  return normalizer->Mutate(body).ValueOrUnchanged(body);
}
} // namespace tvm::tl::ascend
