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

#include "./scheduled_tir.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>

#include <utility>

#include "./kernel_rewriter.h"
#include "./task_analysis.h"
#include "./task_annotations.h"
#include "ascend/transform/attr.h"
#include "layout/annotations.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;
using ffi::GetRef;

namespace {

bool IsSemanticTaskMarker(const RegionStmtNode *op) {
  return ScheduleMarkerKey(op) == attr::kAscendTask ||
         ScheduleMarkerKey(op) == attr::kAscendPerCoreTask;
}

RegionStmt RequireSemanticTaskMarker(const Stmt &stmt, const char *context) {
  const auto *task = SingleScheduleStatement(stmt).as<RegionStmtNode>();
  ICHECK(task != nullptr && IsSemanticTaskMarker(task))
      << context
      << " expected exactly one outer T.Task/T.PerCoreTask marker, got "
      << stmt;
  return GetRef<RegionStmt>(task);
}

Stmt AnnotateSemanticTaskCoreMask(const RegionStmt &task, CoreMask core_mask) {
  TaskMetadata metadata;
  metadata.core_mask = core_mask;
  return WithSchedulePayload(
      task.get(),
      MergeTaskMetadata(metadata, ScheduleMarkerPayload(task.get())));
}

Stmt StripScheduledGuards(Stmt body) {
  while (true) {
    body = SingleScheduleStatement(body);
    const auto *region = body.as<RegionStmtNode>();
    if (!region)
      return body;
    auto key = ScheduleMarkerKey(region);
    if (key != kScheduleCondition && !IsScheduleGuardAttribute(key))
      return body;
    body = UnwrapScheduleMarker(region);
  }
}

ScheduleUnitMetadata MakeScheduleUnitMetadata(const IRStructure *node) {
  ScheduleUnitMetadata metadata;
  metadata.Set(kScheduleUnitStage, node->GetStage());
  return metadata;
}

class ScheduledTIREncoder {
public:
  Stmt Encode(const std::vector<std::shared_ptr<IRStructure>> &root) {
    return EncodeList(root);
  }

private:
  Stmt EncodeList(const std::vector<std::shared_ptr<IRStructure>> &nodes) {
    Array<Stmt> statements;
    statements.reserve(nodes.size());
    for (const auto &node : nodes)
      statements.push_back(EncodeNode(node.get()));
    return SeqStmt(statements);
  }

  Stmt EncodeScheduleUnit(const IRStructure *node, Stmt body,
                          const Location &span) {
    body = WrapScheduledGuards(std::move(body), node->GetGuards());
    return MakeScheduleMarker(attr::kScheduleUnit,
                              MakeScheduleUnitMetadata(node), std::move(body),
                              span);
  }

  Stmt EncodeNode(const IRStructure *node) {
    ICHECK(node != nullptr);
    if (node->IsTask())
      return EncodeTask(static_cast<const TaskNode *>(node));

    const auto *control = static_cast<const ControlNode *>(node);
    For loop = control->control;
    loop.CopyOnWrite()->body = SeqStmt(EncodeList(control->children));
    return EncodeControl(control, std::move(loop));
  }

  Stmt EncodeTask(const TaskNode *task) {
    Stmt body = task->stmt;
    RegionStmt semantic_task =
        RequireSemanticTaskMarker(body, "EncodeScheduledTIR TaskNode");
    if (task->GetCoreMask() != kCoreUnassigned) {
      body = AnnotateSemanticTaskCoreMask(semantic_task, task->GetCoreMask());
    }
    return EncodeScheduleUnit(task, std::move(body), task->stmt->loc);
  }

  Stmt EncodeControl(const ControlNode *control, For loop) {
    return EncodeScheduleUnit(control, std::move(loop), control->control->loc);
  }
};

std::shared_ptr<TaskNode> MakePlainTaskNode() {
  return std::make_shared<TaskNode>();
}

std::shared_ptr<ControlNode> MakePlainControlNode() {
  auto control = std::make_shared<ControlNode>();
  control->task = MakePlainTaskNode();
  return control;
}

class ScheduledTIRDecoder : public TaskAwareConstrVisitor {
public:
  ScheduledTIRDecoder(const ConstrSet &outer_ctx,
                      ScheduledExtraInfoFactory make_extra_info,
                      LayoutMap layouts)
      : make_extra_info_(make_extra_info), layouts_(std::move(layouts)) {
    constr_stack_ = outer_ctx.constrs_;
  }

  std::vector<std::shared_ptr<IRStructure>> Decode(const Stmt &body) {
    return DecodeList(body, nullptr);
  }

private:
  std::vector<std::shared_ptr<IRStructure>> DecodeList(const Stmt &stmt,
                                                       IRStructure *parent) {
    std::vector<std::shared_ptr<IRStructure>> saved;
    saved.swap(result_);
    Visit(stmt);
    std::vector<std::shared_ptr<IRStructure>> decoded;
    decoded.swap(result_);
    result_.swap(saved);
    for (size_t i = 0; i < decoded.size(); ++i) {
      decoded[i]->SetIndex(i);
      decoded[i]->SetParent(parent);
    }
    return decoded;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const RegionStmtNode *op) final {
    ICHECK(IsScheduleUnit(op))
        << "DecodeScheduledTIR expected a tl.schedule_unit. Run "
           "MaterializeScheduleUnits first";
    GuardList guards;
    result_.push_back(DecodeGuardedNode(op, UnwrapScheduleMarker(op), &guards));
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const ForNode *) final {
    FatalUnmarkedTask("DecodeScheduledTIR", "serial loop",
                      "MaterializeScheduleUnits");
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const EvaluateNode *) final {
    FatalUnmarkedTask("DecodeScheduledTIR", "Evaluate",
                      "MaterializeScheduleUnits");
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorStoreNode *) final {
    FatalUnmarkedTask("DecodeScheduledTIR", "TensorStore",
                      "MaterializeScheduleUnits");
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const BindNode *) final {
    FatalUnmarkedTask("DecodeScheduledTIR", "Bind", "MaterializeScheduleUnits");
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const WhileNode *) final {
    FatalUnmarkedTask("DecodeScheduledTIR", "While",
                      "MaterializeScheduleUnits");
  }

  std::shared_ptr<IRStructure> DecodeGuardedNode(const RegionStmtNode *marker,
                                                 const Stmt &statement,
                                                 GuardList *guards) {
    Stmt body = SingleScheduleStatement(statement);
    const auto *region = body.as<RegionStmtNode>();
    if (!region)
      return DecodeNode(marker, body, std::move(*guards));
    auto key = ScheduleMarkerKey(region);
    if (key == attr::kAscendTask || key == attr::kAscendPerCoreTask)
      return DecodeNode(marker, body, std::move(*guards));
    if (key == kScheduleCondition) {
      auto condition = ScheduleMarkerPayload(region).as_or_throw<PrimExpr>();
      guards->push_back(std::make_unique<ConditionGuard>(condition));
      auto scope = MakeGuard(condition);
      return DecodeGuardedNode(marker, UnwrapScheduleMarker(region), guards);
    }
    ICHECK(IsScheduleGuardAttribute(key))
        << "DecodeScheduledTIR found an unsupported guard: " << key;
    auto payload = ScheduleMarkerPayload(region);
    guards->push_back(
        std::make_unique<AttributeGuard>(payload, key, region->loc));
    auto scope = MakeGuard(payload.as_or_throw<PrimExpr>(), true);
    return DecodeGuardedNode(marker, UnwrapScheduleMarker(region), guards);
  }

  std::shared_ptr<IRStructure> DecodeNode(const RegionStmtNode *marker,
                                          const Stmt &body, GuardList guards) {
    const auto *loop_node = body.as<ForNode>();
    if (loop_node == nullptr)
      return DecodeTask(marker, body, std::move(guards));

    ICHECK(loop_node->kind == ForKind::kDefault ||
           loop_node->kind == ForKind::kUnrolled)
        << "Scheduled control marker contains a non-serial/unrolled loop";

    auto control = MakeControlNode();
    control->SetStage(GetScheduleUnitStage(marker));
    control->SetGuards(std::move(guards));
    control->control = GetRef<For>(loop_node);
    For task_loop = control->control;
    task_loop.CopyOnWrite()->body = SeqStmt(Evaluate(IntImm::Int32(0)));
    control->task->stmt = task_loop;
    control->task->outer_ctx = GetConstrSet();
    for (const PrimExpr &expression :
         {loop_node->begin, loop_node->end, loop_node->step})
      ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(expression)),
                        control->task.get());
    ApplyGuardAccesses(control.get(), control->task.get());
    {
      auto loop_scope = MakeGuard(loop_node->var, ScheduleLoopRange(loop_node));
      auto non_empty_scope = MakeGuard(ScheduleLoopTripCount(loop_node) > 0);
      control->children = DecodeList(loop_node->body, control.get());
    }
    control->InitializeLoopBodyContext();
    return control;
  }

  std::shared_ptr<IRStructure> DecodeTask(const RegionStmtNode *marker,
                                          const Stmt &body, GuardList guards) {
    RegionStmt semantic_task = RequireSemanticTaskMarker(
        body, "DecodeScheduledTIR leaf schedule unit");
    auto task = MakeTaskNode();
    task->stmt = body;
    task->SetStage(GetScheduleUnitStage(marker));
    TaskMetadata metadata =
        ParseTaskMetadata(ScheduleMarkerPayload(semantic_task.get()));
    if (metadata.core_mask.has_value())
      task->SetCoreMask(metadata.core_mask.value());
    task->SetGuards(std::move(guards));
    task->outer_ctx = GetConstrSet();
    ApplyTaskResourceUsage(AnalyzeTaskResourceUsage(body), task.get());
    ApplyTaskAccesses(AnalyzeTaskAccesses(body, layouts_), task.get());
    uint16_t pipe_mask = task->GetPipeMask();
    if (ScheduleMarkerKey(semantic_task.get()) == attr::kAscendPerCoreTask) {
      ICHECK_NE(pipe_mask, 0)
          << "Cannot recover the pipe of a scheduled T.PerCoreTask";
      task->MarkPerCoreTask();
    } else {
      task->SetSpecialWriteMask(AnalyzeSpecialRegisterWrites(body));
    }
    if (metadata.latency.has_value())
      task->SetLatency(metadata.latency.value());
    if (metadata.ii.has_value())
      task->SetII(metadata.ii.value());
    ApplyGuardAccesses(task.get(), task.get());
    return task;
  }

  std::shared_ptr<TaskNode> MakeTaskNode() const {
    if (make_extra_info_)
      return std::make_shared<TaskNode>(make_extra_info_());
    return MakePlainTaskNode();
  }

  std::shared_ptr<ControlNode> MakeControlNode() const {
    std::shared_ptr<ControlNode> control =
        make_extra_info_ ? std::make_shared<ControlNode>(make_extra_info_())
                         : MakePlainControlNode();
    if (make_extra_info_)
      control->task = MakeTaskNode();
    control->task->SetParent(control.get());
    return control;
  }

  void ApplyGuardAccesses(const IRStructure *node, TaskNode *task) {
    for (const auto &guard : node->GetGuards()) {
      if (guard->IsCondition()) {
        const auto *condition =
            static_cast<const ConditionGuard *>(guard.get());
        ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(condition->condition)),
                          task);
        continue;
      }
      const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
      if (auto expression = attribute->node.as<PrimExpr>()) {
        ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(expression.value())),
                          task);
      }
    }
  }

  std::vector<std::shared_ptr<IRStructure>> result_;
  ScheduledExtraInfoFactory make_extra_info_;
  LayoutMap layouts_;
};

void ValidateBufferVersions(const BufferVersionMap &buffer_versions,
                            const char *metadata_name) {
  for (const auto &[data, num_versions] : buffer_versions) {
    ICHECK_GT(num_versions, 0) << metadata_name << " version count for " << data
                               << " must be positive";
  }
}

void ValidateBufferVersionModes(const BufferVersionModeMap &buffer_modes) {
  for (const auto &[data, mode] : buffer_modes) {
    ICHECK(mode == "auto" || mode == "iteration" || mode == "counter")
        << "Buffer version mode for " << data
        << " must be auto, iteration, or counter, got " << mode;
  }
}

} // namespace

std::optional<CoreMask>
GetScheduledCoreMask(const RegionStmtNode *schedule_unit) {
  ICHECK(IsScheduleUnit(schedule_unit))
      << "Expected a `tl.schedule_unit` RegionStmt, got "
      << ScheduleMarkerKey(schedule_unit);
  Stmt payload = StripScheduledGuards(UnwrapScheduleMarker(schedule_unit));
  if (const auto *loop = payload.as<ForNode>()) {
    ICHECK(loop->kind == ForKind::kDefault || loop->kind == ForKind::kUnrolled)
        << "Scheduled control marker contains a non-serial/unrolled loop";
    return std::nullopt;
  }
  RegionStmt semantic_task =
      RequireSemanticTaskMarker(payload, "Scheduled leaf unit");
  return ParseTaskMetadata(ScheduleMarkerPayload(semantic_task.get()))
      .core_mask;
}

CoreMask RequireScheduledCoreMask(const RegionStmtNode *schedule_unit) {
  std::optional<CoreMask> core_mask = GetScheduledCoreMask(schedule_unit);
  ICHECK(core_mask.has_value())
      << "Scheduled unit has no core_mask-bearing T.Task";
  return core_mask.value();
}

RegionStmt EncodeScheduledTIR(ScheduledTIR scheduled_tir) {
  ValidateBufferVersions(scheduled_tir.metadata.buffer_versions,
                         "Scheduled buffer");
  ValidateBufferVersions(scheduled_tir.metadata.manual_buffer_versions,
                         "Manual buffer");
  ValidateBufferVersionModes(scheduled_tir.metadata.buffer_version_modes);
  ICHECK(scheduled_tir.metadata.kernel_root.defined())
      << "Scheduled TIR metadata is missing its kernel root shell";

  Stmt body = ScheduledTIREncoder().Encode(scheduled_tir.tree);
  RegionStmt rewritten = scheduled_tir.metadata.kernel_root;
  RegionStmtNode *node = rewritten.CopyOnWrite();
  node->body = SeqStmt(std::move(body));

  Map<String, Any> annotations = node->attrs->dict;
  annotations.erase(kBufferVersionsMap);
  annotations.erase(kManualMultiBuffer);
  annotations.erase(kBufferVersionMode);
  annotations.erase(kUnlimitMemoryScopes);
  annotations.erase(kBufferAliasMap);
  annotations.erase(kVectorCount);
  annotations.erase(kRootConflictHints);
  if (!scheduled_tir.metadata.buffer_versions.empty()) {
    annotations.Set(kBufferVersionsMap,
                    std::move(scheduled_tir.metadata.buffer_versions));
  }
  if (!scheduled_tir.metadata.manual_buffer_versions.empty()) {
    annotations.Set(kManualMultiBuffer,
                    std::move(scheduled_tir.metadata.manual_buffer_versions));
  }
  if (!scheduled_tir.metadata.buffer_version_modes.empty()) {
    annotations.Set(kBufferVersionMode,
                    std::move(scheduled_tir.metadata.buffer_version_modes));
  }
  if (!scheduled_tir.metadata.unlimit_memory_scopes.empty()) {
    annotations.Set(kUnlimitMemoryScopes,
                    std::move(scheduled_tir.metadata.unlimit_memory_scopes));
  }
  if (scheduled_tir.metadata.has_buffer_aliases) {
    ValidateBufferAliasMap(scheduled_tir.metadata.buffer_aliases);
    annotations.Set(kBufferAliasMap,
                    std::move(scheduled_tir.metadata.buffer_aliases));
  }
  if (scheduled_tir.metadata.num_aiv_subcores.has_value()) {
    annotations.Set(kVectorCount,
                    scheduled_tir.metadata.num_aiv_subcores.value());
  }
  if (!scheduled_tir.metadata.root_conflict_hints.empty()) {
    annotations.Set(kRootConflictHints,
                    std::move(scheduled_tir.metadata.root_conflict_hints));
  }
  node->attrs = DictAttrs(std::move(annotations));
  return rewritten;
}

ScheduledTIR DecodeScheduledTIR(const RegionStmt &root,
                                const ConstrSet &outer_ctx,
                                ScheduledExtraInfoFactory make_extra_info) {
  ScheduledTIR scheduled_tir;
  LayoutMap layouts;
  if (auto value = root->attrs->dict.Get("layout_map"))
    layouts = value.value().cast<LayoutMap>();
  layouts = CollectLayoutHints(root->body, layouts);
  auto decoder = ffi::make_object<ScheduledTIRDecoder>(
      outer_ctx, make_extra_info, std::move(layouts));
  scheduled_tir.tree = decoder->Decode(root->body);
  auto storage = ffi::make_object<StorageAliasAnalysis>();
  std::function<void(const std::vector<std::shared_ptr<IRStructure>> &)>
      collect_storage = [&](const auto &nodes) {
        for (const auto &node : nodes) {
          node->SetStorageAnalysis(storage);
          if (node->IsTask()) {
            storage->Visit(static_cast<TaskNode *>(node.get())->stmt);
          } else {
            auto *control = static_cast<ControlNode *>(node.get());
            control->task->SetStorageAnalysis(storage);
            collect_storage(control->children);
          }
        }
      };
  collect_storage(scheduled_tir.tree);
  scheduled_tir.metadata.kernel_root = root;
  if (auto value = root->attrs->dict.Get(kBufferVersionsMap)) {
    scheduled_tir.metadata.buffer_versions =
        value.value().cast<BufferVersionMap>();
    ValidateBufferVersions(scheduled_tir.metadata.buffer_versions,
                           "Scheduled buffer");
  }
  if (auto value = root->attrs->dict.Get(kManualMultiBuffer)) {
    scheduled_tir.metadata.manual_buffer_versions =
        value.value().cast<BufferVersionMap>();
    ValidateBufferVersions(scheduled_tir.metadata.manual_buffer_versions,
                           "Manual buffer");
  }
  if (auto value = root->attrs->dict.Get(kBufferVersionMode)) {
    scheduled_tir.metadata.buffer_version_modes =
        value.value().cast<BufferVersionModeMap>();
    ValidateBufferVersionModes(scheduled_tir.metadata.buffer_version_modes);
  }
  if (auto value = root->attrs->dict.Get(kUnlimitMemoryScopes)) {
    scheduled_tir.metadata.unlimit_memory_scopes =
        value.value().cast<Array<String>>();
  }
  if (auto value = root->attrs->dict.Get(kBufferAliasMap)) {
    scheduled_tir.metadata.buffer_aliases =
        value.value().cast<BufferAliasMap>();
    ValidateBufferAliasMap(scheduled_tir.metadata.buffer_aliases);
    scheduled_tir.metadata.has_buffer_aliases = true;
  }
  if (auto value = root->attrs->dict.Get(kVectorCount)) {
    auto count = GetIntAnnotation(value.value());
    ICHECK(count && (*count == 1 || *count == 2))
        << "Mixed-kernel vector_count must be the constant integer 1 or 2, got "
        << value.value();
    scheduled_tir.metadata.num_aiv_subcores = static_cast<int>(*count);
  }
  if (auto value = root->attrs->dict.Get(kRootConflictHints)) {
    scheduled_tir.metadata.root_conflict_hints =
        value.value().cast<Array<Any>>();
  }
  return scheduled_tir;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.ascend.RoundTripScheduledTIR", [](tirx::Function function) {
        return RewriteTilelangKernels(
            function, "RoundTripScheduledTIR",
            [](const TilelangKernelContext &context) {
              return EncodeScheduledTIR(
                  DecodeScheduledTIR(context.root, context.outer_ctx));
            });
      });
}

} // namespace ascend
} // namespace tl
} // namespace tvm
