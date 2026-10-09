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
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <utility>

#include "./task_analysis.h"
#include "./task_annotations.h"
#include "ascend/transform/attr.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;
using ffi::GetRef;

namespace {

bool IsSemanticTaskMarker(const AttrStmtNode *op) {
  return op->attr_key == attr::kAscendTask ||
         op->attr_key == attr::kAscendPerCoreTask;
}

AttrStmt RequireSemanticTaskMarker(const Stmt &stmt, const char *context) {
  const auto *task = stmt.as<AttrStmtNode>();
  ICHECK(task != nullptr && IsSemanticTaskMarker(task))
      << context
      << " expected exactly one outer T.Task/T.PerCoreTask marker, got "
      << stmt;
  return GetRef<AttrStmt>(task);
}

Stmt AnnotateSemanticTaskCoreMask(const AttrStmt &task, CoreMask core_mask) {
  TaskMetadata metadata;
  metadata.core_mask = core_mask;
  return AttrStmt(MergeTaskMetadata(metadata, task->node), task->attr_key,
                  task->value, task->body, task->span);
}

Stmt StripScheduledGuards(Stmt body) {
  while (true) {
    if (const auto *condition = body.as<IfThenElseNode>()) {
      ICHECK(!condition->else_case.defined())
          << "Scheduled condition guard must not have an else branch";
      body = condition->then_case;
      continue;
    }
    if (const auto *attribute = body.as<AttrStmtNode>()) {
      if (!IsScheduleGuardAttribute(attribute->attr_key))
        break;
      body = attribute->body;
      continue;
    }
    break;
  }
  return body;
}

ScheduleUnitMetadata MakeScheduleUnitMetadata(const IRStructure *node) {
  ScheduleUnitMetadata metadata;
  metadata.Set(kScheduleUnitStage, Integer(node->GetStage()));
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
    return SeqStmt::Flatten(statements);
  }

  Stmt EncodeScheduleUnit(const IRStructure *node, Stmt body,
                          const Span &span) {
    body = WrapScheduledGuards(std::move(body), node->GetGuards());
    return AttrStmt(MakeScheduleUnitMetadata(node), attr::kScheduleUnit,
                    Integer(1), std::move(body), span);
  }

  Stmt EncodeNode(const IRStructure *node) {
    ICHECK(node != nullptr);
    if (node->IsTask())
      return EncodeTask(static_cast<const TaskNode *>(node));

    const auto *control = static_cast<const ControlNode *>(node);
    For loop = control->control;
    loop.CopyOnWrite()->body = EncodeList(control->children);
    return EncodeControl(control, std::move(loop));
  }

  Stmt EncodeTask(const TaskNode *task) {
    Stmt body = task->stmt;
    AttrStmt semantic_task =
        RequireSemanticTaskMarker(body, "EncodeScheduledTIR TaskNode");
    if (task->GetCoreMask() != kCoreUnassigned) {
      body = AnnotateSemanticTaskCoreMask(semantic_task, task->GetCoreMask());
    }
    return EncodeScheduleUnit(task, std::move(body), task->stmt->span);
  }

  Stmt EncodeControl(const ControlNode *control, For loop) {
    return EncodeScheduleUnit(control, std::move(loop), control->control->span);
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
                      ScheduledExtraInfoFactory make_extra_info)
      : make_extra_info_(make_extra_info) {
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
    VisitStmt(stmt);
    std::vector<std::shared_ptr<IRStructure>> decoded;
    decoded.swap(result_);
    result_.swap(saved);
    for (size_t i = 0; i < decoded.size(); ++i) {
      decoded[i]->SetIndex(i);
      decoded[i]->SetParent(parent);
    }
    return decoded;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    ICHECK(IsScheduleUnit(op))
        << "DecodeScheduledTIR expected a `tl.schedule_unit` AttrStmt, got "
           "attr "
        << op->attr_key
        << ". Run tl.transform.MaterializeScheduleUnits before AutoSchedule.";
    GuardList guards;
    result_.push_back(DecodeGuardedNode(op, op->body, &guards));
  }

  void VisitStmt_(const ForNode *) final {
    LOG(FATAL) << "DecodeScheduledTIR found an unmarked serial loop in "
                  "scheduled TIR";
  }

  void VisitStmt_(const EvaluateNode *) final {
    LOG(FATAL)
        << "DecodeScheduledTIR found an unmarked Evaluate in scheduled TIR";
  }

  void VisitStmt_(const BufferStoreNode *) final {
    LOG(FATAL) << "DecodeScheduledTIR found an unmarked BufferStore in "
                  "scheduled TIR";
  }

  void VisitStmt_(const BindNode *) final {
    LOG(FATAL) << "DecodeScheduledTIR found an unmarked Bind in scheduled TIR";
  }

  void VisitStmt_(const WhileNode *) final {
    LOG(FATAL) << "DecodeScheduledTIR found an unmarked While in scheduled TIR";
  }

  void VisitStmt_(const SBlockNode *) final {
    LOG(FATAL) << "DecodeScheduledTIR found an unmarked block in scheduled TIR";
  }

  std::shared_ptr<IRStructure> DecodeGuardedNode(const AttrStmtNode *marker,
                                                 const Stmt &body,
                                                 GuardList *guards) {
    if (const auto *condition = body.as<IfThenElseNode>()) {
      ICHECK(!condition->else_case.defined())
          << "Scheduled condition guard must not have an else branch";
      guards->push_back(std::make_unique<ConditionGuard>(condition->condition));
      auto scope = MakeGuard(condition->condition);
      return DecodeGuardedNode(marker, condition->then_case, guards);
    }
    if (const auto *attribute = body.as<AttrStmtNode>()) {
      if (attribute->attr_key == tl::attr::kAscendTask ||
          attribute->attr_key == tl::attr::kAscendPerCoreTask) {
        return DecodeNode(marker, body, std::move(*guards));
      }
      ICHECK(IsScheduleGuardAttribute(attribute->attr_key))
          << "DecodeScheduledTIR found an unsupported guard AttrStmt key "
          << attribute->attr_key;
      guards->push_back(
          std::make_unique<AttributeGuard>(attribute->node, attribute->attr_key,
                                           attribute->value, attribute->span));
      if (attribute->attr_key == tirx::attr::tilelang_assume) {
        auto scope = MakeGuard(Downcast<PrimExpr>(attribute->node),
                               /*is_assume=*/true);
        return DecodeGuardedNode(marker, attribute->body, guards);
      }
      return DecodeGuardedNode(marker, attribute->body, guards);
    }
    return DecodeNode(marker, body, std::move(*guards));
  }

  std::shared_ptr<IRStructure> DecodeNode(const AttrStmtNode *marker,
                                          const Stmt &body, GuardList guards) {
    const auto *loop_node = body.as<ForNode>();
    if (loop_node == nullptr)
      return DecodeTask(marker, body, std::move(guards));

    ICHECK(loop_node->kind == ForKind::kSerial ||
           loop_node->kind == ForKind::kUnrolled)
        << "Scheduled control marker contains a non-serial/unrolled loop";

    auto control = MakeControlNode();
    control->SetStage(GetScheduleUnitStage(marker));
    control->SetGuards(std::move(guards));
    control->control = GetRef<For>(loop_node);
    control->task->stmt =
        For(loop_node->loop_var, loop_node->min, loop_node->extent,
            loop_node->kind, Evaluate(0), loop_node->thread_binding,
            loop_node->annotations, loop_node->step, loop_node->span);
    control->task->outer_ctx = GetConstrSet();
    ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(loop_node->min)),
                      control->task.get());
    ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(loop_node->extent)),
                      control->task.get());
    if (loop_node->step.defined()) {
      ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(loop_node->step.value())),
                        control->task.get());
    }
    ApplyGuardAccesses(control.get(), control->task.get());

    {
      auto loop_scope =
          MakeGuard(loop_node->loop_var,
                    Range::FromMinExtent(loop_node->min, loop_node->extent));
      auto non_empty_scope = MakeGuard(loop_node->extent > 0);
      control->children = DecodeList(loop_node->body, control.get());
    }
    control->InitializeLoopBodyContext();
    return control;
  }

  std::shared_ptr<IRStructure> DecodeTask(const AttrStmtNode *marker,
                                          const Stmt &body, GuardList guards) {
    AttrStmt semantic_task = RequireSemanticTaskMarker(
        body, "DecodeScheduledTIR leaf schedule unit");
    auto task = MakeTaskNode();
    task->stmt = body;
    task->SetStage(GetScheduleUnitStage(marker));
    TaskMetadata metadata = ParseTaskMetadata(semantic_task->node);
    if (metadata.core_mask.has_value())
      task->SetCoreMask(metadata.core_mask.value());
    task->SetGuards(std::move(guards));
    task->outer_ctx = GetConstrSet();
    ApplyTaskResourceUsage(AnalyzeTaskResourceUsage(body), task.get());
    ApplyTaskAccesses(AnalyzeTaskAccesses(body), task.get());
    uint16_t pipe_mask = task->GetPipeMask();
    if (semantic_task->attr_key == attr::kAscendPerCoreTask) {
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
      if (const auto *expr = attribute->node.as<PrimExprNode>()) {
        ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(GetRef<PrimExpr>(expr))),
                          task);
      }
    }
  }

  std::vector<std::shared_ptr<IRStructure>> result_;
  ScheduledExtraInfoFactory make_extra_info_;
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
GetScheduledCoreMask(const AttrStmtNode *schedule_unit) {
  ICHECK(IsScheduleUnit(schedule_unit))
      << "Expected a `tl.schedule_unit` AttrStmt, got "
      << schedule_unit->attr_key;
  Stmt payload = StripScheduledGuards(schedule_unit->body);
  if (const auto *loop = payload.as<ForNode>()) {
    ICHECK(loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)
        << "Scheduled control marker contains a non-serial/unrolled loop";
    return std::nullopt;
  }
  AttrStmt semantic_task =
      RequireSemanticTaskMarker(payload, "Scheduled leaf unit");
  return ParseTaskMetadata(semantic_task->node).core_mask;
}

CoreMask RequireScheduledCoreMask(const AttrStmtNode *schedule_unit) {
  std::optional<CoreMask> core_mask = GetScheduledCoreMask(schedule_unit);
  ICHECK(core_mask.has_value())
      << "Scheduled unit has no core_mask-bearing T.Task";
  return core_mask.value();
}

SBlock EncodeScheduledTIR(ScheduledTIR scheduled_tir) {
  ValidateBufferVersions(scheduled_tir.metadata.buffer_versions,
                         "Scheduled buffer");
  ValidateBufferVersions(scheduled_tir.metadata.manual_buffer_versions,
                         "Manual buffer");
  ValidateBufferVersionModes(scheduled_tir.metadata.buffer_version_modes);
  ICHECK(scheduled_tir.metadata.kernel_root.defined())
      << "Scheduled TIR metadata is missing its kernel root shell";

  Stmt body = ScheduledTIREncoder().Encode(scheduled_tir.tree);
  SBlock rewritten = scheduled_tir.metadata.kernel_root;
  SBlockNode *node = rewritten.CopyOnWrite();
  node->body = std::move(body);

  Map<String, Any> annotations = node->annotations;
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
                    Integer(scheduled_tir.metadata.num_aiv_subcores.value()));
  }
  if (!scheduled_tir.metadata.root_conflict_hints.empty()) {
    annotations.Set(kRootConflictHints,
                    std::move(scheduled_tir.metadata.root_conflict_hints));
  }
  node->annotations = std::move(annotations);
  return rewritten;
}

ScheduledTIR DecodeScheduledTIR(const SBlock &root, const ConstrSet &outer_ctx,
                                ScheduledExtraInfoFactory make_extra_info) {
  ScheduledTIR scheduled_tir;
  scheduled_tir.tree =
      ScheduledTIRDecoder(outer_ctx, make_extra_info).Decode(root->body);
  scheduled_tir.metadata.kernel_root = root;
  if (auto value = root->annotations.Get(kBufferVersionsMap)) {
    scheduled_tir.metadata.buffer_versions =
        value.value().cast<BufferVersionMap>();
    ValidateBufferVersions(scheduled_tir.metadata.buffer_versions,
                           "Scheduled buffer");
  }
  if (auto value = root->annotations.Get(kManualMultiBuffer)) {
    scheduled_tir.metadata.manual_buffer_versions =
        value.value().cast<BufferVersionMap>();
    ValidateBufferVersions(scheduled_tir.metadata.manual_buffer_versions,
                           "Manual buffer");
  }
  if (auto value = root->annotations.Get(kBufferVersionMode)) {
    scheduled_tir.metadata.buffer_version_modes =
        value.value().cast<BufferVersionModeMap>();
    ValidateBufferVersionModes(scheduled_tir.metadata.buffer_version_modes);
  }
  if (auto value = root->annotations.Get(kUnlimitMemoryScopes)) {
    scheduled_tir.metadata.unlimit_memory_scopes =
        value.value().cast<Array<String>>();
  }
  if (auto value = root->annotations.Get(kBufferAliasMap)) {
    scheduled_tir.metadata.buffer_aliases =
        value.value().cast<BufferAliasMap>();
    ValidateBufferAliasMap(scheduled_tir.metadata.buffer_aliases);
    scheduled_tir.metadata.has_buffer_aliases = true;
  }
  if (auto value = root->annotations.Get(kVectorCount)) {
    const auto *count = value.value().as<IntImmNode>();
    ICHECK(count != nullptr && (count->value == 1 || count->value == 2))
        << "Mixed-kernel vector_count must be the constant integer 1 or 2, got "
        << value.value();
    scheduled_tir.metadata.num_aiv_subcores = static_cast<int>(count->value);
  }
  if (auto value = root->annotations.Get(kRootConflictHints)) {
    scheduled_tir.metadata.root_conflict_hints =
        value.value().cast<Array<Any>>();
  }
  return scheduled_tir;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
