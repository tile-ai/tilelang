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
 * \file materialize_schedule_units.cc
 * \brief Normalize task boundaries and materialize trivial schedule units.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "ascend/op/builtin.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/scheduled_tir.h"
#include "ascend/transform/auto_schedule/task_analysis.h"
#include "ascend/transform/auto_schedule/task_annotations.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;
using namespace ascend;
using ffi::Array;
using ffi::GetRef;

namespace {

Stmt MakeSequence(const std::vector<Stmt> &statements) {
  if (statements.empty())
    return Evaluate(0);
  if (statements.size() == 1)
    return statements[0];
  Array<Stmt> sequence;
  sequence.reserve(statements.size());
  for (const Stmt &stmt : statements)
    sequence.push_back(stmt);
  return SeqStmt::Flatten(sequence);
}

bool IsTaskBoundary(const Stmt &stmt) {
  if (const auto *loop = stmt.as<ForNode>())
    return loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled;
  return stmt.as<EvaluateNode>() || stmt.as<BufferStoreNode>() ||
         stmt.as<BindNode>() || stmt.as<WhileNode>() || stmt.as<SBlockNode>();
}

class TaskNormalizer : public StmtMutator {
public:
  static Stmt Rewrite(const Stmt &body) {
    TaskNormalizer normalizer;
    return normalizer(body);
  }

private:
  Stmt VisitStmt(const Stmt &stmt) final {
    if (const auto *attribute = stmt.as<AttrStmtNode>()) {
      if (attribute->attr_key == tl::attr::kAscendStage) {
        ICHECK_EQ(task_marker_depth_, 0)
            << "T.Stage must wrap a complete scheduler task; it cannot appear "
               "inside T.Task";
        ICHECK_EQ(per_core_task_depth_, 0)
            << "T.Stage must wrap a complete scheduler task; it cannot appear "
               "inside T.PerCoreTask";
      }
      if (attribute->attr_key == tl::attr::kAscendTask) {
        ++task_marker_depth_;
        Stmt result = StmtMutator::VisitStmt(stmt);
        --task_marker_depth_;
        const auto *normalized = result.as<AttrStmtNode>();
        ICHECK(normalized && normalized->attr_key == tl::attr::kAscendTask);
        TaskMetadata metadata;
        return AttrStmt(MergeTaskMetadata(metadata, normalized->node),
                        normalized->attr_key, normalized->value,
                        normalized->body, normalized->span);
      }
      if (attribute->attr_key == tl::attr::kAscendPerCoreTask) {
        ICHECK_EQ(task_marker_depth_, 0)
            << "T.PerCoreTask cannot be nested inside T.Task. Place T.Task "
               "candidate regions inside T.PerCoreTask instead.";
        ++per_core_task_depth_;
        Stmt body = VisitStmt(attribute->body);
        --per_core_task_depth_;
        TaskMetadata metadata;
        return AttrStmt(MergeTaskMetadata(metadata, attribute->node),
                        attribute->attr_key, attribute->value, std::move(body),
                        attribute->span);
      }
    }

    if (task_marker_depth_ > 0)
      return StmtMutator::VisitStmt(stmt);

    if (!IsTaskBoundary(stmt))
      return StmtMutator::VisitStmt(stmt);

    if (per_core_task_depth_ > 0 && stmt.as<BindNode>()) {
      ICHECK(!HasTaskBufferAccess(stmt))
          << "An unmarked scalar Bind inside T.PerCoreTask may not access "
             "buffers: synchronization with statements outside the "
             "PerCoreTask is emitted only at T.Task candidate sites, so an "
             "external buffer dependency could not be synchronized at this "
             "Bind. Dependency-free scalar Bind control is allowed. Move the "
             "buffer access into a supported same-pipe T.Task candidate. Bind="
          << stmt;
    }

    TaskMetadata metadata;
    return AttrStmt(MergeTaskMetadata(metadata, IntImm(DataType::Int(32), 0)),
                    tl::attr::kAscendTask, IntImm(DataType::Int(32), 1), stmt,
                    stmt->span);
  }

  int task_marker_depth_{0};
  int per_core_task_depth_{0};
};

ScheduleUnitMetadata MakeInitialScheduleMetadata() {
  ScheduleUnitMetadata metadata;
  metadata.Set(kScheduleUnitStage, Integer(kUnscheduledStage));
  return metadata;
}

Stmt MakeInitialScheduleUnit(Stmt body) {
  Span span = body->span;
  return AttrStmt(MakeInitialScheduleMetadata(), attr::kScheduleUnit,
                  Integer(1), std::move(body), span);
}

Stmt MakeNoOpTask() {
  TaskMetadata metadata;
  Stmt body = Evaluate(0);
  return AttrStmt(MergeTaskMetadata(metadata, IntImm(DataType::Int(32), 0)),
                  tl::attr::kAscendTask, IntImm(DataType::Int(32), 1),
                  std::move(body));
}

AttrStmt RequireScheduleUnit(const Stmt &stmt) {
  const auto *unit = stmt.as<AttrStmtNode>();
  ICHECK(unit != nullptr && IsScheduleUnit(unit))
      << "MaterializeScheduleUnits expected a schedule unit, got " << stmt;
  return GetRef<AttrStmt>(unit);
}

Stmt RebuildWithPrependedGuard(const AttrStmt &unit, Stmt guarded_body) {
  return AttrStmt(unit->node, unit->attr_key, unit->value,
                  std::move(guarded_body), unit->span);
}

Stmt AssignStage(const Stmt &stmt, int stage) {
  AttrStmt unit = RequireScheduleUnit(stmt);
  ICHECK_EQ(GetScheduleUnitStage(unit.get()), kUnscheduledStage)
      << "Nested T.Stage scopes are not supported";
  ScheduleUnitMetadata metadata = unit->node.cast<ScheduleUnitMetadata>();
  metadata.Set(kScheduleUnitStage, Integer(stage));
  return AttrStmt(metadata, unit->attr_key, unit->value, unit->body,
                  unit->span);
}

Stmt PrependConditionGuard(const Stmt &stmt, const PrimExpr &condition) {
  AttrStmt unit = RequireScheduleUnit(stmt);
  return RebuildWithPrependedGuard(unit, IfThenElse(condition, unit->body));
}

Stmt PrependAttributeGuard(const Stmt &stmt, const AttrStmtNode *attribute) {
  AttrStmt unit = RequireScheduleUnit(stmt);
  return RebuildWithPrependedGuard(
      unit, AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                     unit->body, attribute->span));
}

class PerCoreTaskControlValidator : public StmtExprVisitor {
public:
  explicit PerCoreTaskControlValidator(uint16_t pipe_mask)
      : expected_pipe_("PIPE_" + GetResourcePipeName(pipe_mask)) {}

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::ascend_cross_core_set_flag()) ||
        op->op.same_as(tl::ascend_cross_core_wait_flag())) {
      ICHECK_EQ(op->args.size(), 3);
      const auto *mode = op->args[0].as<IntImmNode>();
      ICHECK(mode) << "Cross-core synchronization mode must be a constant "
                      "integer, got "
                   << op->args[0];
      const auto *pipe = op->args[1].as<StringImmNode>();
      ICHECK(pipe) << "Inter-core synchronization pipe must be a constant "
                      "string, got "
                   << op->args[1];
      ICHECK_EQ(pipe->value, expected_pipe_)
          << "Inter-core synchronization inside T.PerCoreTask must use its "
             "inferred pipe "
          << expected_pipe_ << ", got " << pipe->value;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::string expected_pipe_;
};

class TaskMarkerControlValidator : public StmtExprVisitor {
public:
  TaskMarkerControlValidator(bool reject_candidate_selection,
                             bool allow_cross_core_sync)
      : reject_candidate_selection_(reject_candidate_selection),
        allow_cross_core_sync_(allow_cross_core_sync) {}

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    ICHECK_NE(op->attr_key, tl::attr::kAscendTask)
        << "Nested T.Task regions are not supported";
    ICHECK_NE(op->attr_key, tl::attr::kAscendStage)
        << "T.Stage must wrap a complete scheduler task. Place T.Stage outside "
           "T.Task, T.SimtVF, T.SimdVF, SBlock, and parallel-loop task "
           "boundaries.";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    ICHECK(!reject_candidate_selection_)
        << "Control flow selecting a T.Task candidate must be outside the "
           "T.Task scope";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    ICHECK(allow_cross_core_sync_ ||
           (!op->op.same_as(tl::ascend_cross_core_set_flag()) &&
            !op->op.same_as(tl::ascend_cross_core_wait_flag())))
        << "Inter-core synchronization inside T.Task must be the task's only "
           "statement";
    StmtExprVisitor::VisitExpr_(op);
  }

  bool reject_candidate_selection_;
  bool allow_cross_core_sync_;
};

bool IsStandaloneCrossCoreSync(const Stmt &stmt) {
  const auto *evaluate = stmt.as<EvaluateNode>();
  if (!evaluate)
    return false;
  const auto *call = evaluate->value.as<CallNode>();
  return call && (call->op.same_as(tl::ascend_cross_core_set_flag()) ||
                  call->op.same_as(tl::ascend_cross_core_wait_flag()));
}

struct ValidatedTask {
  uint16_t pipe_mask{0};
  bool has_buffer_access{false};
};

ValidatedTask ValidateTask(const AttrStmtNode *op, bool inside_per_core_task) {
  ICHECK_EQ(op->attr_key, tl::attr::kAscendTask);

  Stmt task_body = op->body;
  TaskMarkerControlValidator control_validator(
      inside_per_core_task, IsStandaloneCrossCoreSync(task_body));
  control_validator(task_body);

  Stmt task = GetRef<Stmt>(op);
  TaskResourceUsage resource_usage = AnalyzeTaskResourceUsage(task);
  uint16_t issue_pipe_mask = resource_usage.pipe_mask;
  if (AnalyzeSpecialRegisterWrites(task) != 0)
    issue_pipe_mask |= static_cast<uint16_t>(ResourcePipe::kScalar);
  ICHECK((issue_pipe_mask & (issue_pipe_mask - 1)) == 0)
      << "T.Task supports exactly one Ascend hardware pipe, but its body uses "
         "multiple pipes (mask="
      << issue_pipe_mask
      << "). Split the statements into separate T.Task regions. Body=" << task;
  uint16_t hbm_core_affinity = resource_usage.hbm_mask & kHbmCoreAffinityMask;
  ICHECK((hbm_core_affinity & (hbm_core_affinity - 1)) == 0)
      << "T.Task requires one Ascend core affinity, but its body uses both AIC "
         "and AIV HBM paths. Split GM-to-L1 and GM-to-UB operations into "
         "separate T.Task regions. Body="
      << task;
  return {resource_usage.pipe_mask, HasTaskBufferAccess(task)};
}

bool IsPerCoreTaskScalarControl(const AttrStmtNode *task) {
  ICHECK_EQ(task->attr_key, tl::attr::kAscendTask);
  Stmt body = task->body;
  if (body.as<BindNode>())
    return true;
  const auto *evaluate = body.as<EvaluateNode>();
  if (!evaluate)
    return false;
  if (is_zero(evaluate->value))
    return true;
  const auto *call = evaluate->value.as<CallNode>();
  return call && (call->op.same_as(tl::ascend_cross_core_set_flag()) ||
                  call->op.same_as(tl::ascend_cross_core_wait_flag()));
}

class PerCoreTaskBodyValidator : public StmtVisitor {
public:
  static void Validate(const Stmt &body, uint16_t expected_pipe_mask) {
    PerCoreTaskBodyValidator validator(expected_pipe_mask);
    validator(body);
    ICHECK(validator.has_dependency_task_)
        << "T.PerCoreTask must contain at least one dependency-bearing task "
           "after task normalization";
  }

private:
  explicit PerCoreTaskBodyValidator(uint16_t expected_pipe_mask)
      : expected_pipe_mask_(expected_pipe_mask) {}

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kAscendTask) {
      ValidatedTask task = ValidateTask(op, /*inside_per_core_task=*/true);
      if (!task.has_buffer_access) {
        ICHECK(IsPerCoreTaskScalarControl(op))
            << "T.PerCoreTask only permits scalar Bind/condition/loop control "
               "and inter-core synchronization in addition to operations on "
               "its inferred pipe; got "
            << GetRef<Stmt>(op);
        return;
      }
      ICHECK_NE(task.pipe_mask, 0)
          << "Cannot infer an Ascend pipe for T.PerCoreTask candidate "
          << GetRef<Stmt>(op);
      ICHECK_EQ(task.pipe_mask & (task.pipe_mask - 1), 0)
          << "Each dependency-bearing statement in T.PerCoreTask must use "
             "exactly one hardware pipe, but statement mask="
          << task.pipe_mask << ", statement=" << GetRef<Stmt>(op);
      ICHECK_EQ(task.pipe_mask, expected_pipe_mask_)
          << "T.PerCoreTask candidate pipe does not match its inferred pipe. "
             "Inferred mask="
          << expected_pipe_mask_ << ", statement mask=" << task.pipe_mask
          << ", statement=" << GetRef<Stmt>(op);
      has_dependency_task_ = true;
      return;
    }
    ICHECK_NE(op->attr_key, tl::attr::kAscendPerCoreTask)
        << "Nested T.PerCoreTask regions are not supported";
    ICHECK_NE(op->attr_key, attr::kScheduleUnit)
        << "T.PerCoreTask contains an already-materialized schedule unit";
    ICHECK(IsScheduleGuardAttribute(op->attr_key))
        << "MaterializeScheduleUnits only supports known scheduling AttrStmt "
           "wrappers inside T.PerCoreTask. Unsupported attr_key: "
        << op->attr_key;
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled)
      FatalUnmarkedTask("MaterializeScheduleUnits", "parallel loop");
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "Evaluate");
  }

  void VisitStmt_(const BufferStoreNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "BufferStore");
  }

  void VisitStmt_(const BindNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "Bind");
  }

  void VisitStmt_(const WhileNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "While");
  }

  void VisitStmt_(const SBlockNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "block");
  }

  uint16_t expected_pipe_mask_{0};
  bool has_dependency_task_{false};
};

void ValidatePerCoreTask(const AttrStmtNode *op) {
  ICHECK_EQ(op->attr_key, tl::attr::kAscendPerCoreTask);
  TaskResourceUsage resource_usage = AnalyzeTaskResourceUsage(op->body);
  uint16_t pipe_mask = resource_usage.pipe_mask;
  ICHECK_NE(pipe_mask, 0) << "Cannot infer an Ascend pipe for T.PerCoreTask";
  ICHECK_EQ(pipe_mask & (pipe_mask - 1), 0)
      << "T.PerCoreTask requires exactly one inferred hardware pipe, but its "
         "mask is "
      << pipe_mask;
  PerCoreTaskBodyValidator::Validate(op->body, pipe_mask);
  PerCoreTaskControlValidator control_validator(pipe_mask);
  control_validator(op->body);
}

// Lower normalized Task TIR into the canonical scheduled-TIR grammar without
// making scheduling decisions. A child list without T.Stage uses stage -1
// throughout. If any child has a requested non-negative stage, unannotated
// siblings are normalized to stage 0 so AutoSchedule can identify manual mode
// from a uniform per-list invariant.
class ScheduleUnitMaterializer {
public:
  static Stmt Rewrite(const Stmt &body) {
    ScheduleUnitMaterializer materializer;
    std::vector<Stmt> units = materializer.MaterializeChildList(body);
    return MakeSequence(units);
  }

private:
  // Only kernel roots and loop bodies define child-list boundaries. Conditions
  // and attributes flatten into their enclosing list and are normalized there.
  std::vector<Stmt> MaterializeChildList(const Stmt &stmt) {
    std::vector<Stmt> units = MaterializeList(stmt);
    EnsureNonEmpty(&units);
    NormalizeRequestedStages(&units);
    return units;
  }

  std::vector<Stmt> MaterializeList(const Stmt &stmt) {
    if (!stmt.defined())
      return {};
    if (const auto *sequence = stmt.as<SeqStmtNode>()) {
      std::vector<Stmt> result;
      for (const Stmt &child : sequence->seq) {
        std::vector<Stmt> child_units = MaterializeList(child);
        result.insert(result.end(), child_units.begin(), child_units.end());
      }
      return result;
    }
    if (const auto *loop = stmt.as<ForNode>())
      return MaterializeLoop(loop);
    if (const auto *condition = stmt.as<IfThenElseNode>())
      return MaterializeCondition(condition);
    if (const auto *attribute = stmt.as<AttrStmtNode>())
      return MaterializeAttribute(attribute);
    if (stmt.as<EvaluateNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "Evaluate");
    if (stmt.as<BufferStoreNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "BufferStore");
    if (stmt.as<BindNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "Bind");
    if (stmt.as<WhileNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "While");
    if (stmt.as<SBlockNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "block");
    LOG(FATAL) << "MaterializeScheduleUnits found an unsupported statement: "
               << stmt;
  }

  std::vector<Stmt> MaterializeLoop(const ForNode *op) {
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled)
      FatalUnmarkedTask("MaterializeScheduleUnits", "parallel loop");
    std::vector<Stmt> body = MaterializeChildList(op->body);
    For loop = GetRef<For>(op);
    loop.CopyOnWrite()->body = MakeSequence(body);
    return {MakeInitialScheduleUnit(std::move(loop))};
  }

  std::vector<Stmt> MaterializeCondition(const IfThenElseNode *op) {
    std::vector<Stmt> result = MaterializeList(op->then_case);
    for (Stmt &unit : result)
      unit = PrependConditionGuard(unit, op->condition);
    if (!op->else_case.defined())
      return result;

    std::vector<Stmt> else_units = MaterializeList(op->else_case.value());
    PrimExpr else_condition = Not(op->condition);
    for (Stmt &unit : else_units)
      unit = PrependConditionGuard(unit, else_condition);
    result.insert(result.end(), else_units.begin(), else_units.end());
    return result;
  }

  std::vector<Stmt> MaterializeAttribute(const AttrStmtNode *op) {
    ICHECK_NE(op->attr_key, attr::kScheduleUnit)
        << "MaterializeScheduleUnits cannot run on an already-materialized "
           "schedule";
    if (op->attr_key == tl::attr::kAscendTask) {
      ValidateTask(op, /*inside_per_core_task=*/false);
      return {MakeInitialScheduleUnit(GetRef<Stmt>(op))};
    }
    if (op->attr_key == tl::attr::kAscendPerCoreTask) {
      ValidatePerCoreTask(op);
      return {MakeInitialScheduleUnit(GetRef<Stmt>(op))};
    }
    if (op->attr_key == tl::attr::kAscendStage) {
      PrimExpr stage_expr = op->node.cast<PrimExpr>();
      const auto *stage = stage_expr.as<IntImmNode>();
      ICHECK(stage != nullptr && stage->value >= 0 &&
             stage->value <= std::numeric_limits<int>::max())
          << "T.Stage requires a non-negative constant integer, got "
          << stage_expr;
      std::vector<Stmt> body = MaterializeList(op->body);
      EnsureNonEmpty(&body);
      for (Stmt &unit : body)
        unit = AssignStage(unit, static_cast<int>(stage->value));
      return body;
    }
    ICHECK(IsScheduleGuardAttribute(op->attr_key))
        << "MaterializeScheduleUnits only supports known scheduling AttrStmt "
           "wrappers inside the schedulable tilelang_root body. Unsupported "
           "attr_key: "
        << op->attr_key;
    std::vector<Stmt> body = MaterializeList(op->body);
    EnsureNonEmpty(&body);
    for (Stmt &unit : body)
      unit = PrependAttributeGuard(unit, op);
    return body;
  }

  void EnsureNonEmpty(std::vector<Stmt> *units) {
    ICHECK(units != nullptr);
    if (units->empty())
      units->push_back(MakeInitialScheduleUnit(MakeNoOpTask()));
  }

  void NormalizeRequestedStages(std::vector<Stmt> *units) {
    ICHECK(units != nullptr);
    bool manual_schedule =
        std::any_of(units->begin(), units->end(), [](const Stmt &unit) {
          return GetScheduleUnitStage(RequireScheduleUnit(unit).get()) !=
                 kUnscheduledStage;
        });
    if (!manual_schedule)
      return;
    for (Stmt &unit : *units) {
      if (GetScheduleUnitStage(RequireScheduleUnit(unit).get()) ==
          kUnscheduledStage) {
        unit = AssignStage(unit, 0);
      }
    }
  }
};

} // namespace

/*! \brief Normalize task boundaries and materialize unscheduled units. */
tvm::transform::Pass MaterializeScheduleUnits() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    return RewriteTilelangKernels(
        std::move(func), "MaterializeScheduleUnits",
        [](const TilelangKernelContext &context) {
          SBlock root = context.root;
          Stmt normalized = TaskNormalizer::Rewrite(root->body);
          root.CopyOnWrite()->body =
              ScheduleUnitMaterializer::Rewrite(normalized);
          return root;
        },
        /*require_kernel=*/false);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MaterializeScheduleUnits", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MaterializeScheduleUnits",
                        MaterializeScheduleUnits);
}

} // namespace tl
} // namespace tvm
