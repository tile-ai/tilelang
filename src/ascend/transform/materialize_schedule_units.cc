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
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/op/builtin.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/schedule_scope.h"
#include "ascend/transform/auto_schedule/task_annotations.h"
#include "ascend/transform/auto_schedule/task_resources.h"

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
    return Evaluate(IntImm::Int32(0));
  if (statements.size() == 1)
    return statements[0];
  Array<Stmt> sequence;
  sequence.reserve(statements.size());
  for (const Stmt &stmt : statements)
    sequence.push_back(stmt);
  return SeqStmt(sequence);
}

bool IsTaskBoundary(const Stmt &statement) {
  if (const auto *loop = statement.as<ForNode>())
    return loop->kind != ForKind::kDefault && loop->kind != ForKind::kUnrolled;
  return statement.as<EvaluateNode>() || statement.as<TensorStoreNode>() ||
         statement.as<BindNode>() || statement.as<WhileNode>() ||
         statement.as<RegionStmtNode>() || statement.as<BreakNode>() ||
         statement.as<ContinueNode>() || statement.as<ReturnNode>() ||
         statement.as<AssertStmtNode>();
}

class TaskNormalizer {
public:
  static Stmt Rewrite(const Stmt &body) {
    TaskNormalizer normalizer;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        body, [&](const Var &variable, TVMFFIDefRegionKind kind) {
          if (kind != kTVMFFIDefRegionKindNone)
            normalizer.local_definitions_.insert(variable);
          return ffi::WalkResult::Advance();
        });
    return normalizer.RewriteStmt(body);
  }

private:
  bool IsDynamicAllocation(const Stmt &statement) const {
    const auto *binding = statement.as<BindNode>();
    const auto *allocation = binding ? binding->value.as<CallNode>() : nullptr;
    if (!allocation || !allocation->op.same_as(tirx::alloc_tensor_op()))
      return false;
    bool dynamic = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        binding->var->ty, [&](const Var &variable) {
          dynamic |= local_definitions_.count(variable) != 0;
          return ffi::WalkResult::Advance();
        });
    return dynamic;
  }

  Stmt RewriteStmt(const Stmt &statement) {
    if (const auto *region = statement.as<RegionStmtNode>()) {
      if (region->op.same_as(Op::Get("tirx.scope")) &&
          region->attrs->dict.empty() && region->args.empty() &&
          region->body_params.empty() && region->result_vars.empty())
        return RewriteStmt(region->body);
      auto key = ScheduleMarkerKey(region);
      if (key == attr::kAscendStage) {
        ICHECK_EQ(task_marker_depth_, 0)
            << "T.Stage must wrap a complete scheduler task; it cannot appear "
               "inside T.Task";
        ICHECK_EQ(per_core_task_depth_, 0)
            << "T.Stage must wrap a complete scheduler task; it cannot appear "
               "inside T.PerCoreTask";
      }
      if (key == attr::kAscendTask || key == attr::kAscendPerCoreTask) {
        bool per_core = key == attr::kAscendPerCoreTask;
        if (per_core) {
          ICHECK_EQ(task_marker_depth_, 0)
              << "T.PerCoreTask cannot be nested inside T.Task";
          ++per_core_task_depth_;
        } else {
          ++task_marker_depth_;
        }
        Stmt body = RewriteStmt(UnwrapScheduleMarker(region));
        if (per_core)
          --per_core_task_depth_;
        else
          --task_marker_depth_;
        auto result = MakeScheduleMarker(
            key,
            MergeTaskMetadata(TaskMetadata{}, ScheduleMarkerPayload(region)),
            std::move(body), region->loc, !region->result_vars.empty());
        auto annotations = region->attrs->dict;
        annotations.erase(kScheduleYield);
        for (const auto &[name, value] : result->attrs->dict)
          annotations.Set(name, value);
        result.CopyOnWrite()->attrs = DictAttrs(annotations);
        return result;
      }
      if (!key.empty()) {
        auto result = GetRef<RegionStmt>(region);
        if (key == kScheduleCondition)
          ++guarded_depth_;
        result.CopyOnWrite()->body = SeqStmt(RewriteStmt(region->body));
        if (key == kScheduleCondition)
          --guarded_depth_;
        return result;
      }
    }
    if (task_marker_depth_ == 0 && IsTaskBoundary(statement)) {
      if (per_core_task_depth_ > 0 && statement.as<BindNode>()) {
        ICHECK(!HasTaskBufferAccess(statement))
            << "An unmarked scalar Bind inside T.PerCoreTask may not access "
               "buffers. "
               "Move the buffer access into a supported same-pipe T.Task "
               "candidate.";
      }
      return MakeScheduleMarker(attr::kAscendTask,
                                ffi::Map<ffi::String, ffi::Any>{}, statement,
                                statement->loc);
    }
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      Array<Stmt> body;
      for (size_t index = 0; index < sequence->seq.size(); ++index) {
        const Stmt &child = sequence->seq[index];
        if (guarded_depth_ != 0 && task_marker_depth_ == 0 &&
            per_core_task_depth_ == 0 && IsDynamicAllocation(child)) {
          Array<Stmt> lifetime;
          for (; index < sequence->seq.size(); ++index)
            lifetime.push_back(RewriteStmt(sequence->seq[index]));
          PrimVar scope("allocation_scope", PrimType::Int(32));
          body.push_back(
              For(scope, IntImm::Int32(0), IntImm::Int32(1), IntImm::Int32(1),
                  ForKind::kUnrolled, SeqStmt(lifetime),
                  {{"num_stages", 1}, {"enable_offset", false}}, child->loc));
          break;
        }
        body.push_back(RewriteStmt(child));
      }
      return SeqStmt(body, sequence->loc);
    }
    if (const auto *loop = statement.as<ForNode>()) {
      auto result = GetRef<For>(loop);
      ++guarded_depth_;
      result.CopyOnWrite()->body = SeqStmt(RewriteStmt(loop->body));
      --guarded_depth_;
      return result;
    }
    if (const auto *condition = statement.as<IfNode>()) {
      auto result = GetRef<If>(condition);
      ++guarded_depth_;
      result.CopyOnWrite()->then_case =
          SeqStmt(RewriteStmt(condition->then_case));
      if (condition->else_case)
        result.CopyOnWrite()->else_case =
            SeqStmt(RewriteStmt(condition->else_case.value()));
      --guarded_depth_;
      return result;
    }
    if (const auto *loop = statement.as<WhileNode>()) {
      auto result = GetRef<While>(loop);
      result.CopyOnWrite()->body = SeqStmt(RewriteStmt(loop->body));
      return result;
    }
    if (const auto *region = statement.as<RegionStmtNode>()) {
      auto result = GetRef<RegionStmt>(region);
      result.CopyOnWrite()->body = SeqStmt(RewriteStmt(region->body));
      return result;
    }
    return statement;
  }

  int task_marker_depth_{0};
  int per_core_task_depth_{0};
  int guarded_depth_{0};
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> local_definitions_;
};

ScheduleUnitMetadata MakeInitialScheduleMetadata() {
  return {{kScheduleUnitStage, kUnscheduledStage}};
}

Stmt MakeInitialScheduleUnit(Stmt body) {
  return MakeScheduleMarker(attr::kScheduleUnit, MakeInitialScheduleMetadata(),
                            body, body->loc);
}

Stmt MakeNoOpTask() {
  return MakeScheduleMarker(attr::kAscendTask,
                            ffi::Map<ffi::String, ffi::Any>{},
                            Evaluate(IntImm::Int32(0)));
}

RegionStmt RequireScheduleUnit(const Stmt &stmt) {
  const auto *unit = stmt.as<RegionStmtNode>();
  ICHECK(unit != nullptr && IsScheduleUnit(unit))
      << "MaterializeScheduleUnits expected a schedule unit, got " << stmt;
  return GetRef<RegionStmt>(unit);
}

Stmt AssignStage(const Stmt &stmt, int stage) {
  RegionStmt unit = RequireScheduleUnit(stmt);
  ICHECK_EQ(GetScheduleUnitStage(unit.get()), kUnscheduledStage)
      << "Nested T.Stage scopes are not supported";
  auto metadata =
      ScheduleMarkerPayload(unit.get()).as_or_throw<ScheduleUnitMetadata>();
  metadata.Set(kScheduleUnitStage, stage);
  return WithSchedulePayload(unit.get(), metadata);
}

Stmt PrependConditionGuard(const Stmt &stmt, const PrimExpr &condition) {
  RegionStmt unit = RequireScheduleUnit(stmt);
  Stmt body = UnwrapScheduleMarker(unit.get());
  return ReplaceScheduleBody(
      unit.get(),
      MakeScheduleMarker(kScheduleCondition, condition, body, unit->loc));
}

Stmt PrependAttributeGuard(const Stmt &stmt, const RegionStmtNode *attribute) {
  RegionStmt unit = RequireScheduleUnit(stmt);
  return ReplaceScheduleBody(
      unit.get(),
      MakeScheduleMarker(ScheduleMarkerKey(attribute),
                         ScheduleMarkerPayload(attribute),
                         UnwrapScheduleMarker(unit.get()), attribute->loc));
}

class PerCoreTaskControlValidator : public AnalysisVisitor {
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
    AnalysisVisitor::VisitExpr_(op);
  }

  std::string expected_pipe_;
};

class TaskMarkerControlValidator : public AnalysisVisitor {
public:
  TaskMarkerControlValidator(bool reject_candidate_selection,
                             bool allow_cross_core_sync)
      : reject_candidate_selection_(reject_candidate_selection),
        allow_cross_core_sync_(allow_cross_core_sync) {}

private:
  void VisitStmt_(const RegionStmtNode *op) final {
    ICHECK_NE(ScheduleMarkerKey(op), tl::attr::kAscendTask)
        << "Nested T.Task regions are not supported";
    ICHECK_NE(ScheduleMarkerKey(op), tl::attr::kAscendStage)
        << "T.Stage must wrap a complete scheduler task. Place T.Stage outside "
           "T.Task, T.SimtVF, T.SimdVF, SBlock, and parallel-loop task "
           "boundaries.";
    AnalysisVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfNode *op) final {
    ICHECK(!reject_candidate_selection_)
        << "Control flow selecting a T.Task candidate must be outside the "
           "T.Task scope";
    AnalysisVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    ICHECK(allow_cross_core_sync_ ||
           (!op->op.same_as(tl::ascend_cross_core_set_flag()) &&
            !op->op.same_as(tl::ascend_cross_core_wait_flag())))
        << "Inter-core synchronization inside T.Task must be the task's only "
           "statement";
    AnalysisVisitor::VisitExpr_(op);
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

ValidatedTask ValidateTask(const RegionStmtNode *op,
                           bool inside_per_core_task) {
  ICHECK_EQ(ScheduleMarkerKey(op), tl::attr::kAscendTask);

  Stmt task_body = SingleScheduleStatement(UnwrapScheduleMarker(op));
  auto control_validator = ffi::make_object<TaskMarkerControlValidator>(
      inside_per_core_task, IsStandaloneCrossCoreSync(task_body));
  control_validator->Visit(task_body);

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

bool IsPerCoreTaskScalarControl(const RegionStmtNode *task) {
  ICHECK_EQ(ScheduleMarkerKey(task), tl::attr::kAscendTask);
  Stmt body = SingleScheduleStatement(UnwrapScheduleMarker(task));
  if (body.as<BindNode>())
    return true;
  const auto *evaluate = body.as<EvaluateNode>();
  if (!evaluate)
    return false;
  if (GetConstInt(evaluate->value) == 0)
    return true;
  const auto *call = evaluate->value.as<CallNode>();
  return call && (call->op.same_as(tl::ascend_cross_core_set_flag()) ||
                  call->op.same_as(tl::ascend_cross_core_wait_flag()));
}

class PerCoreTaskBodyValidator : public AnalysisVisitor {
public:
  static void Validate(const Stmt &body, uint16_t expected_pipe_mask) {
    auto validator =
        ffi::make_object<PerCoreTaskBodyValidator>(expected_pipe_mask);
    validator->Visit(body);
    ICHECK(validator->has_dependency_task_)
        << "T.PerCoreTask must contain at least one dependency-bearing task "
           "after task normalization";
  }

public:
  explicit PerCoreTaskBodyValidator(uint16_t expected_pipe_mask)
      : expected_pipe_mask_(expected_pipe_mask) {}

  void VisitStmt_(const RegionStmtNode *op) final {
    if (ScheduleMarkerKey(op) == tl::attr::kAscendTask) {
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
    ICHECK_NE(ScheduleMarkerKey(op), tl::attr::kAscendPerCoreTask)
        << "Nested T.PerCoreTask regions are not supported";
    ICHECK_NE(ScheduleMarkerKey(op), attr::kScheduleUnit)
        << "T.PerCoreTask contains an already-materialized schedule unit";
    ICHECK(IsScheduleGuardAttribute(ScheduleMarkerKey(op)))
        << "MaterializeScheduleUnits only supports known scheduling AttrStmt "
           "wrappers inside T.PerCoreTask. Unsupported attr_key: "
        << ScheduleMarkerKey(op);
    AnalysisVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kDefault && op->kind != ForKind::kUnrolled)
      FatalUnmarkedTask("MaterializeScheduleUnits", "parallel loop");
    AnalysisVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "Evaluate");
  }

  void VisitStmt_(const TensorStoreNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "BufferStore");
  }

  void VisitStmt_(const BindNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "Bind");
  }

  void VisitStmt_(const WhileNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "While");
  }

  uint16_t expected_pipe_mask_{0};
  bool has_dependency_task_{false};
};

void ValidatePerCoreTask(const RegionStmtNode *op) {
  ICHECK_EQ(ScheduleMarkerKey(op), tl::attr::kAscendPerCoreTask);
  TaskResourceUsage resource_usage = AnalyzeTaskResourceUsage(op->body);
  uint16_t pipe_mask = resource_usage.pipe_mask;
  ICHECK_NE(pipe_mask, 0) << "Cannot infer an Ascend pipe for T.PerCoreTask";
  ICHECK_EQ(pipe_mask & (pipe_mask - 1), 0)
      << "T.PerCoreTask requires exactly one inferred hardware pipe, but its "
         "mask is "
      << pipe_mask;
  PerCoreTaskBodyValidator::Validate(op->body, pipe_mask);
  auto control_validator =
      ffi::make_object<PerCoreTaskControlValidator>(pipe_mask);
  control_validator->Visit(op->body);
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
    if (const auto *condition = stmt.as<IfNode>())
      return MaterializeCondition(condition);
    if (const auto *attribute = stmt.as<RegionStmtNode>())
      return MaterializeAttribute(attribute);
    if (stmt.as<EvaluateNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "Evaluate");
    if (stmt.as<TensorStoreNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "BufferStore");
    if (stmt.as<BindNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "Bind");
    if (stmt.as<WhileNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "While");
    LOG(FATAL) << "MaterializeScheduleUnits found an unsupported statement: "
               << stmt;
  }

  std::vector<Stmt> MaterializeLoop(const ForNode *op) {
    if (op->kind != ForKind::kDefault && op->kind != ForKind::kUnrolled)
      FatalUnmarkedTask("MaterializeScheduleUnits", "parallel loop");
    std::vector<Stmt> body = MaterializeChildList(op->body);
    For loop = GetRef<For>(op);
    loop.CopyOnWrite()->body = SeqStmt(MakeSequence(body));
    return {MakeInitialScheduleUnit(std::move(loop))};
  }

  std::vector<Stmt> MaterializeCondition(const IfNode *op) {
    std::vector<Stmt> result = MaterializeList(op->then_case);
    for (Stmt &unit : result)
      unit = PrependConditionGuard(unit, op->condition);
    if (!op->else_case.has_value())
      return result;

    std::vector<Stmt> else_units = MaterializeList(op->else_case.value());
    PrimExpr else_condition = !op->condition;
    for (Stmt &unit : else_units)
      unit = PrependConditionGuard(unit, else_condition);
    result.insert(result.end(), else_units.begin(), else_units.end());
    return result;
  }

  std::vector<Stmt> MaterializeAttribute(const RegionStmtNode *op) {
    ICHECK_NE(ScheduleMarkerKey(op), attr::kScheduleUnit)
        << "MaterializeScheduleUnits cannot run on an already-materialized "
           "schedule";
    if (ScheduleMarkerKey(op) == tl::attr::kAscendTask) {
      ValidateTask(op, /*inside_per_core_task=*/false);
      return {MakeInitialScheduleUnit(GetRef<Stmt>(op))};
    }
    if (ScheduleMarkerKey(op) == tl::attr::kAscendPerCoreTask) {
      ValidatePerCoreTask(op);
      return {MakeInitialScheduleUnit(GetRef<Stmt>(op))};
    }
    if (ScheduleMarkerKey(op) == tl::attr::kAscendStage) {
      auto stage = GetIntAnnotation(ScheduleMarkerPayload(op));
      ICHECK(stage && *stage >= 0 && *stage <= std::numeric_limits<int>::max())
          << "T.Stage requires a non-negative constant integer, got "
          << ScheduleMarkerPayload(op);
      std::vector<Stmt> body = MaterializeList(UnwrapScheduleMarker(op));
      EnsureNonEmpty(&body);
      for (Stmt &unit : body)
        unit = AssignStage(unit, static_cast<int>(*stage));
      return body;
    }
    ICHECK(IsScheduleGuardAttribute(ScheduleMarkerKey(op)))
        << "MaterializeScheduleUnits only supports known scheduling AttrStmt "
           "wrappers inside the schedulable tilelang_root body. Unsupported "
           "attr_key: "
        << ScheduleMarkerKey(op);
    std::vector<Stmt> body = MaterializeList(UnwrapScheduleMarker(op));
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
  auto pass_func =
      [](tirx::Function func, IRModule,
         tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    return RewriteTilelangKernels(
        std::move(func), "MaterializeScheduleUnits",
        [](const TilelangKernelContext &context) {
          RegionStmt root = context.root;
          Stmt normalized = TaskNormalizer::Rewrite(root->body);
          root.CopyOnWrite()->body =
              SeqStmt(ScheduleUnitMaterializer::Rewrite(normalized));
          return root;
        },
        /*require_kernel=*/false);
  };
  return tirx::transform::CreateFunctionPass(pass_func, 0,
                                             "tl.MaterializeScheduleUnits");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MaterializeScheduleUnits",
                        MaterializeScheduleUnits);
}

} // namespace tl
} // namespace tvm
