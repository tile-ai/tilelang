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
 * \brief Normalize native task boundaries without hiding SSA definitions.
 */

#include "ascend/transform/source_scopes.h"
#include "auto_schedule/kernel_rewriter.h"
#include "auto_schedule/task_resource.h"
#include "ir/stmt_functor.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ascend;
using namespace prim;
using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Optional;

bool IsSerialLoop(const ForObj *loop) {
  return loop->kind == ForKind::kDefault || loop->kind == ForKind::kUnrolled;
}

bool IsMetadataScope(const SourceAttribute &attribute) {
  return attribute.key == "tl.loop.annotations";
}

bool IsStandaloneCrossCoreSync(const Stmt &body) {
  const auto *evaluate = SingleStatement(body).as<EvaluateObj>();
  const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
  return call && task_resource_detail::IsCrossCoreSyncCall(call);
}

// Keep lexical scopes intact while normalizing candidate boundaries. A Bind
// local to a branch or a PerCoreTask must keep dominating its local users.
class TaskNormalizer {
public:
  Stmt Rewrite(const Stmt &stmt, int task_depth = 0, int per_core_depth = 0) {
    if (const auto *sequence = stmt.as<SeqStmtObj>()) {
      Array<Stmt> body;
      for (const Stmt &child : sequence->seq)
        body.push_back(Rewrite(child, task_depth, per_core_depth));
      return SeqStmt(body, stmt->loc);
    }
    if (IsAscendBlock(stmt.as<RegionStmtObj>()) && task_depth == 0)
      return MakeTask(stmt);
    if (const auto *region = stmt.as<RegionStmtObj>()) {
      if (IsStageScope(region))
        TVM_FFI_CHECK(task_depth == 0 && per_core_depth == 0, ValueError)
            << "T.Stage must wrap a complete scheduler task; it cannot appear "
               "inside T.Task, T.PerCoreTask or an atomic VF task boundary";
      if (IsPerCoreTask(region)) {
        TVM_FFI_CHECK_EQ(task_depth, 0, ValueError)
            << "T.PerCoreTask cannot be nested inside T.Task. Place T.Task "
               "candidate regions inside T.PerCoreTask instead.";
        TaskAnnotations(region);
        RegionStmt result = GetRef<RegionStmt>(region);
        result.CopyOnWrite()->body =
            SeqStmt(Rewrite(region->body, task_depth, per_core_depth + 1));
        return result;
      }
      if (IsTask(region)) {
        TaskAnnotations(region);
        RegionStmt result = GetRef<RegionStmt>(region);
        result.CopyOnWrite()->body =
            SeqStmt(Rewrite(region->body, task_depth + 1, per_core_depth));
        return result;
      }
      if (auto source = ReadSourceAttribute(region)) {
        if (source->key == "tl.sblock" && task_depth == 0)
          return MakeTask(stmt);
        return source->WithBody(
            Rewrite(region->body, task_depth, per_core_depth));
      }
    }
    if (const auto *bind = stmt.as<BindObj>()) {
      if (task_depth == 0 && per_core_depth > 0)
        TVM_FFI_CHECK(!HasTaskBufferAccess(stmt), ValueError)
            << "An unmarked scalar Bind inside T.PerCoreTask may not access "
               "buffers: external synchronization is emitted only at T.Task "
               "candidate sites. Move the access into a same-pipe T.Task "
               "candidate. Bind="
            << stmt;
      // Top-level Bind is marked in its value by ScheduleUnitMaterializer;
      // a PerCoreTask-local scalar Bind stays in that task's lexical scope.
      return GetRef<Bind>(bind);
    }
    bool boundary = stmt.as<EvaluateObj>() || stmt.as<TensorStoreObj>() ||
                    stmt.as<WhileObj>() || stmt.as<AssertStmtObj>() ||
                    stmt.as<BreakObj>();
    if (const auto *loop = stmt.as<ForObj>())
      boundary |= !IsSerialLoop(loop);
    if (task_depth == 0 && boundary) {
      if (IsTaskMetadataStatement(stmt))
        return stmt;
      return MakeTask(stmt);
    }
    if (const auto *loop = stmt.as<ForObj>()) {
      For rewritten = GetRef<For>(loop);
      rewritten.CopyOnWrite()->body =
          SeqStmt(Rewrite(loop->body, task_depth, per_core_depth));
      return rewritten;
    }
    if (const auto *branch = stmt.as<IfObj>()) {
      If rewritten = GetRef<If>(branch);
      auto *writer = rewritten.CopyOnWrite();
      writer->then_case =
          SeqStmt(Rewrite(branch->then_case, task_depth, per_core_depth));
      if (branch->else_case)
        writer->else_case = SeqStmt(
            Rewrite(branch->else_case.value(), task_depth, per_core_depth));
      return rewritten;
    }
    if (const auto *loop = stmt.as<WhileObj>()) {
      While rewritten = GetRef<While>(loop);
      rewritten.CopyOnWrite()->body =
          SeqStmt(Rewrite(loop->body, task_depth, per_core_depth));
      return rewritten;
    }
    if (const auto *region = stmt.as<RegionStmtObj>()) {
      RegionStmt rewritten = GetRef<RegionStmt>(region);
      rewritten.CopyOnWrite()->body =
          SeqStmt(Rewrite(region->body, task_depth, per_core_depth));
      return rewritten;
    }
    return stmt;
  }
};

class TaskControlValidator : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  TaskControlValidator(bool reject_selection, bool allow_cross_core)
      : reject_selection_(reject_selection),
        allow_cross_core_(allow_cross_core) {}

  Optional<ffi::VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    TVM_FFI_CHECK(!IsTask(op), ValueError)
        << "Nested T.Task regions are not supported";
    TVM_FFI_CHECK(!IsStageScope(op), ValueError)
        << "T.Stage must wrap a complete scheduler task. Place it outside "
           "T.Task, T.SimtVF, T.SimdVF, SBlock and parallel-loop boundaries.";
    TVM_FFI_CHECK(!IsPerCoreTask(op), ValueError)
        << "T.PerCoreTask cannot be nested inside T.Task";
    return StmtExprVisitor::Visit_(op);
  }

  Optional<ffi::VisitInterrupt> Visit_(const IfObj *op) final {
    TVM_FFI_CHECK(!reject_selection_, ValueError)
        << "Control flow selecting a T.Task candidate must be outside the "
           "T.Task scope";
    return StmtExprVisitor::Visit_(op);
  }

  Optional<ffi::VisitInterrupt> Visit_(const CallObj *op) final {
    TVM_FFI_CHECK(allow_cross_core_ ||
                      !task_resource_detail::IsCrossCoreSyncCall(op),
                  ValueError)
        << "Inter-core synchronization inside T.Task must be the task's only "
           "statement";
    return StmtExprVisitor::Visit_(op);
  }

private:
  bool reject_selection_;
  bool allow_cross_core_;
};

struct ValidatedTask {
  uint16_t pipe_mask;
  bool has_buffer_access;
};

ValidatedTask ValidateTask(const RegionStmt &task, bool inside_per_core_task) {
  TVM_FFI_CHECK(IsTask(task.get()), ValueError);
  ParseTaskMetadata(TaskAnnotations(task.get()));
  auto validator = ffi::make_object<TaskControlValidator>(
      inside_per_core_task, IsStandaloneCrossCoreSync(task->body));
  validator->Visit(task->body);
  auto usage = AnalyzeTaskResourceUsage(task);
  uint16_t issue_pipe_mask = usage.pipe_mask;
  if (AnalyzeSpecialRegisterWrites(task) != 0)
    issue_pipe_mask |= PipeMask(ResourcePipe::kScalar);
  TVM_FFI_CHECK(IsSingleResourcePipe(issue_pipe_mask), ValueError)
      << "T.Task supports exactly one Ascend hardware pipe, but its body uses "
         "multiple pipes (mask="
      << issue_pipe_mask
      << "). Split the statements into separate T.Task regions. Body=" << task;
  uint16_t affinity = usage.hbm_mask & kHbmCoreAffinityMask;
  TVM_FFI_CHECK((affinity & (affinity - 1)) == 0, ValueError)
      << "T.Task requires one Ascend core affinity, but its body uses both AIC "
         "and AIV HBM paths. Split GM-to-L1 and GM-to-UB operations into "
         "separate T.Task regions. Body="
      << task;
  return {usage.pipe_mask, HasTaskBufferAccess(task)};
}

bool IsPerCoreScalarControl(const Stmt &stmt) {
  Stmt body = SingleStatement(stmt);
  if (body.as<BindObj>())
    return !HasTaskBufferAccess(body);
  const auto *evaluate = body.as<EvaluateObj>();
  return evaluate && (tirx::IsNoOp(body) || IsStandaloneCrossCoreSync(body));
}

class PerCoreControlValidator : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  explicit PerCoreControlValidator(uint16_t pipe)
      : expected_("PIPE_" + GetResourcePipeName(pipe)) {}
  Optional<ffi::VisitInterrupt> Visit_(const CallObj *op) final {
    if (task_resource_detail::IsCrossCoreSyncCall(op)) {
      TVM_FFI_CHECK_EQ(op->args.size(), 3U, ValueError);
      TVM_FFI_CHECK(op->args[0].as<IntImmObj>(), ValueError)
          << "Cross-core synchronization mode must be a constant integer";
      const auto *pipe = op->args[1].as<StringImmObj>();
      TVM_FFI_CHECK(pipe && pipe->value == expected_, ValueError)
          << "Inter-core synchronization inside T.PerCoreTask must use its "
             "inferred pipe "
          << expected_;
    }
    return StmtExprVisitor::Visit_(op);
  }

private:
  std::string expected_;
};

void ValidatePerCoreTask(const RegionStmt &task) {
  ParseTaskMetadata(TaskAnnotations(task.get()));
  auto usage = AnalyzeTaskResourceUsage(task->body);
  TVM_FFI_CHECK(IsSingleResourcePipe(usage.pipe_mask), ValueError)
      << "T.PerCoreTask requires exactly one inferred hardware pipe, but its "
         "mask is "
      << usage.pipe_mask;
  bool has_dependency_task = false;
  std::function<void(const Stmt &)> visit = [&](const Stmt &stmt) {
    if (const auto *sequence = stmt.as<SeqStmtObj>()) {
      for (const Stmt &child : sequence->seq)
        visit(child);
      return;
    }
    if (const auto *region = stmt.as<RegionStmtObj>()) {
      if (IsTask(region)) {
        auto candidate = ValidateTask(GetRef<RegionStmt>(region), true);
        if (!candidate.has_buffer_access) {
          TVM_FFI_CHECK(IsPerCoreScalarControl(region->body), ValueError)
              << "T.PerCoreTask only permits scalar Bind/condition/loop "
                 "control and inter-core synchronization in addition to "
                 "operations on its inferred pipe";
          return;
        }
        TVM_FFI_CHECK(candidate.pipe_mask == usage.pipe_mask, ValueError)
            << "T.PerCoreTask candidate pipe does not match its inferred pipe";
        has_dependency_task = true;
        return;
      }
      TVM_FFI_CHECK(!IsPerCoreTask(region), ValueError)
          << "Nested T.PerCoreTask regions are not supported";
      auto source = ReadSourceAttribute(region);
      TVM_FFI_CHECK(source && (IsScheduleGuardAttribute(source->key) ||
                               IsMetadataScope(*source)),
                    ValueError)
          << "Unsupported region inside T.PerCoreTask: " << region->op->name;
      visit(region->body);
      return;
    }
    if (const auto *loop = stmt.as<ForObj>()) {
      TVM_FFI_CHECK(IsSerialLoop(loop), ValueError)
          << "T.PerCoreTask has an unmarked parallel loop";
      visit(loop->body);
      return;
    }
    if (const auto *branch = stmt.as<IfObj>()) {
      visit(branch->then_case);
      if (branch->else_case)
        visit(branch->else_case.value());
      return;
    }
    if (stmt.as<BindObj>()) {
      TVM_FFI_CHECK(!HasTaskBufferAccess(stmt), ValueError)
          << "An unmarked scalar Bind inside T.PerCoreTask may not access "
             "buffers";
      return;
    }
    if (IsTaskMetadataStatement(stmt))
      return;
    FatalUnmarkedTask("MaterializeScheduleUnits", stmt->GetTypeKey());
  };
  visit(task->body);
  TVM_FFI_CHECK(has_dependency_task, ValueError)
      << "T.PerCoreTask must contain at least one dependency-bearing task "
         "after task normalization";
  auto validator = ffi::make_object<PerCoreControlValidator>(usage.pipe_mask);
  validator->Visit(task->body);
}

// Walk units belonging to one scheduling child list. Branches and assumption
// regions retain lexical scope; a serial loop begins a separate child list.
Stmt MapChildUnits(const Stmt &stmt,
                   const std::function<Stmt(const Stmt &)> &map) {
  if (const auto *sequence = stmt.as<SeqStmtObj>()) {
    Array<Stmt> body;
    for (const Stmt &child : sequence->seq)
      body.push_back(MapChildUnits(child, map));
    return SeqStmt(body, stmt->loc);
  }
  if (const auto *bind = stmt.as<BindObj>(); IsScheduledBind(bind))
    return map(stmt);
  if (const auto *region = stmt.as<RegionStmtObj>(); IsScheduleUnit(region))
    return map(stmt);
  if (const auto *branch = stmt.as<IfObj>()) {
    If rewritten = GetRef<If>(branch);
    auto *writer = rewritten.CopyOnWrite();
    writer->then_case = SeqStmt(MapChildUnits(branch->then_case, map));
    if (branch->else_case)
      writer->else_case =
          SeqStmt(MapChildUnits(branch->else_case.value(), map));
    return rewritten;
  }
  if (auto attribute = ReadSourceAttribute(stmt)) {
    TVM_FFI_CHECK(IsScheduleGuardAttribute(attribute->key) ||
                      IsMetadataScope(attribute.value()),
                  ValueError)
        << "Expected a scheduling guard or loop metadata scope, got "
        << attribute->key;
    return attribute->WithBody(MapChildUnits(attribute->region->body, map));
  }
  TVM_FFI_CHECK(IsTaskMetadataStatement(stmt), ValueError)
      << "Incomplete native schedule: unmarked " << stmt->GetTypeKey();
  return stmt;
}

Stmt SetUnitStage(const Stmt &stmt, int stage) {
  if (const auto *bind = stmt.as<BindObj>(); IsScheduledBind(bind)) {
    auto binding = ReadScheduledBinding(bind->value.as<CallObj>());
    return WithScheduledBindingMetadata(GetRef<Bind>(bind), stage,
                                        binding.task);
  }
  return WithScheduleUnitStage(stmt.as_or_throw<RegionStmt>(), stage);
}

void ValidateScheduledList(const Stmt &body) {
  MapChildUnits(body, [](const Stmt &stmt) {
    GetScheduleUnitStage(stmt);
    if (const auto *bind = stmt.as<BindObj>(); IsScheduledBind(bind)) {
      ReadScheduledBinding(bind->value.as<CallObj>());
      return stmt;
    }
    auto unit = stmt.as_or_throw<RegionStmt>();
    Stmt payload = SingleStatement(unit->body);
    if (const auto *loop = payload.as<ForObj>()) {
      TVM_FFI_CHECK(IsSerialLoop(loop), ValueError)
          << "Scheduled control must be a serial or unrolled loop";
      ValidateScheduledList(loop->body);
    } else {
      const auto *task = payload.as<RegionStmtObj>();
      TVM_FFI_CHECK(IsSemanticTask(task), ValueError)
          << "Scheduled leaf must contain exactly one T.Task/T.PerCoreTask";
      if (IsTask(task))
        ValidateTask(GetRef<RegionStmt>(task), false);
      else
        ValidatePerCoreTask(GetRef<RegionStmt>(task));
    }
    return stmt;
  });
}

class ScheduleUnitMaterializer {
public:
  static SeqStmt Rewrite(const SeqStmt &body) {
    bool already_scheduled = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        body,
        [&](const ffi::ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *region = object.as<RegionStmtObj>())
            already_scheduled |= IsScheduleUnit(region);
          if (const auto *call = object.as<CallObj>())
            already_scheduled |= IsScheduledBind(call);
          return ffi::WalkResult::Advance();
        });
    if (already_scheduled) {
      ValidateScheduledList(body);
      return body;
    }
    ScheduleUnitMaterializer materializer;
    Stmt normalized = TaskNormalizer().Rewrite(body);
    return materializer.MaterializeChildList(normalized);
  }

private:
  Stmt EnsureNonEmpty(Stmt body) {
    bool has_unit = false;
    MapChildUnits(body, [&](const Stmt &unit) {
      has_unit = true;
      return unit;
    });
    if (!has_unit) {
      auto empty_task = MakeTask(SeqStmt(Array<Stmt>{}, body->loc));
      body = SeqStmt({body, MakeScheduleUnit(empty_task, kUnscheduledStage)},
                     body->loc);
    }
    return body;
  }

  SeqStmt MaterializeChildList(const Stmt &stmt) {
    Stmt result = EnsureNonEmpty(Materialize(stmt));
    bool manual = false;
    MapChildUnits(result, [&](const Stmt &unit) {
      manual |= GetScheduleUnitStage(unit) != kUnscheduledStage;
      return unit;
    });
    if (manual)
      result = MapChildUnits(result, [](const Stmt &unit) {
        return GetScheduleUnitStage(unit) == kUnscheduledStage
                   ? SetUnitStage(unit, 0)
                   : unit;
      });
    return SeqStmt(result);
  }

  Stmt Materialize(const Stmt &stmt) {
    if (const auto *sequence = stmt.as<SeqStmtObj>()) {
      Array<Stmt> body;
      for (const Stmt &child : sequence->seq)
        body.push_back(Materialize(child));
      return SeqStmt(body, stmt->loc);
    }
    if (IsTaskMetadataStatement(stmt))
      return stmt;
    if (const auto *bind = stmt.as<BindObj>())
      return MakeScheduledBinding(GetRef<Bind>(bind), kUnscheduledStage);
    if (const auto *loop = stmt.as<ForObj>()) {
      TVM_FFI_CHECK(IsSerialLoop(loop), ValueError)
          << "MaterializeScheduleUnits found an unmarked parallel loop";
      For rewritten = GetRef<For>(loop);
      rewritten.CopyOnWrite()->body = MaterializeChildList(loop->body);
      return MakeScheduleUnit(rewritten, kUnscheduledStage);
    }
    if (const auto *branch = stmt.as<IfObj>()) {
      If rewritten = GetRef<If>(branch);
      auto *writer = rewritten.CopyOnWrite();
      writer->then_case = SeqStmt(Materialize(branch->then_case));
      if (branch->else_case)
        writer->else_case = SeqStmt(Materialize(branch->else_case.value()));
      return rewritten;
    }
    const auto *region = stmt.as<RegionStmtObj>();
    TVM_FFI_CHECK(region, ValueError)
        << "MaterializeScheduleUnits found an unmarked " << stmt->GetTypeKey();
    if (IsTask(region)) {
      ValidateTask(GetRef<RegionStmt>(region), false);
      return MakeScheduleUnit(stmt, kUnscheduledStage);
    }
    if (IsPerCoreTask(region)) {
      ValidatePerCoreTask(GetRef<RegionStmt>(region));
      return MakeScheduleUnit(stmt, kUnscheduledStage);
    }
    if (IsStageScope(region)) {
      int stage = GetStageScopeStage(region);
      Stmt body = EnsureNonEmpty(Materialize(region->body));
      return MapChildUnits(body, [&](const Stmt &unit) {
        TVM_FFI_CHECK_EQ(GetScheduleUnitStage(unit), kUnscheduledStage,
                         ValueError)
            << "Nested T.Stage scopes are not supported";
        return SetUnitStage(unit, stage);
      });
    }
    auto attribute = ReadSourceAttribute(region);
    TVM_FFI_CHECK(attribute.has_value(), ValueError)
        << "MaterializeScheduleUnits found an unsupported region "
        << region->op->name;
    TVM_FFI_CHECK(IsScheduleGuardAttribute(attribute->key) ||
                      IsMetadataScope(attribute.value()),
                  ValueError)
        << "MaterializeScheduleUnits only supports known scheduling "
           "attributes inside a kernel; unsupported attr_key: "
        << attribute->key;
    return attribute->WithBody(
        EnsureNonEmpty(Materialize(attribute->region->body)));
  }
};

} // namespace

tvm::transform::Pass MaterializeScheduleUnits() {
  auto pass_func = [](tirx::Function function, const IRModule &,
                      const tvm::transform::PassContext &) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    CheckWellFormed(function);
    auto result = RewriteTilelangKernels(
        std::move(function), "MaterializeScheduleUnits",
        [](const TilelangKernelContext &context) {
          RegionStmt root = context.root;
          SeqStmt body = ScheduleUnitMaterializer::Rewrite(root->body);
          if (!body.same_as(root->body))
            root.CopyOnWrite()->body = std::move(body);
          return root;
        },
        false);
    CheckWellFormed(result);
    return result;
  };
  return tirx::transform::CreateFunctionPass(pass_func, 0,
                                             "tl.MaterializeScheduleUnits");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MaterializeScheduleUnits",
                                   MaterializeScheduleUnits);
}

} // namespace tvm::tl
