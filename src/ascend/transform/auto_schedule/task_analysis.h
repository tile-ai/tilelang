#include "compat/analyzer.h"
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

#include "op/prim.h"
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "./ir_structure.h"
#include "./memory_detector.h"
#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/transform/attr.h"
#include "op/copy.h"
#include "op/fill.h"
#include "op/utils.h"
#include "support/check.h"
#include "task_resources.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

class TaskAwareConstrVisitor : public ConstrVisitor {
protected:
  static Stmt GetUnconditionalStatement(Stmt statement) {
    statement = SingleScheduleStatement(statement);
    if (const auto *region = statement.as<RegionStmtNode>()) {
      auto key = ScheduleMarkerKey(region);
      if (key == attr::kAscendTask || key == attr::kScheduleUnit)
        return GetUnconditionalStatement(UnwrapScheduleMarker(region));
    }
    return statement;
  }

  static ffi::Optional<Bind> GetFlatBind(const Stmt &stmt) {
    return GetFlatTaskBind(stmt);
  }

  ffi::Optional<ffi::VisitInterrupt>
  Visit_(const RegionStmtNode *node) override {
    auto key = ScheduleMarkerKey(node);
    if (key == kScheduleCondition || IsScheduleGuardAttribute(key)) {
      auto condition = ScheduleMarkerPayload(node).as_or_throw<PrimExpr>();
      if (auto interrupt = Visit(condition))
        return interrupt;
      auto guard = MakeGuard(condition, IsScheduleGuardAttribute(key));
      return Visit(UnwrapScheduleMarker(node));
    }
    if (!key.empty())
      return Visit(UnwrapScheduleMarker(node));
    return ConstrVisitor::Visit_(node);
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const SeqStmtNode *node) override {
    size_t old_size = constr_stack_.size();
    for (const Stmt &statement : node->seq) {
      auto interrupt = Visit(statement);
      if (interrupt) {
        constr_stack_.resize(old_size);
        return interrupt;
      }
      Stmt payload = GetUnconditionalStatement(statement);
      if (auto binding = payload.as<Bind>()) {
        auto variable = binding.value()->var.as<PrimVar>();
        auto value = binding.value()->value.as<PrimExpr>();
        if (variable && value)
          constr_stack_.emplace_back(variable.value(), value.value());
      } else if (const auto *assertion = payload.as<AssertStmtNode>()) {
        if (SideEffect(assertion->condition) == CallEffectKind::kPure)
          constr_stack_.emplace_back(assertion->condition);
      } else if (const auto *evaluate = payload.as<EvaluateNode>()) {
        if (const auto *call = evaluate->value.as<CallNode>();
            call && call->op.same_as(prim::assume_op()))
          constr_stack_.emplace_back(call->args[0].as_or_throw<PrimExpr>(),
                                     true);
      }
    }
    constr_stack_.resize(old_size);
    return std::nullopt;
  }
};

struct RegionElementUpperBound {
  int64_t elements{1};
  // False only for the last-resort extent=1 heuristic. Constant, analyzer, and
  // static-buffer-shape bounds are all usable for tightening an equal copy
  // region from the other side.
  bool reliable{true};
};

// Estimate a region's maximum element count under the current task context.
// A symbolic extent first uses Analyzer bounds, then the corresponding static
// buffer dimension, and finally the legacy extent=1 heuristic. Static buffer
// dimensions also cap weak dtype-wide Analyzer bounds.
inline RegionElementUpperBound
EstimateRegionElementUpperBound(const TensorVar &buffer,
                                const ffi::Array<Range> &ranges,
                                sym::Analyzer analyzer) {
  RegionElementUpperBound result;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const PrimExpr &extent_expr = ranges[i]->extent;
    int64_t extent = 1;
    bool extent_reliable = true;

    if (auto extent_int = GetConstInt(extent_expr)) {
      extent = *extent_int;
    } else {
      auto shape_extent = i < buffer.shape().size()
                              ? GetConstInt(buffer.shape()[i])
                              : std::nullopt;
      sym::ConstIntBound bound =
          tl::arith::ConstIntBoundOf(analyzer, extent_expr);

      sym::Analyzer unconstrained_analyzer;
      PrimVar unconstrained_extent("unconstrained_extent", extent_expr.ty());
      int64_t unconstrained_max =
          tl::arith::ConstIntBoundOf(unconstrained_analyzer,
                                     unconstrained_extent)
              ->max_value;
      bool has_finite_bound = bound->max_value >= 0 &&
                              bound->max_value != sym::ConstIntBound::kPosInf;
      bool has_useful_bound =
          has_finite_bound && bound->max_value < unconstrained_max;

      if (shape_extent.has_value()) {
        extent = has_finite_bound ? std::min(bound->max_value, *shape_extent)
                                  : *shape_extent;
      } else if (has_useful_bound) {
        extent = bound->max_value;
      } else {
        extent_reliable = false;
      }
    }

    result.elements *= extent;
    result.reliable = result.reliable && extent_reliable;
  }
  return result;
}

inline PrimExpr RegionElementCountExpr(const ffi::Array<Range> &ranges) {
  PrimExpr elements = tl::MakeConst(PrimType::Int(64), 1);
  for (const Range &range : ranges)
    elements = elements * Cast(PrimType::Int(64), range->extent);
  return elements;
}

inline int64_t ElementsToBytes(int64_t elements, PrimType dtype) {
  int64_t elem_bits = dtype.bits() * dtype.lanes();
  return (elements * elem_bits + 7) / 8;
}

// Hardware resource properties derived from one task body.
// Direct buffer and scalar-variable accesses made by one statement. Accesses
// remain separate from hardware resource classification because AutoSchedule
// also adds accesses from surrounding guards and loop expressions to a task.
struct TaskAccessInfo {
  std::vector<TensorRegion> read_regions;
  std::vector<TensorRegion> write_regions;
  Map<Var, Optional<PrimExpr>> l0_pitches;
  std::vector<Var> read_vars;
  std::vector<Var> write_vars;
};

namespace task_analysis_detail {

// Collect expressions evaluated by scalar control rather than by a task's
// hardware data path. TensorVar-region indices are handled separately because a
// tile operation may legally transfer data between memories owned by different
// cores while its address expressions must still be available on the issuing
// core.
class ScalarControlExpressionCollector : public AnalysisVisitor {
public:
  static std::vector<PrimExpr> Collect(const Stmt &stmt) {
    auto collector = ffi::make_object<ScalarControlExpressionCollector>();
    collector->Visit(stmt);
    return std::move(collector->expressions_);
  }

private:
  void VisitStmt_(const BindNode *op) final {
    if (auto scalar = op->value.as<PrimExpr>())
      expressions_.push_back(scalar.value());
  }

  void VisitStmt_(const RegionStmtNode *op) final {
    auto key = ScheduleMarkerKey(op);
    if (!key.empty()) {
      if (key != attr::kAscendTask && key != attr::kAscendPerCoreTask &&
          key != attr::kScheduleUnit)
        if (auto scalar = ScheduleMarkerPayload(op).as<PrimExpr>())
          expressions_.push_back(scalar.value());
      VisitStmt(UnwrapScheduleMarker(op));
      return;
    }
    for (const Expr &argument : op->args)
      if (auto scalar = argument.as<PrimExpr>())
        expressions_.push_back(scalar.value());
    VisitStmt(op->body);
  }

  void VisitStmt_(const IfNode *op) final {
    expressions_.push_back(op->condition);
    VisitStmt(op->then_case);
    if (op->else_case.has_value())
      VisitStmt(op->else_case.value());
  }

  void VisitStmt_(const ForNode *op) final {
    expressions_.push_back(op->begin);
    expressions_.push_back(op->end);
    expressions_.push_back(op->step);
    VisitStmt(op->body);
  }

  void VisitStmt_(const WhileNode *op) final {
    expressions_.push_back(op->condition);
    VisitStmt(op->body);
  }

  void VisitStmt_(const AssertStmtNode *op) final {
    expressions_.push_back(op->condition);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call != nullptr && IsCrossCoreSyncCall(call)) {
      ICHECK_EQ(call->args.size(), 3)
          << "ascend_cross_core_set_flag/wait_flag expects 3 arguments";
      expressions_.push_back(call->args[2].as_or_throw<PrimExpr>());
    }
  }

  void VisitStmt_(const TensorStoreNode *) final {}
  std::vector<PrimExpr> expressions_;
};

} // namespace task_analysis_detail

inline TaskAccessInfo AnalyzeTaskAccesses(const Stmt &stmt,
                                          const LayoutMap &layouts = {}) {
  MemoryAccessDetector memory_detector(
      MemoryAccessDetector::AccessMode::kLogical, layouts);
  memory_detector.Analyze(stmt);
  TaskAccessInfo result;
  result.read_regions = memory_detector.GetReadRegions();
  result.write_regions = memory_detector.GetWriteRegions();
  result.l0_pitches = memory_detector.GetL0Pitches();
  result.read_vars = memory_detector.GetReadVars();
  result.write_vars = memory_detector.GetWriteVars();
  return result;
}

inline bool
TaskAccessesStorage(const TaskAccessInfo &accesses, const Var &storage,
                    const StorageAliasAnalysis *analysis = nullptr) {
  auto matches = [&](const Expr &expression) {
    if (expression.same_as(storage))
      return true;
    if (analysis)
      if (auto access = analysis->Resolve(expression))
        return access->root.same_as(storage);
    return false;
  };
  auto regions_access = [&](const std::vector<TensorRegion> &regions) {
    return std::any_of(regions.begin(), regions.end(), [&](const auto &region) {
      return matches(region->source);
    });
  };
  if (regions_access(accesses.read_regions) ||
      regions_access(accesses.write_regions)) {
    return true;
  }
  auto vars_access = [&](const std::vector<Var> &vars) {
    return std::any_of(vars.begin(), vars.end(),
                       [&](const Var &var) { return matches(var); });
  };
  return vars_access(accesses.read_vars) || vars_access(accesses.write_vars);
}

inline bool HasCrossCoreSync(const Stmt &stmt) {
  class Detector : public AnalysisVisitor {
  public:
    bool found{false};

  private:
    void VisitExpr_(const CallNode *op) final {
      if (task_analysis_detail::IsCrossCoreSyncCall(op)) {
        found = true;
        return;
      }
      AnalysisVisitor::VisitExpr_(op);
    }
  };
  auto detector = ffi::make_object<Detector>();
  detector->Visit(stmt);
  return detector->found;
}

// Queries where scalar values and control expressions can be evaluated from
// the candidate core masks already attached to their defining tasks. This is
// intentionally stateless with respect to final placement: AssignCore and
// ResolveCore update candidates to a fixed point, while ResolveCore separately
// propagates actual execution demand.
class CoreMaskAvailability {
public:
  CoreMaskAvailability() = default;

  explicit CoreMaskAvailability(const std::vector<TaskNode *> &tasks) {
    AddTasks(tasks);
  }

  void AddTasks(const std::vector<TaskNode *> &tasks) {
    for (const TaskNode *task : tasks) {
      if (!storage_context_)
        storage_context_ = task;
      for (const Var &var : task->GetWriteVars())
        writers_[var].push_back(task);
      for (const TensorRegion &region : task->GetWriteRegions()) {
        if (IsRegisterRegion(region))
          storage_writers_[task->StorageOf(
                               region->source.as_or_throw<TensorVar>())]
              .push_back(task);
      }
    }
  }

  void SetExternalVarCoreMask(const Var &var, CoreMask core_mask) {
    ICHECK(IsValidCoreMask(core_mask));
    external_vars_[var] = core_mask;
  }

  CoreMask GetValueCoreMask(const Var &value,
                            const TaskNode *reader = nullptr) const {
    auto external = external_vars_.find(value);
    if (external != external_vars_.end())
      return external->second;

    return GetWriterCoreMask(writers_, value, reader);
  }

  CoreMask GetExpressionCoreMask(const PrimExpr &expr,
                                 const TaskNode *reader = nullptr) const {
    if (!expr.defined())
      return kCoreAll;
    return GetAccessCoreMask(AnalyzeTaskAccesses(Evaluate(expr)), reader);
  }

  CoreMask GetAccessCoreMask(const TaskAccessInfo &accesses,
                             const TaskNode *reader = nullptr) const {
    CoreMask result = kCoreAll;
    for (const Var &var : accesses.read_vars)
      result &= GetValueCoreMask(var, reader);
    for (const TensorRegion &region : accesses.read_regions) {
      if (IsSharedBuffer(region->source.as_or_throw<TensorVar>())) {
        result &= kCoreVector;
      } else if (IsL1Buffer(region->source.as_or_throw<TensorVar>()) ||
                 IsL0ABuffer(region->source.as_or_throw<TensorVar>()) ||
                 IsL0BBuffer(region->source.as_or_throw<TensorVar>()) ||
                 IsL0CBuffer(region->source.as_or_throw<TensorVar>())) {
        result &= kCoreCube;
      } else if (IsRegisterRegion(region)) {
        result &= GetRegisterCoreMask(region, reader);
      }
    }
    return result;
  }

  CoreMask GetTaskReadCoreMask(const TaskNode *task) const {
    ICHECK(task != nullptr);
    TaskAccessInfo accesses;
    accesses.read_regions = task->GetReadRegions();
    accesses.read_vars = task->GetReadVars();
    return GetAccessCoreMask(accesses, task);
  }

  CoreMask GetTaskInputCoreMask(const TaskNode *task) const {
    ICHECK(task != nullptr);
    CoreMask result = GetValueInputCoreMask(task);

    for (const TensorRegion &region : task->GetReadRegions())
      result &= GetRegionIndexCoreMask(region, task);
    for (const TensorRegion &region : task->GetWriteRegions())
      result &= GetRegionIndexCoreMask(region, task);

    for (const PrimExpr &expr :
         task_analysis_detail::ScalarControlExpressionCollector::Collect(
             task->stmt)) {
      result &= GetExpressionCoreMask(expr, task);
    }
    for (const auto &guard : task->GetGuards()) {
      if (guard->IsCondition()) {
        const auto *condition =
            static_cast<const ConditionGuard *>(guard.get());
        result &= GetExpressionCoreMask(condition->condition, task);
        continue;
      }
      const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
      if (auto expression = attribute->node.as<PrimExpr>())
        result &= GetExpressionCoreMask(expression.value(), task);
    }

    uint16_t scalar_pipe = static_cast<uint16_t>(ResourcePipe::kScalar);
    if (task->GetPipeMask() == scalar_pipe)
      result &= GetTaskReadCoreMask(task);
    return result;
  }

private:
  using WriterMap = std::unordered_map<Var, std::vector<const TaskNode *>,
                                       ObjectPtrHash, ObjectPtrEqual>;
  using CoreMap =
      std::unordered_map<Var, CoreMask, ObjectPtrHash, ObjectPtrEqual>;

  static CoreMask GetWriterCoreMask(const WriterMap &writers, const Var &value,
                                    const TaskNode *reader) {
    auto found = writers.find(value);
    if (found == writers.end())
      return kCoreAll;
    CoreMask result = kCoreAll;
    for (const TaskNode *writer : found->second) {
      if (writer != reader)
        result &= writer->GetCoreMask();
    }
    return result;
  }

  CoreMask GetRegisterCoreMask(const TensorRegion &region,
                               const TaskNode *reader) const {
    auto tensor = region->source.as_or_throw<TensorVar>();
    const TaskNode *context = reader ? reader : storage_context_;
    Var storage =
        context ? context->StorageOf(tensor) : static_cast<Var>(tensor);
    return GetValueCoreMask(tensor, reader) &
           GetWriterCoreMask(storage_writers_, storage, reader);
  }

  CoreMask GetValueInputCoreMask(const TaskNode *task) const {
    CoreMask result = kCoreAll;
    for (const Var &var : task->GetReadVars())
      result &= GetValueCoreMask(var, task);
    for (const TensorRegion &region : task->GetReadRegions()) {
      if (IsRegisterRegion(region))
        result &= GetRegisterCoreMask(region, task);
    }
    return result;
  }

  CoreMask GetRegionIndexCoreMask(const TensorRegion &region,
                                  const TaskNode *reader) const {
    CoreMask result = kCoreAll;
    for (const Range &range : region->region) {
      result &= GetExpressionCoreMask(range->begin, reader);
      result &= GetExpressionCoreMask(range->extent, reader);
    }
    return result;
  }

  WriterMap writers_;
  WriterMap storage_writers_;
  CoreMap external_vars_;
  const TaskNode *storage_context_{nullptr};
};

inline bool HasTaskBufferAccess(const TaskNode *task) {
  ICHECK(task != nullptr);
  return !task->GetReadRegions().empty() || !task->GetWriteRegions().empty();
}

inline void ApplyTaskResourceUsage(const TaskResourceUsage &usage,
                                   TaskNode *task) {
  ICHECK(task != nullptr);
  task->SetPipeMask(usage.pipe_mask);
  task->SetHbmMask(usage.hbm_mask);
  if (usage.reads_pad_value)
    task->SetReadsPadValue(true);
  if (usage.reads_atomic)
    task->SetReadsAtomic(true);
}

inline void ApplyTaskAccesses(const TaskAccessInfo &accesses, TaskNode *task) {
  ICHECK(task != nullptr);
  for (const TensorRegion &region : accesses.read_regions)
    task->AddReadRegion(region);
  for (const TensorRegion &region : accesses.write_regions)
    task->AddWriteRegion(region);
  auto pitches = task->GetL0Pitches();
  for (const auto &[storage, pitch] : accesses.l0_pitches)
    MergeL0Pitch(&pitches, storage, pitch);
  task->SetL0Pitches(pitches);
  for (const Var &var : accesses.read_vars)
    task->AddReadVar(var);
  for (const Var &var : accesses.write_vars)
    task->AddWriteVar(var);
}

} // namespace ascend
} // namespace tl
} // namespace tvm
