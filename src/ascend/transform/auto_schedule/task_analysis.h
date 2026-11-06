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

#include "ascend/ir/task.h"
#include "ascend/transform/source_scopes.h"
#include "ir/stmt_functor.h"
#include "ir_structure.h"
#include "layout/expr_utils.h"
#include "memory_detector.h"
#include "task_resource.h"
#include "transform/common/constr_visitor.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tvm::tl::ascend {
using namespace tirx;
using namespace prim;

inline ffi::Optional<Bind> GetFlatTaskBind(const Stmt &statement) {
  Stmt stmt = SingleStatement(statement);
  if (const auto *bind = stmt.as<BindObj>()) {
    if (IsScheduledBind(bind))
      return Bind(bind->var,
                  ReadScheduledBinding(bind->value.as<CallObj>()).value,
                  bind->loc);
    return ffi::GetRef<Bind>(bind);
  }
  if (const auto *region = stmt.as<RegionStmtObj>();
      IsTask(region) || IsScheduleUnit(region))
    return GetFlatTaskBind(region->body);
  return std::nullopt;
}

inline Stmt ReplaceFlatTaskBind(const Stmt &statement,
                                const Bind &replacement) {
  if (const auto *sequence = statement.as<SeqStmtObj>()) {
    TVM_FFI_CHECK_EQ(sequence->seq.size(), 1U, ValueError);
    return SeqStmt({ReplaceFlatTaskBind(sequence->seq[0], replacement)},
                   statement->loc);
  }
  if (const auto *bind = statement.as<BindObj>()) {
    if (IsScheduledBind(bind)) {
      auto marker = ReadScheduledBinding(bind->value.as<CallObj>());
      return MakeScheduledBinding(replacement, marker.stage, marker.task);
    }
    return replacement;
  }
  const auto *region = statement.as<RegionStmtObj>();
  TVM_FFI_CHECK(IsTask(region) || IsScheduleUnit(region), ValueError)
      << "Expected a flat scheduled binding";
  RegionStmt result = ffi::GetRef<RegionStmt>(region);
  result.CopyOnWrite()->body =
      SeqStmt(ReplaceFlatTaskBind(region->body, replacement));
  return result;
}

inline ffi::Optional<PrimExpr> GetFlatTaskAssumption(const Stmt &statement) {
  Stmt stmt = SingleStatement(statement);
  if (const auto *evaluate = stmt.as<EvaluateObj>()) {
    const auto *call = evaluate->value.as<CallObj>();
    if (call && call->op.same_as(prim::assume_op())) {
      PrimExpr condition = call->args[0].as_or_throw<PrimExpr>();
      // A later store or opaque call may change a memory predicate. Only
      // replayable values can describe subsequent uses; a native Var bound
      // to a load is a stable snapshot and remains eligible here.
      if (SideEffect(condition) <= CallEffectKind::kPure)
        return condition;
    }
  }
  if (const auto *region = stmt.as<RegionStmtObj>();
      IsTask(region) || IsScheduleUnit(region))
    return GetFlatTaskAssumption(region->body);
  return std::nullopt;
}

class TaskAwareConstrVisitor : public ConstrVisitor {
public:
  using ConstrVisitor::Visit;
  using ConstrVisitor::Visit_;

protected:
  Result Visit_(const SeqStmtObj *op) override {
    auto scope = Scope();
    for (const Stmt &stmt : op->seq) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(stmt));
      if (auto bind = GetFlatTaskBind(stmt)) {
        if (auto value = bind->value.as<PrimExpr>())
          constr_stack_.emplace_back(bind->var, value.value());
      } else if (const auto *assertion = stmt.as<AssertStmtObj>()) {
        if (SideEffect(assertion->condition) <= CallEffectKind::kPure)
          constr_stack_.emplace_back(assertion->condition);
      } else if (auto assumption = GetFlatTaskAssumption(stmt)) {
        // A task wrapper preserves the source-order fact established by its
        // single assume statement. A branch or nested sequence does not.
        constr_stack_.emplace_back(assumption.value(), true);
      }
    }
    return std::nullopt;
  }

  Result Visit_(const IfObj *op) override {
    // A memory predicate describes the value read on branch entry. A later
    // read after a store is a different value, so it cannot inherit that
    // predicate. Pure parameters and explicit SSA snapshots remain usable.
    if (SideEffect(op->condition) > CallEffectKind::kPure)
      return StmtExprVisitor::Visit_(op);
    return ConstrVisitor::Visit_(op);
  }

  Result Visit_(const RegionStmtObj *op) override {
    auto attribute = ReadSourceAttribute(op);
    if (attribute && IsScheduleGuardAttribute(attribute->key)) {
      for (const Expr &arg : op->args)
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(arg));
      auto scope = Scope();
      PrimExpr condition = attribute->node.cast<PrimExpr>();
      if (SideEffect(condition) <= CallEffectKind::kPure)
        constr_stack_.emplace_back(condition, true);
      return Visit(op->body);
    }
    return ConstrVisitor::Visit_(op);
  }
};

struct RegionElementUpperBound {
  int64_t elements{1};
  bool reliable{true};
};

inline RegionElementUpperBound
EstimateRegionElementUpperBound(const TensorVar &tensor, const Region &ranges,
                                const sym::AnalyzerObj *analyzer) {
  RegionElementUpperBound result;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const PrimExpr &expression = ranges[i]->extent;
    auto constant = TryConstInt(expression);
    int64_t extent = 1;
    bool reliable = true;
    if (constant) {
      extent = *constant;
    } else {
      auto shape = i < tensor.shape().size() ? TryConstInt(tensor.shape()[i])
                                             : std::nullopt;
      auto bound = analyzer->const_int_bound(expression);
      sym::Analyzer unconstrained;
      PrimVar variable("unconstrained_extent", expression.ty());
      bool finite = bound->max_value >= 0 &&
                    bound->max_value != sym::ConstIntBound::kPosInf;
      if (shape)
        extent = finite ? std::min(bound->max_value, *shape) : *shape;
      else if (finite &&
               bound->max_value <
                   unconstrained->const_int_bound(variable)->max_value)
        extent = bound->max_value;
      else
        reliable = false;
    }
    extent = std::max<int64_t>(0, extent);
    TVM_FFI_CHECK(extent == 0 ||
                      result.elements <=
                          std::numeric_limits<int64_t>::max() / extent,
                  ValueError)
        << "Task region element count overflows int64";
    result.elements *= extent;
    result.reliable &= reliable;
  }
  return result;
}

inline PrimExpr RegionElementCountExpr(const Region &ranges) {
  PrimExpr elements = IntImm::Int64(1);
  for (const Range &range : ranges)
    elements = elements * cast(range->extent, PrimType::Int(64));
  return elements;
}

inline int64_t ElementsToBytes(int64_t elements, PrimType dtype) {
  int64_t bits = dtype.bits() * dtype.lanes();
  TVM_FFI_CHECK(elements >= 0 && bits > 0 &&
                    elements <=
                        (std::numeric_limits<int64_t>::max() - 7) / bits,
                ValueError)
      << "Task byte count overflows int64";
  return (elements * bits + 7) / 8;
}

struct TaskAccessInfo {
  std::vector<TensorRegion> read_regions;
  std::vector<TensorRegion> write_regions;
  Map<Var, Optional<PrimExpr>> l0_pitches;
  std::vector<Var> read_vars;
  std::vector<Var> write_vars;
  std::vector<Var> opaque_access_vars;
  StorageProvenance storage = MakeAscendStorageProvenance();
};

namespace task_analysis_detail {
using task_resource_detail::IsCrossCoreSyncCall;
using task_resource_detail::SpecialRegisterForCrossCorePipe;

class ScalarControlExpressionCollector {
public:
  static std::vector<PrimExpr> Collect(const Stmt &stmt) {
    ScalarControlExpressionCollector collector;
    collector.Visit(stmt);
    return std::move(collector.expressions_);
  }

private:
  void Add(const Expr &value) {
    if (auto primitive = value.as<PrimExpr>())
      expressions_.push_back(*primitive);
  }
  void Visit(const Stmt &stmt) {
    if (const auto *sequence = stmt.as<SeqStmtObj>()) {
      for (const Stmt &child : sequence->seq)
        Visit(child);
    } else if (const auto *bind = stmt.as<BindObj>()) {
      if (IsScheduledBind(bind))
        Add(ReadScheduledBinding(bind->value.as<CallObj>()).value);
      else
        Add(bind->value);
    } else if (const auto *branch = stmt.as<IfObj>()) {
      Add(branch->condition);
      Visit(branch->then_case);
      if (branch->else_case)
        Visit(*branch->else_case);
    } else if (const auto *loop = stmt.as<ForObj>()) {
      Add(loop->begin);
      Add(loop->end);
      Add(loop->step);
      Visit(loop->body);
    } else if (const auto *loop = stmt.as<WhileObj>()) {
      Add(loop->condition);
      Visit(loop->body);
    } else if (const auto *assertion = stmt.as<AssertStmtObj>()) {
      Add(assertion->condition);
    } else if (const auto *evaluate = stmt.as<EvaluateObj>()) {
      const auto *call = evaluate->value.as<CallObj>();
      if (call && IsCrossCoreSyncCall(call))
        Add(call->args[2]);
    } else if (const auto *region = stmt.as<RegionStmtObj>()) {
      if (auto attribute = ReadSourceAttribute(region)) {
        if (auto value = attribute->node.try_cast<Expr>())
          Add(*value);
        if (auto value = attribute->value.try_cast<Expr>())
          Add(*value);
      } else if (!IsAscendBlock(region) && !IsSemanticTask(region) &&
                 !IsScheduleUnit(region))
        for (const Expr &arg : region->args)
          Add(arg);
      Visit(region->body);
    }
  }
  std::vector<PrimExpr> expressions_;
};
} // namespace task_analysis_detail

inline TaskAccessInfo AnalyzeTaskAccesses(
    const Stmt &stmt, const LayoutMap &layouts = {},
    const StorageProvenance &storage = MakeAscendStorageProvenance(),
    const ConstrSet &context = {}) {
  auto detector = ffi::make_object<MemoryAccessDetector>(
      MemoryAccessDetector::AccessMode::kLogical, layouts, storage, context);
  detector->Analyze(stmt);
  TaskAccessInfo result;
  result.read_regions = detector->GetReadRegions();
  result.write_regions = detector->GetWriteRegions();
  result.l0_pitches = detector->GetL0Pitches();
  result.read_vars = detector->GetReadVars();
  result.write_vars = detector->GetWriteVars();
  result.opaque_access_vars = detector->GetOpaqueAccessVars();
  result.storage = storage;
  result.storage.Collect(stmt);
  return result;
}

inline bool TaskAccessesStorage(const TaskAccessInfo &accesses,
                                const Var &storage) {
  auto regions_access = [&](const std::vector<TensorRegion> &regions) {
    return std::any_of(regions.begin(), regions.end(), [&](const auto &region) {
      return accesses.storage.Identity(RegionTensor(region)).same_as(storage);
    });
  };
  if (regions_access(accesses.read_regions) ||
      regions_access(accesses.write_regions)) {
    return true;
  }
  auto vars_access = [&](const std::vector<Var> &vars) {
    return std::any_of(vars.begin(), vars.end(),
                       [&](const Var &var) { return var.same_as(storage); });
  };
  return vars_access(accesses.read_vars) || vars_access(accesses.write_vars);
}

inline bool HasCrossCoreSync(const Stmt &stmt) {
  bool found = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      stmt, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        found |= task_analysis_detail::IsCrossCoreSyncCall(call);
        return ffi::WalkResult::Advance();
      });
  return found;
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
      for (const Var &var : task->GetWriteVars())
        writers_[var].push_back(task);
      for (const TensorRegion &region : task->GetWriteRegions()) {
        if (IsRegisterRegion(region)) {
          writers_[task->StorageIdentity(RegionTensor(region))].push_back(task);
          register_writers_.emplace_back(RegionTensor(region), task);
        }
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

    auto writers = writers_.find(value);
    if (writers == writers_.end())
      return kCoreAll;
    CoreMask result = kCoreAll;
    bool found = false;
    for (const TaskNode *writer : writers->second) {
      if (writer == reader)
        continue;
      found = true;
      result &= writer->GetCoreMask();
    }
    return found ? result : kCoreAll;
  }

  CoreMask GetExpressionCoreMask(const PrimExpr &expr,
                                 const TaskNode *reader = nullptr) const {
    if (!expr.defined())
      return kCoreAll;
    return GetAccessCoreMask(
        AnalyzeTaskAccesses(Evaluate(expr), {},
                            reader ? reader->StorageContext()
                                   : MakeAscendStorageProvenance()),
        reader);
  }

  CoreMask GetAccessCoreMask(const TaskAccessInfo &accesses,
                             const TaskNode *reader = nullptr) const {
    CoreMask result = kCoreAll;
    for (const Var &var : accesses.read_vars)
      result &= GetValueCoreMask(var, reader);
    for (const TensorRegion &region : accesses.read_regions) {
      result &= GetValueCoreMask(RegionTensor(region), reader);
      if (IsSharedBuffer(RegionTensor(region))) {
        result &= kCoreVector;
      } else if (IsL1Buffer(RegionTensor(region)) ||
                 IsL0ABuffer(RegionTensor(region)) ||
                 IsL0BBuffer(RegionTensor(region)) ||
                 IsL0CBuffer(RegionTensor(region))) {
        result &= kCoreCube;
      } else if (IsRegisterRegion(region)) {
        result &=
            GetRegisterCoreMask(RegionTensor(region), accesses.storage, reader);
      }
    }
    return result;
  }

  CoreMask GetTaskReadCoreMask(const TaskNode *task) const {
    ICHECK(task != nullptr);
    TaskAccessInfo accesses;
    accesses.read_regions = task->GetReadRegions();
    accesses.read_vars = task->GetReadVars();
    accesses.storage = task->StorageContext();
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
      if (auto expr = attribute->node.try_cast<PrimExpr>())
        result &= GetExpressionCoreMask(*expr, task);
      if (auto expr = attribute->value.try_cast<PrimExpr>())
        result &= GetExpressionCoreMask(*expr, task);
    }

    uint16_t scalar_pipe = static_cast<uint16_t>(ResourcePipe::kScalar);
    if (task->GetPipeMask() == scalar_pipe)
      result &= GetTaskReadCoreMask(task);
    return result;
  }

private:
  CoreMask GetRegisterCoreMask(const TensorVar &tensor,
                               const StorageProvenance &storage,
                               const TaskNode *reader) const {
    CoreMask result = GetValueCoreMask(storage.Identity(tensor), reader);
    for (const auto &[written_tensor, writer] : register_writers_) {
      if (writer == reader ||
          ProvenDisjointAscendStorage(storage.Address(tensor),
                                      storage.Address(written_tensor)))
        continue;
      // External and integer-derived roots may alias. Only independent
      // alloc_tensor results prove that a writer cannot supply this value.
      result &= writer->GetCoreMask();
    }
    return result;
  }

  using WriterMap = std::unordered_map<Var, std::vector<const TaskNode *>,
                                       ObjectPtrHash, ObjectPtrEqual>;
  using CoreMap =
      std::unordered_map<Var, CoreMask, ObjectPtrHash, ObjectPtrEqual>;
  CoreMask GetValueInputCoreMask(const TaskNode *task) const {
    CoreMask result = kCoreAll;
    for (const Var &var : task->GetReadVars())
      result &= GetValueCoreMask(var, task);
    for (const TensorRegion &region : task->GetReadRegions()) {
      if (IsRegisterRegion(region))
        result &= GetRegisterCoreMask(RegionTensor(region),
                                      task->StorageContext(), task);
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
  std::vector<std::pair<TensorVar, const TaskNode *>> register_writers_;
  CoreMap external_vars_;
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
  for (const Var &var : accesses.opaque_access_vars)
    task->AddOpaqueAccessVar(var);
}

} // namespace tvm::tl::ascend
