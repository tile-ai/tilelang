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

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
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
#include "ascend/op/utils.h"
#include "ascend/transform/attr.h"
#include "op/copy.h"
#include "op/fill.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

inline ffi::Optional<Bind> GetFlatTaskBind(const Stmt &stmt) {
  if (const auto *bind = stmt.as<BindNode>())
    return ffi::GetRef<Bind>(bind);
  const auto *attribute = stmt.as<AttrStmtNode>();
  if (attribute && (attribute->attr_key == tl::attr::kAscendTask ||
                    attribute->attr_key == tl::attr::kScheduleUnit))
    return GetFlatTaskBind(attribute->body);
  return std::nullopt;
}

inline Stmt ReplaceFlatTaskBind(const Stmt &stmt, const Bind &replacement) {
  if (stmt.as<BindNode>())
    return replacement;
  const auto *attribute = stmt.as<AttrStmtNode>();
  ICHECK(attribute && (attribute->attr_key == tl::attr::kAscendTask ||
                       attribute->attr_key == tl::attr::kScheduleUnit))
      << "Expected a flat Task/schedule-unit wrapper around Bind, got " << stmt;
  return AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                  ReplaceFlatTaskBind(attribute->body, replacement),
                  attribute->span);
}

class TaskAwareConstrVisitor : public ConstrVisitor {
protected:
  static ffi::Optional<Bind> GetFlatBind(const Stmt &stmt) {
    return GetFlatTaskBind(stmt);
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    // Mirror ConstrVisitor's own SeqStmt handling: a flat Bind (here wrapped
    // in a Task/schedule-unit attr) defines its variable for the following
    // statements of this sequence, but not while its value is evaluated.
    size_t old_size = constr_stack_.size();
    for (const Stmt &stmt : op->seq) {
      VisitStmt(stmt);
      if (ffi::Optional<Bind> bind = GetFlatBind(stmt); bind.defined()) {
        constr_stack_.emplace_back(bind.value()->var, bind.value()->value);
      }
    }
    constr_stack_.resize(old_size);
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
EstimateRegionElementUpperBound(const Buffer &buffer, const Region &ranges,
                                arith::Analyzer *analyzer) {
  RegionElementUpperBound result;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const PrimExpr &extent_expr = ranges[i]->extent;
    int64_t extent = 1;
    bool extent_reliable = true;

    if (const int64_t *extent_int = as_const_int(extent_expr)) {
      extent = *extent_int;
    } else {
      const int64_t *shape_extent =
          i < buffer->shape.size() ? as_const_int(buffer->shape[i]) : nullptr;
      arith::ConstIntBound bound = analyzer->const_int_bound(extent_expr);

      arith::Analyzer unconstrained_analyzer;
      Var unconstrained_extent("unconstrained_extent", extent_expr.dtype());
      int64_t unconstrained_max =
          unconstrained_analyzer.const_int_bound(unconstrained_extent)
              ->max_value;
      bool has_finite_bound = bound->max_value >= 0 &&
                              bound->max_value != arith::ConstIntBound::kPosInf;
      bool has_useful_bound =
          has_finite_bound && bound->max_value < unconstrained_max;

      if (shape_extent != nullptr) {
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

inline PrimExpr RegionElementCountExpr(const Region &ranges) {
  PrimExpr elements = make_const(DataType::Int(64), 1);
  for (const Range &range : ranges)
    elements = elements * cast(DataType::Int(64), range->extent);
  return elements;
}

inline int64_t ElementsToBytes(int64_t elements, DataType dtype) {
  int64_t elem_bits = dtype.bits() * dtype.lanes();
  return (elements * elem_bits + 7) / 8;
}

// Hardware resource properties derived from one task body.
struct TaskResourceUsage {
  uint16_t pipe_mask{0};
  uint16_t hbm_mask{0};
  bool reads_pad_value{false};
  bool reads_atomic{false};
};

// Direct buffer and scalar-variable accesses made by one statement. Accesses
// remain separate from hardware resource classification because AutoSchedule
// also adds accesses from surrounding guards and loop expressions to a task.
struct TaskAccessInfo {
  std::vector<BufferRegion> read_regions;
  std::vector<BufferRegion> write_regions;
  std::vector<Var> read_vars;
  std::vector<Var> write_vars;
};

namespace task_analysis_detail {

inline bool IsCrossCoreSyncCall(const CallNode *op) {
  return op->op.same_as(tl::ascend_cross_core_set_flag()) ||
         op->op.same_as(tl::ascend_cross_core_wait_flag());
}

inline constexpr uint16_t kSpecializedPipes =
    static_cast<uint16_t>(ResourcePipe::kMTE1) |
    static_cast<uint16_t>(ResourcePipe::kMTE2) |
    static_cast<uint16_t>(ResourcePipe::kMTE3) |
    static_cast<uint16_t>(ResourcePipe::kCube) |
    static_cast<uint16_t>(ResourcePipe::kVector) |
    static_cast<uint16_t>(ResourcePipe::kFixpipe);

inline uint16_t ResolvePipe(uint16_t mask) {
  if ((mask & kSpecializedPipes) == 0)
    return mask | static_cast<uint16_t>(ResourcePipe::kScalar);
  return mask;
}

inline uint16_t GetHbmResourceMask(const Buffer &src, const Buffer &dst) {
  if (IsGlobalBuffer(src) && IsSharedBuffer(dst))
    return static_cast<uint16_t>(HbmPort::kAiv);
  if (IsSharedBuffer(src) && IsGlobalBuffer(dst))
    return static_cast<uint16_t>(HbmPort::kAiv) |
           static_cast<uint16_t>(HbmPort::kStore);
  if (IsGlobalBuffer(src) && IsL1Buffer(dst))
    return static_cast<uint16_t>(HbmPort::kAic);
  if (IsL0CBuffer(src) && IsGlobalBuffer(dst))
    return static_cast<uint16_t>(HbmPort::kAic) |
           static_cast<uint16_t>(HbmPort::kStore);
  return 0;
}

inline SpecialRegister
SpecialRegisterForCrossCorePipe(const PrimExpr &pipe_arg) {
  const auto *pipe = pipe_arg.as<StringImmNode>();
  ICHECK(pipe != nullptr)
      << "Cross-core synchronization pipe must be a constant string, got "
      << pipe_arg;
  switch (ParseAscendPipe(pipe->value)) {
  case ResourcePipe::kScalar:
    return SpecialRegister::kPipeScalar;
  case ResourcePipe::kVector:
    return SpecialRegister::kPipeVector;
  case ResourcePipe::kCube:
    return SpecialRegister::kPipeCube;
  case ResourcePipe::kMTE1:
    return SpecialRegister::kPipeMTE1;
  case ResourcePipe::kMTE2:
    return SpecialRegister::kPipeMTE2;
  case ResourcePipe::kMTE3:
    return SpecialRegister::kPipeMTE3;
  case ResourcePipe::kFixpipe:
    return SpecialRegister::kPipeFixpipe;
  case ResourcePipe::kNone:
  case ResourcePipe::kAll:
    break;
  }
  LOG(FATAL) << "Unsupported cross-core synchronization pipe " << pipe->value;
  return SpecialRegister::kNone;
}

class SpecialRegisterWriteAnalyzer : public StmtExprVisitor {
public:
  static uint16_t Analyze(const Stmt &stmt) {
    SpecialRegisterWriteAnalyzer analyzer;
    analyzer(stmt);
    return analyzer.mask_;
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::loop_break())) {
      Add(SpecialRegister::kLoopControl);
    } else if (op->op.same_as(tl::ascend_set_hf32_mode())) {
      Add(SpecialRegister::kHf32Mode);
    } else if (op->op.same_as(tl::ascend_set_copy_pad_value())) {
      Add(SpecialRegister::kPadValue);
    } else if (op->op.same_as(tl::ascend_set_atomic()) ||
               op->op.same_as(tl::ascend_set_atomic_none())) {
      Add(SpecialRegister::kAtomicMode);
    } else if (IsCrossCoreSyncCall(op)) {
      ICHECK_EQ(op->args.size(), 3)
          << "ascend_cross_core_set_flag/wait_flag expects 3 arguments";
      Add(SpecialRegisterForCrossCorePipe(op->args[1]));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void Add(SpecialRegister reg) { mask_ |= static_cast<uint16_t>(reg); }

  uint16_t mask_{0};
};

class ResourceAnalyzer : public StmtExprVisitor {
public:
  static TaskResourceUsage Analyze(const Stmt &stmt) {
    ResourceAnalyzer analyzer;
    analyzer(stmt);
    analyzer.usage_.pipe_mask = ResolvePipe(analyzer.usage_.pipe_mask);
    return analyzer.usage_;
  }

private:
  void VisitExpr_(const CallNode *op) final {
    static const auto copy_op = Op::Get("tl.tileop.copy");

    usage_.pipe_mask |= GetAscendTaskPipeMask(ffi::GetRef<Call>(op));
    if (op->op.same_as(copy_op)) {
      Copy copy_obj(op->args, op->annotations);
      const CopyNode *copy = copy_obj.get();
      usage_.hbm_mask |= GetHbmResourceMask(copy->src, copy->dst);
      if (copy_obj->GetDataSelect() != 0)
        usage_.reads_pad_value = true;
      if (IsGlobalBuffer(copy->dst))
        usage_.reads_atomic = true;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  // Task-level blocks are atomic to all task analyses.
  void VisitStmt_(const SBlockNode *op) final {
    usage_.pipe_mask |= PipeMask(GetAscendBlockPipe(op->name_hint));
  }

  TaskResourceUsage usage_;
};

// Collect expressions evaluated by scalar control rather than by a task's
// hardware data path. Buffer-region indices are handled separately because a
// tile operation may legally transfer data between memories owned by different
// cores while its address expressions must still be available on the issuing
// core.
class ScalarControlExpressionCollector : public StmtVisitor {
public:
  static std::vector<PrimExpr> Collect(const Stmt &stmt) {
    ScalarControlExpressionCollector collector;
    collector(stmt);
    return std::move(collector.expressions_);
  }

private:
  void VisitExpr(const PrimExpr &) final {}

  void VisitStmt_(const BindNode *op) final {
    expressions_.push_back(op->value);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tl::attr::kAscendTask &&
        op->attr_key != tl::attr::kAscendPerCoreTask &&
        op->attr_key != tl::attr::kScheduleUnit) {
      if (const auto *expr = op->node.as<PrimExprNode>())
        expressions_.push_back(ffi::GetRef<PrimExpr>(expr));
      if (op->value.defined())
        expressions_.push_back(op->value);
    }
    VisitStmt(op->body);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    expressions_.push_back(op->condition);
    VisitStmt(op->then_case);
    if (op->else_case.defined())
      VisitStmt(op->else_case.value());
  }

  void VisitStmt_(const ForNode *op) final {
    expressions_.push_back(op->min);
    expressions_.push_back(op->extent);
    if (op->step.defined())
      expressions_.push_back(op->step.value());
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
      expressions_.push_back(call->args[2]);
    }
  }

  void VisitStmt_(const BufferStoreNode *) final {}
  void VisitStmt_(const SBlockNode *op) final { VisitStmt(op->body); }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    expressions_.push_back(op->predicate);
    VisitStmt(op->block);
  }

  std::vector<PrimExpr> expressions_;
};

} // namespace task_analysis_detail

inline TaskResourceUsage AnalyzeTaskResourceUsage(const Stmt &stmt) {
  return task_analysis_detail::ResourceAnalyzer::Analyze(stmt);
}

inline uint16_t AnalyzeSpecialRegisterWrites(const Stmt &stmt) {
  return task_analysis_detail::SpecialRegisterWriteAnalyzer::Analyze(stmt);
}

inline TaskAccessInfo AnalyzeTaskAccesses(const Stmt &stmt) {
  MemoryAccessDetector memory_detector;
  memory_detector.Analyze(stmt);
  TaskAccessInfo result;
  result.read_regions = memory_detector.GetReadRegions();
  result.write_regions = memory_detector.GetWriteRegions();
  result.read_vars = memory_detector.GetReadVars();
  result.write_vars = memory_detector.GetWriteVars();
  return result;
}

inline bool TaskAccessesStorage(const TaskAccessInfo &accesses,
                                const Var &storage) {
  auto regions_access = [&](const std::vector<BufferRegion> &regions) {
    return std::any_of(regions.begin(), regions.end(), [&](const auto &region) {
      return region->buffer->data.same_as(storage);
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
  class Detector : public StmtExprVisitor {
  public:
    bool found{false};

  private:
    void VisitExpr_(const CallNode *op) final {
      if (task_analysis_detail::IsCrossCoreSyncCall(op)) {
        found = true;
        return;
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  } detector;
  detector(stmt);
  return detector.found;
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
      for (const BufferRegion &region : task->GetWriteRegions()) {
        if (IsRegisterRegion(region))
          writers_[region->buffer->data].push_back(task);
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
      return kCoreBroadcast;
    CoreMask result = kCoreBroadcast;
    bool found = false;
    for (const TaskNode *writer : writers->second) {
      if (writer == reader)
        continue;
      found = true;
      result &= writer->GetCoreMask();
    }
    return found ? result : kCoreBroadcast;
  }

  CoreMask GetExpressionCoreMask(const PrimExpr &expr,
                                 const TaskNode *reader = nullptr) const {
    if (!expr.defined())
      return kCoreBroadcast;
    return GetAccessCoreMask(AnalyzeTaskAccesses(Evaluate(expr)), reader);
  }

  CoreMask GetAccessCoreMask(const TaskAccessInfo &accesses,
                             const TaskNode *reader = nullptr) const {
    CoreMask result = kCoreBroadcast;
    for (const Var &var : accesses.read_vars)
      result &= GetValueCoreMask(var, reader);
    for (const BufferRegion &region : accesses.read_regions) {
      if (IsSharedBuffer(region->buffer)) {
        result &= kCoreVector;
      } else if (IsL1Buffer(region->buffer) || IsL0ABuffer(region->buffer) ||
                 IsL0BBuffer(region->buffer) || IsL0CBuffer(region->buffer)) {
        result &= kCoreCube;
      } else if (IsRegisterRegion(region)) {
        result &= GetValueCoreMask(region->buffer->data, reader);
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

    for (const BufferRegion &region : task->GetReadRegions())
      result &= GetRegionIndexCoreMask(region, task);
    for (const BufferRegion &region : task->GetWriteRegions())
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
      if (const auto *expr = attribute->node.as<PrimExprNode>())
        result &= GetExpressionCoreMask(ffi::GetRef<PrimExpr>(expr), task);
      if (attribute->value.defined())
        result &= GetExpressionCoreMask(attribute->value, task);
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
  CoreMask GetValueInputCoreMask(const TaskNode *task) const {
    CoreMask result = kCoreBroadcast;
    for (const Var &var : task->GetReadVars())
      result &= GetValueCoreMask(var, task);
    for (const BufferRegion &region : task->GetReadRegions()) {
      if (IsRegisterRegion(region))
        result &= GetValueCoreMask(region->buffer->data, task);
    }
    return result;
  }

  CoreMask GetRegionIndexCoreMask(const BufferRegion &region,
                                  const TaskNode *reader) const {
    CoreMask result = kCoreBroadcast;
    for (const Range &range : region->region) {
      result &= GetExpressionCoreMask(range->min, reader);
      result &= GetExpressionCoreMask(range->extent, reader);
    }
    return result;
  }

  WriterMap writers_;
  CoreMap external_vars_;
};

inline bool HasTaskBufferAccess(const Stmt &stmt) {
  TaskAccessInfo accesses = AnalyzeTaskAccesses(stmt);
  return !accesses.read_regions.empty() || !accesses.write_regions.empty();
}

inline bool HasTaskBufferAccess(const TaskNode *task) {
  ICHECK(task != nullptr);
  return !task->GetReadRegions().empty() || !task->GetWriteRegions().empty();
}

[[noreturn]] inline void FatalUnmarkedTask(const char *consumer,
                                           const char *node_kind,
                                           const char *prerequisite = nullptr) {
  if (prerequisite != nullptr) {
    LOG(FATAL) << consumer << " found an unmarked " << node_kind
               << ". Run tl.transform." << prerequisite << " before "
               << consumer << ".";
  }
  LOG(FATAL) << consumer << " found an unmarked " << node_kind
             << " after internal task normalization";
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
  for (const BufferRegion &region : accesses.read_regions)
    task->AddReadRegion(region);
  for (const BufferRegion &region : accesses.write_regions)
    task->AddWriteRegion(region);
  for (const Var &var : accesses.read_vars)
    task->AddReadVar(var);
  for (const Var &var : accesses.write_vars)
    task->AddWriteVar(var);
}

} // namespace ascend
} // namespace tl
} // namespace tvm
