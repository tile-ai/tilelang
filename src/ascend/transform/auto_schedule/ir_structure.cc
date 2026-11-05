#include "ir/loop_math.h"
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
 * \file ir_structure.cc
 * \brief IRStructure node cloning and debug helpers for TileLang.
 */

#include "ir/analyzer.h"
#include "ir_structure.h"
#include "task_analysis.h"
#include <algorithm>
#include <tvm/ffi/extra/structural_equal.h>
#include <unordered_set>

namespace tvm::tl::ascend {
using namespace tirx;
using namespace prim;

namespace {

bool GuardsStructurallyEqual(const PrimExpr &a, const PrimExpr &b) {
  ICHECK(a.defined() && b.defined());
  return a.same_as(b) || ffi::StructuralEqual()(a, b);
}

bool GuardsEquivalentWithAnalyzer(const sym::Analyzer &analyzer,
                                  const PrimExpr &a, const PrimExpr &b) {
  ICHECK(a.defined() && b.defined());
  if (GuardsStructurallyEqual(a, b))
    return true;
  return analyzer->CanProveEqual(a, b);
}

bool GuardImpliesWithAnalyzer(const sym::Analyzer &analyzer,
                              const PrimExpr &premise,
                              const PrimExpr &conclusion) {
  ICHECK(premise.defined() && conclusion.defined());
  if (GuardsStructurallyEqual(premise, conclusion))
    return true;
  ScopedAnalyzerConstraints constraint(analyzer, premise);
  return analyzer->CanProve(conclusion);
}

} // namespace

bool GuardsEquivalent(const PrimExpr &a, const PrimExpr &b,
                      const ConstrSet &outer_ctx) {
  auto analyzer = CreateTileLangAnalyzer();
  auto context = outer_ctx.Populate(analyzer);
  return GuardsEquivalentWithAnalyzer(analyzer, a, b);
}

bool GuardImplies(const PrimExpr &premise, const PrimExpr &conclusion,
                  const ConstrSet &outer_ctx) {
  if (IsZero(premise) || IsOne(conclusion) ||
      GuardsStructurallyEqual(premise, conclusion))
    return true;
  auto analyzer = CreateTileLangAnalyzer();
  auto context = outer_ctx.Populate(analyzer);
  return GuardImpliesWithAnalyzer(analyzer, premise, conclusion);
}

PrimExpr IRStructure::GetConditionGuard() const {
  PrimExpr guard = IntImm::Bool(true);
  for (const auto &item : GetGuards()) {
    if (!item->IsCondition())
      continue;
    const PrimExpr &condition =
        static_cast<const ConditionGuard *>(item.get())->condition;
    guard = IsOne(guard) ? condition : guard && condition;
  }
  return guard;
}

bool IRStructure::GuardsTouchStorage(const Var &storage) const {
  auto touches_storage = [&](const PrimExpr &expression) {
    return expression.defined() &&
           TaskAccessesStorage(
               AnalyzeTaskAccesses(Evaluate(expression), {}, StorageContext()),
               storage);
  };

  for (const auto &guard : GetGuards()) {
    if (guard->IsCondition()) {
      const auto *condition = static_cast<const ConditionGuard *>(guard.get());
      if (touches_storage(condition->condition))
        return true;
      continue;
    }

    const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
    auto node = attribute->node.try_cast<PrimExpr>();
    if (node.has_value() && touches_storage(node.value()))
      return true;
  }
  return false;
}

Stmt WrapWithGuard(Stmt stmt, const PrimExpr &guard) {
  ICHECK(guard.defined());
  return IsOne(guard) ? stmt
                      : If(guard, NativeTaskAttribute::AsSequence(stmt),
                           std::nullopt, stmt->loc);
}

ConstrSet TaskNode::GetContextBeforeConditionGuards() const {
  ConstrSet result = outer_ctx;
  const GuardList &guards = GetGuards();
  for (auto guard_it = guards.rbegin(); guard_it != guards.rend(); ++guard_it) {
    if (!(*guard_it)->IsCondition())
      continue;
    const PrimExpr &condition =
        static_cast<const ConditionGuard *>(guard_it->get())->condition;
    bool removed = false;
    for (size_t index = result.constrs_.size(); index > 0; --index) {
      const Constr &constraint = result.constrs_[index - 1];
      if (constraint.kind == Constr::kConstr &&
          ffi::StructuralEqual()(constraint.value, condition)) {
        result.constrs_.erase(result.constrs_.begin() + index - 1);
        removed = true;
        break;
      }
    }
    ICHECK(removed) << "Task condition is missing from its constraint context: "
                    << condition;
  }
  return result;
}

void ControlNode::InitializeLoopBodyContext() {
  ICHECK(task != nullptr);
  ConstrSet result = task->outer_ctx;
  AppendSerialLoopFacts(&result, control);
  for (const auto &child : children) {
    if (!child->IsTask())
      continue;
    const auto *body_task = static_cast<const TaskNode *>(child.get());
    if (ffi::Optional<Bind> bind = GetFlatTaskBind(body_task->stmt);
        bind.has_value()) {
      PrimExpr definition_guard = body_task->GetConditionGuard();
      if (IsOne(definition_guard)) {
        if (auto value = bind.value()->value.as<PrimExpr>())
          result.AddConstr(bind.value()->var, *value);
      } else if (bind.value()->var->ty.as<PrimType>() &&
                 bind.value()->var->ty.as_or_throw<PrimType>().IsScalar()) {
        // A condition snapshot defined under another condition is arbitrary
        // outside that condition. Preserve its definition domain instead of
        // globally binding the snapshot and proving facts from an assignment
        // that may not execute.
        auto freshen = ffi::make_object<FreshenMutableReads>();
        PrimExpr value =
            (*freshen)(bind.value()->value.as_or_throw<PrimExpr>());
        result.AddConstr(!definition_guard ||
                         bind.value()->var.as_or_throw<PrimExpr>() == value);
      }
    }
  }
  loop_body_ctx_ = std::move(result);
}

IRStructure *IRStructure::GetLowestCommonAncestor(IRStructure *other) {
  ICHECK(other != nullptr) << "Cannot find an ancestor of a null IRStructure";
  std::unordered_set<IRStructure *> ancestors;
  for (IRStructure *node = this; node != nullptr; node = node->GetParent()) {
    ancestors.insert(node);
  }
  for (IRStructure *node = other; node != nullptr; node = node->GetParent()) {
    if (ancestors.count(node))
      return node;
  }
  return nullptr;
}

std::vector<IRStructure *> IRStructure::PathFrom(IRStructure *ancestor) {
  std::vector<IRStructure *> path;
  for (IRStructure *node = this; node != nullptr; node = node->GetParent()) {
    path.push_back(node);
    if (node == ancestor) {
      std::reverse(path.begin(), path.end());
      return path;
    }
  }
  if (ancestor == nullptr) {
    std::reverse(path.begin(), path.end());
    return path;
  }
  LOG(FATAL) << "The requested IRStructure path start is not an ancestor";
  return {};
}

IRStructure *IRStructure::GetChildOnPathFrom(IRStructure *ancestor) {
  ICHECK(ancestor != nullptr) << "IRStructure path ancestor cannot be null";
  IRStructure *node = this;
  while (node != nullptr && node->GetParent() != ancestor)
    node = node->GetParent();
  ICHECK(node != nullptr)
      << "The requested IRStructure path start is not a strict ancestor";
  return node;
}

int CompareIRStructure(IRStructure *lhs, IRStructure *rhs) {
  if (lhs == rhs)
    return 0;
  if (lhs == nullptr)
    return -2;
  if (rhs == nullptr)
    return 2;

  std::vector<IRStructure *> lhs_path = lhs->PathFrom();
  std::vector<IRStructure *> rhs_path = rhs->PathFrom();
  size_t common_size = std::min(lhs_path.size(), rhs_path.size());
  bool same_nodes = true;
  for (size_t i = 0; i < common_size; ++i) {
    same_nodes &= lhs_path[i] == rhs_path[i];
    auto branch_index = [](IRStructure *node) {
      IRStructure *parent = node->GetParent();
      if (!parent || !parent->IsScope())
        return size_t{0};
      auto lists = parent->ChildLists();
      for (size_t list = 0; list < lists.size(); ++list)
        for (const auto &child : *lists[list])
          if (child.get() == node)
            return list;
      return size_t{0};
    };
    size_t lhs_branch = branch_index(lhs_path[i]);
    size_t rhs_branch = branch_index(rhs_path[i]);
    if (lhs_branch != rhs_branch)
      return lhs_branch < rhs_branch ? -1 : 1;
    // A lexical branch or assumption is an evaluation boundary. Its default
    // stage is not a scheduling instruction and cannot move a preceding or
    // following definition across that boundary.
    if ((lhs_path[i]->IsScope() || rhs_path[i]->IsScope()) &&
        lhs_path[i]->GetIndex() != rhs_path[i]->GetIndex())
      return lhs_path[i]->GetIndex() < rhs_path[i]->GetIndex() ? -1 : 1;
    if (lhs_path[i]->GetStage() != rhs_path[i]->GetStage())
      return lhs_path[i]->GetStage() < rhs_path[i]->GetStage() ? -1 : 1;
    if (lhs_path[i]->GetIndex() != rhs_path[i]->GetIndex())
      return lhs_path[i]->GetIndex() < rhs_path[i]->GetIndex() ? -1 : 1;
  }
  if (lhs_path.size() == rhs_path.size())
    return 0;
  if (lhs_path.size() < rhs_path.size())
    return same_nodes ? -2 : -1;
  return same_nodes ? 2 : 1;
}

namespace {

std::optional<std::pair<int64_t, int64_t>>
MachineIntegerBounds(const PrimExpr &expression,
                     const sym::Analyzer &analyzer) {
  auto type_min = TryConstInt(min_value(expression.ty()));
  auto type_max = TryConstInt(max_value(expression.ty()));
  if (!type_min || !type_max)
    return std::nullopt;
  auto bound = analyzer->const_int_bound(expression);
  // ConstIntBound encodes -infinity as -INT64_MAX, not INT64_MIN. It is
  // not a finite lower bound on an int64 machine value. Restore the type's
  // complete range before reasoning about an induction update or difference.
  int64_t lower = bound->min_value <= sym::ConstIntBound::kNegInf
                      ? *type_min
                      : std::max(*type_min, bound->min_value);
  int64_t upper = bound->max_value >= sym::ConstIntBound::kPosInf
                      ? *type_max
                      : std::min(*type_max, bound->max_value);
  return std::pair{lower, upper};
}

Optional<Stmt> SerialTail(Stmt statement) {
  for (;;) {
    if (const auto *sequence = statement.as<SeqStmtObj>()) {
      Optional<Stmt> tail;
      for (const Stmt &child : sequence->seq)
        if (!IsTaskMetadataStatement(child))
          tail = child;
      if (!tail)
        return std::nullopt;
      statement = *tail;
      continue;
    }
    if (auto attribute = ReadTaskAttribute(statement);
        attribute && (attribute->key == attr::kScheduleUnit ||
                      IsSemanticTaskKey(attribute->key) ||
                      attribute->key == "tl.loop.annotations")) {
      if (IsSemanticTaskKey(attribute->key)) {
        auto metadata = ParseTaskMetadata(attribute->node);
        if (attribute->key == attr::kAscendPerCoreTask ||
            (metadata.core_mask && *metadata.core_mask != kCoreUnassigned &&
             *metadata.core_mask != kCoreAll))
          return std::nullopt;
      }
      statement = attribute->region->body;
      continue;
    }
    return statement;
  }
}

bool EndsInCurrentLoopBreak(const Stmt &statement) {
  auto tail = SerialTail(statement);
  return tail && tail.value().as<BreakObj>();
}

const IRStructure *ActiveSerialTail(const IRStructure::ChildList &children) {
  for (auto it = children.rbegin(); it != children.rend(); ++it) {
    const IRStructure *node = it->get();
    if (node->HasGuards())
      return nullptr;
    if (node->IsScope()) {
      const auto *scope = static_cast<const ScopeNode *>(node);
      if (IsTaskMetadataStatement(scope->shell) && scope->children.empty())
        continue;
      if (auto attribute = ReadTaskAttribute(scope->shell);
          attribute && attribute->key == "tl.loop.annotations")
        return ActiveSerialTail(scope->children);
    }
    if (node->IsTask()) {
      const auto *task = static_cast<const TaskNode *>(node);
      if (task->IsPerCoreTask() || (task->GetCoreMask() != kCoreUnassigned &&
                                    task->GetCoreMask() != kCoreAll))
        return nullptr;
    }
    return node;
  }
  return nullptr;
}

Optional<Stmt> ActiveTailStatement(const IRStructure *node) {
  if (!node)
    return std::nullopt;
  if (node->IsTask())
    return SerialTail(static_cast<const TaskNode *>(node)->stmt);
  if (node->IsScope())
    return static_cast<const ScopeNode *>(node)->shell;
  return std::nullopt;
}

bool EndsInActiveBreak(const IRStructure::ChildList &children) {
  const IRStructure *node = ActiveSerialTail(children);
  auto tail = ActiveTailStatement(node);
  if (!tail)
    return false;
  if (tail.value().as<BreakObj>())
    return true;
  if (node->IsScope() && tail.value().as<IfObj>()) {
    const auto *scope = static_cast<const ScopeNode *>(node);
    return scope->has_else && EndsInActiveBreak(scope->children) &&
           EndsInActiveBreak(scope->else_children);
  }
  return false;
}

bool HasInductionContinue(const IRStructure::ChildList &children) {
  std::vector<TaskNode *> tasks;
  for (const auto &child : children)
    CollectTaskNodesAndHeaders(child.get(), tasks);
  bool found = false;
  for (const TaskNode *task : tasks)
    VisitNativePostOrder(task->stmt, [&](const ObjectRef &object) {
      found |= object.as<ContinueObj>() != nullptr;
    });
  // An inner-loop continue could be handled separately; declining that case
  // keeps the current proof independent of implicit jump targets.
  return found;
}

} // namespace

SerialInductionInfo AnalyzeSerialInduction(const ControlNode *control) {
  if (!control || !control->task)
    return {};
  const ForObj *loop = control->control.get();
  if (SideEffect(loop->begin) > CallEffectKind::kPure ||
      SideEffect(loop->end) > CallEffectKind::kPure ||
      SideEffect(loop->step) > CallEffectKind::kPure)
    return {};
  auto analyzer = CreateTileLangAnalyzer();
  auto outer = control->task->outer_ctx.Populate(analyzer);
  bool header_range =
      CanUseLoopBounds(loop->begin, loop->end, loop->step, analyzer.get());
  if (analyzer->CanProve(loop->begin >= loop->end))
    return {true, std::nullopt};
  auto type_min = TryConstInt(min_value(loop->var.ty()));
  auto type_max = TryConstInt(max_value(loop->var.ty()));
  if (!type_min || !type_max)
    return {header_range, std::nullopt};
  auto start = MachineIntegerBounds(loop->begin, analyzer);
  auto stop = MachineIntegerBounds(loop->end, analyzer);
  if (!start || !stop)
    return {header_range, std::nullopt};
  auto [start_lo, start_hi] = *start;
  auto clipped = [&](ffi::BigInt lower, ffi::BigInt upper) {
    lower = std::max(lower, ffi::BigInt(*type_min));
    upper = std::min(upper, ffi::BigInt(*type_max));
    upper = std::min(upper, ffi::BigInt(stop->second) - 1);
    if (lower > upper)
      return SerialInductionInfo{true, std::nullopt};
    ffi::BigInt delta_lo = lower - start_hi;
    ffi::BigInt delta_hi = upper - start_lo;
    return SerialInductionInfo{
        true, std::pair{*lower.as<int64_t>(), *upper.as<int64_t>()},
        delta_lo.as<int64_t>().has_value() &&
            delta_hi.as<int64_t>().has_value()};
  };
  if (header_range)
    return clipped(start_lo, ffi::BigInt(stop->second) - 1);
  auto step = TryConstInt(analyzer->Simplify(loop->step));
  if (step && *step == 0)
    return clipped(start_lo, start_hi);
  if (!step)
    return {};
  if (HasInductionContinue(control->children))
    return {};
  const IRStructure *tail_node = ActiveSerialTail(control->children);
  auto tail = ActiveTailStatement(tail_node);
  if (!tail)
    return {};
  if (tail.value().as<BreakObj>())
    return clipped(start_lo, start_hi);
  const auto *branch = tail.value().as<IfObj>();
  if (!branch)
    return {};
  const auto *scope = tail_node->IsScope()
                          ? static_cast<const ScopeNode *>(tail_node)
                          : nullptr;
  bool then_break = scope ? EndsInActiveBreak(scope->children)
                          : EndsInCurrentLoopBreak(branch->then_case);
  bool else_break =
      scope ? scope->has_else && EndsInActiveBreak(scope->else_children)
            : branch->else_case && EndsInCurrentLoopBreak(*branch->else_case);
  if (then_break && else_break)
    return clipped(start_lo, start_hi);
  if ((!then_break && !else_break) ||
      SideEffect(branch->condition) > CallEffectKind::kPure)
    return {};
  // Induction: the initial coordinate is begin. On every continuing path the
  // monotone invariant and this mandatory tail guard must make the next
  // update representable; otherwise no mathematical-coordinate proof exists.
  ConstrAnalyzerContext continuation;
  continuation.Enter(analyzer, loop->var < loop->end, false);
  continuation.Enter(
      analyzer, *step > 0 ? loop->var >= loop->begin : loop->var <= loop->begin,
      false);
  continuation.Enter(analyzer,
                     analyzer->Simplify(then_break ? Not(branch->condition)
                                                   : branch->condition),
                     false);
  if (analyzer->CanProve(IntImm::Bool(false)))
    return clipped(start_lo, start_hi);
  auto current = MachineIntegerBounds(loop->var, analyzer);
  if (!current)
    return {};
  ffi::BigInt next_lo = ffi::BigInt(current->first) + *step;
  ffi::BigInt next_hi = ffi::BigInt(current->second) + *step;
  if (next_lo < *type_min || next_hi > *type_max)
    return {};
  return clipped(std::min(ffi::BigInt(start_lo), next_lo),
                 std::max(ffi::BigInt(start_hi), next_hi));
}

Optional<PrimExpr> TryCalculateIterationCount(ControlNode *loop) {
  // A dead ancestor also makes an otherwise unknown inner coordinate dead.
  // Establish this before constructing any divisions or span products.
  for (auto *ancestor = loop; ancestor;
       ancestor = ancestor->GetParentControl()) {
    const ForObj *node = ancestor->control.get();
    auto analyzer = CreateTileLangAnalyzer();
    auto context = ancestor->task->outer_ctx.Populate(analyzer);
    if (SideEffect(node->begin) <= CallEffectKind::kPure &&
        SideEffect(node->end) <= CallEffectKind::kPure &&
        analyzer->CanProve(node->begin >= node->end))
      return IntImm::Int64(0);
  }
  PrimExpr total = IntImm::Int64(0), multiplier = IntImm::Int64(1);
  ffi::BigInt total_bound = 0, multiplier_bound = 1;
  for (; loop != nullptr; loop = loop->GetParentControl()) {
    const ForObj *node = loop->control.get();
    auto analyzer = CreateTileLangAnalyzer();
    auto context = loop->task->outer_ctx.Populate(analyzer);
    if (SideEffect(node->begin) > CallEffectKind::kPure ||
        SideEffect(node->end) > CallEffectKind::kPure ||
        SideEffect(node->step) > CallEffectKind::kPure)
      return std::nullopt;
    if (analyzer->CanProve(node->begin >= node->end))
      return IntImm::Int64(0);
    if (!analyzer->CanProve(node->step != 0))
      return std::nullopt;
    auto induction = AnalyzeSerialInduction(loop);
    if (!induction.no_wrap || !induction.bounds)
      return std::nullopt;
    auto begin = MachineIntegerBounds(node->begin, analyzer);
    auto step_bounds = MachineIntegerBounds(node->step, analyzer);
    if (!begin || !step_bounds)
      return std::nullopt;
    ffi::BigInt position_lo =
        ffi::BigInt(induction.bounds->first) - begin->second;
    ffi::BigInt position_hi =
        ffi::BigInt(induction.bounds->second) - begin->first;
    bool positive = analyzer->CanProve(node->step > 0);
    if (positive)
      position_lo = std::max(position_lo, ffi::BigInt(0));
    else
      position_hi = std::min(position_hi, ffi::BigInt(0));
    if (!position_lo.as<int64_t>() || !position_hi.as<int64_t>())
      return std::nullopt;
    ffi::BigInt stride = positive ? ffi::BigInt(step_bounds->first)
                                  : -ffi::BigInt(step_bounds->second);
    if (stride <= 0)
      return std::nullopt;
    ffi::BigInt ordinal_bound =
        (positive ? position_hi : -position_lo) / stride;
    total_bound += ordinal_bound * multiplier_bound;
    if (!total_bound.as<int64_t>())
      return std::nullopt;
    PrimExpr step = cast(node->step, PrimType::Int(64));
    PrimExpr position = cast(node->var, PrimType::Int(64)) -
                        cast(node->begin, PrimType::Int(64));
    total = total + FloorDiv(position, step) * multiplier;
    if (loop->GetParentControl()) {
      if (!positive)
        return std::nullopt;
      auto end = MachineIntegerBounds(node->end, analyzer);
      if (!end)
        return std::nullopt;
      ffi::BigInt distance_bound = ffi::BigInt(end->second) - begin->first;
      if (distance_bound < 0 || !distance_bound.as<int64_t>())
        return std::nullopt;
      ffi::BigInt span_bound = (distance_bound + stride - 1) / stride;
      multiplier_bound *= span_bound;
      if (!multiplier_bound.as<int64_t>())
        return std::nullopt;
      // Positive ranges supply nominal spans. Break may leave holes, as
      // required by the normalized-while version-clock protocol.
      PrimExpr distance = cast(node->end, PrimType::Int(64)) -
                          cast(node->begin, PrimType::Int(64));
      // ceildiv(distance, step) forms distance + step - 1, which can
      // overflow even when the distance and the resulting span both fit.
      PrimExpr span =
          FloorDiv(distance, step) +
          cast(FloorMod(distance, step) != IntImm::Int64(0), PrimType::Int(64));
      multiplier = multiplier * max(span, IntImm::Int64(0));
    }
  }
  return total;
}

PrimExpr CalculateIterationCount(ControlNode *loop) {
  auto result = TryCalculateIterationCount(loop);
  TVM_FFI_CHECK(result.has_value(), ValueError)
      << "Flattened scheduled iteration requires pure bounds, a nonzero "
         "step, a positive inner nominal span, and non-wrapping int64 "
         "coordinates";
  return *result;
}

std::shared_ptr<IRStructure> TaskNode::Clone() const {
  auto new_task = std::make_shared<TaskNode>(CloneExtraInfo());
  // Copy statement
  new_task->stmt = stmt;
  // Copy resource pipe mask
  new_task->SetPipeMask(GetPipeMask());
  new_task->SetHbmMask(GetHbmMask());
  new_task->SetSpecialWriteMask(GetSpecialWriteMask());
  new_task->SetReadsPadValue(ReadsPadValueRegister());
  new_task->SetReadsAtomic(ReadsAtomicRegister());
  if (IsPerCoreTask())
    new_task->MarkPerCoreTask();
  // Copy memory access regions
  new_task->SetReadRegions(GetReadRegions());
  new_task->SetWriteRegions(GetWriteRegions());
  new_task->SetL0Pitches(GetL0Pitches());
  new_task->SetReadVars(GetReadVars());
  new_task->SetWriteVars(GetWriteVars());
  new_task->SetOpaqueAccessVars(GetOpaqueAccessVars());
  new_task->outer_ctx = outer_ctx;
  new_task->SetCoreMask(GetCoreMask());
  new_task->CopyBaseMetadataFrom(*this);
  return new_task;
}

std::shared_ptr<IRStructure> ControlNode::Clone() const {
  auto new_ctrl = std::make_shared<ControlNode>(CloneExtraInfo());
  // Copy For control (For is a TVM object with reference counting)
  new_ctrl->control = control;
  // Clone children
  new_ctrl->children.reserve(children.size());
  for (size_t i = 0; i < children.size(); ++i) {
    std::shared_ptr<IRStructure> child =
        children[i] ? children[i]->Clone() : nullptr;
    if (child) {
      child->SetIndex(i);
      child->SetParent(new_ctrl.get());
    }
    new_ctrl->children.push_back(std::move(child));
  }
  if (task) {
    new_ctrl->task = std::static_pointer_cast<TaskNode>(task->Clone());
    new_ctrl->task->SetParent(new_ctrl.get());
  }
  new_ctrl->loop_body_ctx_ = loop_body_ctx_;
  new_ctrl->CopyBaseMetadataFrom(*this);
  return new_ctrl;
}

std::shared_ptr<IRStructure> ScopeNode::Clone() const {
  auto result = std::make_shared<ScopeNode>(CloneExtraInfo());
  result->shell = shell;
  result->outer_ctx = outer_ctx;
  result->condition_snapshot = condition_snapshot;
  result->has_else = has_else;
  result->CopyBaseMetadataFrom(*this);
  if (task) {
    result->task = std::static_pointer_cast<TaskNode>(task->Clone());
    result->task->SetParent(result.get());
  }
  auto destination_lists = result->ChildLists();
  auto source_lists = ChildLists();
  for (size_t list = 0; list < source_lists.size(); ++list)
    for (const auto &child : *source_lists[list]) {
      auto clone = child->Clone();
      clone->SetParent(result.get());
      destination_lists[list]->push_back(std::move(clone));
    }
  return result;
}

void CollectAllTaskNodes(IRStructure *node, std::vector<TaskNode *> &tasks) {
  if (!node)
    return;
  if (node->IsTask()) {
    tasks.push_back(static_cast<TaskNode *>(node));
    return;
  }
  for (const auto *list : node->ChildLists())
    for (const auto &child : *list)
      CollectAllTaskNodes(child.get(), tasks);
}

void CollectTaskNodesAndHeaders(IRStructure *node,
                                std::vector<TaskNode *> &tasks) {
  if (!node)
    return;
  if (node->IsTask()) {
    tasks.push_back(static_cast<TaskNode *>(node));
    return;
  }
  if (auto *header = node->HeaderTask())
    tasks.push_back(header);
  for (const auto *list : node->ChildLists())
    for (const auto &child : *list)
      CollectTaskNodesAndHeaders(child.get(), tasks);
}

// Original helper function to print IRStructure (kept for backward
// compatibility)
void PrintIRStructure(const IRStructure *node, int indent) {
  if (!node)
    return;

  std::string indent_str(indent * 2, ' ');

  for (const auto &guard : node->GetGuards()) {
    if (guard->IsAttribute()) {
      const auto *attribute_guard =
          static_cast<const AttributeGuard *>(guard.get());
      LOG(INFO) << indent_str << "  [attr " << attribute_guard->key << "] "
                << attribute_guard->node;
    } else {
      const auto *condition_guard =
          static_cast<const ConditionGuard *>(guard.get());
      LOG(INFO) << indent_str << "  [guard] " << condition_guard->condition;
    }
  }

  if (node->IsTask()) {
    const TaskNode *task = static_cast<const TaskNode *>(node);
    LOG(INFO) << indent_str << "TaskNode:";
    LOG(INFO) << indent_str << "  stmt: " << task->stmt;
    for (auto &region : task->GetReadRegions()) {
      LOG(INFO) << indent_str << "  Read Region: " << region;
    }
    for (auto &region : task->GetWriteRegions()) {
      LOG(INFO) << indent_str << "  Write Region: " << region;
    }
    for (auto &var : task->GetReadVars()) {
      LOG(INFO) << indent_str << "  Read Var: " << var;
    }
    for (auto &var : task->GetWriteVars()) {
      LOG(INFO) << indent_str << "  Write Var: " << var;
    }
    LOG(INFO) << indent_str << "  pipe_mask: 0x" << std::hex
              << task->GetPipeMask() << std::dec;
    LOG(INFO) << indent_str << "  latency: " << task->GetLatency() << " cycles";
    LOG(INFO) << indent_str << "  II: " << task->GetII() << " cycles";
    LOG(INFO) << indent_str
              << "  core_mask: " << static_cast<int>(task->GetCoreMask());
  } else {
    LOG(INFO) << indent_str
              << (node->IsControl() ? "ControlNode:" : "ScopeNode:");
    for (const auto *list : node->ChildLists())
      for (const auto &child : *list)
        PrintIRStructure(child.get(), indent + 2);
  }
}

} // namespace tvm::tl::ascend
