#include "ascend/transform/source_scopes.h"
#include "ir/annotations.h"
#include "ir/loop_annotations.h"
#include "ir/loop_math.h"
#include "layout/expr_utils.h"
#include "op/builtin.h"
#include <tvm/ffi/extra/structural_visit.h>
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
 * \file dependency_analysis.cc
 * \brief Shared schedule and synchronization dependency analysis.
 */

#include "./dependency_analysis.h"
#include "./task_analysis.h"
#include "./task_annotations.h"
#include "ascend/op/utils.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include "ascend/layout/ascend_layouts.h"
#include "support/check.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;
using namespace prim;

// An undefined storage key groups explicit conflicts between distinct storage.
using CoveredDependencyMap =
    std::unordered_map<const VarObj *, std::set<DependencyTaskPair>>;

// A bare tensor/storage hint covers all of its logical aliases. A region
// hint names one tensor and one exact logical footprint, not every view.
static bool HintOperandMatches(const Any &hint, const Var &storage,
                               const Optional<TensorRegion> &query,
                               const StorageProvenance &context) {
  if (auto var = hint.try_cast<Var>()) {
    if (var->same_as(storage) || (query && var->same_as(RegionTensor(*query))))
      return true;
    // A local decl_tensor or pointer alias may be the operand at the marker's
    // definition site. Resolve it using the same proven storage context as
    // the queried access; unknown pointer roots never establish equivalence.
    auto declared = context.Address(*var);
    auto accessed =
        context.Address(query ? Expr(RegionTensor(*query)) : Expr(storage));
    return declared.exact && accessed.exact &&
           declared.root.same_as(accessed.root);
  }
  auto region = hint.try_cast<TensorRegion>();
  TVM_FFI_CHECK(region.has_value(), ValueError)
      << "conflict_hint requires a native tensor/storage or TensorRegion";
  return query && RegionTensor(*region).same_as(RegionTensor(*query)) &&
         RegionsEqual(region.value()->region, query.value()->region);
}

namespace {

constexpr const char *kScopedConflictHints = "tl.scoped_conflict_hints";
constexpr const char *kConflictScopeId = "tl.conflict_scope_id";
using HintFields = ffi::Map<ffi::String, ffi::Any>;

bool SameHintContext(const ConstrSet &a, const ConstrSet &b) {
  if (a.constrs_.size() != b.constrs_.size())
    return false;
  for (size_t i = 0; i < a.constrs_.size(); ++i) {
    const Constr &x = a.constrs_[i], &y = b.constrs_[i];
    if (x.kind != y.kind || x.is_assume != y.is_assume)
      return false;
    if (x.kind != Constr::kConstr && !x.var.same_as(y.var))
      return false;
    if (x.kind == Constr::kBindRange ? !x.range.same_as(y.range)
                                     : !x.value.same_as(y.value))
      return false;
  }
  return true;
}

template <class T>
bool SameHintObjects(const std::vector<T> &a, const std::vector<T> &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!a[i].same_as(b[i]))
      return false;
  return true;
}

Stmt HintGuardState(const IRStructure *node) {
  Stmt result = SeqStmt({});
  for (auto it = node->GetGuards().rbegin(); it != node->GetGuards().rend();
       ++it)
    result = (*it)->Wrap(result);
  return result;
}

// Substitute pure definitions only. Source predicates and source ranges must
// not simplify themselves to true before the query is asked to imply them.
Map<Var, PrimExpr> HintPureDefinitions(const ConstrSet &context) {
  Map<Var, PrimExpr> result;
  for (const Constr &constraint : context.constrs_)
    if (constraint.kind == Constr::kBindValue &&
        !constraint.var.same_as(constraint.value) &&
        SideEffect(constraint.value) <= CallEffectKind::kPure)
      result.Set(constraint.var,
                 SubstituteConstraintExpr(constraint.value, result));
  return result;
}

bool HintStatementMayAbandon(const Stmt &statement, const ConstrSet &context) {
  bool result = false;
  sym::Analyzer analyzer;
  auto scope = context.Populate(analyzer);
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      statement,
      [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
        if (object.as<BreakObj>() || object.as<ContinueObj>() ||
            object.as<ReturnObj>() || object.as<WhileObj>())
          result = true;
        if (const auto *loop = object.as<ForObj>())
          result |= !CanUseLoopBounds(loop->begin, loop->end, loop->step,
                                      analyzer.get());
        return ffi::WalkResult::Advance();
      });
  return result;
}

struct HintSource {
  const IRStructure *site;
  const TaskNode *task;
  ConstrSet context;
  std::vector<const ForObj *> raw_loops;
  bool known_execution{true};
};

// A grouped task has its own lexical definitions. Its internal loops do not
// become outer scheduler clocks merely because their copied tags match.
class TaskHintCollector : public TaskAwareConstrVisitor {
public:
  using TaskAwareConstrVisitor::Visit;
  using TaskAwareConstrVisitor::Visit_;
  using Callback = std::function<void(const HintFields &, const HintSource &)>;

  TaskHintCollector(const TaskNode *task, const ConstrSet &context,
                    bool known_execution, Callback collect)
      : task_(task), known_execution_(known_execution),
        collect_(std::move(collect)) {
    constr_stack_ = context.constrs_;
  }

private:
  Result Visit_(const SeqStmtObj *op) final {
    auto scope = Scope();
    bool old_execution = known_execution_;
    for (const Stmt &statement : op->seq) {
      Visit(statement);
      known_execution_ &= !HintStatementMayAbandon(statement, GetConstrSet());
      if (auto bind = GetFlatTaskBind(statement)) {
        if (auto value = bind->value.as<PrimExpr>())
          constr_stack_.emplace_back(bind->var, *value);
      } else if (const auto *assertion = statement.as<AssertStmtObj>()) {
        if (SideEffect(assertion->condition) <= CallEffectKind::kPure)
          constr_stack_.emplace_back(assertion->condition);
      }
    }
    known_execution_ = old_execution;
    return std::nullopt;
  }

  Result Visit_(const EvaluateObj *op) final {
    const auto *call = op->value.as<CallObj>();
    if (IsAnnotationCall(call)) {
      collect_(ReadAnnotations(call), HintSource{task_, task_, GetConstrSet(),
                                                 raw_loops_, known_execution_});
      return std::nullopt;
    }
    return TaskAwareConstrVisitor::Visit_(op);
  }

  Result Visit_(const ForObj *op) final {
    raw_loops_.push_back(op);
    auto result = TaskAwareConstrVisitor::Visit_(op);
    raw_loops_.pop_back();
    return result;
  }

  const TaskNode *task_;
  bool known_execution_;
  Callback collect_;
  std::vector<const ForObj *> raw_loops_;
};

CoreMask HintCoreMask(const HintSource &source) {
  CoreMask mask = source.task ? source.task->GetCoreMask() : kCoreUnassigned;
  return mask == kCoreUnassigned ? kCoreAll : mask;
}

std::optional<int> HintSubcore(const HintSource &source) {
  return source.task ? GetTaskSubcore(source.task->stmt) : std::nullopt;
}

bool CompatibleHintCores(const HintSource &a, const HintSource &b) {
  auto a_sid = HintSubcore(a), b_sid = HintSubcore(b);
  return (HintCoreMask(a) & HintCoreMask(b)) != kCoreUnassigned &&
         (!a_sid || !b_sid || a_sid == b_sid);
}

const IRStructure *CommonHintAncestor(const IRStructure *a,
                                      const IRStructure *b) {
  for (const IRStructure *ancestor = a; ancestor;
       ancestor = ancestor->GetParent())
    if (b->IsWithin(ancestor))
      return ancestor;
  return nullptr;
}

} // namespace

struct ScopedConflictHintIndex::Impl {
  struct NodeState {
    const IRStructure *node;
    const IRStructure *parent;
    const TaskNode *header;
    const StorageProvenance *storage;
    Stmt statement;
    Stmt guards;
    ConstrSet context;
    Optional<PrimExpr> condition_snapshot;
    int stage;
    size_t index;
    CoreMask core;
    std::vector<std::vector<const IRStructure *>> children;
    std::vector<TensorRegion> reads, writes;
    std::vector<Var> read_vars, write_vars, opaque;
    Map<Var, Optional<PrimExpr>> pitches;
    uint16_t special_read, special_write;
    bool per_core{false};

    explicit NodeState(const IRStructure *value)
        : node(value), parent(value->GetParent()), header(value->HeaderTask()),
          storage(&value->StorageContext()), statement(ffi::UnsafeInit{}),
          guards(HintGuardState(value)), stage(value->GetStage()),
          index(value->GetIndex()), core(kCoreUnassigned),
          special_read(value->GetSpecialReadMask()),
          special_write(value->GetSpecialWriteMask()) {
      if (value->IsTask()) {
        const auto *task = static_cast<const TaskNode *>(value);
        statement = task->stmt;
        context = task->outer_ctx;
        core = task->GetCoreMask();
        reads = task->GetReadRegions();
        writes = task->GetWriteRegions();
        read_vars = task->GetReadVars();
        write_vars = task->GetWriteVars();
        opaque = task->GetOpaqueAccessVars();
        pitches = task->GetL0Pitches();
        per_core = task->IsPerCoreTask();
      } else if (value->IsControl()) {
        statement = static_cast<const ControlNode *>(value)->control;
      } else {
        const auto *scope = static_cast<const ScopeNode *>(value);
        statement = scope->shell;
        context = scope->outer_ctx;
        condition_snapshot = scope->condition_snapshot;
      }
      for (const auto *list : value->ChildLists()) {
        children.emplace_back();
        for (const auto &child : *list)
          children.back().push_back(child.get());
      }
    }

    bool Current() const {
      NodeState now(node);
      return now.parent == parent && now.header == header &&
             now.storage == storage && now.statement.same_as(statement) &&
             now.stage == stage && now.index == index && now.core == core &&
             now.children == children && now.special_read == special_read &&
             now.special_write == special_write && now.per_core == per_core &&
             ffi::StructuralEqual()(now.pitches, pitches) &&
             SameHintContext(now.context, context) &&
             now.condition_snapshot.has_value() ==
                 condition_snapshot.has_value() &&
             (!condition_snapshot ||
              now.condition_snapshot.value().same_as(*condition_snapshot)) &&
             ffi::StructuralEqual()(now.guards, guards) &&
             SameHintObjects(now.reads, reads) &&
             SameHintObjects(now.writes, writes) &&
             SameHintObjects(now.read_vars, read_vars) &&
             SameHintObjects(now.write_vars, write_vars) &&
             SameHintObjects(now.opaque, opaque);
    }
  };

  struct Record {
    HintSource source;
    Array<Any> operands;
    String role;
  };
  struct Group {
    const ControlNode *target;
    String id;
    int cross;
    bool polarity;
    std::vector<Record> records;
  };

  IRStructure::ChildList *tree;
  IRStructure::ChildList roots;
  // A mutation may detach a child or replace a header. Keep every captured
  // node alive until stale-index validation has diagnosed the changed tree.
  IRStructure::ChildList retained_nodes;
  std::shared_ptr<const void> cache_identity{std::make_shared<int>(0)};
  std::vector<NodeState> states;
  std::unordered_set<const IRStructure *> membership;
  std::unordered_map<const IRStructure *, ConstrSet> contexts;
  std::vector<Group> groups;

  Impl(IRStructure::ChildList &forest, const ConstrSet &outer)
      : tree(&forest), roots(forest) {
    for (const auto &node : forest)
      Capture(node);
    CollectList(forest, outer, {}, true);
  }

  void Capture(const std::shared_ptr<IRStructure> &owned) {
    const IRStructure *node = owned.get();
    TVM_FFI_CHECK(membership.insert(node).second, ValueError)
        << "Scoped conflict index requires a tree with unique current nodes";
    retained_nodes.push_back(owned);
    states.emplace_back(node);
    if (node->HeaderTask())
      Capture(node->IsControl() ? static_cast<const ControlNode *>(node)->task
                                : static_cast<const ScopeNode *>(node)->task);
    for (const auto *list : node->ChildLists())
      for (const auto &child : *list)
        Capture(child);
  }

  void Validate(const std::vector<IRStructure *> &nodes,
                const ControlNode *loop) const {
    bool current = tree->size() == roots.size();
    for (size_t i = 0; current && i < roots.size(); ++i)
      current &= (*tree)[i].get() == roots[i].get();
    for (const NodeState &state : states)
      current &= state.Current();
    for (const auto *node : nodes)
      current &= membership.count(node) != 0;
    current &= !loop || membership.count(loop) != 0;
    TVM_FFI_CHECK(current, ValueError)
        << "Scoped conflict index is stale or belongs to another kernel; "
           "rebuild it after changing the scheduler tree";
  }

  bool NodeMayAbandon(const IRStructure *node) const {
    if (node->IsTask()) {
      const auto *task = static_cast<const TaskNode *>(node);
      return HintStatementMayAbandon(task->stmt, contexts.at(task));
    }
    if (node->IsControl()) {
      const auto *control = static_cast<const ControlNode *>(node);
      sym::Analyzer analyzer;
      auto outer = contexts.at(control).Populate(analyzer);
      if (!CanUseLoopBounds(control->control->begin, control->control->end,
                            control->control->step, analyzer.get()))
        return true;
    }
    for (const auto *list : node->ChildLists())
      for (const auto &child : *list)
        if (NodeMayAbandon(child.get()))
          return true;
    return false;
  }

  PrimExpr CaptureGate(const PrimExpr &condition, ConstrSet *context) {
    if (SideEffect(condition) <= CallEffectKind::kPure)
      return condition;
    PrimVar snapshot("conflict_hint_condition", condition.ty());
    context->AddConstr(snapshot, condition);
    return snapshot;
  }

  void CollectList(const IRStructure::ChildList &list,
                   const ConstrSet &incoming, Map<Var, PrimExpr> captures,
                   bool known_execution) {
    ConstrSet current = incoming;
    for (const auto &node : list) {
      ConstrSet active = current;
      for (const auto &guard : node->GetGuards()) {
        if (guard->IsCondition()) {
          auto condition =
              static_cast<const ConditionGuard *>(guard.get())->condition;
          active.AddConstr(CaptureGate(condition, &active));
        } else {
          const auto *attribute =
              static_cast<const AttributeGuard *>(guard.get());
          if (attribute->key == "tilelang_assume")
            active.AddConstr(
                CaptureGate(attribute->node.cast<PrimExpr>(), &active), true);
        }
      }
      contexts.emplace(node.get(), active);
      if (const auto *header = node->HeaderTask())
        contexts.emplace(header, active);
      if (node->IsTask()) {
        const auto *task = static_cast<const TaskNode *>(node.get());
        ConstrSet source_context =
            task->outer_ctx.Substitute(captures).Merge(active);
        auto collector = ffi::make_object<TaskHintCollector>(
            task, source_context, known_execution,
            [&](const HintFields &fields, const HintSource &source) {
              Add(fields, source);
            });
        collector->Visit(task->stmt);
        if (auto bind = GetFlatTaskBind(task->stmt))
          if (auto value = bind->value.as<PrimExpr>())
            current.AddConstr(bind->var, *value);
      } else if (node->IsControl()) {
        const auto *control = static_cast<const ControlNode *>(node.get());
        ConstrSet body = active;
        AppendSerialLoopFacts(&body, control->control);
        CollectList(control->children, body, captures, known_execution);
      } else if (node->IsScope()) {
        const auto *scope = static_cast<const ScopeNode *>(node.get());
        HintSource source{scope,
                          nullptr,
                          scope->outer_ctx.Substitute(captures).Merge(active),
                          {},
                          known_execution};
        if (const auto *evaluate = scope->shell.as<EvaluateObj>()) {
          if (const auto *call = evaluate->value.as<CallObj>())
            Add(ReadAnnotations(call), source);
        } else if (const auto *branch = scope->shell.as<IfObj>()) {
          ConstrSet body = active;
          PrimExpr gate = CaptureGate(branch->condition, &body);
          auto inner_captures = captures;
          if (scope->condition_snapshot &&
              SideEffect(branch->condition) > CallEffectKind::kPure)
            if (auto variable = scope->condition_snapshot.value().as<Var>())
              inner_captures.Set(*variable, gate);
          ConstrSet then_context = body;
          then_context.AddConstr(gate);
          CollectList(scope->children, then_context, inner_captures,
                      known_execution);
          if (scope->has_else) {
            body.AddConstr(Not(gate));
            CollectList(scope->else_children, body, inner_captures,
                        known_execution);
          }
        } else if (auto attribute = ReadSourceAttribute(scope->shell)) {
          ConstrSet body = active;
          auto inner_captures = captures;
          if (attribute->key == "tl.loop.annotations")
            Add(attribute->value.cast<HintFields>(), source);
          if (attribute->key == "tilelang_assume") {
            PrimExpr original = attribute->node.cast<PrimExpr>();
            PrimExpr gate = CaptureGate(original, &body);
            if (scope->condition_snapshot &&
                SideEffect(original) > CallEffectKind::kPure)
              if (auto variable = scope->condition_snapshot.value().as<Var>())
                inner_captures.Set(*variable, gate);
            body.AddConstr(gate, true);
          }
          CollectList(scope->children, body, inner_captures, known_execution);
        }
      }
      known_execution &= !NodeMayAbandon(node.get());
    }
  }

  void Add(const HintFields &annotations, const HintSource &source) {
    auto payload = annotations.Get(kScopedConflictHints);
    if (!payload)
      return;
    for (const Any &entry : payload->cast<Array<Any>>()) {
      auto fields = entry.cast<HintFields>();
      for (const char *name : {"schema_version", "hint_id", "role", "operands",
                               "target_scopes", "cross", "polarity"})
        TVM_FFI_CHECK(fields.count(name), ValueError)
            << "Scoped conflict hint is missing " << name;
      auto version = fields.at("schema_version").try_cast<int64_t>();
      if (!version)
        if (auto value = fields.at("schema_version").try_cast<PrimExpr>())
          version = TryConstInt(*value);
      auto cross = fields.at("cross").try_cast<PrimExpr>();
      auto polarity = fields.at("polarity").try_cast<PrimExpr>();
      TVM_FFI_CHECK(version && *version == 1 && cross && polarity &&
                        cross->ty() == PrimType::Int(32) &&
                        polarity->ty() == PrimType::Bool() &&
                        TryConstInt(*cross) && TryConstInt(*polarity),
                    ValueError)
          << "Scoped conflict hint requires schema_version=1, int32 cross and "
             "bool polarity";
      int64_t cross_code = *TryConstInt(*cross);
      TVM_FFI_CHECK(cross_code >= -1 && cross_code <= 1, ValueError)
          << "Scoped conflict cross mode must be -1, 0 or 1";
      String id = fields.at("hint_id").cast<String>();
      String role = fields.at("role").cast<String>();
      auto operands = fields.at("operands").cast<Array<Any>>();
      TVM_FFI_CHECK(!id.empty() && ((role == "pair" && operands.size() == 2) ||
                                    ((role == "lhs" || role == "rhs") &&
                                     operands.size() == 1)),
                    ValueError)
          << "Scoped conflict hint has an invalid role or operand count";
      for (const Any &operand : operands) {
        auto variable = operand.try_cast<Var>();
        TVM_FFI_CHECK(
            operand.try_cast<TensorRegion>().has_value() ||
                (variable && (variable.value()->ty.as<TensorTypeObj>() ||
                              variable.value()->ty.as<PtrTypeObj>())),
            ValueError)
            << "Scoped conflict operands must be native tensors, pointers or "
               "regions";
      }
      std::unordered_set<std::string> seen;
      for (const String &target_id :
           fields.at("target_scopes").cast<Array<String>>()) {
        TVM_FFI_CHECK(seen.insert(std::string(target_id)).second, ValueError)
            << "Scoped conflict hint repeats a target scope";
        const ControlNode *target = nullptr;
        size_t matches = target_id == "root" ? 1 : 0;
        bool internal = false;
        if (target_id != "root") {
          for (const ForObj *loop : source.raw_loops)
            if (auto name = loop->annotations.Get(kConflictScopeId);
                name && name->cast<String>() == target_id) {
              ++matches;
              internal = true;
            }
          for (const IRStructure *ancestor = source.site; ancestor;
               ancestor = ancestor->GetParent())
            if (ancestor->IsControl()) {
              const auto *control = static_cast<const ControlNode *>(ancestor);
              if (auto name =
                      control->control->annotations.Get(kConflictScopeId);
                  name && name->cast<String>() == target_id) {
                ++matches;
                target = control;
              }
            }
        }
        TVM_FFI_CHECK(matches == 1, ValueError)
            << "Scoped conflict target must resolve to one actual ancestor: "
            << target_id;
        if (internal)
          continue;
        auto found =
            std::find_if(groups.begin(), groups.end(), [&](const Group &group) {
              return group.target == target && group.id == id;
            });
        bool forced = *TryConstInt(*polarity) != 0;
        if (found == groups.end()) {
          groups.push_back(
              Group{target, id, static_cast<int>(cross_code), forced, {}});
          found = std::prev(groups.end());
        }
        TVM_FFI_CHECK(
            found->cross == cross_code && found->polarity == forced &&
                (found->records.empty() ||
                 (found->records.front().role == "pair") == (role == "pair")),
            ValueError)
            << "Scoped conflict group has incompatible records";
        found->records.push_back(Record{source, operands, role});
      }
    }
  }

  bool SourceQualifies(const Record &record, const ControlNode *target,
                       const TaskNode &query) const {
    const HintSource &source = record.source;
    if (!source.known_execution || !source.raw_loops.empty() ||
        (target && !query.IsWithin(target)))
      return false;
    for (const IRStructure *ancestor = source.site;
         ancestor && ancestor != target; ancestor = ancestor->GetParent())
      if (ancestor->IsControl() && !query.IsWithin(ancestor))
        return false;
    CoreMask query_core = query.GetCoreMask();
    if (query_core == kCoreUnassigned)
      query_core = kCoreAll;
    if ((query_core & HintCoreMask(source)) != query_core)
      return false;
    if (auto sid = HintSubcore(source);
        sid && GetTaskSubcore(query.stmt) != sid)
      return false;
    auto definitions = HintPureDefinitions(source.context);
    sym::Analyzer analyzer;
    analyzer->z3_prover.SetRLimit(50000);
    auto context = contexts.at(&query).Populate(analyzer);
    for (const Constr &constraint : source.context.constrs_) {
      if (constraint.kind != Constr::kConstr)
        continue;
      PrimExpr guard = SubstituteConstraintExpr(constraint.value, definitions);
      if (SideEffect(guard) > CallEffectKind::kPure ||
          !(analyzer->CanProve(guard) || analyzer->z3_prover.CanProve(guard)))
        return false;
    }
    return true;
  }

  bool Matches(const Record &record, size_t operand, const ControlNode *target,
               const TaskNode &query, const Var &storage,
               const Optional<TensorRegion> &region) const {
    if (!SourceQualifies(record, target, query))
      return false;
    const Any &hint = record.operands[operand];
    if (auto variable = hint.try_cast<Var>()) {
      auto declared = record.source.site->StorageContext().Address(*variable);
      auto accessed = query.StorageContext().Address(
          region ? Expr(RegionTensor(*region)) : Expr(storage));
      return declared.exact && accessed.exact &&
             declared.root.same_as(accessed.root);
    }
    auto declared = hint.cast<TensorRegion>();
    if (!region || !RegionTensor(declared).same_as(RegionTensor(*region)))
      return false;
    auto source_defs = HintPureDefinitions(record.source.context);
    auto query_defs = HintPureDefinitions(contexts.at(&query));
    Region a, b;
    auto normalize = [&](const Region &input, const Map<Var, PrimExpr> &defs,
                         Region *output) {
      for (const Range &axis : input) {
        PrimExpr begin = SubstituteConstraintExpr(axis->begin, defs);
        PrimExpr extent = SubstituteConstraintExpr(axis->extent, defs);
        if (SideEffect(begin) > CallEffectKind::kPure ||
            SideEffect(extent) > CallEffectKind::kPure)
          return false;
        output->push_back(Range::FromMinExtent(begin, extent));
      }
      return true;
    };
    if (!normalize(declared->region, source_defs, &a) ||
        !normalize(region.value()->region, query_defs, &b))
      return false;
    return RegionsEqual(a, b);
  }

  bool UniqueClonePair(const Group &group, const Record &a,
                       const Record &b) const {
    if (!CompatibleHintCores(a.source, b.source))
      return false;
    const IRStructure *anchor =
        CommonHintAncestor(a.source.site, b.source.site);
    // A branch or metadata wrapper can contain mixed halves from different
    // copies after transformation. Only a retained actual iteration domain
    // (plus the core domain checked above) can partition a copied family.
    while (anchor && !anchor->IsControl())
      anchor = anchor->GetParent();
    size_t left = 0, right = 0;
    for (const Record &record : group.records)
      if ((!anchor || record.source.site->IsWithin(anchor)) &&
          CompatibleHintCores(record.source, a.source) &&
          CompatibleHintCores(record.source, b.source)) {
        left += record.role == "lhs";
        right += record.role == "rhs";
      }
    // Copies in another retained loop/core domain do not make this original
    // pair a third declaration. An ambiguous root family has no pair proof.
    return left == 1 && right == 1;
  }

  std::optional<bool> Query(ControlNode *loop, int offset,
                            const TaskNode &a_task, const Var &a_storage,
                            const Optional<TensorRegion> &a_region,
                            const TaskNode &b_task, const Var &b_storage,
                            const Optional<TensorRegion> &b_region) const {
    if (groups.empty())
      return std::nullopt;
    Validate({}, loop);
    TVM_FFI_CHECK(membership.count(&a_task) && membership.count(&b_task),
                  ValueError)
        << "Scoped conflict query belongs to another kernel";
    bool no_conflict = false;
    for (const Group &group : groups) {
      if (group.target != loop ||
          (group.cross != -1 && (group.cross == 1) != (offset != 0)))
        continue;
      auto match = [&](const Record &record, size_t operand, bool first) {
        return first
                   ? Matches(record, operand, loop, a_task, a_storage, a_region)
                   : Matches(record, operand, loop, b_task, b_storage,
                             b_region);
      };
      bool applies = false;
      std::set<std::pair<const Record *, const Record *>> candidates;
      for (const Record &left : group.records) {
        if (left.role == "pair") {
          applies |= (match(left, 0, true) && match(left, 1, false)) ||
                     (match(left, 0, false) && match(left, 1, true));
        } else if (left.role == "lhs") {
          for (const Record &right : group.records)
            if (right.role == "rhs" && UniqueClonePair(group, left, right) &&
                ((match(left, 0, true) && match(right, 0, false)) ||
                 (match(left, 0, false) && match(right, 0, true))))
              candidates.emplace(&left, &right);
        }
      }
      applies |= candidates.size() == 1;
      if (applies && group.polarity)
        return true;
      no_conflict |= applies;
    }
    return no_conflict ? std::optional<bool>(false) : std::nullopt;
  }
};

ScopedConflictHintIndex::ScopedConflictHintIndex(IRStructure::ChildList &tree,
                                                 const ConstrSet &outer)
    : impl_(std::make_unique<Impl>(tree, outer)) {}

ScopedConflictHintIndex::~ScopedConflictHintIndex() = default;

const std::shared_ptr<const void> &
ScopedConflictHintIndex::CacheIdentity() const {
  return impl_->cache_identity;
}

void ScopedConflictHintIndex::Validate(const std::vector<IRStructure *> &nodes,
                                       const ControlNode *loop) const {
  impl_->Validate(nodes, loop);
}

std::optional<bool> ScopedConflictHintIndex::Query(
    ControlNode *loop, int offset, const TaskNode &a_task, const Var &a_storage,
    const Optional<TensorRegion> &a_region, const TaskNode &b_task,
    const Var &b_storage, const Optional<TensorRegion> &b_region) const {
  return impl_->Query(loop, offset, a_task, a_storage, a_region, b_task,
                      b_storage, b_region);
}

static ConflictHintList
ActiveConflictHints(ControlNode *loop, const ConflictHintList &root_hints) {
  if (loop == nullptr)
    return root_hints;
  auto ann = ReadLoopAnnotations(loop->control.get()).Get("conflict_hint");
  return ann ? ann->cast<ConflictHintList>() : ConflictHintList{};
}

static std::optional<bool> DeclaredConflict(
    ControlNode *loop, int offset, const ConflictHintList &root_hints,
    const Var &a_storage, const Optional<TensorRegion> &a_region,
    const Var &b_storage, const Optional<TensorRegion> &b_region,
    const StorageProvenance &a_context, const StorageProvenance &b_context,
    const ScopedConflictHintIndex *scoped_hints = nullptr,
    const TaskNode *a_task = nullptr, const TaskNode *b_task = nullptr) {
  bool declared_no_conflict = false;
  for (const Any &entry : ActiveConflictHints(loop, root_hints)) {
    auto hint = entry.cast<Array<Any>>();
    TVM_FFI_CHECK_EQ(hint.size(), 4U, ValueError)
        << "conflict_hint annotation expects 4-element entries";
    auto polarity = hint[3].try_cast<PrimExpr>();
    auto cross_value = hint[2].try_cast<PrimExpr>();
    TVM_FFI_CHECK(polarity && polarity->ty() == PrimType::Bool() &&
                      TryConstInt(*polarity).has_value() && cross_value &&
                      TryConstInt(*cross_value).has_value(),
                  ValueError)
        << "conflict_hint requires constant integer cross mode and boolean "
           "polarity";
    int64_t cross_code = *TryConstInt(*cross_value);
    TVM_FFI_CHECK(cross_code >= -1 && cross_code <= 1, ValueError)
        << "conflict_hint cross mode must be -1, 0 or 1";
    if (cross_code != -1 && (cross_code == 1) != (offset != 0))
      continue;
    bool matches =
        (HintOperandMatches(hint[0], a_storage, a_region, a_context) &&
         HintOperandMatches(hint[1], b_storage, b_region, b_context)) ||
        (HintOperandMatches(hint[0], b_storage, b_region, b_context) &&
         HintOperandMatches(hint[1], a_storage, a_region, a_context));
    if (!matches)
      continue;
    // A forced conflict wins over contradictory no-conflict declarations.
    if (*TryConstInt(*polarity))
      return true;
    declared_no_conflict = true;
  }
  if (scoped_hints && a_task && b_task)
    if (auto result =
            scoped_hints->Query(loop, offset, *a_task, a_storage, a_region,
                                *b_task, b_storage, b_region)) {
      if (*result)
        return true;
      declared_no_conflict = true;
    }
  return declared_no_conflict ? std::optional<bool>(false) : std::nullopt;
}

// Opaque access summaries retain canonical root Vars, not the pointer
// expressions that escaped. Check those expressions before recovering the
// root's precision, otherwise a scope-changing cast or unknown pointer call
// can appear to be an ordinary global/on-chip allocation again.
static std::vector<Expr> UnresolvedOpaquePointers(const TaskNode *task) {
  std::vector<Expr> unresolved;
  const auto &storage = task->StorageContext();
  auto inspect =
      [&](const ffi::ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
    auto expression = object.as<Expr>();
    if (!expression || !expression.value()->ty.as<PtrTypeObj>())
      return ffi::WalkResult::Advance();
    Expr value = expression.value();
    if (const auto *call = value.as<CallObj>()) {
      if (call->op.same_as(tl::ptr_access())) {
        value = call->args[0];
      } else if (call->op.same_as(tl::access_ptr())) {
        value = call->args[0].as_or_throw<TensorLoad>()->source;
      } else if (IsScheduledBind(call)) {
        value = ReadScheduledBinding(call).value;
      }
    }
    auto address = storage.Address(value);
    if (!address.exact ||
        !CompatibleStorageScopes(address.root->ty, expression.value()->ty))
      unresolved.push_back(value);
    return ffi::WalkResult::Advance();
  };
  auto definition = GetFlatTaskBind(task->stmt);
  if (definition && (definition->var->ty.as<PtrTypeObj>() ||
                     definition->var->ty.as<TensorTypeObj>())) {
    // Forming an address does not access its pointee. A pointer-producing
    // call may still access memory through its arguments, including a
    // rootless pointer or a cast whose original scope was lost.
    std::function<void(const Expr &)> visit_inputs = [&](const Expr &value) {
      if (value.as<VarObj>())
        return;
      if (const auto *call = value.as<CallObj>()) {
        if (IsScheduledBind(call)) {
          visit_inputs(ReadScheduledBinding(call).value);
          return;
        }
        if (call->op.same_as(tirx::tensor_data_ptr_op()) ||
            call->op.same_as(tirx::alloc_tensor_op()))
          return;
        size_t first = 0;
        if (call->op.same_as(tirx::decl_tensor_op()) ||
            call->op.same_as(tirx::reinterpret_op()) ||
            call->op.same_as(tirx::ptr_byte_offset_op())) {
          visit_inputs(call->args[0]);
          first = 1;
        }
        for (size_t i = first; i < call->args.size(); ++i)
          ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(call->args[i],
                                                          inspect);
        return;
      }
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(value, inspect);
    };
    visit_inputs(definition->value);
  } else {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(task->stmt, inspect);
  }
  return unresolved;
}

// Preserve the complete lexical path between adjacent loops. In native IR an
// If, assume or annotation scope can sit between those loops, and its other
// child lists/header can access the storage too.
static std::vector<const ControlNode *>
StorageIterationNest(ControlNode *loop, const StorageAddress &address) {
  auto controls = loop->GetAncestorControls();
  auto storage = address.exact ? address.root.as<Var>() : Optional<Var>();
  if (!storage)
    return controls;
  size_t first = 0;
  for (size_t i = controls.size(); i > 1; --i) {
    const IRStructure *outer = controls[i - 2];
    const IRStructure *path_child = controls[i - 1];
    bool external = false;
    for (const IRStructure *parent = path_child->GetParent(); parent;
         path_child = parent, parent = parent->GetParent()) {
      if (auto *header = parent->HeaderTask())
        external |= header->TouchesStorage(*storage);
      for (const auto *children : parent->ChildLists())
        for (const auto &child : *children)
          if (child.get() != path_child)
            external |= child->TouchesStorage(*storage);
      if (parent == outer)
        break;
    }
    if (external) {
      first = i - 1;
      break;
    }
  }
  return {controls.begin() + first, controls.end()};
}

struct ConflictProof {
  sym::Analyzer analyzer;
  Map<Var, PrimExpr> a_sub, b_sub;
  std::unique_ptr<ConstrAnalyzerContext> context;

  ConflictProof(const ConstrSet &a_ctx, const ConstrSet &b_ctx,
                ControlNode *loop, int offset, size_t num_storage_owners,
                const StorageAddress &storage) {
    analyzer->z3_prover.SetRLimit(50000);
    ConstrSet a_local = a_ctx, b_local = b_ctx, combined;
    std::unordered_map<const ControlNode *, SerialInductionInfo> inductions;
    auto induction =
        [&](const ControlNode *control) -> const SerialInductionInfo & {
      auto [entry, inserted] = inductions.try_emplace(control);
      if (inserted)
        entry->second = AnalyzeSerialInduction(control);
      return entry->second;
    };
    auto seed = [&](const IRStructure *scope) {
      for (const Var &variable : scope->GetWriteVars()) {
        if (!variable.as<PrimExpr>())
          continue;
        a_sub.Set(variable,
                  variable.CopyWithSuffix("_p").as_or_throw<PrimVar>());
        b_sub.Set(variable,
                  variable.CopyWithSuffix("_c").as_or_throw<PrimVar>());
      }
    };
    if (offset != 0) {
      ICHECK(loop);
      auto nest = offset < 0 && num_storage_owners <= 1
                      ? StorageIterationNest(loop, storage)
                      : loop->GetAncestorControls();
      if (offset > 0)
        nest = {loop};
      ICHECK(!nest.empty());
      seed(nest.front());
      Var pivot = nest.front()->control->var;
      a_local = a_local.RenameFrom("_p", a_sub, pivot);
      b_local = b_local.RenameFrom("_c", b_sub, pivot);
      if (offset < 0 && num_storage_owners <= 1) {
        PrimExpr different = IntImm::Bool(false);
        bool distinct_coordinates = true;
        for (const auto *control : nest) {
          PrimExpr iteration = control->control->var;
          different =
              different || (SubstituteConstraintExpr(iteration, a_sub) !=
                            SubstituteConstraintExpr(iteration, b_sub));
          sym::Analyzer progress;
          auto outer = control->task->outer_ctx.Populate(progress);
          distinct_coordinates &=
              induction(control).no_wrap &&
              SideEffect(control->control->step) <= CallEffectKind::kPure &&
              progress->CanProve(control->control->step != 0);
        }
        // A zero-step loop can enter repeatedly at the same coordinate until
        // Break. Unknown steps must permit that execution as well.
        if (distinct_coordinates)
          combined.AddConstr(different);
      } else if (offset > 0) {
        const auto &facts = induction(loop);
        auto step = TryConstInt(loop->control->step);
        if (facts.no_wrap && facts.bounds && step) {
          ffi::BigInt shift = ffi::BigInt(*step) * offset;
          ffi::BigInt lower = ffi::BigInt(facts.bounds->first) + shift;
          ffi::BigInt upper = ffi::BigInt(facts.bounds->second) + shift;
          if (shift.as<int64_t>() && lower.as<int64_t>() &&
              upper.as<int64_t>()) {
            PrimExpr iteration = loop->control->var;
            PrimExpr producer = cast(SubstituteConstraintExpr(iteration, a_sub),
                                     PrimType::Int(64));
            PrimExpr consumer = cast(SubstituteConstraintExpr(iteration, b_sub),
                                     PrimType::Int(64));
            combined.AddConstr(consumer ==
                               producer + IntImm::Int64(*shift.as<int64_t>()));
          }
        }
      }
    }
    if (loop) {
      // The decoder records interval bounds. Constant-step congruences keep
      // impossible intermediate coordinates out of manual-ring proofs.
      for (const auto *control : loop->GetAncestorControls()) {
        auto step = TryConstInt(control->control->step);
        if (!step || *step == 0 ||
            *step == std::numeric_limits<int64_t>::min() ||
            !induction(control).no_wrap || !induction(control).int64_delta ||
            SideEffect(control->control->begin) > CallEffectKind::kPure)
          continue;
        int64_t stride = *step < 0 ? -*step : *step;
        for (const auto *sub : {&a_sub, &b_sub}) {
          PrimExpr iteration =
              cast(SubstituteConstraintExpr(control->control->var, *sub),
                   PrimType::Int(64));
          PrimExpr begin =
              cast(SubstituteConstraintExpr(control->control->begin, *sub),
                   PrimType::Int(64));
          combined.AddConstr(
              FloorMod(iteration - begin, IntImm::Int64(stride)) ==
              IntImm::Int64(0));
        }
      }
    }
    combined = combined.Merge(a_local).Merge(b_local);
    context = combined.Populate(analyzer);
  }

  bool Prove(const PrimExpr &expression) const {
    return analyzer->CanProve(expression) ||
           analyzer->z3_prover.CanProve(expression);
  }
};

bool RegionsMayConflict(const ConstrSet &a_ctx, const TensorRegion &a_region,
                        const ConstrSet &b_ctx, const TensorRegion &b_region,
                        ControlNode *loop, int offset,
                        size_t num_storage_owners,
                        const ConflictHintList &root_conflicts,
                        Optional<PrimExpr> a_pitch, Optional<PrimExpr> b_pitch,
                        const StorageProvenance &a_storage,
                        const StorageProvenance &b_storage,
                        const ScopedConflictHintIndex *scoped_hints,
                        const TaskNode *a_task, const TaskNode *b_task) {
  if (offset != 0 && loop == nullptr)
    return false;
  TensorVar a_tensor = RegionTensor(a_region),
            b_tensor = RegionTensor(b_region);
  Var a_identity = a_storage.Identity(a_tensor),
      b_identity = b_storage.Identity(b_tensor);
  if (auto declared = DeclaredConflict(
          loop, offset, root_conflicts, a_identity, a_region, b_identity,
          b_region, a_storage, b_storage, scoped_hints, a_task, b_task))
    return *declared;
  auto a_address = a_storage.Address(a_tensor),
       b_address = b_storage.Address(b_tensor);
  if (ProvenDisjointAscendStorage(a_address, b_address))
    return false;

  ConflictProof proof(a_ctx, b_ctx, loop, offset, num_storage_owners,
                      a_address);
  if (!a_address.exact || !b_address.exact ||
      !a_address.root.same_as(b_address.root))
    return !proof.Prove(IntImm::Bool(false));

  auto freshen_a = ffi::make_object<FreshenMutableReads>(
      FreshenMutableReads::Mode::kSnapshot);
  auto freshen_b = ffi::make_object<FreshenMutableReads>(
      FreshenMutableReads::Mode::kSnapshot);
  auto a_value = [&](const PrimExpr &value) {
    return (*freshen_a)(SubstituteConstraintExpr(value, proof.a_sub));
  };
  auto b_value = [&](const PrimExpr &value) {
    return (*freshen_b)(SubstituteConstraintExpr(value, proof.b_sub));
  };

  TensorRegion a_footprint = a_region, b_footprint = b_region;
  bool matrix = IsL0ABuffer(a_tensor) || IsL0BBuffer(a_tensor) ||
                IsL0CBuffer(a_tensor) || IsL0ABuffer(b_tensor) ||
                IsL0BBuffer(b_tensor) || IsL0CBuffer(b_tensor);
  if (matrix && !(a_pitch && b_pitch &&
                  proof.Prove(a_value(*a_pitch) == b_value(*b_pitch)))) {
    // Unknown compact L0 pitches cannot justify disjoint matrix tiles. Expand
    // the trailing matrix axes, then prove separation in the actual layout so
    // even a zero-stride leading axis cannot masquerade as a version slot.
    if (!a_tensor.same_as(b_tensor) ||
        a_region->region.size() != b_region->region.size())
      return !proof.Prove(IntImm::Bool(false));
    auto expanded = [&](const TensorRegion &region) {
      Region ranges = region->region;
      size_t leading = ranges.size() > 2 ? ranges.size() - 2 : 0;
      for (size_t i = leading; i < ranges.size(); ++i)
        ranges.Set(i, Range::FromMinExtent(0, a_tensor.shape()[i]));
      return TensorRegion(a_tensor, ranges, TensorRegionType(), region->loc);
    };
    a_footprint = expanded(a_region);
    b_footprint = expanded(b_region);
  }

  // Compare arbitrary points in the two logical rectangles after applying
  // their real native layouts and byte-pointer offsets. This handles aliases
  // with different ranks, strides and element widths, including packed FP4.
  auto address_bits = [&](const TensorRegion &region,
                          const StorageProvenance &storage, const auto &value,
                          const char *prefix) -> Optional<PrimExpr> {
    TensorVar tensor = RegionTensor(region);
    Array<PrimExpr> indices;
    for (size_t axis = 0; axis < region->region.size(); ++axis) {
      PrimVar index(std::string(prefix) + std::to_string(axis),
                    PrimType::Int(64));
      PrimExpr begin =
          cast(value(region->region[axis]->begin), PrimType::Int(64));
      PrimExpr extent =
          cast(value(region->region[axis]->extent), PrimType::Int(64));
      proof.context->Enter(proof.analyzer,
                           index >= begin && index < begin + extent, false);
      indices.push_back(index);
    }
    auto offsets = tensor.ty()->ElemOffset(indices);
    if (offsets.size() != 1)
      return std::nullopt;
    int bits = tensor.dtype().bits() * tensor.dtype().lanes();
    return value(cast(storage.Address(tensor).byte_offset, PrimType::Int(64)) *
                     IntImm::Int64(8) +
                 cast(offsets[0], PrimType::Int(64)) * IntImm::Int64(bits));
  };
  auto a_bits = address_bits(a_footprint, a_storage, a_value, "access_p_");
  auto b_bits = address_bits(b_footprint, b_storage, b_value, "access_c_");
  if (!a_bits || !b_bits)
    return !proof.Prove(IntImm::Bool(false));
  int a_width = a_tensor.dtype().bits() * a_tensor.dtype().lanes();
  int b_width = b_tensor.dtype().bits() * b_tensor.dtype().lanes();
  return !proof.Prove(*a_bits + IntImm::Int64(a_width) <= *b_bits ||
                      *b_bits + IntImm::Int64(b_width) <= *a_bits);
}

std::vector<DepInfo>
AnalyzeDependencies(std::vector<IRStructure *> nodes, ControlNode *loop,
                    const BufferVersionMap &manual_buffer_versions,
                    const MultiBufferOwnerMap &multi_buffer_owners,
                    DependencyCache *dependency_cache,
                    const ConflictHintList &root_conflicts,
                    const ScopedConflictHintIndex *scoped_hints) {
  if (scoped_hints)
    scoped_hints->Validate(nodes, loop);
  DependencyCache local_cache;
  DependencyCache &cache = dependency_cache ? *dependency_cache : local_cache;
  DependencyCacheKey cache_key{
      loop, nodes, scoped_hints ? scoped_hints->CacheIdentity() : nullptr};
  if (auto cached = cache.find(cache_key); cached != cache.end())
    return cached->second;
  if (loop) {
    // A lexical header is evaluated in source order. Its default stage must
    // not move it before a preceding definition or after a following write.
    auto first = nodes.begin();
    for (auto end = nodes.begin();; ++end) {
      if (end == nodes.end() || (*end)->IsScope()) {
        std::stable_sort(first, end,
                         [](const IRStructure *lhs, const IRStructure *rhs) {
                           return lhs->GetStage() < rhs->GetStage();
                         });
        if (end == nodes.end())
          break;
        first = std::next(end);
      }
    }
  }

  const size_t n = nodes.size();
  std::vector<DepInfo> deps;

  struct RegionAccess {
    TaskNode *task;
    Optional<TensorRegion> region;
    Optional<Var> storage;
    StorageAddress address;
  };
  struct NodeAccesses {
    std::vector<RegionAccess> reads;
    std::vector<RegionAccess> writes;
  };
  std::vector<NodeAccesses> accesses(n);
  for (size_t i = 0; i < n; ++i) {
    std::vector<TaskNode *> tasks;
    CollectTaskNodesAndHeaders(nodes[i], tasks);
    for (TaskNode *task : tasks) {
      auto access = [&](const TensorRegion &region) {
        TensorVar tensor = RegionTensor(region);
        return RegionAccess{task, region, task->StorageIdentity(tensor),
                            task->StorageContext().Address(tensor)};
      };
      for (const TensorRegion &region : task->GetReadRegions())
        accesses[i].reads.push_back(access(region));
      for (const TensorRegion &region : task->GetWriteRegions())
        accesses[i].writes.push_back(access(region));
      auto unresolved = UnresolvedOpaquePointers(task);
      for (const Var &storage : task->GetOpaqueAccessVars()) {
        auto represented = [&](const std::vector<RegionAccess> &side) {
          return std::any_of(side.begin(), side.end(), [&](const auto &item) {
            return item.task == task && item.storage &&
                   item.storage.value().same_as(storage);
          });
        };
        bool reads = represented(accesses[i].reads);
        bool writes = represented(accesses[i].writes);
        RegionAccess opaque{task, std::nullopt, storage,
                            task->StorageContext().Address(storage)};
        opaque.address.exact &= unresolved.empty();
        // Retain a known access direction, but never trust an incomplete
        // physical extent. An unrepresented pointer effect may read or write.
        if (reads || !writes)
          accesses[i].reads.push_back(opaque);
        if (writes || !reads)
          accesses[i].writes.push_back(opaque);
      }
      for (const Expr &pointer : unresolved) {
        // An integer-derived or rootless pointer can escape without leaving
        // an opaque Var in the access summary. Retain that unknown effect;
        // no allocation identity or bounded access direction can be inferred.
        auto address = task->StorageContext().Address(pointer);
        address.exact = false;
        RegionAccess opaque{task, std::nullopt, pointer.as<Var>(), address};
        accesses[i].reads.push_back(opaque);
        accesses[i].writes.push_back(opaque);
      }
    }
  }

  std::function<void(IRStructure *, CoveredDependencyMap &)>
      collect_subtree_deps = [&](IRStructure *node, CoveredDependencyMap &out) {
        if (!node || node->IsTask())
          return;
        auto *clock = node->IsControl() ? static_cast<ControlNode *>(node)
                                        : node->GetParentControl();
        for (const auto *list : node->ChildLists()) {
          std::vector<IRStructure *> children;
          for (const auto &child : *list)
            children.push_back(child.get());
          auto child_deps = AnalyzeDependencies(
              children, clock, manual_buffer_versions, multi_buffer_owners,
              &cache, root_conflicts, scoped_hints);
          for (const auto &cdep : child_deps) {
            auto &covered_pairs =
                out[cdep.storage ? cdep.storage.value().get() : nullptr];
            covered_pairs.insert(cdep.task_pairs.begin(),
                                 cdep.task_pairs.end());
          }
          for (auto *child : children)
            collect_subtree_deps(child, out);
        }
      };
  auto subtree_covered = [&](IRStructure *node) {
    CoveredDependencyMap covered;
    collect_subtree_deps(node, covered);
    return covered;
  };

  auto get_manual_versions = [&](const ffi::Optional<Var> &storage) {
    if (!storage.has_value())
      return 0;
    auto it = manual_buffer_versions.find(storage.value());
    return it == manual_buffer_versions.end() ? 0 : (*it).second;
  };
  auto get_num_storage_owners =
      [&](const ffi::Optional<Var> &storage) -> size_t {
    if (loop == nullptr || !storage.has_value())
      return 0;
    auto owners = multi_buffer_owners.find(storage.value());
    if (owners == multi_buffer_owners.end() ||
        std::find(owners->second.begin(), owners->second.end(), loop) ==
            owners->second.end()) {
      return 0;
    }
    return owners->second.size();
  };
  auto find_owner = [](const std::vector<ControlNode *> &owners,
                       const TaskNode *task) -> ControlNode * {
    ControlNode *result = nullptr;
    for (ControlNode *owner : owners) {
      if (!task->IsWithin(owner) || task == owner->HeaderTask())
        continue;
      // Nested owners deliberately remain data dependencies here;
      // PrepareMultiBuffer diagnoses their overlapping ownership.
      if (result != nullptr)
        return nullptr;
      result = owner;
    }
    return result;
  };
  auto is_owner_exclusion = [&](const ffi::Optional<Var> &storage,
                                const TaskNode *lhs, const TaskNode *rhs) {
    if (!storage.has_value())
      return false;
    auto owners = multi_buffer_owners.find(storage.value());
    if (owners == multi_buffer_owners.end() || owners->second.size() <= 1)
      return false;
    ControlNode *lhs_owner = find_owner(owners->second, lhs);
    ControlNode *rhs_owner = find_owner(owners->second, rhs);
    return lhs_owner != nullptr && rhs_owner != nullptr &&
           lhs_owner != rhs_owner;
  };

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (loop == nullptr && i >= j)
        continue;
      struct PendingDependency {
        ffi::Optional<Var> storage;
        int distance;
        DependencyKind kind;
        std::set<DependencyTaskPair> task_pairs;
      };
      std::vector<PendingDependency> pending;

      auto add_pair = [&](const ffi::Optional<Var> &storage, int distance,
                          TaskNode *producer, TaskNode *consumer,
                          DependencyKind kind = DependencyKind::kData) {
        auto existing = std::find_if(
            pending.begin(), pending.end(), [&](const PendingDependency &dep) {
              return dep.storage.same_as(storage) && dep.distance == distance &&
                     dep.kind == kind;
            });
        if (existing == pending.end()) {
          pending.push_back({storage, distance, kind, {}});
          existing = std::prev(pending.end());
        }
        existing->task_pairs.emplace(producer, consumer);
      };

      auto consider = [&](const std::vector<RegionAccess> &lhs_accesses,
                          const std::vector<RegionAccess> &rhs_accesses) {
        for (const RegionAccess &lhs : lhs_accesses) {
          for (const RegionAccess &rhs : rhs_accesses) {
            if (lhs.region && rhs.region && lhs.address.exact &&
                rhs.address.exact &&
                IsSharedBuffer(RegionTensor(*lhs.region)) &&
                IsSharedBuffer(RegionTensor(*rhs.region))) {
              // A common UB offset (e.g. a hardware dual-copy destination)
              // still names different physical memory on the two AIVs.
              auto local_sid = [](const TaskNode *task) -> std::optional<int> {
                if (IsVectorCore(task->GetCoreMask()))
                  return VectorSid(task->GetCoreMask());
                return GetTaskSubcore(task->stmt);
              };
              auto lhs_sid = local_sid(lhs.task), rhs_sid = local_sid(rhs.task);
              if (lhs_sid && rhs_sid && *lhs_sid != *rhs_sid)
                continue;
            }
            const auto &lhs_storage = lhs.storage;
            const auto &rhs_storage = rhs.storage;
            ffi::Optional<Var> storage;
            if (lhs_storage && rhs_storage && lhs.address.exact &&
                rhs.address.exact &&
                lhs.address.root.same_as(rhs.address.root) &&
                lhs_storage.same_as(rhs_storage))
              storage = lhs_storage;
            auto may_conflict = [&](int distance, size_t owners) {
              if (lhs.region && rhs.region)
                return RegionsMayConflict(
                    lhs.task->outer_ctx, *lhs.region, rhs.task->outer_ctx,
                    *rhs.region, loop, distance, owners, root_conflicts,
                    lhs.task->GetL0Pitch(lhs_storage.value()),
                    rhs.task->GetL0Pitch(rhs_storage.value()),
                    lhs.task->StorageContext(), rhs.task->StorageContext(),
                    scoped_hints, lhs.task, rhs.task);
              if (distance != 0 && loop == nullptr)
                return false;
              if (lhs_storage && rhs_storage)
                if (auto declared = DeclaredConflict(
                        loop, distance, root_conflicts, lhs_storage.value(),
                        lhs.region, rhs_storage.value(), rhs.region,
                        lhs.task->StorageContext(), rhs.task->StorageContext(),
                        scoped_hints, lhs.task, rhs.task))
                  return *declared;
              if (ProvenDisjointAscendStorage(lhs.address, rhs.address))
                return false;
              ConflictProof proof(lhs.task->outer_ctx, rhs.task->outer_ctx,
                                  loop, distance, owners, lhs.address);
              return !proof.Prove(IntImm::Bool(false));
            };
            bool owner_exclusion =
                is_owner_exclusion(storage, lhs.task, rhs.task);
            // The i<j visit sees every cross-owner write/read, write/write,
            // and read/write conflict once. Such conflicts are undirected:
            // the scheduler chooses one physical owner order.
            if (owner_exclusion && i >= j)
              continue;
            int manual_versions = get_manual_versions(storage);
            if (manual_versions && !owner_exclusion) {
              for (int d = (i < j ? 0 : 1); d <= manual_versions; ++d) {
                if (may_conflict(d, /*owners=*/0)) {
                  add_pair(storage, d, lhs.task, rhs.task);
                  break;
                }
              }
              continue;
            }
            for (int distance : {0, -1}) {
              if (distance == 0 && i >= j)
                continue;
              if (distance < 0 && loop == nullptr)
                continue;
              if (may_conflict(distance, get_num_storage_owners(storage))) {
                add_pair(storage, owner_exclusion ? 0 : distance, lhs.task,
                         rhs.task,
                         owner_exclusion ? DependencyKind::kOwnerExclusion
                                         : DependencyKind::kData);
                break;
              }
            }
          }
        }
      };
      consider(accesses[i].writes, accesses[j].reads);
      consider(accesses[i].writes, accesses[j].writes);
      consider(accesses[i].reads, accesses[j].writes);

      if (i == j && !nodes[i]->IsTask()) {
        CoveredDependencyMap covered = subtree_covered(nodes[i]);
        for (PendingDependency &dep : pending) {
          auto covered_it =
              covered.find(dep.storage ? dep.storage.value().get() : nullptr);
          if (covered_it == covered.end())
            continue;
          for (auto it = dep.task_pairs.begin(); it != dep.task_pairs.end();) {
            if (covered_it->second.count(*it)) {
              it = dep.task_pairs.erase(it);
            } else {
              ++it;
            }
          }
        }
      }

      for (PendingDependency &dep : pending) {
        if (dep.task_pairs.empty())
          continue;
        deps.push_back({nodes[i], nodes[j], std::move(dep.storage),
                        std::vector<DependencyTaskPair>(dep.task_pairs.begin(),
                                                        dep.task_pairs.end()),
                        dep.distance, dep.kind});
      }
    }
  }
  cache.emplace(std::move(cache_key), deps);
  return deps;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
