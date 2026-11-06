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

/*! \file prepare_multi_buffer.cc
 *  \brief Prepare logical storage epochs and physical multi-buffer protocols.
 */

#include "ascend/transform/source_scopes.h"
#include "ir/analyzer.h"
#include "ir/loop_annotations.h"
#include "layout/expr_utils.h"
#include "transform/common/constr_visitor.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/multi_buffer.h"
#include "./auto_schedule/scheduled_tir.h"
#include "./auto_schedule/task_analysis.h"
#include "./auto_schedule/task_annotations.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/buffer_version.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ascend;

namespace {

using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

String StorageScopeName(const Var &storage) {
  if (auto tensor = storage.as<TensorVar>())
    return tensor->scope().name();
  return storage->ty.as_or_throw<PtrType>()->scope.name();
}

bool GuardExprEqual(const PrimExpr &a, const PrimExpr &b) {
  return a.same_as(b) || ffi::StructuralEqual()(a, b);
}

PrimExpr DisjoinGuards(const PrimExpr &a, const PrimExpr &b,
                       const sym::Analyzer &analyzer) {
  if (IsZero(a) || IsOne(b))
    return b;
  if (IsZero(b) || IsOne(a) || GuardExprEqual(a, b))
    return a;

  auto implies = [&](const PrimExpr &premise, const PrimExpr &conclusion) {
    ScopedAnalyzerConstraints premise_context(analyzer, premise);
    bool result = analyzer->CanProve(conclusion);
    return result;
  };
  if (analyzer->CanProveEqual(a, b) || implies(b, a))
    return a;
  if (implies(a, b))
    return b;
  return a || b;
}

struct GuardLiteral {
  PrimExpr expr;
  bool negated{false};
};

using GuardCube = std::vector<GuardLiteral>;

void FlattenGuardDisjunction(const PrimExpr &expr,
                             std::vector<PrimExpr> *terms) {
  if (const auto *op = expr.as<OrObj>()) {
    FlattenGuardDisjunction(op->a, terms);
    FlattenGuardDisjunction(op->b, terms);
    return;
  }
  terms->push_back(expr);
}

void FlattenGuardConjunction(const PrimExpr &expr,
                             std::vector<PrimExpr> *literals) {
  if (const auto *op = expr.as<AndObj>()) {
    FlattenGuardConjunction(op->a, literals);
    FlattenGuardConjunction(op->b, literals);
    return;
  }
  literals->push_back(expr);
}

bool AddGuardLiteral(const PrimExpr &expr, GuardCube *cube) {
  PrimExpr base = expr;
  bool negated = false;
  while (const auto *op = base.as<NotObj>()) {
    negated = !negated;
    base = op->a;
  }

  if (IsOne(base))
    return !negated;
  if (IsZero(base))
    return negated;

  for (const GuardLiteral &literal : *cube) {
    if (!GuardExprEqual(literal.expr, base))
      continue;
    return literal.negated == negated;
  }
  cube->push_back({std::move(base), negated});
  return true;
}

std::optional<GuardCube> NormalizeGuardCube(const PrimExpr &term) {
  std::vector<PrimExpr> literals;
  FlattenGuardConjunction(term, &literals);
  GuardCube cube;
  cube.reserve(literals.size());
  for (const PrimExpr &literal : literals) {
    if (!AddGuardLiteral(literal, &cube))
      return std::nullopt;
  }
  return cube;
}

bool GuardCubeIsSubset(const GuardCube &subset, const GuardCube &superset) {
  for (const GuardLiteral &expected : subset) {
    bool found = false;
    for (const GuardLiteral &candidate : superset) {
      if (expected.negated == candidate.negated &&
          GuardExprEqual(expected.expr, candidate.expr)) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

void RemoveAbsorbedGuardCubes(std::vector<GuardCube> *cubes) {
  // A || (A && B) == A.
  std::vector<GuardCube> kept;
  kept.reserve(cubes->size());
  for (size_t i = 0; i < cubes->size(); ++i) {
    bool absorbed = false;
    for (size_t j = 0; j < cubes->size(); ++j) {
      if (i == j || !GuardCubeIsSubset((*cubes)[j], (*cubes)[i]))
        continue;
      if ((*cubes)[j].size() < (*cubes)[i].size() || j < i) {
        absorbed = true;
        break;
      }
    }
    if (!absorbed)
      kept.push_back((*cubes)[i]);
  }
  *cubes = std::move(kept);
}

std::optional<GuardCube> TryMergeGuardCubes(const GuardCube &a,
                                            const GuardCube &b) {
  // (X && p) || (X && !p) == X.
  if (a.size() != b.size() || a.empty())
    return std::nullopt;

  GuardCube merged;
  merged.reserve(a.size() - 1);
  int complementary_literals = 0;
  for (const GuardLiteral &a_literal : a) {
    const GuardLiteral *matching = nullptr;
    for (const GuardLiteral &b_literal : b) {
      if (GuardExprEqual(a_literal.expr, b_literal.expr)) {
        matching = &b_literal;
        break;
      }
    }
    if (matching == nullptr)
      return std::nullopt;
    if (a_literal.negated == matching->negated) {
      merged.push_back(a_literal);
    } else if (++complementary_literals > 1) {
      return std::nullopt;
    }
  }
  if (complementary_literals != 1)
    return std::nullopt;
  return merged;
}

PrimExpr RebuildGuardCube(const GuardCube &cube) {
  PrimExpr result = IntImm::Bool(true);
  for (const GuardLiteral &literal : cube) {
    PrimExpr expr = literal.negated ? !literal.expr : literal.expr;
    result = IsOne(result) ? expr : result && expr;
  }
  return result;
}

PrimExpr SimplifyGuardUnion(const PrimExpr &guard) {
  // Treat an existing OR-of-ANDs as a set of cubes. Every rewrite below only
  // removes a cube or a literal; unlike normal-form conversion, it never
  // distributes terms or grows the expression.
  std::vector<PrimExpr> terms;
  FlattenGuardDisjunction(guard, &terms);

  std::vector<GuardCube> cubes;
  cubes.reserve(terms.size());
  for (const PrimExpr &term : terms) {
    std::optional<GuardCube> cube = NormalizeGuardCube(term);
    if (cube.has_value())
      cubes.push_back(std::move(cube.value()));
  }
  if (cubes.empty())
    return IntImm::Bool(false);

  RemoveAbsorbedGuardCubes(&cubes);
  while (true) {
    bool changed = false;
    for (size_t i = 0; i < cubes.size() && !changed; ++i) {
      for (size_t j = i + 1; j < cubes.size(); ++j) {
        std::optional<GuardCube> merged =
            TryMergeGuardCubes(cubes[i], cubes[j]);
        if (!merged.has_value())
          continue;
        cubes[i] = std::move(merged.value());
        cubes.erase(cubes.begin() + j);
        RemoveAbsorbedGuardCubes(&cubes);
        changed = true;
        break;
      }
    }
    if (!changed)
      break;
  }

  PrimExpr result = IntImm::Bool(false);
  for (const GuardCube &cube : cubes) {
    PrimExpr term = RebuildGuardCube(cube);
    result = IsZero(result) ? term : result || term;
  }
  return result;
}

PrimExpr CombineGuardTerms(const std::vector<PrimExpr> &terms,
                           const sym::Analyzer &analyzer) {
  PrimExpr result = IntImm::Bool(false);
  for (const PrimExpr &term : terms) {
    result = DisjoinGuards(result, term, analyzer);
    if (IsOne(result))
      break;
  }
  return SimplifyGuardUnion(result);
}

struct StorageEpochAnalysis {
  PrimExpr active_guard{IntImm::Bool(false)};
  // An unavailable guard uses a conservative lexical clock. That fallback is
  // not proof that another storage has the same actual activity predicate.
  bool precise_guard{true};
  // Latest loop-body position that defines a variable used by active_guard.
  // Earlier representatives dominate every later proof-equivalent snapshot.
  int guard_ready_position{-1};
  bool has_write{false};
};

struct EpochBody {
  IRStructure *parent;
  IRStructure::ChildList *children;
};

EpochBody GetEpochBody(ControlNode *loop) {
  EpochBody result{loop, &loop->children};
  // Match AppendLoopAnnotation's placement inside a sole metadata-wrapper
  // chain. Keep the native region scope: its local snapshot bindings must
  // dominate both the epoch metadata and the counter advances that use them.
  while (result.children->size() == 1) {
    IRStructure *node = result.children->front().get();
    if (!node->IsScope())
      break;
    auto *scope = static_cast<ScopeNode *>(node);
    auto attribute = ReadSourceAttribute(scope->shell);
    if (!attribute || attribute->key != "tl.loop.annotations")
      break;
    result = {scope, &scope->children};
  }
  return result;
}

std::set<int> OwnerRelativeStages(const IRStructure *node) {
  // Nested loops are single operations in the outer owner's schedule. Lexical
  // scopes, including loop metadata, retain the stages of their actual tasks.
  if (!node->IsScope())
    return {node->GetStage()};
  std::set<int> result;
  for (const auto *list : node->ChildLists())
    for (const auto &child : *list) {
      auto stages = OwnerRelativeStages(child.get());
      result.insert(stages.begin(), stages.end());
    }
  if (result.empty() && node->HeaderTask())
    result.insert(node->GetStage());
  return result;
}

std::set<int> StorageAccessStages(const IRStructure *node, const Var &storage,
                                  const L0StorageGroups &groups) {
  auto touches = [&](const IRStructure *access) {
    for (const Var &member : groups.Members(storage))
      if (access->TouchesStorage(member))
        return true;
    return false;
  };
  if (!node->IsScope())
    return touches(node) ? std::set<int>{node->GetStage()} : std::set<int>{};
  std::set<int> result;
  for (const auto *list : node->ChildLists())
    for (const auto &child : *list) {
      auto stages = StorageAccessStages(child.get(), storage, groups);
      result.insert(stages.begin(), stages.end());
    }
  if (node->HeaderTask() && touches(node->HeaderTask())) {
    // A memory-reading header follows a uniform descendant stage. If its
    // descendants span stages, do not invent a single counter epoch for it.
    auto stages = OwnerRelativeStages(node);
    result.insert(stages.begin(), stages.end());
  }
  return result;
}

// Replay only total integer predicates. A pure operation can still trap when
// moved outside its defining branch, so calls, memory reads, arbitrary division
// and arithmetic that can overflow are deliberately not expanded here.
bool CanReplayEpochPredicate(const PrimExpr &value) {
  bool safe = true;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      value, [&](const ffi::ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
        auto expression = node.as<PrimExpr>();
        if (!expression)
          return ffi::WalkResult::Advance();
        const PrimType &type = expression->ty();
        if (!type.IsScalar() ||
            !(type.MatchesCode(kDLInt) || type.MatchesCode(kDLUInt) ||
              type.MatchesCode(kDLBool))) {
          safe = false;
          return ffi::WalkResult::Advance();
        }
        if (node.as<IntImmObj>() || node.as<VarObj>() || node.as<EQObj>() ||
            node.as<NEObj>() || node.as<LTObj>() || node.as<LEObj>() ||
            node.as<GTObj>() || node.as<GEObj>() || node.as<AndObj>() ||
            node.as<OrObj>() || node.as<NotObj>())
          return ffi::WalkResult::Advance();
        PrimExpr divisor{ffi::UnsafeInit{}};
        if (const auto *op = node.as<FloorDivObj>())
          divisor = op->b;
        else if (const auto *op = node.as<FloorModObj>())
          divisor = op->b;
        if (divisor.defined()) {
          auto constant = TryConstInt(divisor);
          safe &= constant && *constant > 0;
        } else {
          safe = false;
        }
        return ffi::WalkResult::Advance();
      });
  return safe;
}

PrimExpr ReplayEpochPredicate(const PrimExpr &guard, const TaskNode *task) {
  if (!task)
    return guard;
  ffi::Map<Var, PrimExpr> bindings;
  for (const Constr &entry : task->outer_ctx.constrs_) {
    if (entry.kind != Constr::kBindValue || entry.var.same_as(entry.value) ||
        !CanReplayEpochPredicate(entry.value))
      continue;
    bindings.Set(entry.var, SubstituteConstraintExpr(entry.value, bindings));
  }
  // Use bindings only, never the branch predicates themselves. Simplifying
  // under a branch's full context could replace its activation with true.
  return SubstituteConstraintExpr(guard, bindings);
}

StorageEpochAnalysis
AnalyzeStorageEpoch(const Var &storage, ControlNode *loop,
                    const L0StorageGroups &groups = L0StorageGroups()) {
  std::vector<PrimExpr> guard_terms;
  std::vector<const TaskNode *> guard_tasks;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> guard_definitions;
  sym::Analyzer analyzer;
  auto loop_context = loop->GetLoopBodyContext().Populate(analyzer);

  int first_access = -1;
  bool has_write = false;
  std::optional<int> access_stage;
  bool has_multiple_access_stages = false;
  bool precise_guard = true;
  auto record = [&](const IRStructure *node, int position, int stage,
                    const PrimExpr &guard, bool known_guard) {
    bool touches = false;
    for (const Var &member : groups.Members(storage))
      touches |= node->TouchesStorage(member);
    if (!touches)
      return;
    precise_guard &= known_guard;
    if (!access_stage)
      access_stage = stage;
    else
      has_multiple_access_stages |= *access_stage != stage;
    if (first_access < 0)
      first_access = position;
    for (const Var &member : groups.Members(storage))
      has_write |= node->WritesStorage(member);
    guard_terms.push_back(guard);
    const TaskNode *task = node->IsTask() ? static_cast<const TaskNode *>(node)
                                          : node->HeaderTask();
    guard_tasks.push_back(task);
  };
  std::function<void(const IRStructure *, int, PrimExpr, bool)> collect;
  collect = [&](const IRStructure *node, int position, PrimExpr guard,
                bool known_guard) {
    guard = guard && node->GetConditionGuard();
    if (!node->IsScope()) {
      // A whole inner loop may execute zero or many times. Its internal
      // induction variables do not dominate the outer owner's protocol.
      record(node, position, node->GetStage(), guard, known_guard);
      return;
    }
    const auto *scope = static_cast<const ScopeNode *>(node);
    if (scope->task)
      for (int stage : OwnerRelativeStages(scope))
        record(scope->task.get(), position, stage, guard, known_guard);
    PrimExpr condition = IntImm::Bool(true);
    bool branch = false;
    if (const auto *conditional = scope->shell.as<IfObj>()) {
      // A captured mutable header is not a definition available after the
      // scope. Advancing every iteration is conservative in that case.
      branch = SideEffect(conditional->condition) <= CallEffectKind::kPure;
      if (branch)
        condition = conditional->condition;
      else
        known_guard = false;
    }
    for (const auto &child : scope->children)
      collect(child.get(), position, guard && condition, known_guard);
    for (const auto &child : scope->else_children)
      collect(child.get(), position, branch ? guard && !condition : guard,
              known_guard);
  };
  for (const auto &child : *GetEpochBody(loop).children) {
    int position = child->GetIndex();
    for (const Var &var : child->GetWriteVars())
      guard_definitions[var] = position;
    collect(child.get(), position, IntImm::Bool(true), true);
  }
  ICHECK_GE(first_access, 0) << "Storage epoch analysis for " << storage->name
                             << " contains no access";

  StorageEpochAnalysis result;
  result.has_write = has_write;
  result.precise_guard = precise_guard;
  if (has_multiple_access_stages) {
    // Snapshot guards from different stages describe different logical
    // iterations after pipeline lowering and cannot form one loop-local active
    // epoch. Keep the lexical domain unconditional instead.
    result.active_guard = IntImm::Bool(true);
    result.precise_guard = false;
    return result;
  }
  result.active_guard = CombineGuardTerms(guard_terms, analyzer);
  for (const Var &var : UndefinedVars(result.active_guard)) {
    auto definition = guard_definitions.find(var);
    if (definition != guard_definitions.end()) {
      result.guard_ready_position =
          std::max(result.guard_ready_position, definition->second);
    }
  }
  if (result.guard_ready_position >= first_access) {
    // A branch-local SSA predicate can be replayed at the owner only when its
    // computation is total and all remaining operands already dominate the
    // first access. Keep the original captures and side-effecting expressions
    // in place; this changes only the generated epoch predicate.
    std::vector<PrimExpr> replayed_guard_terms;
    replayed_guard_terms.reserve(guard_terms.size());
    for (size_t i = 0; i < guard_terms.size(); ++i)
      replayed_guard_terms.push_back(
          ReplayEpochPredicate(guard_terms[i], guard_tasks[i]));
    PrimExpr replayed = CombineGuardTerms(replayed_guard_terms, analyzer);
    int ready = -1;
    for (const Var &var : UndefinedVars(replayed)) {
      auto definition = guard_definitions.find(var);
      if (definition != guard_definitions.end())
        ready = std::max(ready, definition->second);
    }
    if (ready < first_access) {
      result.active_guard = replayed;
      result.guard_ready_position = ready;
      return result;
    }
    // An unavailable snapshot still cannot guard a loop-boundary protocol.
    result.active_guard = IntImm::Bool(true);
    result.precise_guard = false;
    result.guard_ready_position = -1;
  }
  return result;
}

std::vector<Var> CollectLoopStorages(ControlNode *loop) {
  std::vector<Var> result;
  VarSet seen;
  auto append = [&](const std::vector<TensorRegion> &regions) {
    for (const TensorRegion &region : regions) {
      Var storage = loop->StorageIdentity(RegionTensor(region));
      if (seen.insert(storage).second)
        result.push_back(storage);
    }
  };
  for (const auto &child : loop->children) {
    append(child->GetReadRegions());
    append(child->GetWriteRegions());
  }
  return result;
}

CoreMask GetStorageAccessCoreMask(const Var &storage, IRStructure *root,
                                  bool skip_broadcast_fills) {
  auto subtree_core_mask = [](const IRStructure *subtree) {
    CoreMask result = kCoreUnassigned;
    std::function<void(const IRStructure *)> visit =
        [&](const IRStructure *node) {
          if (node->IsTask()) {
            result |= static_cast<const TaskNode *>(node)->GetCoreMask();
            return;
          }
          for (const auto *list : node->ChildLists())
            for (const auto &child : *list)
              visit(child.get());
        };
    visit(subtree);
    return result;
  };

  CoreMask result = kCoreUnassigned;
  std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
    if (node->IsTask()) {
      auto *task = static_cast<TaskNode *>(node);
      if (task->TouchesStorage(storage) &&
          (!skip_broadcast_fills ||
           !IsMultiBufferBroadcastFill(task, storage))) {
        result |= task->GetCoreMask();
      }
      return;
    }
    if (const auto *header = node->HeaderTask();
        header && header->TouchesStorage(storage))
      result |= subtree_core_mask(node);
    for (const auto *list : node->ChildLists())
      for (const auto &child : *list)
        visit(child.get());
  };
  visit(root);
  return result;
}

bool ExprUsesAnyVar(const PrimExpr &expr, const VarSet &vars) {
  if (!expr.defined() || vars.empty())
    return false;
  for (const Var &var : UndefinedVars(expr))
    if (vars.count(var))
      return true;
  return false;
}

struct StorageState {
  Var storage{ffi::UnsafeInit{}};
  int num_versions{1};
  BufferVersionMode requested{BufferVersionMode::kAuto};
  std::vector<ControlNode *> owners;
};

struct PreparedOwner {
  MultiBufferOwnerInfo info;
  bool needs_counter{false};
  bool can_share_counter{true};
};

struct PreparedStorage {
  MultiBufferInfo info;
  bool uses_counter{false};
  bool can_share_counter{true};
  std::vector<int> storage_stages;
  CoreMask protocol_core_mask{kCoreUnassigned};
};

class MultiBufferPlanBuilder {
public:
  MultiBufferPlanBuilder(const std::vector<std::shared_ptr<IRStructure>> &root,
                         const BufferVersionMap &selected_versions,
                         const BufferVersionModeTable &requested_modes,
                         const ffi::Optional<PrimVar> &outer_sid,
                         L0StorageGroups groups)
      : root_(root), selected_versions_(selected_versions),
        requested_modes_(requested_modes), groups_(std::move(groups)) {
    if (outer_sid.has_value())
      availability_.SetExternalVarCoreMask(outer_sid.value(), kCoreVector);
  }

  MultiBufferPlan Build() {
    InitializeStorages();
    std::vector<TaskNode *> all_tasks;
    CollectAllTaskNodes(root_, all_tasks);
    availability_.AddTasks(all_tasks);
    for (const TaskNode *task : all_tasks) {
      if (Optional<Bind> bind = GetFlatTaskBind(task->stmt); bind.has_value())
        bind_definitions_.emplace(bind.value()->var, task);
    }
    MultiBufferOwnerMap owners_by_storage =
        CollectMultiBufferOwners(root_, groups_);
    for (StorageState &state : storages_) {
      auto owners_it = owners_by_storage.find(state.storage);
      if (owners_it != owners_by_storage.end())
        state.owners = owners_it->second;
      for (const Var &member : groups_.Members(state.storage)) {
        if (member.same_as(state.storage))
          continue;
        auto member_owners = owners_by_storage.find(member);
        ICHECK(member_owners != owners_by_storage.end() &&
               member_owners->second.size() == state.owners.size() &&
               std::is_permutation(member_owners->second.begin(),
                                   member_owners->second.end(),
                                   state.owners.begin()))
            << "Bound L0 data/SF group " << state.storage->name
            << " must have the same owner loops for every member";
      }
    }

    std::vector<PreparedStorage> prepared;
    for (StorageState &state : storages_) {
      ICHECK(!state.owners.empty())
          << "Automatic multi-buffer storage " << state.storage->name
          << " has no annotated owner loop";
      ValidateOwnersDisjoint(state);
      for (const Var &member : groups_.Members(state.storage)) {
        StorageState member_state = state;
        member_state.storage = member;
        ValidateCoverage(member_state);
      }

      PreparedStorage storage;
      storage.info.storage = state.storage;
      storage.info.num_versions = state.num_versions;
      bool needs_counter = state.owners.size() > 1;
      for (ControlNode *owner : state.owners) {
        PreparedOwner prepared_owner = AnalyzeOwner(state.storage, owner);
        needs_counter |= prepared_owner.needs_counter;
        storage.can_share_counter &= prepared_owner.can_share_counter;
        storage.info.owners.push_back(std::move(prepared_owner.info));
        storage.storage_stages.push_back(-1);
      }

      std::vector<std::optional<int>> counter_stages;
      counter_stages.reserve(storage.info.owners.size());
      bool counter_available = true;
      for (const MultiBufferOwnerInfo &owner : storage.info.owners) {
        std::optional<int> stage =
            CounterStorageStage(state.storage, owner.loop);
        counter_available &= stage.has_value();
        counter_stages.push_back(stage);
      }

      ICHECK(state.requested != BufferVersionMode::kIteration ||
             state.owners.size() == 1U)
          << "Explicit iteration buffer versioning for storage "
          << state.storage->name
          << " requires exactly one owner; use counter mode for disjoint "
             "owners";
      bool can_fallback_to_iteration =
          state.owners.size() == 1U &&
          TryCalculateIterationCount(state.owners.front()).has_value();
      ICHECK(state.requested != BufferVersionMode::kIteration ||
             can_fallback_to_iteration)
          << "Explicit iteration buffer versioning requires a provable native "
             "loop ordinal; use counter mode for zero-step or wrapping loops";
      storage.uses_counter =
          state.requested == BufferVersionMode::kCounter ||
          (state.requested == BufferVersionMode::kAuto && needs_counter &&
           (counter_available || !can_fallback_to_iteration));
      if (storage.uses_counter) {
        for (size_t i = 0; i < storage.info.owners.size(); ++i) {
          if (!counter_stages[i].has_value()) {
            LOG(FATAL)
                << "Counter multi-buffer storage " << state.storage->name
                << " is accessed at multiple schedule stages in one owner "
                   "loop. Align their T.Stage values, provide a compatible "
                   "manual owner, or use iteration mode with one owner";
          }
          storage.storage_stages[i] = counter_stages[i].value();
        }
        storage.protocol_core_mask =
            GetCounterProtocolCoreMask(state.storage, storage.info.owners);
      } else {
        ICHECK_EQ(storage.info.owners.size(), 1U)
            << "Iteration multi-buffering requires exactly one owner";
        storage.info.owners.front().active_guard = IntImm::Bool(true);
        ControlNode *owner = storage.info.owners.front().loop;
        if (HasNonAffineLoopPath(owner)) {
          LOG(WARNING) << "Iteration buffer versioning for storage "
                       << state.storage->name << " uses a non-affine loop nest"
                       << (counter_available ? "; consider counter mode."
                                             : ".");
        }
      }
      prepared.push_back(std::move(storage));
    }
    AssignCounterGroups(&prepared);
    for (const PreparedStorage &storage : prepared) {
      for (const Var &member : groups_.Members(storage.info.storage)) {
        MultiBufferInfo info = storage.info;
        info.storage = member;
        plan_.AddInfo(std::move(info));
      }
    }
    return std::move(plan_);
  }

private:
  void InitializeStorages() {
    storages_.reserve(selected_versions_.size());
    for (const auto &[storage, num_versions] : selected_versions_) {
      if (!groups_.Representative(storage).same_as(storage))
        continue;
      StorageState state;
      state.storage = storage;
      state.num_versions = num_versions;
      auto mode = requested_modes_.find(storage);
      if (mode != requested_modes_.end())
        state.requested = mode->second;
      storages_.push_back(std::move(state));
    }
  }

  bool RangeUsesLoopVariantValue(
      const PrimExpr &expr, const VarSet &outer_loop_vars,
      const std::vector<const ControlNode *> &outer_loops) const {
    if (SideEffect(expr) > CallEffectKind::kPure ||
        ExprUsesAnyVar(expr, outer_loop_vars)) {
      return true;
    }
    // A range may look like a pure Var while its Bind is re-evaluated inside
    // an enclosing loop body and can therefore change between iterations.
    for (const Var &var : UndefinedVars(expr)) {
      auto definition = bind_definitions_.find(var);
      if (definition == bind_definitions_.end())
        continue;
      for (const ControlNode *outer_loop : outer_loops) {
        if (definition->second->IsWithin(outer_loop))
          return true;
      }
    }
    return false;
  }

  bool HasNonAffineLoopPath(const ControlNode *owner) const {
    VarSet outer_loop_vars;
    std::vector<const ControlNode *> outer_loops;
    for (const ControlNode *control : owner->GetAncestorControls()) {
      if (control->HasConditions() || control->BodyHasLoopBreak() ||
          !AnalyzeSerialInduction(control).no_wrap)
        return true;
      const For &loop = control->control;
      bool non_affine =
          RangeUsesLoopVariantValue(loop->begin, outer_loop_vars,
                                    outer_loops) ||
          RangeUsesLoopVariantValue(loop->end, outer_loop_vars, outer_loops);
      non_affine |=
          RangeUsesLoopVariantValue(loop->step, outer_loop_vars, outer_loops);
      if (non_affine)
        return true;
      outer_loop_vars.insert(loop->var);
      outer_loops.push_back(control);
    }
    return false;
  }

  std::optional<int> CounterStorageStage(const Var &storage,
                                         ControlNode *owner) const {
    std::set<int> storage_stages;
    for (const auto &child : owner->children) {
      auto stages = StorageAccessStages(child.get(), storage, groups_);
      storage_stages.insert(stages.begin(), stages.end());
    }
    ICHECK(!storage_stages.empty());
    if (storage_stages.size() != 1U)
      return std::nullopt;
    ICHECK_GE(*storage_stages.begin(), 0);
    return *storage_stages.begin();
  }

  CoreMask GetCounterProtocolCoreMask(
      const Var &storage,
      const std::vector<MultiBufferOwnerInfo> &owners) const {
    // Prepare runs before ResolveCore, so these are conservative candidate
    // masks. They are used only to decide whether two storages can safely share
    // one physical counter protocol; ResolveCore chooses the final subset.
    CoreMask result = kCoreUnassigned;
    for (const MultiBufferOwnerInfo &owner : owners)
      for (const Var &member : groups_.Members(storage))
        result |= GetStorageAccessCoreMask(member, owner.loop,
                                           /*skip_broadcast_fills=*/true);
    ICHECK_NE(result, kCoreUnassigned)
        << "Counter multi-buffer storage " << storage->name
        << " has no core-assigned owner access";
    return result;
  }

  PreparedOwner AnalyzeOwner(const Var &storage, ControlNode *owner) {
    StorageEpochAnalysis analysis =
        AnalyzeStorageEpoch(storage, owner, groups_);
    bool has_non_affine_loop_path =
        HasNonAffineLoopPath(owner) || !TryCalculateIterationCount(owner);
    bool needs_counter =
        !IsOne(analysis.active_guard) || has_non_affine_loop_path;

    PreparedOwner result;
    result.info.loop = owner;
    result.info.active_guard = analysis.active_guard;
    result.needs_counter = needs_counter;
    result.can_share_counter = analysis.precise_guard;
    return result;
  }

  void AssignCounterGroups(std::vector<PreparedStorage> *storages) const {
    struct CounterGroup {
      // These owners define the shared loop structure and guard-equivalence
      // classes. The physical advance reads the canonical guard selected by
      // AnnotateStorageEpochs for each owner.
      std::vector<MultiBufferOwnerInfo> representative_owners;
      std::vector<CoreMask> guard_core_masks;
      std::vector<int> storage_stages;
      CoreMask protocol_core_mask;
      bool can_share_counter;
      int counter_group_id;
      TensorVar counter{ffi::UnsafeInit{}};
    };

    auto compatible = [&](const CounterGroup &group,
                          const PreparedStorage &storage) {
      if (!group.can_share_counter || !storage.can_share_counter ||
          group.representative_owners.size() != storage.info.owners.size() ||
          group.storage_stages != storage.storage_stages) {
        return false;
      }
      CoreMask combined_core_mask =
          group.protocol_core_mask | storage.protocol_core_mask;
      for (size_t i = 0; i < group.representative_owners.size(); ++i) {
        const MultiBufferOwnerInfo &group_owner =
            group.representative_owners[i];
        const MultiBufferOwnerInfo &storage_owner = storage.info.owners[i];
        CoreMask guard_core_mask =
            group.guard_core_masks[i] &
            availability_.GetExpressionCoreMask(storage_owner.active_guard);
        if (group_owner.loop != storage_owner.loop ||
            !GuardsEquivalent(group_owner.active_guard,
                              storage_owner.active_guard,
                              storage_owner.loop->GetLoopBodyContext()) ||
            (guard_core_mask & combined_core_mask) != combined_core_mask) {
          return false;
        }
      }
      return true;
    };

    std::vector<CounterGroup> groups;
    int next_counter_group_id = 1;
    for (PreparedStorage &storage : *storages) {
      if (!storage.uses_counter)
        continue;

      CounterGroup *selected = nullptr;
      for (CounterGroup &group : groups) {
        if (compatible(group, storage)) {
          selected = &group;
          break;
        }
      }
      if (selected == nullptr) {
        // Counter IDs identify a shared physical counter. InsertSync combines
        // this ID with the concrete owner loop, which uniquely selects that
        // owner's active guard.
        int counter_group_id = next_counter_group_id++;
        TensorVar counter(storage.info.storage->name + "_version_counter_" +
                              std::to_string(counter_group_id),
                          TensorType({IntImm::Int32(1)}, PrimType::Int(32),
                                     runtime::MemoryScope::Get("local.var")));
        std::vector<CoreMask> guard_core_masks;
        for (const MultiBufferOwnerInfo &owner : storage.info.owners) {
          guard_core_masks.push_back(
              availability_.GetExpressionCoreMask(owner.active_guard));
        }
        groups.push_back(CounterGroup{
            storage.info.owners, std::move(guard_core_masks),
            storage.storage_stages, storage.protocol_core_mask,
            storage.can_share_counter, counter_group_id, std::move(counter)});
        selected = &groups.back();
      } else {
        // compatible() rechecks every possible canonical guard against the
        // enlarged mask, preserving availability after every accepted merge.
        selected->protocol_core_mask |= storage.protocol_core_mask;
        for (size_t i = 0; i < storage.info.owners.size(); ++i) {
          selected->guard_core_masks[i] &= availability_.GetExpressionCoreMask(
              storage.info.owners[i].active_guard);
        }
      }
      // AddInfo requires one canonical guard per shared counter. Semantic
      // equivalence was proved above; the executable representative is chosen
      // separately by AnnotateStorageEpochs using definition readiness.
      for (size_t i = 0; i < storage.info.owners.size(); ++i)
        storage.info.owners[i].active_guard =
            selected->representative_owners[i].active_guard;
      storage.info.counter_group_id = selected->counter_group_id;
      storage.info.counter = selected->counter;
    }
  }

  static void ValidateOwnersDisjoint(const StorageState &state) {
    for (size_t i = 0; i < state.owners.size(); ++i) {
      for (size_t j = i + 1; j < state.owners.size(); ++j) {
        ICHECK(!state.owners[i]->IsWithin(state.owners[j]) &&
               !state.owners[j]->IsWithin(state.owners[i]))
            << "Automatic multi-buffer storage " << state.storage->name
            << " has nested owner loops; owners for one storage must be "
               "disjoint";
      }
    }
  }

  static bool ExprTouchesStorage(const PrimExpr &expr, const Var &storage) {
    return expr.defined() &&
           TaskAccessesStorage(AnalyzeTaskAccesses(Evaluate(expr)), storage);
  }

  static bool IsL1Storage(const Var &storage) {
    ffi::String scope = StorageScopeName(storage);
    return scope == "shared.l1" || scope == "shared.l1.dyn";
  }

  static bool IsStrictlyWithinOneOwner(const StorageState &state,
                                       const IRStructure *control) {
    bool found_owner = false;
    for (const ControlNode *candidate : state.owners) {
      if (!control->IsWithin(candidate))
        continue;
      if (control == candidate || found_owner)
        return false;
      found_owner = true;
    }
    return found_owner;
  }

  void ValidateCoverage(const StorageState &state) const {
    std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
      if (!node)
        return;
      if (node->IsTask()) {
        auto *task = static_cast<TaskNode *>(node);
        if (!task->TouchesStorage(state.storage))
          return;
        size_t owner_count = 0;
        for (ControlNode *owner : state.owners)
          owner_count += task->IsWithin(owner);
        ICHECK_LE(owner_count, 1U)
            << "Automatic multi-buffer storage " << state.storage->name
            << " has a task inside nested owner loops";
        if (owner_count == 1)
          return;
        ICHECK(!task->GuardsTouchStorage(state.storage))
            << "Automatic multi-buffer storage " << state.storage->name
            << " is accessed by a task guard outside its annotated owner "
               "loops; guards can follow a physical version only when they "
               "are strictly inside one owner";
        ICHECK(CanBroadcastFillToStorage(task->stmt, state.storage,
                                         task->StorageContext()))
            << "Automatic multi-buffer storage " << state.storage->name
            << " is accessed outside its annotated owner loops; "
               "only T.fill accesses that can be rewritten independently "
               "may target every physical version; offending task: "
            << task->stmt;
        ICHECK(state.num_versions <= 1 || !IsL1Storage(state.storage))
            << "Broadcast fill for L1 storage " << state.storage->name
            << " with " << state.num_versions << " versions is not supported";
        // ResolveCore needs this marker to exempt the owner-external task for
        // every logical multi-buffer plan. MaterializeMultiBuffer later clears
        // it without rewriting storage when the selected version count is one.
        AddMultiBufferBroadcastFill(task, state.storage);
        return;
      }
      if (node->IsScope()) {
        if (const auto *header = node->HeaderTask();
            header && header->TouchesStorage(state.storage))
          ICHECK(IsStrictlyWithinOneOwner(state, node))
              << "A storage-dependent lexical header must remain inside its "
                 "multi-buffer owner";
        for (const auto *list : node->ChildLists())
          for (const auto &child : *list)
            visit(child.get());
        return;
      }
      auto *control = static_cast<ControlNode *>(node);
      const For &loop = control->control;
      bool range_touches_storage =
          ExprTouchesStorage(loop->begin, state.storage) ||
          ExprTouchesStorage(loop->end, state.storage) ||
          ExprTouchesStorage(loop->step, state.storage);
      ICHECK(!range_touches_storage)
          << "Automatic multi-buffer storage " << state.storage->name
          << " is used by an unnormalized loop bound; run "
             "NormalizeControlFlowForSchedule before MaterializeScheduleUnits "
             "so buffer-dependent loop bounds become schedulable tasks";
      bool guard_touches_storage = control->GuardsTouchStorage(state.storage);
      if (guard_touches_storage) {
        ICHECK(IsStrictlyWithinOneOwner(state, control))
            << "Automatic multi-buffer storage " << state.storage->name
            << " is used by a guard outside its owner; only guards strictly "
               "inside one owner can follow that owner's physical version";
      }
      for (const auto &child : control->children)
        visit(child.get());
    };
    for (const auto &node : root_)
      visit(node.get());
  }

  const std::vector<std::shared_ptr<IRStructure>> &root_;
  const BufferVersionMap &selected_versions_;
  const BufferVersionModeTable &requested_modes_;
  L0StorageGroups groups_;
  std::vector<StorageState> storages_;
  std::unordered_map<Var, const TaskNode *, ObjectPtrHash, ObjectPtrEqual>
      bind_definitions_;
  CoreMaskAvailability availability_;
  MultiBufferPlan plan_;
};

MultiBufferPlan
BuildMultiBufferPlan(const std::vector<std::shared_ptr<IRStructure>> &root,
                     const BufferVersionMap &selected_versions,
                     const BufferVersionModeTable &requested_modes,
                     const ffi::Optional<PrimVar> &outer_sid,
                     const L0StorageGroups &groups) {
  return MultiBufferPlanBuilder(root, selected_versions, requested_modes,
                                outer_sid, groups)
      .Build();
}

BufferVersionModeTable
ParseBufferVersionModes(const BufferVersionModeMap &serialized_modes) {
  BufferVersionModeTable result;
  for (const auto &[storage, mode] : serialized_modes) {
    BufferVersionMode parsed;
    if (mode == "auto") {
      parsed = BufferVersionMode::kAuto;
    } else if (mode == "iteration") {
      parsed = BufferVersionMode::kIteration;
    } else {
      ICHECK_EQ(mode, "counter");
      parsed = BufferVersionMode::kCounter;
    }
    result.emplace(storage, parsed);
  }
  return result;
}

Stmt MakeCompilerTask(const Stmt &body) {
  TaskMetadata metadata;
  auto fields = MergeTaskMetadata(metadata, IntImm::Int32(0));
  if (auto bind = body.as<Bind>())
    return MakeScheduledBinding(*bind, 0, fields.cast<ScheduleUnitMetadata>());
  return MakeTask(body, fields.cast<ScheduleUnitMetadata>());
}

std::shared_ptr<TaskNode> MakeCounterTask(const Stmt &body, int stage) {
  auto task = std::make_shared<TaskNode>();
  task->stmt = MakeCompilerTask(body);
  task->SetStage(stage);
  // Counter state is local.var and can be cloned to either core. ResolveCore
  // narrows this candidate from the prepared owner/core protocol.
  task->SetCoreMask(kCoreAll);
  task->SetPipeMask(static_cast<uint16_t>(ResourcePipe::kScalar));
  ApplyTaskAccesses(AnalyzeTaskAccesses(body), task.get());
  return task;
}

void AnnotateStorageEpochs(
    const std::vector<std::shared_ptr<IRStructure>> &root,
    const MultiBufferPlan &plan, const BufferVersionMap &manual_buffer_versions,
    const ffi::Optional<PrimVar> &outer_sid, const L0StorageGroups &groups) {
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(root, tasks);
  CoreMaskAvailability availability(tasks);
  if (outer_sid.has_value())
    availability.SetExternalVarCoreMask(outer_sid.value(), kCoreVector);

  std::unordered_map<int, CoreMask> counter_group_core_masks;
  for (const MultiBufferInfo &info : plan.Infos()) {
    if (info.UsesCounter()) {
      CoreMask storage_core_mask = kCoreUnassigned;
      for (const MultiBufferOwnerInfo &owner : info.owners) {
        for (const Var &member : groups.Members(info.storage))
          storage_core_mask |= GetStorageAccessCoreMask(
              member, owner.loop, /*skip_broadcast_fills=*/true);
      }
      ICHECK_NE(storage_core_mask, kCoreUnassigned);
      counter_group_core_masks[info.counter_group_id] |= storage_core_mask;
    }
  }

  std::vector<ControlNode *> loops;
  std::function<void(IRStructure *)> collect_loops = [&](IRStructure *node) {
    if (!node)
      return;
    if (node->IsControl())
      loops.push_back(static_cast<ControlNode *>(node));
    for (const auto *list : node->ChildLists())
      for (const auto &child : *list)
        collect_loops(child.get());
  };
  for (const auto &node : root)
    collect_loops(node.get());

  struct StorageEpoch {
    Var storage{ffi::UnsafeInit{}};
    PrimExpr active_guard;
    int guard_ready_position;
    CoreMask required_core_mask{kCoreUnassigned};
  };

  for (ControlNode *loop : loops) {
    MultiBufferCounterMap counter_map;
    std::vector<StorageEpoch> counter_epochs;
    std::vector<StorageEpoch> lexical_epochs;

    std::vector<Var> storages;
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen;
    for (const Var &touched : CollectLoopStorages(loop)) {
      Array<Var> members =
          plan.Find(touched) ? groups.Members(touched) : Array<Var>{touched};
      for (const Var &member : members)
        if (seen.insert(member).second)
          storages.push_back(member);
    }
    for (const Var &storage : storages) {
      const MultiBufferInfo *info = plan.Find(storage);
      StorageEpochAnalysis analysis =
          info ? AnalyzeStorageEpoch(storage, loop, groups)
               : AnalyzeStorageEpoch(storage, loop);
      if (info) {
        // Iteration-mode rings retain the unconditional lexical clock.
        if (!info->UsesCounter())
          continue;
        if (const MultiBufferOwnerInfo *owner = info->FindOwner(loop)) {
          // Counter domains cover the owner and all of its descendants. Only
          // the concrete owner serializes the physical counter binding.
          if (owner->loop == loop) {
            counter_map.Set(storage, info->counter);
            // Keep the guard and readiness position from the same storage.
            // Counter-group compatibility proves semantic equivalence, while
            // the merge below selects one representative that dominates every
            // member access in this owner.
            counter_epochs.push_back(
                {storage, analysis.active_guard, analysis.guard_ready_position,
                 counter_group_core_masks.at(info->counter_group_id)});
          }
          continue;
        }
        // Above or between disjoint owners, the same storage participates in
        // an ordinary lexical domain at this loop.
      }
      if (manual_buffer_versions.count(storage)) {
        continue;
      }
      std::string storage_scope = StorageScopeName(storage);
      if (storage_scope == "local" || storage_scope == "local.var" ||
          storage_scope == "local.fragment") {
        continue;
      }

      // A read-only storage cannot induce an intra-loop RAW/WAW/WAR epoch.
      // Its enclosing dependency is represented in an ancestor lexical
      // domain, so avoid creating loop-local domains that no edge can select.
      if (analysis.has_write && !IsOne(analysis.active_guard)) {
        lexical_epochs.push_back({
            storage,
            analysis.active_guard,
            analysis.guard_ready_position,
            availability.GetExpressionCoreMask(analysis.active_guard),
        });
      }
    }

    std::vector<StorageEpoch> storage_epochs;
    storage_epochs.reserve(counter_epochs.size() + lexical_epochs.size());
    storage_epochs.insert(storage_epochs.end(), counter_epochs.begin(),
                          counter_epochs.end());
    storage_epochs.insert(storage_epochs.end(), lexical_epochs.begin(),
                          lexical_epochs.end());
    if (storage_epochs.empty())
      continue;

    std::stable_sort(storage_epochs.begin(), storage_epochs.end(),
                     [](const StorageEpoch &lhs, const StorageEpoch &rhs) {
                       return lhs.guard_ready_position <
                              rhs.guard_ready_position;
                     });

    // Prove guard equivalence once in PrepareMultiBuffer and serialize one
    // representative expression. Later passes reconstruct both counter and
    // lexical domains using structural equality only.
    struct GuardRepresentative {
      PrimExpr expression;
      CoreMask available_core_mask;
    };
    std::vector<GuardRepresentative> guard_representatives{
        {IntImm::Bool(true), kCoreAll}};
    sym::Analyzer domain_analyzer;
    auto loop_context = loop->GetLoopBodyContext().Populate(domain_analyzer);
    StorageEpochGuardMap guard_map;
    for (const StorageEpoch &epoch : storage_epochs) {
      size_t representative_index = guard_representatives.size();
      for (size_t i = 0; i < guard_representatives.size(); ++i) {
        const GuardRepresentative &representative = guard_representatives[i];
        // A counter advance is replicated on every protocol core. Preserve
        // cross-kind canonicalization only when that entire group can evaluate
        // the earlier equivalent guard.
        if ((representative.available_core_mask & epoch.required_core_mask) !=
            epoch.required_core_mask) {
          continue;
        }
        if (GuardExprEqual(representative.expression, epoch.active_guard) ||
            domain_analyzer->CanProveEqual(representative.expression,
                                           epoch.active_guard)) {
          representative_index = i;
          break;
        }
      }
      if (representative_index == guard_representatives.size()) {
        guard_representatives.push_back(
            {epoch.active_guard,
             availability.GetExpressionCoreMask(epoch.active_guard)});
        representative_index = guard_representatives.size() - 1;
      }
      guard_map.Set(epoch.storage,
                    guard_representatives[representative_index].expression);
    }

    ScheduleUnitMetadata fields;
    if (!counter_map.empty())
      fields.Set(kMultiBufferCounterMap, std::move(counter_map));
    fields.Set(kStorageEpochGuardMap, std::move(guard_map));
    // The analysis shell and current scheduler children share the same
    // operand-bearing directive. Encoding uses current children, so the
    // metadata is emitted once after the definitions it references.
    loop->control = AppendLoopAnnotation(loop->control, fields);
    auto carrier = std::make_shared<ScopeNode>();
    carrier->shell = MakeAnnotation(fields, loop->control->loc);
    carrier->outer_ctx = loop->GetLoopBodyContext();
    carrier->SetStorageContext(
        std::make_shared<StorageProvenance>(loop->StorageContext()));
    auto body = GetEpochBody(loop);
    carrier->SetParent(body.parent);
    carrier->SetIndex(body.children->size());
    body.children->push_back(carrier);
  }

  // Every counter owner must have been encountered through its storage access
  // order above. This also guarantees that its representative guard was
  // serialized together with the physical counter binding.
  for (const MultiBufferInfo &info : plan.Infos()) {
    if (!info.UsesCounter())
      continue;
    for (const MultiBufferOwnerInfo &owner : info.owners) {
      For loop = owner.loop->control;
      auto counters =
          ReadLoopAnnotations(loop.get()).Get(kMultiBufferCounterMap);
      auto guards = ReadLoopAnnotations(loop.get()).Get(kStorageEpochGuardMap);
      ICHECK(counters.has_value() && guards.has_value())
          << "Counter owner is missing prepared epoch metadata for storage "
          << info.storage->name;
      ICHECK(
          counters.value().cast<MultiBufferCounterMap>().count(info.storage));
      ICHECK(guards.value().cast<StorageEpochGuardMap>().count(info.storage));
    }
  }
}

void MaterializeCounterProtocol(ScheduledTIR *scheduled_tir,
                                const MultiBufferPlan &plan,
                                const L0StorageGroups &groups) {
  struct CounterProtocol {
    TensorVar counter{ffi::UnsafeInit{}};
    std::vector<MultiBufferOwnerInfo> owners;
    std::vector<Var> storages;
  };

  std::map<int, CounterProtocol> protocols;
  for (const MultiBufferInfo &info : plan.Infos()) {
    if (!info.UsesCounter())
      continue;
    CounterProtocol &protocol = protocols[info.counter_group_id];
    if (!protocol.counter.defined()) {
      protocol.counter = info.counter;
      protocol.owners = info.owners;
    } else {
      ICHECK(protocol.counter.same_as(info.counter));
      ICHECK_EQ(protocol.owners.size(), info.owners.size());
      for (size_t i = 0; i < protocol.owners.size(); ++i)
        ICHECK_EQ(protocol.owners[i].loop, info.owners[i].loop);
    }
    protocol.storages.push_back(info.storage);
  }

  std::vector<std::shared_ptr<IRStructure>> *root = &scheduled_tir->tree;
  std::unordered_map<
      ControlNode *,
      std::map<IRStructure *, std::vector<std::shared_ptr<IRStructure>>>>
      advances;
  for (const auto &[_, protocol] : protocols) {
    for (const MultiBufferOwnerInfo &owner : protocol.owners) {
      auto guards = ReadLoopAnnotations(owner.loop->control.get())
                        .Get(kStorageEpochGuardMap);
      ICHECK(guards.has_value())
          << "Counter owner is missing its canonical epoch guard";
      StorageEpochGuardMap guard_map =
          guards.value().cast<StorageEpochGuardMap>();
      PrimExpr active_guard{ffi::UnsafeInit{}};
      for (const Var &storage : protocol.storages) {
        auto guard = guard_map.Get(storage);
        ICHECK(guard.has_value())
            << "Counter owner is missing the epoch guard for storage "
            << storage->name;
        if (!active_guard.defined()) {
          active_guard = guard.value();
        } else {
          // This annotation is produced and consumed within this pass, so its
          // canonical guards retain ObjectRef identity without an IR
          // round-trip.
          ICHECK(active_guard.same_as(guard.value()))
              << "Shared counter storages must use one canonical epoch guard";
        }
      }
      ICHECK(active_guard.defined());

      IRStructure *anchor = nullptr;
      std::set<int> storage_stages;
      for (const auto &child : *GetEpochBody(owner.loop).children) {
        for (const Var &storage : protocol.storages) {
          auto stages = StorageAccessStages(child.get(), storage, groups);
          if (!stages.empty()) {
            anchor = child.get();
            storage_stages.insert(stages.begin(), stages.end());
          }
        }
      }
      ICHECK(anchor != nullptr)
          << "Counter multi-buffer owner has no storage access";
      ICHECK_EQ(storage_stages.size(), 1U)
          << "Counter multi-buffer accesses must retain their validated stage";
      PrimExpr zero = IntImm(PrimType::Int(32), 0);
      PrimExpr value = TensorLoad(protocol.counter, {zero});
      Stmt advance =
          TensorStore(protocol.counter, {zero}, value + IntImm(value.ty(), 1));
      advance = WrapWithGuard(std::move(advance), active_guard);
      advances[owner.loop][anchor].push_back(
          MakeCounterTask(advance, *storage_stages.begin()));
    }
  }

  for (auto &[owner, owner_advances] : advances) {
    auto body = GetEpochBody(owner);
    std::vector<std::shared_ptr<IRStructure>> rewritten;
    rewritten.reserve(body.children->size() + owner_advances.size());
    for (const auto &child : *body.children) {
      rewritten.push_back(child);
      auto it = owner_advances.find(child.get());
      if (it != owner_advances.end()) {
        rewritten.insert(rewritten.end(), it->second.begin(), it->second.end());
      }
    }
    for (size_t i = 0; i < rewritten.size(); ++i) {
      rewritten[i]->SetIndex(i);
      rewritten[i]->SetParent(body.parent);
    }
    *body.children = std::move(rewritten);
  }

  std::vector<std::shared_ptr<IRStructure>> initializers;
  for (const auto &[_, protocol] : protocols) {
    auto allocation =
        Bind(protocol.counter, Call(protocol.counter.ty(), alloc_tensor_op(),
                                    {}, std::nullopt, {protocol.counter.ty()}));
    initializers.push_back(MakeCounterTask(allocation, 0));
    PrimExpr zero = IntImm(PrimType::Int(32), 0);
    Stmt initialize = TensorStore(protocol.counter, {zero},
                                  IntImm(protocol.counter.dtype(), 0));
    initializers.push_back(MakeCounterTask(initialize, 0));
  }

  root->insert(root->begin(), initializers.begin(), initializers.end());
  for (size_t i = 0; i < root->size(); ++i) {
    (*root)[i]->SetIndex(i);
    (*root)[i]->SetParent(nullptr);
  }
}

} // namespace

tvm::transform::Pass PrepareMultiBuffer() {
  using namespace tirx::transform;
  auto pass_func = [](tirx::Function func, const IRModule &,
                      const tvm::transform::PassContext &) -> tirx::Function {
    return RewriteTilelangKernels(
        std::move(func), "PrepareMultiBuffer",
        [](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir =
              DecodeScheduledTIR(context.root, context.outer_ctx, nullptr, {},
                                 context.parameter_alias_contract);
          std::vector<TaskNode *> tasks;
          CollectAllTaskNodes(scheduled_tir.tree, tasks);
          for (TaskNode *task : tasks) {
            ICHECK_NE(task->GetCoreMask(), kCoreUnassigned)
                << "PrepareMultiBuffer requires AssignCore to run first";
          }
          L0StorageGroups groups(CollectL0SFBindings(context.root));
          auto &metadata = scheduled_tir.metadata;
          metadata.buffer_versions = ExpandL0StorageGroupValues(
              metadata.buffer_versions, groups, "version counts");
          metadata.buffer_version_modes = ExpandL0StorageGroupValues(
              metadata.buffer_version_modes, groups, "version modes");
          BufferVersionModeTable modes =
              ParseBufferVersionModes(metadata.buffer_version_modes);
          MultiBufferPlan plan = BuildMultiBufferPlan(
              scheduled_tir.tree, scheduled_tir.metadata.buffer_versions, modes,
              context.outer_sid, groups);
          scheduled_tir.metadata.buffer_version_modes = {};
          AnnotateStorageEpochs(scheduled_tir.tree, plan,
                                scheduled_tir.metadata.manual_buffer_versions,
                                context.outer_sid, groups);
          MaterializeCounterProtocol(&scheduled_tir, plan, groups);
          return EncodeScheduledTIR(std::move(scheduled_tir));
        });
  };
  return CreateFunctionPass(pass_func, 0, "tl.PrepareMultiBuffer");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PrepareMultiBuffer", PrepareMultiBuffer);
}

} // namespace tl
} // namespace tvm
