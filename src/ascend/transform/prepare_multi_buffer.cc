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

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
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
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;

namespace {

using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

bool GuardExprEqual(const PrimExpr &a, const PrimExpr &b) {
  return a.same_as(b) || ffi::StructuralEqual()(a, b);
}

PrimExpr DisjoinGuards(const PrimExpr &a, const PrimExpr &b,
                       arith::Analyzer *analyzer) {
  if (is_zero(a) || is_one(b))
    return b;
  if (is_zero(b) || is_one(a) || GuardExprEqual(a, b))
    return a;

  auto implies = [&](const PrimExpr &premise, const PrimExpr &conclusion) {
    auto exit_constraint = analyzer->EnterConstraint(premise);
    bool result = analyzer->CanProve(conclusion);
    exit_constraint();
    return result;
  };
  if (analyzer->CanProve(a == b) || implies(b, a))
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
  if (const auto *op = expr.as<OrNode>()) {
    FlattenGuardDisjunction(op->a, terms);
    FlattenGuardDisjunction(op->b, terms);
    return;
  }
  terms->push_back(expr);
}

void FlattenGuardConjunction(const PrimExpr &expr,
                             std::vector<PrimExpr> *literals) {
  if (const auto *op = expr.as<AndNode>()) {
    FlattenGuardConjunction(op->a, literals);
    FlattenGuardConjunction(op->b, literals);
    return;
  }
  literals->push_back(expr);
}

bool AddGuardLiteral(const PrimExpr &expr, GuardCube *cube) {
  PrimExpr base = expr;
  bool negated = false;
  while (const auto *op = base.as<NotNode>()) {
    negated = !negated;
    base = op->a;
  }

  if (is_one(base))
    return !negated;
  if (is_zero(base))
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
  PrimExpr result = Bool(true);
  for (const GuardLiteral &literal : cube) {
    PrimExpr expr = literal.negated ? !literal.expr : literal.expr;
    result = is_one(result) ? expr : result && expr;
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
    return Bool(false);

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

  PrimExpr result = Bool(false);
  for (const GuardCube &cube : cubes) {
    PrimExpr term = RebuildGuardCube(cube);
    result = is_zero(result) ? term : result || term;
  }
  return result;
}

PrimExpr CombineGuardTerms(const std::vector<PrimExpr> &terms,
                           arith::Analyzer *analyzer) {
  PrimExpr result = Bool(false);
  for (const PrimExpr &term : terms) {
    result = DisjoinGuards(result, term, analyzer);
    if (is_one(result))
      break;
  }
  return SimplifyGuardUnion(result);
}

struct StorageEpochAnalysis {
  PrimExpr active_guard{Bool(false)};
  // Latest loop-body position that defines a variable used by active_guard.
  // Earlier representatives dominate every later proof-equivalent snapshot.
  int guard_ready_position{-1};
  bool has_write{false};
  bool needs_write_flag{false};
};

StorageEpochAnalysis AnalyzeStorageEpoch(const Array<Var> &storages,
                                         ControlNode *loop,
                                         bool writes_only = false) {
  std::vector<PrimExpr> guard_terms;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> guard_definitions;
  arith::Analyzer analyzer;
  loop->GetLoopBodyContext().Populate(analyzer);

  int first_access = -1;
  bool has_write = false;
  bool needs_write_flag = false;
  bool saw_read = false;
  std::optional<int> access_stage;
  bool has_multiple_access_stages = false;
  for (const auto &child : loop->children) {
    int child_index = child->GetIndex();
    for (const Var &var : child->GetWriteVars())
      guard_definitions[var] = child_index;
    bool touches = false, writes = false, reads = false;
    for (const Var &member : storages) {
      touches |= child->TouchesStorage(member);
      writes |= child->WritesStorage(member);
      reads |= child->ReadsStorage(member);
    }
    // Advancing an OR of writer guards must not redirect an intervening read
    // to a value whose first writer has not executed yet. Nested writers and
    // late predicates use a first-executed-Task latch instead.
    if (writes_only) {
      needs_write_flag |= writes && (saw_read || child->IsControl());
      saw_read |= reads;
      touches = writes;
    }
    if (!touches)
      continue;
    if (!access_stage.has_value()) {
      access_stage = child->GetStage();
    } else if (access_stage.value() != child->GetStage()) {
      has_multiple_access_stages = true;
    }
    if (first_access < 0)
      first_access = child_index;
    for (const Var &member : storages)
      has_write |= child->WritesStorage(member);
    guard_terms.push_back(child->GetConditionGuard());
  }
  if (writes_only && first_access < 0)
    return {}; // A disjoint read-only owner retains the preceding generation.
  ICHECK_GE(first_access, 0)
      << "Storage epoch analysis for " << storages << " contains no access";

  StorageEpochAnalysis result;
  result.has_write = has_write;
  result.needs_write_flag = needs_write_flag;
  if (has_multiple_access_stages) {
    // Snapshot guards from different stages describe different logical
    // iterations after pipeline lowering and cannot form one loop-local active
    // epoch. Keep the lexical domain unconditional instead.
    result.active_guard = Bool(true);
    result.needs_write_flag |= writes_only;
    return result;
  }
  result.active_guard = CombineGuardTerms(guard_terms, &analyzer);
  for (const Var &var : UndefinedVars(result.active_guard)) {
    auto definition = guard_definitions.find(var);
    if (definition != guard_definitions.end()) {
      result.guard_ready_position =
          std::max(result.guard_ready_position, definition->second);
    }
  }
  if (result.guard_ready_position >= first_access) {
    result.needs_write_flag |= writes_only;
    // A guard defined after the first storage access cannot safely guard a
    // loop-boundary protocol.
    result.active_guard = Bool(true);
    result.guard_ready_position = -1;
  }
  return result;
}

StorageEpochAnalysis
AnalyzeStorageEpoch(const Var &storage, ControlNode *loop,
                    const L0StorageGroups &groups = L0StorageGroups()) {
  return AnalyzeStorageEpoch(groups.Members(storage), loop);
}

std::vector<Var> CollectLoopStorages(ControlNode *loop) {
  std::vector<Var> result;
  VarSet seen;
  auto append = [&](const std::vector<BufferRegion> &regions) {
    for (const BufferRegion &region : regions) {
      const Var &storage = region->buffer->data;
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
          const auto *control = static_cast<const ControlNode *>(node);
          for (const auto &child : control->children)
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
    auto *control = static_cast<ControlNode *>(node);
    if (control->task->TouchesStorage(storage))
      result |= subtree_core_mask(control);
    for (const auto &child : control->children)
      visit(child.get());
  };
  visit(root);
  return result;
}

bool ExprUsesAnyVar(const PrimExpr &expr, const VarSet &vars) {
  if (!expr.defined() || vars.empty())
    return false;
  return UsesVar(expr, [&](const VarNode *var) {
    return vars.count(ffi::GetRef<Var>(var));
  });
}

struct StorageState {
  Var storage;
  int num_versions{1};
  BufferVersionMode requested{BufferVersionMode::kAuto};
  std::vector<ControlNode *> owners;
};

struct PreparedOwner {
  MultiBufferOwnerInfo info;
  bool needs_counter{false};
};

struct PreparedStorage {
  MultiBufferInfo info;
  bool uses_counter{false};
  std::vector<int> storage_stages;
  CoreMask protocol_core_mask{kCoreUnassigned};
  std::vector<TaskNode *> write_events;
};

class MultiBufferPlanBuilder {
public:
  MultiBufferPlanBuilder(const std::vector<std::shared_ptr<IRStructure>> &root,
                         const BufferVersionMap &selected_versions,
                         const BufferVersionModeTable &requested_modes,
                         const ffi::Optional<Var> &outer_sid,
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
      if (Optional<Bind> bind = GetFlatTaskBind(task->stmt); bind.defined())
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
            << "Bound L0 data/SF group " << state.storage->name_hint
            << " must have the same owner loops for every member";
      }
    }

    std::vector<PreparedStorage> prepared;
    for (StorageState &state : storages_) {
      ICHECK(!state.owners.empty())
          << "Automatic multi-buffer storage " << state.storage->name_hint
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
      storage.info.write_counter =
          state.requested == BufferVersionMode::kWriteCounter;
      if (storage.info.write_counter) {
        for (TaskNode *task : all_tasks) {
          bool writes = false;
          for (const Var &member : groups_.Members(state.storage))
            writes |= task->WritesStorage(member);
          if (writes)
            storage.write_events.push_back(task);
        }
        ICHECK(!storage.write_events.empty())
            << "write_counter storage " << state.storage->name_hint
            << " has no writing Task";
      }
      bool needs_counter = state.owners.size() > 1;
      for (ControlNode *owner : state.owners) {
        PreparedOwner prepared_owner = AnalyzeOwner(state.storage, owner);
        if (storage.info.write_counter)
          prepared_owner.info.active_guard = Bool(true);
        needs_counter |= prepared_owner.needs_counter;
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
          << state.storage->name_hint
          << " requires exactly one owner; use counter mode for disjoint "
             "owners";
      bool can_fallback_to_iteration = state.owners.size() == 1U;
      storage.uses_counter =
          storage.info.write_counter ||
          state.requested == BufferVersionMode::kCounter ||
          (state.requested == BufferVersionMode::kAuto && needs_counter &&
           (counter_available || !can_fallback_to_iteration));
      if (storage.uses_counter) {
        for (size_t i = 0; i < storage.info.owners.size(); ++i) {
          if (!counter_stages[i].has_value()) {
            LOG(FATAL)
                << "Counter multi-buffer storage " << state.storage->name_hint
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
        storage.info.owners.front().active_guard = Bool(true);
        ControlNode *owner = storage.info.owners.front().loop;
        if (HasNonAffineLoopPath(owner)) {
          LOG(WARNING) << "Iteration buffer versioning for storage "
                       << state.storage->name_hint
                       << " uses a non-affine loop nest"
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
    return UsesVar(expr, [&](const VarNode *node) {
      auto definition = bind_definitions_.find(ffi::GetRef<Var>(node));
      if (definition == bind_definitions_.end())
        return false;
      for (const ControlNode *outer_loop : outer_loops) {
        if (definition->second->IsWithin(outer_loop))
          return true;
      }
      return false;
    });
  }

  bool HasNonAffineLoopPath(const ControlNode *owner) const {
    VarSet outer_loop_vars;
    std::vector<const ControlNode *> outer_loops;
    for (const ControlNode *control : owner->GetAncestorControls()) {
      if (control->HasConditions() || control->BodyHasLoopBreak())
        return true;
      const For &loop = control->control;
      bool non_affine =
          RangeUsesLoopVariantValue(loop->min, outer_loop_vars, outer_loops) ||
          RangeUsesLoopVariantValue(loop->extent, outer_loop_vars, outer_loops);
      if (loop->step.has_value()) {
        non_affine |= RangeUsesLoopVariantValue(loop->step.value(),
                                                outer_loop_vars, outer_loops);
      }
      if (non_affine)
        return true;
      outer_loop_vars.insert(loop->loop_var);
      outer_loops.push_back(control);
    }
    return false;
  }

  std::optional<int> CounterStorageStage(const Var &storage,
                                         ControlNode *owner) const {
    int storage_stage = -1;
    for (const auto &child : owner->children) {
      bool touches = false;
      for (const Var &member : groups_.Members(storage))
        touches |= child->TouchesStorage(member);
      if (!touches)
        continue;
      if (storage_stage < 0) {
        storage_stage = child->GetStage();
        continue;
      }
      if (storage_stage != child->GetStage())
        return std::nullopt;
    }
    ICHECK_GE(storage_stage, 0);
    return storage_stage;
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
        << "Counter multi-buffer storage " << storage->name_hint
        << " has no core-assigned owner access";
    return result;
  }

  PreparedOwner AnalyzeOwner(const Var &storage, ControlNode *owner) {
    StorageEpochAnalysis analysis =
        AnalyzeStorageEpoch(storage, owner, groups_);
    bool has_non_affine_loop_path = HasNonAffineLoopPath(owner);
    bool needs_counter =
        !is_one(analysis.active_guard) || has_non_affine_loop_path;

    PreparedOwner result;
    result.info.loop = owner;
    result.info.active_guard = analysis.active_guard;
    result.needs_counter = needs_counter;
    return result;
  }

  void AssignCounterGroups(std::vector<PreparedStorage> *storages) const {
    struct CounterGroup {
      // These owners define the shared loop structure and guard-equivalence
      // classes. The physical advance reads the canonical guard selected by
      // AnalyzeStorageEpochs for each owner.
      std::vector<MultiBufferOwnerInfo> representative_owners;
      std::vector<CoreMask> guard_core_masks;
      std::vector<int> storage_stages;
      CoreMask protocol_core_mask;
      int counter_group_id;
      Buffer counter;
      bool write_counter;
      std::vector<TaskNode *> write_events;
    };

    auto compatible = [&](const CounterGroup &group,
                          const PreparedStorage &storage) {
      if (group.representative_owners.size() != storage.info.owners.size() ||
          group.storage_stages != storage.storage_stages ||
          group.write_counter != storage.info.write_counter ||
          (group.write_counter && group.write_events != storage.write_events)) {
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
        Buffer counter = decl_buffer(
            {IntImm(DataType::Int(32), 1)},
            storage.info.write_counter ? DataType::Int(64) : DataType::Int(32),
            storage.info.storage->name_hint + "_version_counter_" +
                std::to_string(counter_group_id),
            "local.var");
        std::vector<CoreMask> guard_core_masks;
        for (const MultiBufferOwnerInfo &owner : storage.info.owners) {
          guard_core_masks.push_back(
              availability_.GetExpressionCoreMask(owner.active_guard));
        }
        groups.push_back(
            CounterGroup{storage.info.owners, std::move(guard_core_masks),
                         storage.storage_stages, storage.protocol_core_mask,
                         counter_group_id, std::move(counter),
                         storage.info.write_counter, storage.write_events});
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
      storage.info.counter_group_id = selected->counter_group_id;
      storage.info.counter = selected->counter;
    }
  }

  static void ValidateOwnersDisjoint(const StorageState &state) {
    for (size_t i = 0; i < state.owners.size(); ++i) {
      for (size_t j = i + 1; j < state.owners.size(); ++j) {
        ICHECK(!state.owners[i]->IsWithin(state.owners[j]) &&
               !state.owners[j]->IsWithin(state.owners[i]))
            << "Automatic multi-buffer storage " << state.storage->name_hint
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
    ffi::String scope = GetPtrStorageScope(storage);
    return scope == "shared.l1" || scope == "shared.l1.dyn";
  }

  static bool IsStrictlyWithinOneOwner(const StorageState &state,
                                       const ControlNode *control) {
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
            << "Automatic multi-buffer storage " << state.storage->name_hint
            << " has a task inside nested owner loops";
        if (owner_count == 1)
          return;
        ICHECK(!task->GuardsTouchStorage(state.storage))
            << "Automatic multi-buffer storage " << state.storage->name_hint
            << " is accessed by a task guard outside its annotated owner "
               "loops; guards can follow a physical version only when they "
               "are strictly inside one owner";
        ICHECK(CanBroadcastFillToStorage(task->stmt, state.storage))
            << "Automatic multi-buffer storage " << state.storage->name_hint
            << " is accessed outside its annotated owner loops; "
               "only T.fill accesses that can be rewritten independently "
               "may target every physical version; offending task: "
            << task->stmt;
        ICHECK(state.num_versions <= 1 || !IsL1Storage(state.storage))
            << "Broadcast fill for L1 storage " << state.storage->name_hint
            << " with " << state.num_versions << " versions is not supported";
        // ResolveCore needs this marker to exempt the owner-external task for
        // every logical multi-buffer plan. MaterializeMultiBuffer later clears
        // it without rewriting storage when the selected version count is one.
        AddMultiBufferBroadcastFill(task, state.storage);
        return;
      }
      auto *control = static_cast<ControlNode *>(node);
      const For &loop = control->control;
      bool range_touches_storage =
          ExprTouchesStorage(loop->min, state.storage) ||
          ExprTouchesStorage(loop->extent, state.storage) ||
          (loop->step.has_value() &&
           ExprTouchesStorage(loop->step.value(), state.storage));
      ICHECK(!range_touches_storage)
          << "Automatic multi-buffer storage " << state.storage->name_hint
          << " is used by an unnormalized loop bound; run "
             "NormalizeControlFlowForSchedule before MaterializeScheduleUnits "
             "so buffer-dependent loop bounds become schedulable tasks";
      bool guard_touches_storage = control->GuardsTouchStorage(state.storage);
      if (guard_touches_storage) {
        ICHECK(IsStrictlyWithinOneOwner(state, control))
            << "Automatic multi-buffer storage " << state.storage->name_hint
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
                     const ffi::Optional<Var> &outer_sid,
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
    } else if (mode == "write_counter") {
      parsed = BufferVersionMode::kWriteCounter;
    } else {
      ICHECK_EQ(mode, "counter");
      parsed = BufferVersionMode::kCounter;
    }
    result.emplace(storage, parsed);
  }
  return result;
}

// Preparation plans retain storage identities, guards and insertion anchors.
// Only MaterializeMultiBufferPreparation changes the scheduled tree or emits
// executable counter statements.
struct LoopEpochPlan {
  MultiBufferCounterMap counters;
  StorageEpochGuardMap guards;
  Array<Var> write_counters;
  StorageEpochGuardMap write_updates;
};

using LoopEpochPlans = std::unordered_map<ControlNode *, LoopEpochPlan>;

struct CounterAdvancePlan {
  IRStructure *anchor;
  PrimExpr guard;
};

struct CounterOwnerPlan {
  ControlNode *loop;
  Optional<Buffer> advanced;
  int stage{0};
  std::vector<CounterAdvancePlan> advances;
};

struct CounterProtocolPlan {
  const MultiBufferInfo *info{nullptr};
  std::vector<Var> storages;
  std::vector<CounterOwnerPlan> owners;
};

using CounterProtocolPlans = std::map<int, CounterProtocolPlan>;

LoopEpochPlans AnalyzeStorageEpochs(
    const std::vector<std::shared_ptr<IRStructure>> &root,
    const MultiBufferPlan &plan, const BufferVersionMap &manual_buffer_versions,
    const ffi::Optional<Var> &outer_sid, const L0StorageGroups &groups) {
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
    if (!node || !node->IsControl())
      return;
    auto *loop = static_cast<ControlNode *>(node);
    loops.push_back(loop);
    for (const auto &child : loop->children)
      collect_loops(child.get());
  };
  for (const auto &node : root)
    collect_loops(node.get());

  struct StorageEpoch {
    Var storage;
    PrimExpr active_guard;
    int guard_ready_position;
    CoreMask required_core_mask{kCoreUnassigned};
  };

  LoopEpochPlans result;
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
          if (!info->write_counter || owner->loop == loop)
            continue;
        }
        // Above or between disjoint owners, the same storage participates in
        // an ordinary lexical domain at this loop.
      }
      if (manual_buffer_versions.count(storage)) {
        continue;
      }
      std::string storage_scope = GetPtrStorageScope(storage);
      if (storage_scope == "local" || storage_scope == "local.var" ||
          storage_scope == "local.fragment") {
        continue;
      }

      // A read-only storage cannot induce an intra-loop RAW/WAW/WAR epoch.
      // Its enclosing dependency is represented in an ancestor lexical
      // domain, so avoid creating loop-local domains that no edge can select.
      if (analysis.has_write && !is_one(analysis.active_guard)) {
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
        {Bool(true), kCoreAll}};
    arith::Analyzer domain_analyzer;
    loop->GetLoopBodyContext().Populate(domain_analyzer);
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
            domain_analyzer.CanProve(representative.expression ==
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

    Array<Var> write_counters;
    for (const MultiBufferInfo &info : plan.Infos())
      if (info.write_counter)
        for (const auto &owner : info.owners)
          if (owner.loop == loop)
            write_counters.push_back(info.storage);
    StorageEpochGuardMap write_updates;
    if (auto map = loop->control->annotations.Get(kWriteCounterUpdateGuardMap))
      write_updates = map.value().cast<StorageEpochGuardMap>();
    result.emplace(loop,
                   LoopEpochPlan{std::move(counter_map), std::move(guard_map),
                                 std::move(write_counters),
                                 std::move(write_updates)});
  }

  // Every counter owner must be represented by the same analysis that chose
  // its canonical guard. Later planning consumes this result directly, without
  // depending on annotations already written back to the IR.
  for (const MultiBufferInfo &info : plan.Infos()) {
    if (!info.UsesCounter())
      continue;
    for (const MultiBufferOwnerInfo &owner : info.owners) {
      auto epoch = result.find(owner.loop);
      ICHECK(epoch != result.end())
          << "Counter owner is missing prepared epoch metadata for storage "
          << info.storage->name_hint;
      ICHECK(epoch->second.counters.count(info.storage));
      ICHECK(epoch->second.guards.count(info.storage));
    }
  }
  return result;
}

// An explicit write_counter annotation promises that reads see an initialized
// generation and that an update does not start by reading the previous value.
// Trust this storage-level contract, like an explicit multi_buffer_eligible
// claim: do not re-prove data initialization, first-write dominance or region
// coverage here. Such proofs reject valid multi-Task/loop/sparse updates.
// Automatic owner inference is still a heuristic; users can select the owner
// explicitly when it fails. Violating the contract can read stale/uninitialized
// data without a diagnostic. Allocation reuse keeps its own must-write proof.
// Only structural requirements of counter lowering are checked below.
class WriteCounterValidator {
public:
  WriteCounterValidator(const MultiBufferInfo &info, const LayoutMap &layouts)
      : info_(info), layouts_(layouts) {}

  void Validate(const std::vector<std::shared_ptr<IRStructure>> &root) {
    std::vector<TaskNode *> tasks;
    CollectAllTaskNodes(root, tasks);
    VarSet varying_geometry;
    for (const auto &owner : info_.owners) {
      VarSet variables = CollectIterationVariantVars(owner.loop);
      varying_geometry.insert(variables.begin(), variables.end());
    }
    auto check_geometry = [&](const BufferRegion &region) {
      if (!region->buffer->data.same_as(info_.storage))
        return;
      const Buffer &buffer = region->buffer;
      auto stable = [&](const PrimExpr &expr) {
        return SideEffect(expr) <= CallEffectKind::kPure &&
               !UsesVar(expr, [&](const VarNode *var) {
                 return varying_geometry.count(ffi::GetRef<Var>(var));
               });
      };
      bool invariant = stable(buffer->elem_offset);
      for (const PrimExpr &extent : buffer->shape)
        invariant &= stable(extent);
      for (const PrimExpr &stride : buffer->strides)
        invariant &= stable(stride);
      ICHECK(invariant)
          << "write_counter requires loop-invariant buffer view geometry for "
          << info_.storage->name_hint
          << "; express changing footprints with explicit regions instead of "
             "a changing shape, stride or elem_offset";
    };
    for (TaskNode *task : tasks) {
      MemoryAccessDetector accesses(MemoryAccessDetector::AccessMode::kPhysical,
                                    layouts_);
      accesses.Analyze(task->stmt);
      for (const auto &region : accesses.GetReadRegions())
        check_geometry(region);
      for (const auto &region : accesses.GetWriteRegions())
        check_geometry(region);
      for (const Var &opaque : accesses.GetOpaqueAccessVars())
        ICHECK(!opaque.same_as(info_.storage))
            << "write_counter cannot bind an opaque pointer access to "
            << info_.storage->name_hint << "; use explicit buffer regions";
      ICHECK(!task->GuardsTouchStorage(info_.storage))
          << "write_counter storage " << info_.storage->name_hint
          << " is read by an unsnapshotted guard; bind the predicate in a "
             "separate Task before the write";
      if (!task->TouchesStorage(info_.storage))
        continue;
      ICHECK(SideEffect(task->GetConditionGuard()) <= CallEffectKind::kPure)
          << "write_counter requires snapshotted task predicates for "
          << info_.storage->name_hint
          << "; run NormalizeControlFlowForSchedule before preparation";
      ICHECK(info_.FindOwner(task))
          << "write_counter requires all accesses, including fills, inside "
             "its owner loops: "
          << info_.storage->name_hint;
      ICHECK(!task->IsPerCoreTask())
          << "write_counter requires a uniform writing/reading Task for "
          << info_.storage->name_hint;
    }
    CheckStructure(root);
  }

private:
  void
  CheckStructure(const std::vector<std::shared_ptr<IRStructure>> &nodes) const {
    int stage = -1;
    for (const auto &node : nodes) {
      if (!node->TouchesStorage(info_.storage))
        continue;
      if (stage < 0)
        stage = node->GetStage();
      ICHECK_EQ(stage, node->GetStage())
          << "write_counter accesses must have equal T.Stage values in each "
             "containing loop: "
          << info_.storage->name_hint;
      if (node->IsTask())
        continue;
      auto *loop = static_cast<ControlNode *>(node.get());
      ICHECK(!loop->GuardsTouchStorage(info_.storage))
          << "write_counter control guards must snapshot reads of "
          << info_.storage->name_hint << " in a separate Task";
      const MultiBufferOwnerInfo *owner = info_.FindOwner(loop);
      ICHECK(!owner || owner->loop != loop || !loop->BodyHasLoopBreak())
          << "write_counter does not yet support loop_break in its owner "
             "loop: "
          << info_.storage->name_hint;
      CheckStructure(loop->children);
    }
  }

  const MultiBufferInfo &info_;
  const LayoutMap &layouts_;
};

CounterProtocolPlans PlanCounterProtocols(const ScheduledTIR &scheduled,
                                          const MultiBufferPlan &plan,
                                          LoopEpochPlans *epochs) {
  LayoutMap layouts;
  if (auto value =
          scheduled.metadata.kernel_root->annotations.Get("layout_map"))
    layouts = value.value().cast<LayoutMap>();
  CounterProtocolPlans protocols;
  for (const MultiBufferInfo &info : plan.Infos()) {
    if (!info.UsesCounter())
      continue;
    if (info.write_counter)
      WriteCounterValidator(info, layouts).Validate(scheduled.tree);
    CounterProtocolPlan &protocol = protocols[info.counter_group_id];
    if (!protocol.info) {
      protocol.info = &info;
    } else {
      ICHECK(protocol.info->counter.same_as(info.counter));
      ICHECK_EQ(protocol.info->write_counter, info.write_counter);
      ICHECK_EQ(protocol.info->owners.size(), info.owners.size());
      for (size_t i = 0; i < info.owners.size(); ++i)
        ICHECK_EQ(protocol.info->owners[i].loop, info.owners[i].loop);
    }
    protocol.storages.push_back(info.storage);
  }

  for (auto &[_, protocol] : protocols) {
    const MultiBufferInfo &info = *protocol.info;
    auto touches = [&](const IRStructure *node, bool writes_only) {
      for (const Var &storage : protocol.storages)
        if (writes_only ? node->WritesStorage(storage)
                        : node->TouchesStorage(storage))
          return true;
      return false;
    };
    for (const MultiBufferOwnerInfo &owner : info.owners) {
      CounterOwnerPlan placement{owner.loop, {}, 0, {}};
      LoopEpochPlan &epoch_plan = epochs->at(owner.loop);
      if (!info.write_counter) {
        PrimExpr active_guard;
        for (const Var &storage : protocol.storages) {
          auto guard = epoch_plan.guards.Get(storage);
          ICHECK(guard.has_value())
              << "Counter owner is missing the epoch guard for storage "
              << storage->name_hint;
          if (!active_guard.defined()) {
            active_guard = guard.value();
          } else {
            ICHECK(active_guard.same_as(guard.value()))
                << "Shared counter storages must use one canonical epoch guard";
          }
        }
        ICHECK(active_guard.defined());
        IRStructure *anchor = nullptr;
        for (const auto &child : owner.loop->children)
          if (touches(child.get(), /*writes_only=*/false))
            anchor = child.get();
        ICHECK(anchor) << "Counter multi-buffer owner has no storage access";
        placement.advances.push_back({anchor, active_guard});
      } else {
        // Advancing the shared clock redirects every member's reads. A read
        // of any member between writers must therefore keep the old version
        // until the first writing Task actually executes.
        StorageEpochAnalysis epoch = AnalyzeStorageEpoch(
            Array<Var>(protocol.storages), owner.loop, /*writes_only=*/true);
        PrimExpr active_guard = epoch.active_guard;
        for (const auto &child : owner.loop->children)
          if (touches(child.get(), /*writes_only=*/true)) {
            placement.stage = child->GetStage();
            break;
          }
        if (epoch.needs_write_flag) {
          Buffer advanced =
              decl_buffer({1}, DataType::Bool(),
                          info.storage->name_hint + "_advanced", "local.var");
          placement.advanced = advanced;
          active_guard = BufferLoad(advanced, {0});
        }
        for (const Var &storage : protocol.storages)
          epoch_plan.write_updates.Set(storage, active_guard);
        std::function<void(const std::vector<std::shared_ptr<IRStructure>> &)>
            collect = [&](const auto &nodes) {
              for (const auto &node : nodes) {
                if (node->IsControl()) {
                  collect(static_cast<ControlNode *>(node.get())->children);
                } else if (touches(node.get(), /*writes_only=*/true) &&
                           (placement.advanced.defined() ||
                            placement.advances.empty())) {
                  PrimExpr guard =
                      placement.advanced.defined()
                          ? node->GetConditionGuard() && !active_guard
                          : active_guard;
                  placement.advances.push_back({node.get(), guard});
                }
              }
            };
        collect(owner.loop->children);
      }
      protocol.owners.push_back(std::move(placement));
    }
  }
  return protocols;
}

Stmt MakeCompilerTask(const Stmt &body) {
  TaskMetadata metadata;
  return AttrStmt(MergeTaskMetadata(metadata, IntImm(DataType::Int(32), 0)),
                  attr::kAscendTask, Integer(1), body, body->span);
}

std::shared_ptr<TaskNode> MakeCounterTask(const Stmt &body, int stage) {
  auto task = std::make_shared<TaskNode>();
  task->stmt = MakeCompilerTask(body);
  task->SetStage(stage);
  // Counter state is local.var and can be cloned to either core. ResolveCore
  // narrows this candidate from the prepared owner/core protocol.
  task->SetCoreMask(kCoreAll);
  task->SetPipeMask(static_cast<uint16_t>(ResourcePipe::kScalar));
  return task;
}

void MaterializeMultiBufferPreparation(ScheduledTIR *scheduled,
                                       const LoopEpochPlans &epochs,
                                       const CounterProtocolPlans &protocols) {
  for (const auto &[loop, epoch] : epochs) {
    For rewritten = loop->control;
    ForNode *node = rewritten.CopyOnWrite();
    if (!epoch.counters.empty())
      node->annotations.Set(kMultiBufferCounterMap, epoch.counters);
    if (!epoch.write_counters.empty())
      node->annotations.Set(kWriteCounterBuffers, epoch.write_counters);
    if (!epoch.write_updates.empty())
      node->annotations.Set(kWriteCounterUpdateGuardMap, epoch.write_updates);
    node->annotations.Set(kStorageEpochGuardMap, epoch.guards);
    loop->control = std::move(rewritten);
  }

  using NodeList = std::vector<std::shared_ptr<IRStructure>>;
  struct CounterInsertions {
    NodeList before;
    NodeList after;
  };
  std::unordered_map<IRStructure *, CounterInsertions> insertions;
  std::unordered_map<ControlNode *, NodeList> prefixes;
  NodeList initializers;
  Array<Buffer> buffers = scheduled->metadata.kernel_root->alloc_buffers;
  // Keep the existing allocation and initialization order for mixed modes.
  // Both modes use the same insertion machinery and edit the tree only once.
  for (bool write_counter : {false, true}) {
    NodeList group_initializers;
    for (const auto &[_, protocol] : protocols) {
      const MultiBufferInfo &info = *protocol.info;
      if (info.write_counter != write_counter)
        continue;
      buffers.push_back(info.counter);
      group_initializers.push_back(MakeCounterTask(
          BufferStore(info.counter,
                      make_const(info.counter->dtype, write_counter ? -1 : 0),
                      {0}),
          0));
      for (const CounterOwnerPlan &owner : protocol.owners) {
        if (owner.advanced.defined()) {
          const Buffer &advanced = owner.advanced.value();
          buffers.push_back(advanced);
          NodeList &prefix = prefixes[owner.loop];
          prefix.insert(prefix.begin(),
                        MakeCounterTask(BufferStore(advanced, Bool(false), {0}),
                                        owner.stage));
        }
        for (const CounterAdvancePlan &placement : owner.advances) {
          IRStructure *anchor = placement.anchor;
          const PrimExpr &guard = placement.guard;
          PrimExpr value = BufferLoad(info.counter, {0});
          Stmt body = BufferStore(info.counter,
                                  value + make_const(value.dtype(), 1), {0});
          if (owner.advanced.defined()) {
            body = SeqStmt::Flatten(
                body, BufferStore(owner.advanced.value(), Bool(true), {0}));
          }
          // Ordinary counters advance after their last access; their guard is
          // part of the scalar task body. Write counters advance before their
          // first executed writer, exposing the same guard to InsertSync.
          if (!write_counter)
            body = WrapWithGuard(std::move(body), guard);
          auto advance = MakeCounterTask(body, anchor->GetStage());
          if (write_counter && !is_one(guard)) {
            GuardList guards;
            guards.push_back(std::make_unique<ConditionGuard>(guard));
            advance->SetGuards(std::move(guards));
          }
          CounterInsertions &at = insertions[anchor];
          (write_counter ? at.before : at.after).push_back(std::move(advance));
        }
      }
    }
    initializers.insert(initializers.begin(), group_initializers.begin(),
                        group_initializers.end());
  }

  std::function<void(NodeList &, ControlNode *)> rewrite =
      [&](NodeList &nodes, ControlNode *parent) {
        NodeList result;
        if (auto prefix = prefixes.find(parent); prefix != prefixes.end())
          result = prefix->second;
        for (const auto &node : nodes) {
          if (node->IsControl()) {
            auto *child = static_cast<ControlNode *>(node.get());
            rewrite(child->children, child);
          }
          auto insertion = insertions.find(node.get());
          if (insertion != insertions.end())
            result.insert(result.end(), insertion->second.before.begin(),
                          insertion->second.before.end());
          result.push_back(node);
          if (insertion != insertions.end())
            result.insert(result.end(), insertion->second.after.begin(),
                          insertion->second.after.end());
        }
        if (!parent)
          result.insert(result.begin(), initializers.begin(),
                        initializers.end());
        for (size_t i = 0; i < result.size(); ++i) {
          result[i]->SetIndex(i);
          result[i]->SetParent(parent);
        }
        nodes = std::move(result);
      };
  rewrite(scheduled->tree, nullptr);
  SBlock root = scheduled->metadata.kernel_root;
  root.CopyOnWrite()->alloc_buffers = std::move(buffers);
  scheduled->metadata.kernel_root = std::move(root);
}

} // namespace

tvm::transform::Pass PrepareMultiBuffer() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    return RewriteTilelangKernels(
        std::move(func), "PrepareMultiBuffer",
        [](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir =
              DecodeScheduledTIR(context.root, context.outer_ctx);
          std::vector<TaskNode *> tasks;
          CollectAllTaskNodes(scheduled_tir.tree, tasks);
          for (TaskNode *task : tasks) {
            ICHECK_NE(task->GetCoreMask(), kCoreUnassigned)
                << "PrepareMultiBuffer requires AssignCore to run first";
          }
          L0StorageGroups groups(CollectL0SFBindings(context.root));
          auto &metadata = scheduled_tir.metadata;
          BufferVersionMap selected_versions = ExpandL0StorageGroupValues(
              metadata.buffer_versions, groups, "version counts");
          BufferVersionModeMap requested_modes = ExpandL0StorageGroupValues(
              metadata.buffer_version_modes, groups, "version modes");
          BufferVersionModeTable modes =
              ParseBufferVersionModes(requested_modes);
          for (const auto &[storage, mode] : modes)
            ICHECK(mode != BufferVersionMode::kWriteCounter ||
                   selected_versions.count(storage))
                << "write_counter storage " << storage->name_hint
                << " has no selected version count/owner; the requested "
                   "mode cannot be silently disabled";
          MultiBufferPlan plan =
              BuildMultiBufferPlan(scheduled_tir.tree, selected_versions, modes,
                                   context.outer_sid, groups);
          LoopEpochPlans epochs = AnalyzeStorageEpochs(
              scheduled_tir.tree, plan,
              scheduled_tir.metadata.manual_buffer_versions, context.outer_sid,
              groups);
          CounterProtocolPlans protocols =
              PlanCounterProtocols(scheduled_tir, plan, &epochs);
          metadata.buffer_versions = std::move(selected_versions);
          metadata.buffer_version_modes = {};
          MaterializeMultiBufferPreparation(&scheduled_tir, epochs, protocols);
          return EncodeScheduledTIR(std::move(scheduled_tir));
        });
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PrepareMultiBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PrepareMultiBuffer", PrepareMultiBuffer);
}

} // namespace tl
} // namespace tvm
