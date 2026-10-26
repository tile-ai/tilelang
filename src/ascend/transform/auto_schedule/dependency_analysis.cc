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

/*!
 * \file dependency_analysis.cc
 * \brief Shared schedule and synchronization dependency analysis.
 */

#include "./dependency_analysis.h"
#include "./task_annotations.h"
#include "ascend/op/utils.h"
#include "op/prim.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./scheduled_tir.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include "ascend/layout/ascend_layouts.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

// An undefined storage key groups explicit conflicts between distinct storage.
using CoveredDependencyMap =
    std::unordered_map<const VarNode *, std::set<DependencyTaskPair>>;

// A bare buffer hint matches every region using its storage. A region hint
// matches only the same logical buffer and exact footprint.
static Var RegionStorage(const TensorRegion &region,
                         const StorageAliasAnalysis *analysis) {
  auto tensor = region->source.as_or_throw<TensorVar>();
  if (analysis)
    if (auto access = analysis->Resolve(tensor))
      return access->root;
  return tensor;
}

static bool HintOperandMatches(const Any &hint, const TensorRegion &query,
                               const StorageAliasAnalysis *analysis) {
  if (auto var = hint.as<Var>())
    return var.value().same_as(RegionStorage(query, analysis)) ||
           (analysis && analysis->Resolve(var.value()) &&
            analysis->Resolve(var.value())
                ->root.same_as(RegionStorage(query, analysis)));
  TensorRegion hint_region = hint.as_or_throw<TensorRegion>();
  return hint_region->source.as_or_throw<TensorVar>().same_as(
             query->source.as_or_throw<TensorVar>()) &&
         RegionsEqual(hint_region->region, query->region);
}

static ConflictHintList
ActiveConflictHints(ControlNode *loop, const ConflictHintList &root_hints) {
  if (loop == nullptr)
    return root_hints;
  auto ann = loop->control->annotations.Get("conflict_hint");
  return ann.has_value() ? ann.value().cast<ConflictHintList>()
                         : ConflictHintList{};
}

static bool IsConflictHint(const Array<Any> &hint) {
  ICHECK_EQ(hint.size(), 4u)
      << "conflict_hint annotation expects 4-element entries";
  auto is_conflict = GetIntAnnotation(hint[3]);
  ICHECK(is_conflict && (*is_conflict == 0 || *is_conflict == 1))
      << "conflict_hint annotation expects a constant boolean polarity";
  return *is_conflict != 0;
}

// Return the suffix of the enclosing loop nest that forms one storage-local
// iteration space. Walk outward from `loop` until the parent contains another
// access to `storage` outside the child on this path. That parent is the scope
// where separate access sequences meet, so only its child-to-`loop` suffix is
// renamed when comparing two iterations. If no such parent exists, use the
// complete enclosing loop nest.
std::vector<const ControlNode *> StorageIterationNest(ControlNode *loop,
                                                      const Var &storage) {
  std::vector<const ControlNode *> controls = loop->GetAncestorControls();
  size_t first = 0;
  for (size_t i = controls.size(); i > 1; --i) {
    const ControlNode *parent = controls[i - 2];
    const IRStructure *path_child = controls[i - 1];
    ICHECK_EQ(path_child->GetParent(), parent);

    bool has_external_access =
        parent->task && parent->task->TouchesStorage(storage);
    for (const auto &child : parent->children) {
      if (child.get() != path_child && child->TouchesStorage(storage)) {
        has_external_access = true;
        break;
      }
    }
    if (has_external_access) {
      first = i - 1;
      break;
    }
  }
  return {controls.begin() + first, controls.end()};
}

bool RegionsMayConflict(const ConstrSet &a_ctx, const TensorRegion &a_region,
                        const ConstrSet &b_ctx, const TensorRegion &b_region,
                        ControlNode *loop, int offset,
                        size_t num_storage_owners,
                        const ConflictHintList &root_conflicts,
                        Optional<PrimExpr> a_pitch, Optional<PrimExpr> b_pitch,
                        const StorageAliasAnalysis *storage_analysis) {
  const bool cross = offset != 0;
  if (cross && loop == nullptr)
    return false;

  bool declared_no_conflict = false;
  for (const Any &entry : ActiveConflictHints(loop, root_conflicts)) {
    Array<Any> hint = entry.as_or_throw<Array<Any>>();
    bool is_conflict = IsConflictHint(hint);
    int64_t cross_code = GetIntAnnotation(hint[2]).value();
    if (cross_code != -1 && (cross_code == 1) != cross)
      continue;
    bool matches = (HintOperandMatches(hint[0], a_region, storage_analysis) &&
                    HintOperandMatches(hint[1], b_region, storage_analysis)) ||
                   (HintOperandMatches(hint[0], b_region, storage_analysis) &&
                    HintOperandMatches(hint[1], a_region, storage_analysis));
    if (!matches)
      continue;
    if (is_conflict)
      return true;
    declared_no_conflict = true;
  }
  // A forced conflict wins if contradictory declarations match.
  if (declared_no_conflict)
    return false;

  if (!RegionStorage(a_region, storage_analysis)
           .same_as(RegionStorage(b_region, storage_analysis)))
    return false;

  // Same storage but different logical buffer (T.view/T.reshape): regions may
  // differ in rank/dtype/strides, so keep the dependency conservatively.
  if (!a_region->source.as_or_throw<TensorVar>().same_as(
          b_region->source.as_or_throw<TensorVar>()))
    return true;

  const Array<Range> &a_rng = a_region->region;
  const Array<Range> &b_rng = b_region->region;
  if (a_rng.size() != b_rng.size())
    return true; // shape mismatch — be safe
  const size_t ndim = a_rng.size();

  sym::Analyzer ana;
  ana->z3_prover.SetRLimit(50000);
  Map<PrimVar, PrimExpr> a_sub, b_sub;
  ConstrSet a_local_ctx = a_ctx;
  ConstrSet b_local_ctx = b_ctx;
  ConstrSet loop_ctx;
  // Scheduled tasks can define condition snapshots outside the constraint
  // visitor's Bind stack. Rename all variables written in the compared
  // iteration scope, including these otherwise-free predicates. Sharing one
  // predicate between producer and consumer would incorrectly make opposite
  // branches mutually exclusive across different iterations.
  auto seed_iteration_vars = [&](const ControlNode *scope) {
    for (const Var &variable : scope->GetWriteVars()) {
      auto scalar = variable.as<PrimVar>();
      if (!scalar)
        continue;
      PrimVar var = scalar.value();
      if (!a_sub.count(var))
        a_sub.Set(var, PrimVar(var->name + "_p", var.ty()));
      if (!b_sub.count(var))
        b_sub.Set(var, PrimVar(var->name + "_c", var.ty()));
    }
  };
  if (cross) {
    if (offset < 0) {
      std::vector<const ControlNode *> iteration_nest =
          num_storage_owners > 1
              ? loop->GetAncestorControls()
              : StorageIterationNest(loop,
                                     RegionStorage(a_region, storage_analysis));
      ICHECK(!iteration_nest.empty());
      seed_iteration_vars(iteration_nest.front());
      PrimVar pivot_lv = iteration_nest.front()->control->var;
      a_local_ctx = a_local_ctx.RenameFrom("_p", a_sub, pivot_lv);
      b_local_ctx = b_local_ctx.RenameFrom("_c", b_sub, pivot_lv);
      if (num_storage_owners <= 1) {
        PrimExpr different_iteration = IntImm::Bool(false);
        for (const ControlNode *control : iteration_nest) {
          PrimVar iteration = control->control->var;
          PrimExpr p = Substitute(iteration, a_sub);
          PrimExpr c = Substitute(iteration, b_sub);
          different_iteration = different_iteration || (p != c);
        }
        loop_ctx.AddConstr(different_iteration);
      }
    } else {
      seed_iteration_vars(loop);
      PrimVar iteration = loop->control->var;
      a_local_ctx = a_local_ctx.RenameFrom("_p", a_sub, iteration);
      b_local_ctx = b_local_ctx.RenameFrom("_c", b_sub, iteration);
      PrimExpr p = Substitute(iteration, a_sub);
      PrimExpr c = Substitute(iteration, b_sub);
      loop_ctx.AddConstr(c == p + tl::MakeConst(iteration.ty(), offset) *
                                      Substitute(loop->control->step, a_sub));
    }
  }
  auto constraints =
      loop_ctx.Merge(a_local_ctx).Merge(b_local_ctx).Populate(ana);

  FreshenMutableReads freshen_a(FreshenMutableReads::Mode::kSnapshot);
  FreshenMutableReads freshen_b(FreshenMutableReads::Mode::kSnapshot);
  size_t compared_dims = ndim;
  TensorVar buffer = a_region->source.as_or_throw<TensorVar>();
  if (IsL0ABuffer(buffer) || IsL0BBuffer(buffer) || IsL0CBuffer(buffer)) {
    bool same_pitch = false;
    if (a_pitch.has_value() && b_pitch.has_value()) {
      PrimExpr equal = freshen_a(Substitute(a_pitch.value(), a_sub)) ==
                       freshen_b(Substitute(b_pitch.value(), b_sub));
      same_pitch =
          tl::arith::CanProve(ana, equal) || tl::arith::Z3CanProve(ana, equal);
    }
    // L0 matrix regions may overlap unless their pitches are provably equal.
    if (!same_pitch)
      compared_dims = ndim > 2 ? ndim - 2 : 0;
  }
  PrimExpr any_dim_disjoint = IntImm::Bool(false);
  for (size_t i = 0; i < compared_dims; ++i) {
    PrimExpr a_min = freshen_a(Substitute(a_rng[i]->begin, a_sub));
    PrimExpr a_extent = freshen_a(Substitute(a_rng[i]->extent, a_sub));
    PrimExpr b_min = freshen_b(Substitute(b_rng[i]->begin, b_sub));
    PrimExpr b_extent = freshen_b(Substitute(b_rng[i]->extent, b_sub));
    // A half-open endpoint can exceed the int32 index range by one even
    // for an ordinary single-element access.
    PrimExpr a_end = tl::CastValue(a_min, PrimType::Int(64)) +
                     tl::CastValue(a_extent, PrimType::Int(64));
    PrimExpr b_end = tl::CastValue(b_min, PrimType::Int(64)) +
                     tl::CastValue(b_extent, PrimType::Int(64));
    any_dim_disjoint = any_dim_disjoint || (a_extent <= 0) || (b_extent <= 0) ||
                       (a_end <= b_min) || (b_end <= a_min);
  }
  any_dim_disjoint = tl::arith::Simplify(ana, any_dim_disjoint);
  // Raw Z3 as a fallback catches the infeasible-context case that Analyzer's
  // constant-fold short-circuit misses.
  return !(tl::arith::CanProve(ana, any_dim_disjoint) ||
           tl::arith::Z3CanProve(ana, any_dim_disjoint));
}

std::vector<DepInfo>
AnalyzeDependencies(std::vector<IRStructure *> nodes, ControlNode *loop,
                    const BufferVersionMap &manual_buffer_versions,
                    const MultiBufferOwnerMap &multi_buffer_owners,
                    DependencyCache *dependency_cache,
                    const ConflictHintList &root_conflicts) {
  DependencyCache local_cache;
  DependencyCache &cache = dependency_cache ? *dependency_cache : local_cache;
  if (loop) {
    auto cache_it = cache.find(loop);
    if (cache_it != cache.end())
      return cache_it->second;
    std::stable_sort(nodes.begin(), nodes.end(),
                     [](const IRStructure *lhs, const IRStructure *rhs) {
                       return lhs->GetStage() < rhs->GetStage();
                     });
  }

  const size_t n = nodes.size();
  std::vector<DepInfo> deps;

  struct RegionAccess {
    TaskNode *task;
    TensorRegion region;
  };
  struct NodeAccesses {
    std::vector<RegionAccess> reads;
    std::vector<RegionAccess> writes;
  };
  std::vector<NodeAccesses> accesses(n);
  for (size_t i = 0; i < n; ++i) {
    std::vector<TaskNode *> tasks;
    CollectAllTaskNodes(nodes[i], tasks);
    for (TaskNode *task : tasks) {
      for (const TensorRegion &region : task->GetReadRegions())
        accesses[i].reads.push_back({task, region});
      for (const TensorRegion &region : task->GetWriteRegions())
        accesses[i].writes.push_back({task, region});
    }
  }

  std::function<void(IRStructure *, CoveredDependencyMap &)>
      collect_subtree_deps = [&](IRStructure *node, CoveredDependencyMap &out) {
        if (!node || !node->IsControl())
          return;
        auto *ctrl = static_cast<ControlNode *>(node);
        std::vector<IRStructure *> children;
        for (const auto &child : ctrl->children)
          children.push_back(child.get());
        auto child_deps =
            AnalyzeDependencies(children, ctrl, manual_buffer_versions,
                                multi_buffer_owners, &cache, root_conflicts);
        for (const auto &cdep : child_deps) {
          auto &covered_pairs =
              out[cdep.storage ? cdep.storage.value().get() : nullptr];
          covered_pairs.insert(cdep.task_pairs.begin(), cdep.task_pairs.end());
        }
        for (auto *child : children)
          collect_subtree_deps(child, out);
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
      if (!task->IsWithin(owner))
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
            if (IsSharedBuffer(lhs.region->source.as_or_throw<TensorVar>()) &&
                IsSharedBuffer(rhs.region->source.as_or_throw<TensorVar>())) {
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
            Var lhs_storage = lhs.task->StorageOf(
                lhs.region->source.as_or_throw<TensorVar>());
            Var rhs_storage = rhs.task->StorageOf(
                rhs.region->source.as_or_throw<TensorVar>());
            ffi::Optional<Var> storage;
            if (lhs_storage.same_as(rhs_storage))
              storage = lhs_storage;
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
                if (RegionsMayConflict(
                        lhs.task->outer_ctx, lhs.region, rhs.task->outer_ctx,
                        rhs.region, loop, d,
                        /*num_storage_owners=*/0, root_conflicts,
                        lhs.task->GetL0Pitch(
                            lhs.region->source.as_or_throw<TensorVar>()),
                        rhs.task->GetL0Pitch(
                            rhs.region->source.as_or_throw<TensorVar>()),
                        lhs.task->GetStorageAnalysis())) {
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
              if (RegionsMayConflict(
                      lhs.task->outer_ctx, lhs.region, rhs.task->outer_ctx,
                      rhs.region, loop, distance,
                      get_num_storage_owners(storage), root_conflicts,
                      lhs.task->GetL0Pitch(
                          lhs.region->source.as_or_throw<TensorVar>()),
                      rhs.task->GetL0Pitch(
                          rhs.region->source.as_or_throw<TensorVar>()),
                      lhs.task->GetStorageAnalysis())) {
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

      if (i == j && nodes[i]->IsControl()) {
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
  if (loop)
    cache.emplace(loop, deps);
  return deps;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.ascend.AnalyzeScheduleDependencies", [](RegionStmt root) {
        auto scheduled = DecodeScheduledTIR(root, {});
        ffi::Array<ffi::Map<ffi::String, ffi::Any>> result;
        DependencyCache cache;
        std::function<void(const std::vector<std::shared_ptr<IRStructure>> &,
                           ControlNode *)>
            collect = [&](const auto &nodes, ControlNode *loop) {
              std::vector<IRStructure *> ordered;
              for (const auto &node : nodes)
                ordered.push_back(node.get());
              for (const auto &dependency : AnalyzeDependencies(
                       ordered, loop, scheduled.metadata.manual_buffer_versions,
                       {}, &cache, scheduled.metadata.root_conflict_hints)) {
                ffi::Array<ffi::Array<Stmt>> pairs;
                for (const auto &[producer, consumer] : dependency.task_pairs)
                  pairs.push_back({producer->stmt, consumer->stmt});
                result.push_back(
                    {{"producer",
                      static_cast<int64_t>(dependency.prod_node->GetIndex())},
                     {"consumer",
                      static_cast<int64_t>(dependency.cons_node->GetIndex())},
                     {"distance", dependency.distance},
                     {"kind", static_cast<int>(dependency.kind)},
                     {"storage", dependency.storage},
                     {"loop", loop ? ffi::Any(loop->control) : ffi::Any()},
                     {"task_pairs", pairs}});
              }
              for (const auto &node : nodes)
                if (node->IsControl()) {
                  auto *control = static_cast<ControlNode *>(node.get());
                  collect(control->children, control);
                }
            };
        collect(scheduled.tree, nullptr);
        return result;
      });
}

} // namespace ascend
} // namespace tl
} // namespace tvm
