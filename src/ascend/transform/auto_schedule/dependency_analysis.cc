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

#include <algorithm>
#include <functional>
#include <iterator>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include "support/check.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

// An undefined storage key groups explicit conflicts between distinct storage.
using CoveredDependencyMap =
    std::unordered_map<ffi::Optional<Var>, std::set<DependencyTaskPair>,
                       ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

// A bare buffer hint matches every region using its storage. A region hint
// matches only the same logical buffer and exact footprint.
static bool HintOperandMatches(const Any &hint, const BufferRegion &query) {
  if (auto var = hint.as<Var>())
    return var.value().same_as(query->buffer->data);
  BufferRegion hint_region = Downcast<BufferRegion>(hint);
  return hint_region->buffer.same_as(query->buffer) &&
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
  const auto *is_conflict = hint[3].as<IntImmNode>();
  ICHECK(is_conflict != nullptr && is_conflict->dtype.is_bool())
      << "conflict_hint annotation expects a constant boolean polarity";
  return is_conflict->value;
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

bool RegionsMayConflict(const ConstrSet &a_ctx, const BufferRegion &a_region,
                        const ConstrSet &b_ctx, const BufferRegion &b_region,
                        ControlNode *loop, int offset,
                        size_t num_storage_owners,
                        const ConflictHintList &root_conflicts) {
  const bool cross = offset != 0;
  if (cross && loop == nullptr)
    return false;

  bool declared_no_conflict = false;
  for (const Any &entry : ActiveConflictHints(loop, root_conflicts)) {
    Array<Any> hint = Downcast<Array<Any>>(entry);
    bool is_conflict = IsConflictHint(hint);
    int64_t cross_code = Downcast<IntImm>(hint[2])->value;
    if (cross_code != -1 && (cross_code == 1) != cross)
      continue;
    bool matches = (HintOperandMatches(hint[0], a_region) &&
                    HintOperandMatches(hint[1], b_region)) ||
                   (HintOperandMatches(hint[0], b_region) &&
                    HintOperandMatches(hint[1], a_region));
    if (!matches)
      continue;
    if (is_conflict)
      return true;
    declared_no_conflict = true;
  }
  // A forced conflict wins if contradictory declarations match.
  if (declared_no_conflict)
    return false;

  if (!a_region->buffer->data.same_as(b_region->buffer->data))
    return false;

  // Same storage but different logical buffer (T.view/T.reshape): regions may
  // differ in rank/dtype/strides, so keep the dependency conservatively.
  if (!a_region->buffer.same_as(b_region->buffer))
    return true;

  const Array<Range> &a_rng = a_region->region;
  const Array<Range> &b_rng = b_region->region;
  if (a_rng.size() != b_rng.size())
    return true; // shape mismatch — be safe
  const size_t ndim = a_rng.size();

  arith::Analyzer ana;
  ana.z3_prover.SetRLimit(50000);
  Map<Var, PrimExpr> a_sub, b_sub;
  ConstrSet a_local_ctx = a_ctx;
  ConstrSet b_local_ctx = b_ctx;
  ConstrSet loop_ctx;
  // Scheduled tasks can define condition snapshots outside the constraint
  // visitor's Bind stack. Rename all variables written in the compared
  // iteration scope, including these otherwise-free predicates. Sharing one
  // predicate between producer and consumer would incorrectly make opposite
  // branches mutually exclusive across different iterations.
  auto seed_iteration_vars = [&](const ControlNode *scope) {
    for (const Var &var : scope->GetWriteVars()) {
      if (!a_sub.count(var))
        a_sub.Set(var, var.copy_with_suffix("_p"));
      if (!b_sub.count(var))
        b_sub.Set(var, var.copy_with_suffix("_c"));
    }
  };
  if (cross) {
    if (offset < 0) {
      std::vector<const ControlNode *> iteration_nest =
          num_storage_owners > 1
              ? loop->GetAncestorControls()
              : StorageIterationNest(loop, a_region->buffer->data);
      ICHECK(!iteration_nest.empty());
      seed_iteration_vars(iteration_nest.front());
      Var pivot_lv = iteration_nest.front()->control->loop_var;
      a_local_ctx = a_local_ctx.RenameFrom("_p", a_sub, pivot_lv);
      b_local_ctx = b_local_ctx.RenameFrom("_c", b_sub, pivot_lv);
      if (num_storage_owners <= 1) {
        PrimExpr different_iteration = Bool(false);
        for (const ControlNode *control : iteration_nest) {
          Var iteration = control->control->loop_var;
          PrimExpr p = Substitute(iteration, a_sub);
          PrimExpr c = Substitute(iteration, b_sub);
          different_iteration = different_iteration || (p != c);
        }
        loop_ctx.AddConstr(different_iteration);
      }
    } else {
      seed_iteration_vars(loop);
      Var iteration = loop->control->loop_var;
      a_local_ctx = a_local_ctx.RenameFrom("_p", a_sub, iteration);
      b_local_ctx = b_local_ctx.RenameFrom("_c", b_sub, iteration);
      PrimExpr p = Substitute(iteration, a_sub);
      PrimExpr c = Substitute(iteration, b_sub);
      loop_ctx.AddConstr(c == p + IntImm(iteration.dtype(), offset));
    }
  }
  loop_ctx.Merge(a_local_ctx).Merge(b_local_ctx).Populate(ana);

  // Freshen mutable reads (BufferLoads etc.) inline in the region bounds, per
  // side: a read written verbatim in both bounds is not provably equal across
  // the two access points (a store may sit between them), so it must become an
  // independent unknown on each side.
  FreshenMutableReads freshen_a(FreshenMutableReads::Mode::kSnapshot);
  FreshenMutableReads freshen_b(FreshenMutableReads::Mode::kSnapshot);
  PrimExpr any_dim_disjoint = Bool(false);
  for (size_t i = 0; i < ndim; ++i) {
    PrimExpr a_min = freshen_a(Substitute(a_rng[i]->min, a_sub));
    PrimExpr a_max =
        freshen_a(Substitute(a_rng[i]->min + a_rng[i]->extent - 1, a_sub));
    PrimExpr b_min = freshen_b(Substitute(b_rng[i]->min, b_sub));
    PrimExpr b_max =
        freshen_b(Substitute(b_rng[i]->min + b_rng[i]->extent - 1, b_sub));
    any_dim_disjoint = any_dim_disjoint || (a_max < b_min) || (b_max < a_min);
  }
  any_dim_disjoint = ana.Simplify(any_dim_disjoint);
  // Raw Z3 as a fallback catches the infeasible-context case that Analyzer's
  // constant-fold short-circuit misses.
  return !(ana.CanProve(any_dim_disjoint) ||
           ana.z3_prover.CanProve(any_dim_disjoint));
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
    BufferRegion region;
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
      for (const BufferRegion &region : task->GetReadRegions())
        accesses[i].reads.push_back({task, region});
      for (const BufferRegion &region : task->GetWriteRegions())
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
          auto &covered_pairs = out[cdep.storage];
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

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (loop == nullptr && i >= j)
        continue;
      struct PendingDependency {
        ffi::Optional<Var> storage;
        int distance;
        std::set<DependencyTaskPair> task_pairs;
      };
      std::vector<PendingDependency> pending;

      auto add_pair = [&](const ffi::Optional<Var> &storage, int distance,
                          TaskNode *producer, TaskNode *consumer) {
        auto existing = std::find_if(
            pending.begin(), pending.end(), [&](const PendingDependency &dep) {
              return dep.storage.same_as(storage) && dep.distance == distance;
            });
        if (existing == pending.end()) {
          pending.push_back({storage, distance, {}});
          existing = std::prev(pending.end());
        }
        existing->task_pairs.emplace(producer, consumer);
      };

      auto consider = [&](const std::vector<RegionAccess> &lhs_accesses,
                          const std::vector<RegionAccess> &rhs_accesses) {
        for (const RegionAccess &lhs : lhs_accesses) {
          for (const RegionAccess &rhs : rhs_accesses) {
            const Var &lhs_storage = lhs.region->buffer->data;
            const Var &rhs_storage = rhs.region->buffer->data;
            ffi::Optional<Var> storage;
            if (lhs_storage.same_as(rhs_storage))
              storage = lhs_storage;
            int manual_versions = get_manual_versions(storage);
            if (manual_versions) {
              for (int d = (i < j ? 0 : 1); d <= manual_versions; ++d) {
                if (RegionsMayConflict(lhs.task->outer_ctx, lhs.region,
                                       rhs.task->outer_ctx, rhs.region, loop, d,
                                       /*num_storage_owners=*/0,
                                       root_conflicts)) {
                  add_pair(storage, d, lhs.task, rhs.task);
                  break;
                }
              }
              continue;
            }
            for (int distance : {0, -1}) {
              if (distance == 0 && i >= j)
                continue;
              if (RegionsMayConflict(lhs.task->outer_ctx, lhs.region,
                                     rhs.task->outer_ctx, rhs.region, loop,
                                     distance, get_num_storage_owners(storage),
                                     root_conflicts)) {
                add_pair(storage, distance, lhs.task, rhs.task);
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
          auto covered_it = covered.find(dep.storage);
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
                        dep.distance});
      }
    }
  }
  if (loop)
    cache.emplace(loop, deps);
  return deps;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
