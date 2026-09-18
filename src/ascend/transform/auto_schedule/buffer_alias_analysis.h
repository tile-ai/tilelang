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
 * \file buffer_alias_analysis.h
 * \brief Conditional and periodic buffer-alias analysis for AutoSchedule.
 */

#pragma once

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../buffer_alias.h"
#include "./buffer_lifetime_analysis.h"
#include "./ir_structure.h"
#include "./memory_detector.h"
#include "ascend/op/utils.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {
namespace ascend {

// A candidate address period, which still needs a region-periodicity proof.
struct StorageIterationPeriod {
  int64_t iterations{1};
  bool implicit_slots{false};
};

// Scope-aware view of guaranteed synchronization. The context owns site and
// clock identities and must outlive the BufferAliasAnalyzer that reads it.
class BufferAliasAnalysisContext {
public:
  virtual ~BufferAliasAnalysisContext() = default;

  virtual std::vector<size_t>
  ResolveLifetimeEndpoints(const tirx::Var &storage, TaskNode *task,
                           bool at_beginning, ControlNode *scope) const = 0;
  virtual bool HappensBefore(ControlNode *scope, size_t src, size_t dst,
                             int64_t distance) const = 0;
  virtual std::optional<int64_t> MinimumDistance(ControlNode *scope, size_t src,
                                                 size_t dst) const = 0;
  virtual std::optional<StorageIterationPeriod>
  StoragePeriod(const tirx::Var &storage, ControlNode *scope) const = 0;
  virtual size_t PromoteLifetimeSite(const tirx::Var &storage, size_t site_id,
                                     ControlNode *control) const = 0;
};

namespace buffer_alias_analysis_detail {

using namespace tirx;
using ffi::GetRef;

// Local to buffer-lifetime analysis; never copied into TaskNode or scheduled
// TIR.
struct PhysicalTaskAccesses {
  std::vector<BufferRegion> reads;
  std::vector<BufferRegion> writes;
  std::vector<BufferRegion> must_writes;
  std::vector<Var> opaque_vars;
  std::vector<Var> read_vars;
  std::vector<Var> write_vars;
};

inline PhysicalTaskAccesses AnalyzePhysicalTaskAccesses(const TaskNode *task) {
  PhysicalTaskAccesses result;
  auto append = [](auto &dst, const auto &src) {
    dst.insert(dst.end(), src.begin(), src.end());
  };
  auto analyze = [&](const Stmt &stmt) {
    MemoryAccessDetector detector(MemoryAccessDetector::AccessMode::kPhysical);
    detector.Analyze(stmt);
    append(result.reads, detector.GetReadRegions());
    append(result.writes, detector.GetWriteRegions());
    append(result.must_writes, detector.GetMustWriteRegions());
    append(result.opaque_vars, detector.GetOpaqueAccessVars());
    append(result.read_vars, detector.GetReadVars());
    append(result.write_vars, detector.GetWriteVars());
  };
  const IRStructure *guard_owner = task;
  if (const IRStructure *parent = task->GetParent();
      parent && parent->IsControl() &&
      static_cast<const ControlNode *>(parent)->task.get() == task) {
    guard_owner = parent;
    const auto *loop = task->stmt.as<ForNode>();
    ICHECK(loop);
    analyze(Evaluate(loop->min));
    analyze(Evaluate(loop->extent));
    if (loop->step.defined())
      analyze(Evaluate(loop->step.value()));
  } else {
    analyze(task->stmt);
  }
  for (const auto &guard : guard_owner->GetGuards()) {
    if (guard->IsCondition()) {
      analyze(Evaluate(
          static_cast<const ConditionGuard *>(guard.get())->condition));
    } else {
      const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
      if (auto expr = attribute->node.as<PrimExpr>())
        analyze(Evaluate(expr.value()));
    }
  }
  return result;
}

struct TaskStorageAccess {
  TaskNode *task;
  std::vector<BufferRegion> read_regions;
  std::vector<BufferRegion> write_regions;
  std::vector<BufferRegion> must_write_regions;

  bool Reads() const { return !read_regions.empty(); }
  bool Writes() const { return !write_regions.empty(); }
};

struct StorageAccessInfo {
  Var storage;
  std::vector<TaskStorageAccess> accesses;
  bool opaque{false};
};

using StorageAccessInfoMap =
    std::unordered_map<Var, StorageAccessInfo, ObjectPtrHash, ObjectPtrEqual>;

struct ControlLifetimeComponent {
  ControlNode *control;
  std::vector<GenerationLifetime> generations;
};

struct LifetimeSummary {
  Var storage;
  bool valid{false};
  bool live_in{false};
  bool live_out{false};
  bool live_through{false};
  bool promote_live_through_as_live_in{false};
  std::vector<size_t> live_in_end_sites;
  std::vector<size_t> live_out_begin_sites;
  std::vector<GenerationLifetime> generations;
  // The direct child control collapsed to form this summary. Composite
  // sibling-control summaries instead retain one component per child.
  ControlNode *summary_control{nullptr};
  std::vector<ControlLifetimeComponent> components;
};

struct ConditionSignature {
  std::vector<PrimExpr> conditions;
  bool iteration_invariant{true};
};

using LifetimeSummaryMap =
    std::unordered_map<Var, LifetimeSummary, ObjectPtrHash, ObjectPtrEqual>;

inline std::optional<BufferRegion>
TryMergeDenseRegions(const BufferRegion &lhs, const BufferRegion &rhs) {
  if (!lhs->buffer.same_as(rhs->buffer) ||
      lhs->region.size() != rhs->region.size()) {
    return std::nullopt;
  }
  if (RegionCovers(lhs->region, rhs->region))
    return lhs;
  if (RegionCovers(rhs->region, lhs->region))
    return rhs;

  arith::Analyzer analyzer;
  std::optional<size_t> differing_axis;
  for (size_t i = 0; i < lhs->region.size(); ++i) {
    const Range &lhs_range = lhs->region[i];
    const Range &rhs_range = rhs->region[i];
    if (analyzer.CanProveEqual(lhs_range->min, rhs_range->min) &&
        analyzer.CanProveEqual(lhs_range->extent, rhs_range->extent)) {
      continue;
    }
    if (differing_axis.has_value())
      return std::nullopt;
    differing_axis = i;
  }
  if (!differing_axis.has_value())
    return lhs;

  size_t axis = differing_axis.value();
  const Range &lhs_range = lhs->region[axis];
  const Range &rhs_range = rhs->region[axis];
  PrimExpr lhs_end = lhs_range->min + lhs_range->extent;
  PrimExpr rhs_end = rhs_range->min + rhs_range->extent;
  PrimExpr merged_min;
  if (analyzer.CanProve(lhs_range->min <= rhs_range->min) &&
      analyzer.CanProve(rhs_range->min <= lhs_end)) {
    merged_min = lhs_range->min;
  } else if (analyzer.CanProve(rhs_range->min <= lhs_range->min) &&
             analyzer.CanProve(lhs_range->min <= rhs_end)) {
    merged_min = rhs_range->min;
  } else {
    return std::nullopt;
  }

  PrimExpr merged_end;
  if (analyzer.CanProve(lhs_end >= rhs_end)) {
    merged_end = lhs_end;
  } else if (analyzer.CanProve(rhs_end >= lhs_end)) {
    merged_end = rhs_end;
  } else {
    return std::nullopt;
  }
  Region merged = lhs->region;
  merged.Set(axis, Range::FromMinExtent(merged_min, merged_end - merged_min));
  return BufferRegion(lhs->buffer, std::move(merged));
}

inline bool DenseRegionsCover(const std::vector<BufferRegion> &writes,
                              const BufferRegion &read) {
  // Regions describe different access points. A store can change an index
  // buffer between them, so identical BufferLoads are not an equality proof.
  // Snapshot each write independently before merging: otherwise two changing
  // boundaries could falsely appear adjacent and merge into a constant range.
  auto snapshot = [](const BufferRegion &region) {
    FreshenMutableReads freshen(FreshenMutableReads::Mode::kSnapshot);
    Region bounds;
    bounds.reserve(region->region.size());
    for (const Range &axis : region->region) {
      bounds.push_back(
          Range::FromMinExtent(freshen(axis->min), freshen(axis->extent)));
    }
    return BufferRegion(region->buffer, std::move(bounds));
  };
  BufferRegion read_snapshot = snapshot(read);
  std::vector<BufferRegion> merged;
  for (const BufferRegion &write : writes) {
    if (write->buffer.same_as(read->buffer))
      merged.push_back(snapshot(write));
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < merged.size() && !changed; ++i) {
      for (size_t j = i + 1; j < merged.size(); ++j) {
        std::optional<BufferRegion> combined =
            TryMergeDenseRegions(merged[i], merged[j]);
        if (!combined.has_value())
          continue;
        merged[i] = std::move(combined.value());
        merged.erase(merged.begin() + j);
        changed = true;
        break;
      }
    }
  }
  return std::any_of(merged.begin(), merged.end(), [&](const auto &write) {
    return RegionCovers(write->region, read_snapshot->region);
  });
}

inline bool IsGuaranteedSingleTrip(const ControlNode *control) {
  if (control == nullptr || control->BodyHasLoopBreak())
    return false;

  const ForNode *loop = control->control.get();
  PrimExpr extent = loop->extent;
  PrimExpr step = loop->step.has_value() ? loop->step.value()
                                         : make_const(extent.dtype(), 1);
  if (step.dtype() != extent.dtype())
    step = cast(extent.dtype(), step);

  arith::Analyzer analyzer;
  if (control->task)
    control->task->outer_ctx.Populate(analyzer);
  PrimExpr zero = make_const(extent.dtype(), 0);
  return analyzer.CanProve(extent > zero) && analyzer.CanProve(step > zero) &&
         analyzer.CanProve(extent <= step);
}

} // namespace buffer_alias_analysis_detail

namespace buffer_alias_analysis_detail {

class BufferAliasAnalyzerImpl {
public:
  explicit BufferAliasAnalyzerImpl(const BufferAliasAnalysisContext &context)
      : context_(&context) {}

  void Collect(const std::vector<std::shared_ptr<IRStructure>> &root) {
    storage_access_info_.clear();
    storage_order_.clear();
    lifetime_summaries_.clear();
    buffer_aliases_ = {};

    std::vector<TaskNode *> tasks;
    CollectAllTaskNodes(root, tasks);
    std::unordered_map<const TaskNode *, PhysicalTaskAccesses>
        physical_accesses;
    for (TaskNode *task : tasks) {
      const PhysicalTaskAccesses &physical =
          physical_accesses.emplace(task, AnalyzePhysicalTaskAccesses(task))
              .first->second;
      struct TaskAccessRegions {
        std::vector<BufferRegion> reads;
        std::vector<BufferRegion> writes;
        std::vector<BufferRegion> must_writes;
      };
      std::unordered_map<Var, TaskAccessRegions, ObjectPtrHash, ObjectPtrEqual>
          task_accesses;
      std::vector<Var> task_storage_order;
      auto get_task_access =
          [&](const BufferRegion &region) -> TaskAccessRegions * {
        if (!IsAscendOnChipBuffer(region->buffer))
          return nullptr;
        const Var &storage = region->buffer->data;
        auto [it, inserted] = task_accesses.try_emplace(storage);
        if (inserted)
          task_storage_order.push_back(storage);
        return &it->second;
      };
      for (const BufferRegion &region : physical.reads) {
        if (TaskAccessRegions *access = get_task_access(region))
          access->reads.push_back(region);
      }
      for (const BufferRegion &region : physical.writes) {
        if (TaskAccessRegions *access = get_task_access(region))
          access->writes.push_back(region);
      }
      for (const BufferRegion &region : physical.must_writes) {
        if (TaskAccessRegions *access = get_task_access(region))
          access->must_writes.push_back(region);
      }
      for (const Var &storage : task_storage_order) {
        const TaskAccessRegions &mask = task_accesses.at(storage);
        auto info = storage_access_info_.find(storage);
        if (info == storage_access_info_.end()) {
          StorageAccessInfo storage_info;
          storage_info.storage = storage;
          info = storage_access_info_.emplace(storage, std::move(storage_info))
                     .first;
          storage_order_.push_back(storage);
          AddBufferAliasStorage(&buffer_aliases_, storage);
        }
        info->second.accesses.push_back(
            {task, mask.reads, mask.writes, mask.must_writes});
      }
    }

    for (TaskNode *task : tasks) {
      const PhysicalTaskAccesses &physical = physical_accesses.at(task);
      for (const Var &var : physical.opaque_vars) {
        auto info = storage_access_info_.find(var);
        if (info != storage_access_info_.end())
          info->second.opaque = true;
      }
      auto mark_unrepresented = [&](const Var &var) {
        auto info = storage_access_info_.find(var);
        if (info == storage_access_info_.end())
          return;
        bool represented = false;
        for (const TaskStorageAccess &access : info->second.accesses) {
          if (access.task == task) {
            represented = true;
            break;
          }
        }
        if (!represented)
          info->second.opaque = true;
      };
      for (const Var &var : physical.read_vars)
        mark_unrepresented(var);
      for (const Var &var : physical.write_vars)
        mark_unrepresented(var);
    }
  }

  void Analyze(const std::vector<std::shared_ptr<IRStructure>> &root) {
    lifetime_summaries_.clear();
    AnalyzeScope(root, /*loop=*/nullptr);
  }

  const BufferAliasMap &GetAliases() const { return buffer_aliases_; }
  void Validate() const { ValidateBufferAliasMap(buffer_aliases_); }

private:
  HappensBeforeQuery MakeHappensBeforeQuery(ControlNode *scope) const {
    return [this, scope](size_t src, size_t dst, int64_t distance) {
      return context_->HappensBefore(scope, src, dst, distance);
    };
  }

  MinimumIterationDistance MakeMinimumDistanceQuery(ControlNode *scope) const {
    return [this, scope](size_t src, size_t dst) {
      return context_->MinimumDistance(scope, src, dst);
    };
  }

  void AnalyzeScope(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                    ControlNode *loop) {
    // Cache every child scope before summarizing this scope. A parent can then
    // promote the child's already-built lifetime instead of recursively
    // rediscovering the same nested control chain.
    for (const std::shared_ptr<IRStructure> &node : nodes) {
      if (!node->IsControl())
        continue;
      auto *control = static_cast<ControlNode *>(node.get());
      if (!control->control.defined() || control->children.empty())
        continue;
      AnalyzeScope(control->children, control);
    }

    HappensBeforeQuery current_happens_before = MakeHappensBeforeQuery(loop);

    std::vector<LifetimeSummary> lifetimes;
    for (const Var &storage : storage_order_) {
      auto info = storage_access_info_.find(storage);
      ICHECK(info != storage_access_info_.end());
      std::optional<LifetimeSummary> lifetime =
          BuildStorageLifetime(info->second, loop, current_happens_before);
      if (lifetime.has_value())
        lifetimes.push_back(std::move(lifetime.value()));
    }

    for (size_t i = 0; i < lifetimes.size(); ++i) {
      for (size_t j = i + 1; j < lifetimes.size(); ++j) {
        if (CanAliasStorageLifetimes(lifetimes[i], lifetimes[j], loop,
                                     current_happens_before)) {
          AddBufferAlias(&buffer_aliases_, lifetimes[i].storage,
                         lifetimes[j].storage);
        }
      }
    }

    LifetimeSummaryMap &scope_summaries = lifetime_summaries_[loop];
    for (const LifetimeSummary &lifetime : lifetimes)
      scope_summaries.emplace(lifetime.storage, lifetime);
  }

  std::optional<LifetimeSummary>
  BuildStorageLifetime(const StorageAccessInfo &info, ControlNode *loop,
                       const HappensBeforeQuery &happens_before) const {
    LifetimeSummary direct = BuildGenerations(info, loop, loop, happens_before);
    if (direct.valid)
      return direct;

    ControlNode *child = FindSummaryControl(info, loop);
    if (child != nullptr) {
      auto child_scope = lifetime_summaries_.find(child);
      if (child_scope != lifetime_summaries_.end()) {
        auto child_lifetime = child_scope->second.find(info.storage);
        if (child_lifetime != child_scope->second.end() &&
            child_lifetime->second.components.empty()) {
          return PromoteControlLifetimeSummary(child_lifetime->second, child,
                                               loop);
        }
      }
    }
    return BuildCompositeStorageLifetime(info, loop);
  }

  bool CanAliasStorageLifetimes(
      const LifetimeSummary &lhs, const LifetimeSummary &rhs, ControlNode *loop,
      const HappensBeforeQuery &current_happens_before) const {
    if (lhs.summary_control != nullptr &&
        lhs.summary_control == rhs.summary_control) {
      return false;
    }
    if (!lhs.components.empty() || !rhs.components.empty()) {
      if (lhs.components.size() != rhs.components.size() ||
          lhs.components.empty()) {
        return false;
      }
      for (size_t i = 0; i < lhs.components.size(); ++i) {
        const ControlLifetimeComponent &lhs_component = lhs.components[i];
        const ControlLifetimeComponent &rhs_component = rhs.components[i];
        if (lhs_component.control != rhs_component.control ||
            lhs_component.generations.empty() ||
            rhs_component.generations.empty()) {
          return false;
        }
        HappensBeforeQuery component_happens_before =
            MakeHappensBeforeQuery(lhs_component.control);
        MutuallyExclusiveQuery component_mutually_exclusive =
            [&](const PrimExpr &a, const PrimExpr &b) {
              return ProveMutuallyExclusive(a, b, lhs_component.control);
            };
        if (!CanAliasGenerations(
                lhs_component.generations, rhs_component.generations,
                /*periodic=*/true, component_happens_before,
                MakeMinimumDistanceQuery(lhs_component.control),
                component_mutually_exclusive)) {
          return false;
        }
      }
    }
    auto boundary_compatible = [&](const LifetimeSummary &boundary,
                                   const LifetimeSummary &other) {
      if (!boundary.live_in_end_sites.empty()) {
        if (other.live_in)
          return false;
        if (!other.live_out_begin_sites.empty() &&
            !AllHappenBefore(boundary.live_in_end_sites,
                             other.live_out_begin_sites, 0,
                             current_happens_before)) {
          return false;
        }
        for (const GenerationLifetime &generation : other.generations) {
          if (!IsLocalGeneration(generation) ||
              !AllEventsBefore(MakeIterationEvents(boundary.live_in_end_sites),
                               generation.begin_sites, 0,
                               current_happens_before)) {
            return false;
          }
        }
      }
      if (!boundary.live_out_begin_sites.empty()) {
        if (other.live_out)
          return false;
        if (!other.live_in_end_sites.empty() &&
            !AllHappenBefore(other.live_in_end_sites,
                             boundary.live_out_begin_sites, 0,
                             current_happens_before)) {
          return false;
        }
        for (const GenerationLifetime &generation : other.generations) {
          if (!IsLocalGeneration(generation) ||
              !AllEventsBefore(
                  generation.end_sites,
                  MakeIterationEvents(boundary.live_out_begin_sites), 0,
                  current_happens_before)) {
            return false;
          }
        }
      }
      return true;
    };

    bool lhs_represented = !lhs.generations.empty() ||
                           !lhs.live_in_end_sites.empty() ||
                           !lhs.live_out_begin_sites.empty();
    bool rhs_represented = !rhs.generations.empty() ||
                           !rhs.live_in_end_sites.empty() ||
                           !rhs.live_out_begin_sites.empty();
    bool overlap_at_iteration_boundary =
        loop != nullptr &&
        ((lhs.live_in && rhs.live_out) || (rhs.live_in && lhs.live_out));
    if (lhs.live_through || rhs.live_through || overlap_at_iteration_boundary ||
        !lhs_represented || !rhs_represented ||
        !boundary_compatible(lhs, rhs) || !boundary_compatible(rhs, lhs)) {
      return false;
    }
    if (lhs.generations.empty() || rhs.generations.empty())
      return true;

    MutuallyExclusiveQuery mutually_exclusive = [&](const PrimExpr &a,
                                                    const PrimExpr &b) {
      return ProveMutuallyExclusive(a, b, loop);
    };
    return CanAliasGenerations(lhs.generations, rhs.generations,
                               loop != nullptr, current_happens_before,
                               MakeMinimumDistanceQuery(loop),
                               mutually_exclusive);
  }

  using VarExprMap =
      std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
  using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

  VarSet CollectIterationVariantVars(ControlNode *scope) const {
    VarSet result;
    if (scope == nullptr)
      return result;

    ControlNode *root = scope;
    while (root->GetParentControl() != nullptr)
      root = root->GetParentControl();
    std::function<void(IRStructure *)> collect = [&](IRStructure *node) {
      if (node == nullptr)
        return;
      if (node->IsTask()) {
        for (const Var &var : node->GetWriteVars())
          result.insert(var);
        return;
      }
      auto *control = static_cast<ControlNode *>(node);
      result.insert(control->control->loop_var);
      if (control->task) {
        for (const Var &var : control->task->GetWriteVars())
          result.insert(var);
      }
      for (const std::shared_ptr<IRStructure> &child : control->children)
        collect(child.get());
    };
    collect(root);
    return result;
  }

  bool ExpressionDependsOnIteration(const PrimExpr &expr,
                                    const VarExprMap &bindings,
                                    const VarSet &variant_vars,
                                    VarSet *visiting) const {
    bool varies = false;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (varies)
        return;
      if (node.as<BufferLoadNode>() || node.as<ProducerLoadNode>() ||
          node.as<ReduceNode>()) {
        varies = true;
        return;
      }
      if (const auto *call = node.as<CallNode>()) {
        if (SideEffect(GetRef<PrimExpr>(call)) > CallEffectKind::kPure)
          varies = true;
        return;
      }
      const auto *var_node = node.as<VarNode>();
      if (var_node == nullptr)
        return;
      Var var = GetRef<Var>(var_node);
      auto binding = bindings.find(var);
      if (binding != bindings.end()) {
        if (!visiting->insert(var).second) {
          varies = true;
          return;
        }
        varies = ExpressionDependsOnIteration(binding->second, bindings,
                                              variant_vars, visiting);
        visiting->erase(var);
        return;
      }
      if (variant_vars.count(var))
        varies = true;
    });
    return varies;
  }

  bool AreConditionsIterationInvariant(TaskNode *task,
                                       const std::vector<PrimExpr> &conditions,
                                       ControlNode *scope) const {
    if (scope == nullptr)
      return true;

    VarExprMap bindings;
    if (task != nullptr) {
      for (const Constr &constraint : task->outer_ctx.constrs_) {
        if (constraint.kind == Constr::kBindValue)
          bindings[constraint.var] = constraint.value;
      }
    }
    VarSet variant_vars = CollectIterationVariantVars(scope);
    for (const PrimExpr &condition : conditions) {
      VarSet visiting;
      if (ExpressionDependsOnIteration(condition, bindings, variant_vars,
                                       &visiting)) {
        return false;
      }
    }
    return true;
  }

  std::optional<ConditionSignature>
  CollectConditionSignature(TaskNode *task, ControlNode *scope) const {
    std::vector<IRStructure *> path;
    for (IRStructure *node = task; node != scope; node = node->GetParent()) {
      if (node == nullptr)
        return std::nullopt;
      path.push_back(node);
      if (node->GetParent() == nullptr && scope == nullptr)
        break;
    }
    std::reverse(path.begin(), path.end());

    arith::Analyzer analyzer;
    for (const Constr &constraint : task->outer_ctx.constrs_) {
      if (constraint.kind != Constr::kConstr)
        constraint.Populate(analyzer);
    }
    ConditionSignature signature;
    std::vector<PrimExpr> raw_conditions;
    for (IRStructure *node : path) {
      for (const std::unique_ptr<Guard> &guard : node->GetGuards()) {
        if (!guard->IsCondition())
          continue;
        const auto *condition =
            static_cast<const ConditionGuard *>(guard.get());
        raw_conditions.push_back(condition->condition);
        signature.conditions.push_back(analyzer.Simplify(condition->condition));
      }
    }
    signature.iteration_invariant =
        AreConditionsIterationInvariant(task, raw_conditions, scope);
    return signature;
  }

  PrimExpr ConditionPredicate(const std::vector<PrimExpr> &signature) const {
    PrimExpr predicate = Bool(true);
    for (const PrimExpr &condition : signature)
      predicate = predicate && condition;
    return predicate;
  }

  bool ProveGuardImplication(const std::vector<PrimExpr> &premise,
                             const std::vector<PrimExpr> &conclusion,
                             ControlNode *scope) const {
    if (conclusion.empty())
      return true;
    arith::Analyzer analyzer;
    analyzer.z3_prover.SetRLimit(50000);
    if (scope != nullptr && scope->task)
      scope->task->outer_ctx.Populate(analyzer);
    PrimExpr implication =
        !ConditionPredicate(premise) || ConditionPredicate(conclusion);
    implication = analyzer.Simplify(implication);
    return analyzer.CanProve(implication) ||
           analyzer.z3_prover.CanProve(implication);
  }

  bool ProveMutuallyExclusive(const PrimExpr &lhs, const PrimExpr &rhs,
                              ControlNode *scope) const {
    if (!lhs.defined() || !rhs.defined())
      return false;
    arith::Analyzer analyzer;
    analyzer.z3_prover.SetRLimit(50000);
    if (scope != nullptr && scope->task)
      scope->task->outer_ctx.Populate(analyzer);
    PrimExpr exclusive = analyzer.Simplify(!(lhs && rhs));
    return analyzer.CanProve(exclusive) ||
           analyzer.z3_prover.CanProve(exclusive);
  }

  LifetimeSummary
  BuildGenerations(const StorageAccessInfo &info, ControlNode *analysis_scope,
                   ControlNode *endpoint_scope,
                   const HappensBeforeQuery &happens_before) const {
    ICHECK(analysis_scope == endpoint_scope)
        << "Lifetime endpoints and ordering queries must use the same scope";
    LifetimeSummary result;
    result.storage = info.storage;
    if (info.opaque)
      return result;

    auto can_promote_to_scope = [&](TaskNode *task) {
      IRStructure *node = task;
      while (node->GetParent() != analysis_scope) {
        IRStructure *parent = node->GetParent();
        if (parent == nullptr || !parent->IsControl())
          return false;
        auto *control = static_cast<ControlNode *>(parent);
        if (!IsGuaranteedSingleTrip(control))
          return false;
        node = parent;
      }
      return true;
    };
    auto is_nested_in_scope = [&](TaskNode *task) {
      if (analysis_scope == nullptr)
        return true;
      for (IRStructure *node = task; node != nullptr;
           node = node->GetParent()) {
        if (node == analysis_scope)
          return true;
      }
      return false;
    };

    std::vector<TaskStorageAccess> accesses;
    std::vector<TaskStorageAccess> reads_after_scope;
    bool has_access_before_scope = false;
    bool has_write_after_scope = false;
    for (const TaskStorageAccess &access : info.accesses) {
      if (can_promote_to_scope(access.task)) {
        accesses.push_back(access);
        continue;
      }
      if (is_nested_in_scope(access.task))
        return result;
      if (analysis_scope != nullptr) {
        int order = CompareIRStructure(analysis_scope, access.task);
        if (order > 0) {
          has_access_before_scope = true;
        } else if (order < 0) {
          if (access.Reads())
            reads_after_scope.push_back(access);
          has_write_after_scope |= access.Writes();
        }
      }
    }
    if (accesses.empty())
      return result;

    std::vector<ConditionSignature> conditions;
    conditions.reserve(accesses.size());
    for (const TaskStorageAccess &access : accesses) {
      std::optional<ConditionSignature> signature =
          CollectConditionSignature(access.task, analysis_scope);
      if (!signature.has_value())
        return result;
      conditions.push_back(std::move(signature.value()));
    }
    auto common_control_scope = [](TaskNode *lhs, TaskNode *rhs) {
      std::vector<IRStructure *> lhs_path = lhs->PathFrom();
      std::vector<IRStructure *> rhs_path = rhs->PathFrom();
      IRStructure *common = nullptr;
      size_t count = std::min(lhs_path.size(), rhs_path.size());
      for (size_t index = 0; index < count; ++index) {
        if (lhs_path[index] != rhs_path[index])
          break;
        common = lhs_path[index];
      }
      if (common == nullptr || !common->IsControl())
        return static_cast<ControlNode *>(nullptr);
      return static_cast<ControlNode *>(common);
    };
    auto writer_guard_dominates = [&](TaskNode *writer, TaskNode *reader) {
      ControlNode *common_scope = common_control_scope(writer, reader);
      std::optional<ConditionSignature> writer_signature =
          CollectConditionSignature(writer, common_scope);
      std::optional<ConditionSignature> reader_signature =
          CollectConditionSignature(reader, common_scope);
      if (!writer_signature.has_value() || !reader_signature.has_value())
        return false;
      return ProveGuardImplication(reader_signature->conditions,
                                   writer_signature->conditions, common_scope);
    };
    auto covers = [](const TaskStorageAccess &writer,
                     const TaskStorageAccess &reader) {
      for (const BufferRegion &read : reader.read_regions) {
        if (!DenseRegionsCover(writer.must_write_regions, read))
          return false;
      }
      return true;
    };

    std::vector<size_t> writer_access_indices;
    std::vector<GenerationLifetime> generations;
    std::vector<std::vector<size_t>> writer_begins;
    std::vector<std::vector<size_t>> writer_ends;
    for (size_t access_index = 0; access_index < accesses.size();
         ++access_index) {
      const TaskStorageAccess &access = accesses[access_index];
      if (!access.Writes())
        continue;
      writer_access_indices.push_back(access_index);
      GenerationLifetime generation;
      generation.begin_sites =
          MakeIterationEvents(context_->ResolveLifetimeEndpoints(
              info.storage, access.task, true, endpoint_scope));
      generation.end_sites =
          MakeIterationEvents(context_->ResolveLifetimeEndpoints(
              info.storage, access.task, false, endpoint_scope));
      generation.guard =
          ConditionPredicate(conditions[access_index].conditions);
      generation.guard_iteration_invariant =
          conditions[access_index].iteration_invariant;
      writer_begins.push_back(context_->ResolveLifetimeEndpoints(
          info.storage, access.task, true, analysis_scope));
      writer_ends.push_back(context_->ResolveLifetimeEndpoints(
          info.storage, access.task, false, analysis_scope));
      generations.push_back(std::move(generation));
    }

    bool has_reads = false;
    for (const TaskStorageAccess &access : accesses)
      has_reads |= access.Reads();
    if (writer_access_indices.empty()) {
      if (!has_reads)
        return result;
      result.valid = true;
      result.live_in = true;
      if (has_access_before_scope && reads_after_scope.empty() &&
          !has_write_after_scope) {
        result.live_out = true;
        result.live_through = true;
        result.promote_live_through_as_live_in = true;
        for (const TaskStorageAccess &access : accesses) {
          if (!access.Reads())
            continue;
          std::vector<size_t> read_ends = context_->ResolveLifetimeEndpoints(
              info.storage, access.task, false, endpoint_scope);
          result.live_in_end_sites.insert(result.live_in_end_sites.end(),
                                          read_ends.begin(), read_ends.end());
        }
        SortUnique(&result.live_in_end_sites);
      } else {
        result.live_out = true;
        result.live_through = true;
      }
      return result;
    }

    // A physical/access period is not a distance horizon. After proving that
    // the regions repeat, one nearest ordered writer per residue represents
    // every older writer in that residue, by same-pipe event recurrence.
    auto period = context_->StoragePeriod(info.storage, analysis_scope);
    auto shift_regions = [&](const std::vector<BufferRegion> &regions,
                             int64_t distance) {
      if (analysis_scope == nullptr || distance == 0)
        return regions;
      const ForNode *loop = analysis_scope->control.get();
      PrimExpr step =
          loop->step.value_or(make_const(loop->loop_var.dtype(), 1));
      ffi::Map<Var, PrimExpr> substitution{
          {loop->loop_var,
           loop->loop_var +
               make_const(loop->loop_var.dtype(), distance) * step}};
      std::vector<BufferRegion> shifted;
      for (const BufferRegion &region : regions)
        shifted.emplace_back(region->buffer,
                             Substitute(region->region, substitution));
      return shifted;
    };
    int64_t max_offset = 0;
    if (analysis_scope != nullptr) {
      DataType dtype = analysis_scope->control->loop_var.dtype();
      if (dtype.is_int())
        max_offset = dtype.bits() >= 64
                         ? std::numeric_limits<int64_t>::max()
                         : (int64_t{1} << (dtype.bits() - 1)) - 1;
    }
    bool regions_periodic = analysis_scope != nullptr && period &&
                            period->iterations > 0 &&
                            period->iterations <= max_offset;
    if (regions_periodic) {
      arith::Analyzer analyzer;
      // Do not bind the loop variable to its finite range: periodicity must
      // hold for shifted generations as well as the current iteration.
      VarSet variant_vars = CollectIterationVariantVars(analysis_scope);
      const Var &loop_var = analysis_scope->control->loop_var;
      Var explicit_ordinal("generation_ordinal", loop_var.dtype());
      for (const TaskStorageAccess &access : accesses) {
        VarExprMap bindings;
        for (const Constr &constraint : access.task->outer_ctx.constrs_) {
          if (constraint.kind == Constr::kBindValue)
            bindings[constraint.var] = constraint.value;
        }
        auto has_stable_parameters = [&](const PrimExpr &expr) {
          // Only the explicit coordinate is shifted below. An unresolved let
          // alias of the loop variable must not make a changing slot appear
          // invariant merely because substitution did not reach its binding.
          PrimExpr explicit_coordinate =
              Substitute(expr, {{loop_var, explicit_ordinal}});
          VarSet visiting;
          return !ExpressionDependsOnIteration(explicit_coordinate, bindings,
                                               variant_vars, &visiting);
        };
        for (const auto *regions :
             {&access.read_regions, &access.must_write_regions}) {
          auto shifted = shift_regions(*regions, period->iterations);
          for (size_t i = 0; i < regions->size(); ++i) {
            for (size_t axis = 0; axis < (*regions)[i]->region.size(); ++axis) {
              const Range &original = (*regions)[i]->region[axis];
              const Range &next = shifted[i]->region[axis];
              regions_periodic &= has_stable_parameters(original->min) &&
                                  has_stable_parameters(original->extent);
              regions_periodic &=
                  analyzer.CanProveEqual(original->min, next->min) &&
                  analyzer.CanProveEqual(original->extent, next->extent);
            }
          }
        }
      }
    }

    struct WriterInstance {
      size_t index;
      int64_t distance;
    };
    bool has_carried_generation = false;
    std::vector<size_t> unresolved_read_indices;
    std::vector<size_t> unresolved_read_ends;
    for (size_t reader_index = 0; reader_index < accesses.size();
         ++reader_index) {
      const TaskStorageAccess &reader = accesses[reader_index];
      if (!reader.Reads())
        continue;
      std::vector<size_t> read_begins = context_->ResolveLifetimeEndpoints(
          info.storage, reader.task, true, analysis_scope);
      std::vector<size_t> read_ends = context_->ResolveLifetimeEndpoints(
          info.storage, reader.task, false, endpoint_scope);
      std::vector<WriterInstance> candidates;
      for (size_t writer_index = 0; writer_index < writer_access_indices.size();
           ++writer_index) {
        size_t access_index = writer_access_indices[writer_index];
        const TaskStorageAccess &writer = accesses[access_index];
        if (!ProveGuardImplication(conditions[reader_index].conditions,
                                   conditions[access_index].conditions,
                                   analysis_scope))
          continue;
        if (writer.task != reader.task && covers(writer, reader) &&
            AllHappenBefore(writer_ends[writer_index], read_begins, 0,
                            happens_before)) {
          candidates.push_back({writer_index, 0});
          continue;
        }
        if (!regions_periodic ||
            !conditions[access_index].iteration_invariant ||
            !conditions[reader_index].iteration_invariant)
          continue;
        auto minimum = SeparateIterationEvents(
            MakeIterationEvents(writer_ends[writer_index]),
            MakeIterationEvents(read_begins),
            MakeMinimumDistanceQuery(analysis_scope));
        if (!minimum)
          continue;
        int64_t lower = std::max<int64_t>(*minimum, 1);
        int64_t versions = period->iterations;
        for (int64_t residue = 0; residue < versions; ++residue) {
          if (period->implicit_slots && residue != 0)
            continue;
          int64_t adjustment = residue - lower % versions;
          if (adjustment < 0)
            adjustment += versions;
          auto distance = AddIterationOffsets(lower, adjustment);
          if (!distance || *distance > max_offset)
            continue;
          auto writes = shift_regions(writer.must_write_regions, -*distance);
          bool covers_read = std::all_of(
              reader.read_regions.begin(), reader.read_regions.end(),
              [&](const BufferRegion &read) {
                return DenseRegionsCover(writes, read);
              });
          if (covers_read)
            candidates.push_back({writer_index, *distance});
        }
      }
      std::optional<WriterInstance> latest;
      for (const WriterInstance &candidate : candidates) {
        bool after_all = true;
        for (const WriterInstance &other : candidates) {
          if (candidate.index == other.index &&
              candidate.distance == other.distance)
            continue;
          if (!AllHappenBefore(
                  writer_ends[other.index], writer_begins[candidate.index],
                  other.distance - candidate.distance, happens_before)) {
            after_all = false;
            break;
          }
        }
        if (after_all) {
          if (latest) {
            result.valid = result.live_in = result.live_out =
                result.live_through = true;
            return result;
          }
          latest = candidate;
        }
      }
      if (!candidates.empty() && !latest) {
        result.valid = result.live_in = result.live_out = result.live_through =
            true;
        return result;
      }
      if (latest) {
        auto &ends = generations[latest->index].end_sites;
        for (size_t site : read_ends)
          ends.push_back({site, latest->distance});
        if (latest->distance > 0) {
          has_carried_generation = true;
          result.live_in = result.live_out = true;
        }
        continue;
      }
      result.live_in = true;
      unresolved_read_indices.push_back(reader_index);
      unresolved_read_ends.insert(unresolved_read_ends.end(), read_ends.begin(),
                                  read_ends.end());
    }

    if (!unresolved_read_indices.empty()) {
      // A later writer may feed an unresolved reader in the next iteration.
      // Killing the incoming value locally does not end that carried lifetime,
      // even when the regions repeat or the later writer is unconditional.
      // Only a root-scope incoming value can be reduced to a live-in prefix.
      if (has_carried_generation || analysis_scope != nullptr) {
        result.live_out = true;
        result.live_through = true;
      } else {
        bool all_incoming_values_killed = true;
        for (size_t reader_index : unresolved_read_indices) {
          const TaskStorageAccess &reader = accesses[reader_index];
          std::vector<size_t> read_ends = context_->ResolveLifetimeEndpoints(
              info.storage, reader.task, false, analysis_scope);
          bool killed = false;
          for (size_t writer_index = 0;
               writer_index < writer_access_indices.size(); ++writer_index) {
            size_t writer_access_index = writer_access_indices[writer_index];
            const TaskStorageAccess &writer = accesses[writer_access_index];
            bool guard_is_stable =
                conditions[writer_access_index].conditions.empty() ||
                (conditions[writer_access_index].iteration_invariant &&
                 conditions[reader_index].iteration_invariant);
            if (!ProveGuardImplication(
                    conditions[reader_index].conditions,
                    conditions[writer_access_index].conditions,
                    analysis_scope) ||
                !guard_is_stable || !covers(writer, reader)) {
              continue;
            }
            if (AllHappenBefore(read_ends, writer_begins[writer_index], 0,
                                happens_before)) {
              killed = true;
              break;
            }
          }
          if (!killed) {
            all_incoming_values_killed = false;
            break;
          }
        }
        if (all_incoming_values_killed) {
          result.live_in_end_sites = std::move(unresolved_read_ends);
        } else {
          result.live_out = true;
          result.live_through = true;
        }
      }
    }

    if (!reads_after_scope.empty()) {
      result.live_out = true;
      // An early exit can skip a writer later in the loop body even when the
      // loop itself is unconditional and provably non-empty.
      bool scope_must_execute = !analysis_scope->HasConditions() &&
                                !analysis_scope->BodyHasLoopBreak();
      if (scope_must_execute) {
        const ForNode *loop = analysis_scope->control.get();
        PrimExpr step = loop->step.has_value()
                            ? loop->step.value()
                            : make_const(loop->extent.dtype(), 1);
        arith::Analyzer analyzer;
        if (analysis_scope->task)
          analysis_scope->task->outer_ctx.Populate(analyzer);
        scope_must_execute =
            analyzer.CanProve(loop->extent > 0) && analyzer.CanProve(step > 0);
      }
      bool outside_reads_have_must_def = scope_must_execute;
      for (const TaskStorageAccess &reader : reads_after_scope) {
        bool defined = false;
        for (size_t writer_index = 0;
             writer_index < writer_access_indices.size(); ++writer_index) {
          size_t writer_access_index = writer_access_indices[writer_index];
          const TaskStorageAccess &writer = accesses[writer_access_index];
          if (covers(writer, reader) &&
              writer_guard_dominates(writer.task, reader.task)) {
            defined = true;
            break;
          }
        }
        if (!defined) {
          outside_reads_have_must_def = false;
          break;
        }
      }
      if (has_carried_generation || !outside_reads_have_must_def) {
        result.live_in = true;
        result.live_through = true;
      } else {
        for (const GenerationLifetime &generation : generations) {
          for (const IterationEvent &event : generation.begin_sites) {
            ICHECK_EQ(event.offset, 0);
            result.live_out_begin_sites.push_back(event.site);
          }
        }
      }
    }
    if (has_access_before_scope) {
      result.live_in = true;
      if (result.live_in_end_sites.empty() || has_carried_generation)
        result.live_through = true;
    }
    if (has_write_after_scope) {
      result.live_out = true;
      result.live_through = true;
    }

    for (GenerationLifetime &generation : generations)
      SortUnique(&generation.end_sites);
    SortUnique(&result.live_in_end_sites);
    SortUnique(&result.live_out_begin_sites);
    result.valid = true;
    result.generations = std::move(generations);
    return result;
  }

  ControlNode *FindSummaryControl(const StorageAccessInfo &info,
                                  ControlNode *parent_scope) const {
    ControlNode *control = nullptr;
    for (const TaskStorageAccess &access : info.accesses) {
      IRStructure *child = access.task;
      while (child != nullptr && child->GetParent() != parent_scope)
        child = child->GetParent();
      if (child == nullptr || !child->IsControl())
        continue;
      auto *candidate = static_cast<ControlNode *>(child);
      if (IsGuaranteedSingleTrip(candidate))
        continue;
      if (control == nullptr) {
        control = candidate;
      } else if (control != candidate) {
        return nullptr;
      }
    }
    if (control == nullptr || control->BodyHasLoopBreak())
      return nullptr;
    return control;
  }

  std::optional<LifetimeSummary>
  PromoteControlLifetimeSummary(LifetimeSummary child_result,
                                ControlNode *control,
                                ControlNode *parent_scope) const {
    ICHECK(child_result.valid);
    ICHECK(child_result.components.empty());

    For loop = control->control;
    PrimExpr extent = loop->extent;
    PrimExpr step = loop->step.has_value() ? loop->step.value()
                                           : make_const(extent.dtype(), 1);
    if (step.dtype() != extent.dtype())
      step = cast(extent.dtype(), step);
    arith::Analyzer analyzer;
    if (control->task)
      control->task->outer_ctx.Populate(analyzer);
    if (!analyzer.CanProve(step > 0) || !analyzer.CanProve(extent > 0))
      return std::nullopt;
    // An inner-loop offset is not an outer-loop offset. Collapsing a local
    // generation to the child's entry/exit encloses every child iteration;
    // carried generations require a clock mapping we do not currently prove.
    for (const GenerationLifetime &generation : child_result.generations) {
      if (!IsLocalGeneration(generation))
        return std::nullopt;
    }

    PromoteSites(child_result.storage, &child_result.live_in_end_sites,
                 control);
    PromoteSites(child_result.storage, &child_result.live_out_begin_sites,
                 control);
    for (GenerationLifetime &generation : child_result.generations) {
      PromoteSites(child_result.storage, &generation.begin_sites, control);
      PromoteSites(child_result.storage, &generation.end_sites, control);
    }

    bool parent_live_through = child_result.live_through;
    if (child_result.promote_live_through_as_live_in) {
      // The storage spans the whole child control, so it cannot alias inside
      // that child. At the parent level the same lifetime is a finite prefix
      // ending at the child's last read and may alias a later sibling
      // generation.
      parent_live_through = false;
      child_result.live_out = false;
    }
    std::vector<GenerationLifetime> generations;
    if (!parent_live_through) {
      arith::Analyzer guard_analyzer;
      if (control->task) {
        for (const Constr &constraint : control->task->outer_ctx.constrs_) {
          if (constraint.kind != Constr::kConstr)
            constraint.Populate(guard_analyzer);
        }
      }
      PrimExpr control_guard = Bool(true);
      std::vector<PrimExpr> raw_control_conditions;
      for (const std::unique_ptr<Guard> &guard : control->GetGuards()) {
        if (!guard->IsCondition())
          continue;
        const auto *condition =
            static_cast<const ConditionGuard *>(guard.get());
        raw_control_conditions.push_back(condition->condition);
        control_guard =
            control_guard && guard_analyzer.Simplify(condition->condition);
      }
      bool control_guard_iteration_invariant = AreConditionsIterationInvariant(
          control->task.get(), raw_control_conditions, parent_scope);

      ffi::StructuralEqual structural_equal;
      std::vector<GenerationLifetime> local_summaries;
      for (const GenerationLifetime &generation : child_result.generations) {
        PrimExpr generation_guard =
            generation.guard.defined() ? generation.guard : Bool(true);
        PrimExpr promoted_guard =
            guard_analyzer.Simplify(control_guard && generation_guard);
        auto existing = std::find_if(
            local_summaries.begin(), local_summaries.end(),
            [&](const GenerationLifetime &summary) {
              return structural_equal(summary.guard, promoted_guard);
            });
        if (existing == local_summaries.end()) {
          GenerationLifetime summary;
          summary.guard = promoted_guard;
          summary.guard_iteration_invariant =
              generation.guard_iteration_invariant &&
              control_guard_iteration_invariant;
          local_summaries.push_back(std::move(summary));
          existing = std::prev(local_summaries.end());
        } else {
          existing->guard_iteration_invariant =
              existing->guard_iteration_invariant &&
              generation.guard_iteration_invariant &&
              control_guard_iteration_invariant;
        }
        existing->begin_sites.insert(existing->begin_sites.end(),
                                     generation.begin_sites.begin(),
                                     generation.begin_sites.end());
        existing->end_sites.insert(existing->end_sites.end(),
                                   generation.end_sites.begin(),
                                   generation.end_sites.end());
      }
      for (GenerationLifetime &summary : local_summaries) {
        SortUnique(&summary.begin_sites);
        SortUnique(&summary.end_sites);
        generations.push_back(std::move(summary));
      }
    }
    child_result.live_through = parent_live_through;
    child_result.promote_live_through_as_live_in = false;
    child_result.generations = std::move(generations);
    child_result.summary_control = control;
    return child_result;
  }

  std::optional<LifetimeSummary>
  BuildControlLifetimeSummaryUncached(const StorageAccessInfo &info,
                                      ControlNode *parent_scope) const {
    ControlNode *control = FindSummaryControl(info, parent_scope);
    if (control == nullptr)
      return std::nullopt;

    HappensBeforeQuery child_happens_before = MakeHappensBeforeQuery(control);
    LifetimeSummary child_result =
        BuildGenerations(info, control, control, child_happens_before);
    if (!child_result.valid) {
      std::optional<LifetimeSummary> nested =
          BuildControlLifetimeSummaryUncached(info, control);
      if (!nested.has_value() || !nested->components.empty())
        return std::nullopt;
      child_result = std::move(nested.value());
    }
    return PromoteControlLifetimeSummary(std::move(child_result), control,
                                         parent_scope);
  }

  std::optional<LifetimeSummary>
  BuildCompositeStorageLifetime(const StorageAccessInfo &info,
                                ControlNode *parent_scope) const {
    if (info.opaque)
      return std::nullopt;

    std::vector<ControlNode *> controls;
    std::unordered_map<ControlNode *, StorageAccessInfo> local_infos;
    for (const TaskStorageAccess &access : info.accesses) {
      IRStructure *child = access.task;
      while (child != nullptr && child->GetParent() != parent_scope)
        child = child->GetParent();
      if (child == nullptr || !child->IsControl())
        return std::nullopt;
      auto *control = static_cast<ControlNode *>(child);
      if (IsGuaranteedSingleTrip(control))
        return std::nullopt;
      auto [it, inserted] = local_infos.try_emplace(control);
      if (inserted) {
        it->second.storage = info.storage;
        it->second.opaque = false;
        controls.push_back(control);
      }
      it->second.accesses.push_back(access);
    }
    if (controls.size() < 2)
      return std::nullopt;
    std::stable_sort(controls.begin(), controls.end(),
                     [](ControlNode *lhs, ControlNode *rhs) {
                       return CompareIRStructure(lhs, rhs) < 0;
                     });

    LifetimeSummary lifetime;
    lifetime.storage = info.storage;
    lifetime.valid = true;
    for (ControlNode *control : controls) {
      HappensBeforeQuery local_happens_before = MakeHappensBeforeQuery(control);
      LifetimeSummary local = BuildGenerations(local_infos.at(control), control,
                                               control, local_happens_before);
      if (!local.valid || local.live_in || local.live_out ||
          local.live_through || local.generations.empty()) {
        return std::nullopt;
      }
      std::optional<LifetimeSummary> summary =
          BuildControlLifetimeSummaryUncached(local_infos.at(control),
                                              parent_scope);
      if (!summary.has_value() || summary->summary_control != control ||
          summary->live_in || summary->live_out || summary->live_through ||
          summary->generations.empty()) {
        return std::nullopt;
      }
      ControlLifetimeComponent component;
      component.control = control;
      component.generations = std::move(local.generations);
      lifetime.generations.insert(lifetime.generations.end(),
                                  summary->generations.begin(),
                                  summary->generations.end());
      lifetime.components.push_back(std::move(component));
    }
    return lifetime;
  }

  template <typename Event> static void SortUnique(std::vector<Event> *sites) {
    std::sort(sites->begin(), sites->end());
    sites->erase(std::unique(sites->begin(), sites->end()), sites->end());
  }

  void PromoteSites(const Var &storage, std::vector<size_t> *sites,
                    ControlNode *control) const {
    for (size_t &site : *sites)
      site = context_->PromoteLifetimeSite(storage, site, control);
    SortUnique(sites);
  }

  void PromoteSites(const Var &storage, std::vector<IterationEvent> *events,
                    ControlNode *control) const {
    for (IterationEvent &event : *events) {
      ICHECK_EQ(event.offset, 0);
      event.site = context_->PromoteLifetimeSite(storage, event.site, control);
    }
    SortUnique(events);
  }

  const BufferAliasAnalysisContext *context_;
  StorageAccessInfoMap storage_access_info_;
  std::vector<Var> storage_order_;
  std::map<ControlNode *, LifetimeSummaryMap> lifetime_summaries_;
  BufferAliasMap buffer_aliases_;
};

} // namespace buffer_alias_analysis_detail

class BufferAliasAnalyzer {
public:
  explicit BufferAliasAnalyzer(const BufferAliasAnalysisContext &context)
      : impl_(context) {}

  BufferAliasAnalyzer(const BufferAliasAnalyzer &) = delete;
  BufferAliasAnalyzer &operator=(const BufferAliasAnalyzer &) = delete;

  void Collect(const std::vector<std::shared_ptr<IRStructure>> &root) {
    impl_.Collect(root);
  }
  void Analyze(const std::vector<std::shared_ptr<IRStructure>> &root) {
    impl_.Analyze(root);
  }

  const BufferAliasMap &GetAliases() const { return impl_.GetAliases(); }
  void Validate() const { impl_.Validate(); }

private:
  buffer_alias_analysis_detail::BufferAliasAnalyzerImpl impl_;
};

} // namespace ascend
} // namespace tl
} // namespace tvm
