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
 * \file buffer_alias_analysis.h
 * \brief Conditional and periodic buffer-alias analysis for AutoSchedule.
 */

#pragma once

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
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
#include "./task_analysis.h"
#include "ascend/op/utils.h"
#include "ascend/transform/buffer_version.h"
#include "support/check.h"

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
  ResolveLifetimeEndpoints(const Var &storage, TaskNode *task,
                           bool at_beginning, ControlNode *scope) const = 0;
  virtual bool HappensBefore(ControlNode *scope, size_t src, size_t dst,
                             int64_t distance) const = 0;
  virtual std::optional<int64_t> MinimumDistance(ControlNode *scope, size_t src,
                                                 size_t dst) const = 0;
  virtual std::optional<StorageIterationPeriod>
  StoragePeriod(const Var &storage, ControlNode *scope) const = 0;
  virtual size_t PromoteLifetimeSite(const Var &storage, size_t site_id,
                                     ControlNode *control) const = 0;
};

namespace buffer_alias_analysis_detail {

using namespace tirx;
using namespace prim;
using ffi::GetRef;

// Local to buffer-lifetime analysis; never copied into TaskNode or scheduled
// TIR.
struct PhysicalTaskAccesses {
  std::vector<TensorRegion> reads;
  std::vector<TensorRegion> writes;
  std::vector<TensorRegion> must_writes;
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
    auto detector = ffi::make_object<MemoryAccessDetector>(
        MemoryAccessDetector::AccessMode::kPhysical, LayoutMap{},
        task->StorageContext(), task->outer_ctx);
    detector->Analyze(stmt);
    append(result.reads, detector->GetReadRegions());
    append(result.writes, detector->GetWriteRegions());
    append(result.must_writes, detector->GetMustWriteRegions());
    append(result.opaque_vars, detector->GetOpaqueAccessVars());
    append(result.read_vars, detector->GetReadVars());
    append(result.write_vars, detector->GetWriteVars());
  };
  // Native headers already contain only the evaluated condition/bounds. They
  // are real task nodes for every lexical scope, not a legacy For shell.
  analyze(task->stmt);
  const IRStructure *guard_owner = task;
  if (const IRStructure *parent = task->GetParent();
      parent && parent->HeaderTask() == task)
    guard_owner = parent;
  for (const auto &guard : guard_owner->GetGuards()) {
    if (guard->IsCondition()) {
      analyze(Evaluate(
          static_cast<const ConditionGuard *>(guard.get())->condition));
    } else {
      const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
      if (auto expr = attribute->node.try_cast<Expr>())
        analyze(Evaluate(*expr));
      if (auto expr = attribute->value.try_cast<Expr>())
        analyze(Evaluate(*expr));
    }
  }
  return result;
}

struct TaskStorageAccess {
  TaskNode *task;
  std::vector<TensorRegion> read_regions;
  std::vector<TensorRegion> write_regions;
  std::vector<TensorRegion> must_write_regions;

  bool Reads() const { return !read_regions.empty(); }
  bool Writes() const { return !write_regions.empty(); }
};

struct StorageAccessInfo {
  Var storage{ffi::UnsafeInit{}};
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
  Var storage{ffi::UnsafeInit{}};
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

inline std::optional<TensorRegion>
TryMergeDenseRegions(const TensorRegion &lhs, const TensorRegion &rhs) {
  if (!RegionTensor(lhs).same_as(RegionTensor(rhs)) ||
      lhs->region.size() != rhs->region.size()) {
    return std::nullopt;
  }
  if (RegionCovers(lhs->region, rhs->region))
    return lhs;
  if (RegionCovers(rhs->region, lhs->region))
    return rhs;

  auto analyzer = CreateTileLangAnalyzer();
  std::optional<size_t> differing_axis;
  for (size_t i = 0; i < lhs->region.size(); ++i) {
    const Range &lhs_range = lhs->region[i];
    const Range &rhs_range = rhs->region[i];
    if (analyzer->CanProveEqual(lhs_range->begin, rhs_range->begin) &&
        analyzer->CanProveEqual(lhs_range->extent, rhs_range->extent)) {
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
  PrimExpr lhs_end = lhs_range->begin + lhs_range->extent;
  PrimExpr rhs_end = rhs_range->begin + rhs_range->extent;
  PrimExpr merged_min{ffi::UnsafeInit{}};
  if (analyzer->CanProve(lhs_range->begin <= rhs_range->begin) &&
      analyzer->CanProve(rhs_range->begin <= lhs_end)) {
    merged_min = lhs_range->begin;
  } else if (analyzer->CanProve(rhs_range->begin <= lhs_range->begin) &&
             analyzer->CanProve(lhs_range->begin <= rhs_end)) {
    merged_min = rhs_range->begin;
  } else {
    return std::nullopt;
  }

  PrimExpr merged_end{ffi::UnsafeInit{}};
  if (analyzer->CanProve(lhs_end >= rhs_end)) {
    merged_end = lhs_end;
  } else if (analyzer->CanProve(rhs_end >= lhs_end)) {
    merged_end = rhs_end;
  } else {
    return std::nullopt;
  }
  Region merged = lhs->region;
  merged.Set(axis, Range::FromMinExtent(merged_min, merged_end - merged_min));
  return TensorRegion(RegionTensor(lhs), std::move(merged), TensorRegionType(),
                      lhs->loc);
}

inline bool DenseRegionsCover(const std::vector<TensorRegion> &writes,
                              const TensorRegion &read) {
  // Regions describe different access points. A store can change an index
  // buffer between them, so identical TensorLoads are not an equality proof.
  // Snapshot each write independently before merging: otherwise two changing
  // boundaries could falsely appear adjacent and merge into a constant range.
  auto snapshot = [](const TensorRegion &region) {
    auto freshen = ffi::make_object<FreshenMutableReads>(
        FreshenMutableReads::Mode::kSnapshot);
    Region bounds;
    bounds.reserve(region->region.size());
    for (const Range &axis : region->region) {
      bounds.push_back(Range::FromMinExtent((*freshen)(axis->begin),
                                            (*freshen)(axis->extent)));
    }
    return TensorRegion(RegionTensor(region), std::move(bounds),
                        TensorRegionType(), region->loc);
  };
  TensorRegion read_snapshot = snapshot(read);
  std::vector<TensorRegion> merged;
  for (const TensorRegion &write : writes) {
    if (RegionTensor(write).same_as(RegionTensor(read)))
      merged.push_back(snapshot(write));
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < merged.size() && !changed; ++i) {
      for (size_t j = i + 1; j < merged.size(); ++j) {
        std::optional<TensorRegion> combined =
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

inline bool LoopHasKnownProgress(const ControlNode *control, bool single_trip) {
  if (control == nullptr || control->BodyHasLoopBreak())
    return false;
  const auto *loop = control->control.get();
  if (SideEffect(loop->begin) > CallEffectKind::kPure ||
      SideEffect(loop->end) > CallEffectKind::kPure ||
      SideEffect(loop->step) > CallEffectKind::kPure)
    return false;
  auto analyzer = CreateTileLangAnalyzer();
  auto context =
      control->task ? control->task->outer_ctx.Populate(analyzer) : nullptr;
  // The loop condition remains var < end for every step sign. A negative
  // distance is zero-trip; a negative step with a positive distance does not
  // provide the finite forward progress needed for a storage lifetime proof.
  if (!CanUseLoopBounds(loop->begin, loop->end, loop->step, analyzer.get()) ||
      !analyzer->CanProve(loop->begin < loop->end))
    return false;
  if (!single_trip)
    return true;
  if (loop->var.ty().bits() > 64)
    return false;
  // With begin < end the unsigned difference is the mathematical distance,
  // including a signed range crossing zero whose distance exceeds INT64_MAX.
  PrimExpr distance = cast(loop->end, PrimType::UInt(64)) -
                      cast(loop->begin, PrimType::UInt(64));
  return analyzer->CanProve(distance <= cast(loop->step, PrimType::UInt(64)));
}

inline bool IsGuaranteedSingleTrip(const ControlNode *control) {
  return LoopHasKnownProgress(control, true);
}

} // namespace buffer_alias_analysis_detail

namespace buffer_alias_analysis_detail {

class BufferAliasAnalyzerImpl {
public:
  explicit BufferAliasAnalyzerImpl(const BufferAliasAnalysisContext &context,
                                   L0StorageGroups groups)
      : context_(&context), groups_(std::move(groups)) {}

  void Collect(const std::vector<std::shared_ptr<IRStructure>> &root) {
    storage_access_info_.clear();
    storage_order_.clear();
    lifetime_summaries_.clear();
    buffer_aliases_ = {};

    std::vector<TaskNode *> tasks;
    CollectTaskNodesAndHeaders(root, tasks);
    std::unordered_map<const TaskNode *, PhysicalTaskAccesses>
        physical_accesses;
    std::unordered_map<Var, NativeStorageAddress, ObjectPtrHash, ObjectPtrEqual>
        storage_addresses;
    for (TaskNode *task : tasks) {
      const PhysicalTaskAccesses &physical =
          physical_accesses.emplace(task, AnalyzePhysicalTaskAccesses(task))
              .first->second;
      struct TaskAccessRegions {
        std::vector<TensorRegion> reads;
        std::vector<TensorRegion> writes;
        std::vector<TensorRegion> must_writes;
      };
      std::unordered_map<Var, TaskAccessRegions, ObjectPtrHash, ObjectPtrEqual>
          task_accesses;
      std::vector<Var> task_storage_order;
      auto get_task_access =
          [&](const TensorRegion &region) -> TaskAccessRegions * {
        // SS allocations are packed without lifetime reuse.
        if (IsSSBuffer(RegionTensor(region)))
          return nullptr;
        if (!IsAscendOnChipBuffer(RegionTensor(region)) &&
            !IsL0SFBuffer(RegionTensor(region)))
          return nullptr;
        Var storage = task->StorageIdentity(RegionTensor(region));
        auto address = task->StorageContext().Address(RegionTensor(region));
        auto [origin, first_origin] =
            storage_addresses.try_emplace(storage, address);
        if (!first_origin && (!origin->second.root.same_as(address.root) ||
                              !address.exact || !address.fresh_allocation)) {
          origin->second.exact = false;
          origin->second.fresh_allocation = false;
        }
        auto [it, inserted] = task_accesses.try_emplace(storage);
        if (inserted)
          task_storage_order.push_back(storage);
        return &it->second;
      };
      for (const TensorRegion &region : physical.reads) {
        if (TaskAccessRegions *access = get_task_access(region))
          access->reads.push_back(region);
      }
      for (const TensorRegion &region : physical.writes) {
        if (TaskAccessRegions *access = get_task_access(region))
          access->writes.push_back(region);
      }
      for (const TensorRegion &region : physical.must_writes) {
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
          AddBufferAliasStorage(&buffer_aliases_,
                                groups_.Representative(storage));
        }
        info->second.accesses.push_back(
            {task, mask.reads, mask.writes, mask.must_writes});
      }
    }

    auto proven_disjoint = [](const NativeStorageAddress &lhs,
                              const NativeStorageAddress &rhs) {
      return lhs.exact && rhs.exact && lhs.fresh_allocation &&
             rhs.fresh_allocation && !lhs.root.same_as(rhs.root);
    };
    // Address reuse is only a contract between fresh allocation roots. An
    // on-chip external/integer-derived view may touch any candidate storage;
    // its different diagnostic identity does not exclude that hidden lifetime.
    for (auto &[storage, info] : storage_access_info_) {
      const auto &address = storage_addresses.at(storage);
      info.opaque |= !address.exact || !address.fresh_allocation;
      for (const auto &[other, origin] : storage_addresses)
        if (!other.same_as(storage) && !proven_disjoint(address, origin))
          info.opaque = true;
    }

    for (TaskNode *task : tasks) {
      const PhysicalTaskAccesses &physical = physical_accesses.at(task);
      auto mark_opaque = [&](const Var &var) {
        auto address = task->StorageContext().Address(var);
        for (auto &[storage, info] : storage_access_info_)
          if (!proven_disjoint(address, storage_addresses.at(storage)))
            info.opaque = true;
      };
      for (const Var &var : physical.opaque_vars)
        mark_opaque(var);
      auto definition = GetFlatTaskBind(task->stmt);
      bool address_definition =
          definition && (definition->var->ty.as<PtrTypeObj>() ||
                         definition->var->ty.as<TensorTypeObj>());
      auto mark_unrepresented = [&](const Var &var) {
        if (!var->ty.as<PtrTypeObj>() && !var->ty.as<TensorTypeObj>())
          return;
        // A flat allocation/view/pointer Bind defines an address. Its bound
        // variable and address inputs are not memory reads/writes; any real
        // load in its value or tensor metadata already has its own region.
        if (address_definition)
          return;
        auto address = task->StorageContext().Address(var);
        bool represented = false;
        for (const auto *regions : {&physical.reads, &physical.writes})
          for (const TensorRegion &region : *regions)
            represented |= address.root.same_as(
                task->StorageContext().Address(RegionTensor(region)).root);
        if (!represented)
          mark_opaque(var);
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
    std::function<void(IRStructure *)> analyze_child = [&](IRStructure *node) {
      if (node->IsControl()) {
        auto *control = static_cast<ControlNode *>(node);
        AnalyzeScope(control->children, control);
        return;
      }
      for (const auto *children : node->ChildLists())
        for (const auto &child : *children)
          analyze_child(child.get());
    };
    for (const auto &node : nodes)
      analyze_child(node.get());

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

    std::unordered_map<Var, const LifetimeSummary *, ObjectPtrHash,
                       ObjectPtrEqual>
        by_storage;
    for (const LifetimeSummary &lifetime : lifetimes)
      by_storage.emplace(lifetime.storage, &lifetime);
    auto can_alias_groups = [&](const Var &lhs, const Var &rhs) {
      for (const Var &a : groups_.Members(lhs)) {
        for (const Var &b : groups_.Members(rhs)) {
          auto a_lifetime = by_storage.find(a);
          auto b_lifetime = by_storage.find(b);
          // A partial group summary cannot authorize reusing its address.
          if (a_lifetime == by_storage.end() ||
              b_lifetime == by_storage.end() ||
              !CanAliasStorageLifetimes(*a_lifetime->second,
                                        *b_lifetime->second, loop,
                                        current_happens_before))
            return false;
        }
      }
      return true;
    };
    for (size_t i = 0; i < lifetimes.size(); ++i) {
      const Var &lhs = lifetimes[i].storage;
      if (!groups_.Representative(lhs).same_as(lhs))
        continue;
      for (size_t j = i + 1; j < lifetimes.size(); ++j) {
        const Var &rhs = lifetimes[j].storage;
        if (groups_.Representative(rhs).same_as(rhs) &&
            can_alias_groups(lhs, rhs))
          AddBufferAlias(&buffer_aliases_, lhs, rhs);
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
      if (node->IsControl())
        result.insert(static_cast<ControlNode *>(node)->control->var);
      if (auto *header = node->HeaderTask())
        for (const Var &var : header->GetWriteVars())
          result.insert(var);
      for (const auto *children : node->ChildLists())
        for (const auto &child : *children)
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
    VisitNativePostOrder(expr, [&](const ObjectRef &node) {
      if (varies)
        return;
      if (node.as<TensorLoadObj>()) {
        varies = true;
        return;
      }
      if (const auto *call = node.as<CallObj>()) {
        if (SideEffect(GetRef<Expr>(call)) > CallEffectKind::kPure)
          varies = true;
        return;
      }
      const auto *var_node = node.as<VarObj>();
      if (var_node == nullptr)
        return;
      Var var = GetRef<Var>(var_node);
      auto binding = bindings.find(var);
      if (binding != bindings.end()) {
        // A native SSA snapshot defined outside the repeated tree is stable,
        // even when its original value was a mutable load.
        if (!variant_vars.count(var))
          return;
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
          bindings.insert_or_assign(constraint.var, constraint.value);
      }
    }
    VarSet variant_vars = CollectIterationVariantVars(scope);
    auto depends = [&](const PrimExpr &expression) {
      VarSet visiting;
      return ExpressionDependsOnIteration(expression, bindings, variant_vars,
                                          &visiting);
    };
    // Captured branch predicates are fresh symbols in outer_ctx. Inspect the
    // original header only for invariance; use the decoder's captured symbol
    // for implication/exclusion so a condition is never evaluated twice.
    for (const IRStructure *node = task; node && node != scope;
         node = node->GetParent()) {
      if (!node->IsScope())
        continue;
      const Stmt &shell = static_cast<const ScopeNode *>(node)->shell;
      if (const auto *branch = shell.as<IfObj>()) {
        if (depends(branch->condition))
          return false;
      } else if (auto attribute = ReadTaskAttribute(shell)) {
        if (auto expr = attribute->node.try_cast<PrimExpr>();
            expr && depends(*expr))
          return false;
        if (auto expr = attribute->value.try_cast<PrimExpr>();
            expr && depends(*expr))
          return false;
      }
    }
    for (const PrimExpr &condition : conditions) {
      if (depends(condition)) {
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

    auto analyzer = CreateTileLangAnalyzer();
    ConstrAnalyzerContext bind_context;
    for (const Constr &constraint : task->outer_ctx.constrs_) {
      // A mutable Bind is one SSA snapshot. Keep its variable shared between
      // signatures instead of inventing a new read for each proof.
      if (constraint.kind != Constr::kConstr &&
          (constraint.kind != Constr::kBindValue ||
           SideEffect(constraint.value) <= CallEffectKind::kPure))
        constraint.Populate(analyzer, &bind_context);
    }
    ConditionSignature signature;
    std::vector<PrimExpr> raw_conditions;
    auto append = [&](const PrimExpr &condition) {
      raw_conditions.push_back(condition);
      PrimExpr simplified = analyzer->Simplify(condition);
      if (std::none_of(signature.conditions.begin(), signature.conditions.end(),
                       [&](const PrimExpr &existing) {
                         return ffi::StructuralEqual()(existing, simplified);
                       }))
        signature.conditions.push_back(simplified);
    };
    // The native decoder captures each If/assume condition in outer_ctx.
    // Guard deltas cover additional wrappers introduced by later scheduling.
    for (const Constr &constraint : task->outer_ctx.constrs_)
      if (constraint.kind == Constr::kConstr)
        append(constraint.value);
    for (IRStructure *node : path)
      for (const auto &guard : node->GetGuards())
        if (guard->IsCondition())
          append(static_cast<const ConditionGuard *>(guard.get())->condition);
    signature.iteration_invariant =
        AreConditionsIterationInvariant(task, raw_conditions, scope);
    return signature;
  }

  PrimExpr ConditionPredicate(const std::vector<PrimExpr> &signature) const {
    PrimExpr predicate = IntImm::Bool(true);
    for (const PrimExpr &condition : signature)
      predicate = predicate && condition;
    return predicate;
  }

  bool ProveGuardImplication(const std::vector<PrimExpr> &premise,
                             const std::vector<PrimExpr> &conclusion,
                             ControlNode *scope) const {
    if (conclusion.empty())
      return true;
    auto analyzer = CreateTileLangAnalyzer();
    analyzer->z3_prover.SetRLimit(50000);
    auto context = scope && scope->task
                       ? scope->task->outer_ctx.Populate(analyzer)
                       : nullptr;
    PrimExpr implication =
        !ConditionPredicate(premise) || ConditionPredicate(conclusion);
    implication = analyzer->Simplify(implication);
    return analyzer->CanProve(implication) ||
           analyzer->z3_prover.CanProve(implication);
  }

  bool ProveMutuallyExclusive(const PrimExpr &lhs, const PrimExpr &rhs,
                              ControlNode *scope) const {
    if (!lhs.defined() || !rhs.defined())
      return false;
    auto analyzer = CreateTileLangAnalyzer();
    analyzer->z3_prover.SetRLimit(50000);
    auto context = scope && scope->task
                       ? scope->task->outer_ctx.Populate(analyzer)
                       : nullptr;
    PrimExpr exclusive = analyzer->Simplify(!(lhs && rhs));
    return analyzer->CanProve(exclusive) ||
           analyzer->z3_prover.CanProve(exclusive);
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
      if (analysis_scope && task == analysis_scope->HeaderTask())
        return false;
      IRStructure *node = task;
      while (node->GetParent() != analysis_scope) {
        IRStructure *parent = node->GetParent();
        if (parent == nullptr)
          return false;
        if (parent->IsControl() &&
            !IsGuaranteedSingleTrip(static_cast<ControlNode *>(parent)))
          return false;
        ICHECK(parent->IsControl() || parent->IsScope());
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
      if (analysis_scope && access.task == analysis_scope->HeaderTask()) {
        has_access_before_scope = true;
        continue;
      }
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
      while (common && !common->IsControl())
        common = common->GetParent();
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
      for (const TensorRegion &read : reader.read_regions) {
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
    auto shift_regions = [&](const std::vector<TensorRegion> &regions,
                             int64_t distance) {
      if (analysis_scope == nullptr || distance == 0)
        return regions;
      const ForObj *loop = analysis_scope->control.get();
      PrimExpr step = loop->step;
      ffi::Map<Var, PrimExpr> substitution{
          {loop->var, loop->var + MakeConst(loop->var.ty(), distance) * step}};
      std::vector<TensorRegion> shifted;
      for (const TensorRegion &region : regions)
        shifted.emplace_back(RegionTensor(region),
                             SubstituteTaskRanges(region->region, substitution),
                             TensorRegionType(), region->loc);
      return shifted;
    };
    int64_t max_offset = 0;
    if (analysis_scope != nullptr) {
      PrimType dtype = analysis_scope->control->var.ty();
      if (dtype.MatchesCode(DLDataTypeCode::kDLInt))
        max_offset = dtype.bits() >= 64
                         ? std::numeric_limits<int64_t>::max()
                         : (int64_t{1} << (dtype.bits() - 1)) - 1;
    }
    bool regions_periodic = analysis_scope != nullptr && period &&
                            period->iterations > 0 &&
                            period->iterations <= max_offset;
    if (regions_periodic) {
      auto step = TryConstInt(analysis_scope->control->step);
      uint64_t stride =
          step ? (*step < 0 ? uint64_t(0) - uint64_t(*step) : uint64_t(*step))
               : 0;
      regions_periodic = stride && uint64_t(period->iterations) <=
                                       uint64_t(max_offset) / stride;
    }
    if (regions_periodic) {
      auto analyzer = CreateTileLangAnalyzer();
      // Do not bind the loop variable to its finite range: periodicity must
      // hold for shifted generations as well as the current iteration.
      VarSet variant_vars = CollectIterationVariantVars(analysis_scope);
      const PrimVar &loop_var = analysis_scope->control->var;
      PrimVar explicit_ordinal("generation_ordinal", loop_var->ty);
      for (const TaskStorageAccess &access : accesses) {
        VarExprMap bindings;
        for (const Constr &constraint : access.task->outer_ctx.constrs_) {
          if (constraint.kind == Constr::kBindValue)
            bindings.insert_or_assign(constraint.var, constraint.value);
        }
        auto has_stable_parameters = [&](const PrimExpr &expr) {
          // Only the explicit coordinate is shifted below. An unresolved let
          // alias of the loop variable must not make a changing slot appear
          // invariant merely because substitution did not reach its binding.
          PrimExpr explicit_coordinate =
              SubstituteConstraintExpr(expr, {{loop_var, explicit_ordinal}});
          VarSet visiting;
          return !ExpressionDependsOnIteration(explicit_coordinate, bindings,
                                               variant_vars, &visiting);
        };
        for (const auto *regions :
             {&access.read_regions, &access.must_write_regions}) {
          auto shifted = shift_regions(*regions, period->iterations);
          for (size_t i = 0; i < regions->size(); ++i) {
            const TensorRegion &region = (*regions)[i];
            for (size_t axis = 0; axis < (*regions)[i]->region.size(); ++axis) {
              const Range &original = (*regions)[i]->region[axis];
              const Range &next = shifted[i]->region[axis];
              regions_periodic &= has_stable_parameters(original->begin) &&
                                  has_stable_parameters(original->extent);
              regions_periodic &=
                  analyzer->CanProveEqual(original->begin, next->begin) &&
                  analyzer->CanProveEqual(original->extent, next->extent);
            }
            // Repeating logical coordinates do not prove repeating physical
            // addresses for a loop-dependent view, stride or pointer offset.
            Array<PrimExpr> indices;
            for (size_t axis = 0; axis < region->region.size(); ++axis)
              indices.push_back(PrimVar("period_axis" + std::to_string(axis),
                                        PrimType::Int(64)));
            TensorVar tensor = RegionTensor(region);
            auto offsets = tensor.ty()->ElemOffset(indices);
            if (offsets.size() != 1 ||
                !access.task->StorageContext().Exact(tensor)) {
              regions_periodic = false;
              continue;
            }
            PrimExpr bits =
                access.task->StorageContext().ElementBitOffset(tensor, indices);
            PrimExpr next_bits = SubstituteConstraintExpr(
                bits,
                {{loop_var,
                  loop_var + MakeConst(loop_var.ty(), period->iterations) *
                                 analysis_scope->control->step}});
            regions_periodic &= has_stable_parameters(bits) &&
                                analyzer->CanProveEqual(bits, next_bits);
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
              [&](const TensorRegion &read) {
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
        auto *parent = analysis_scope->GetParentControl();
        auto signature =
            CollectConditionSignature(analysis_scope->HeaderTask(), parent);
        scope_must_execute =
            LoopHasKnownProgress(analysis_scope, false) && signature &&
            ProveGuardImplication({}, signature->conditions, parent);
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

  static ControlNode *SummaryChild(TaskNode *task, ControlNode *parent_scope) {
    ControlNode *candidate = nullptr;
    IRStructure *node = task;
    for (; node && node != parent_scope; node = node->GetParent()) {
      if (node->IsControl()) {
        auto *control = static_cast<ControlNode *>(node);
        if (!IsGuaranteedSingleTrip(control))
          candidate = control;
      }
    }
    if (parent_scope && node != parent_scope)
      return nullptr;
    if (candidate && task == candidate->HeaderTask())
      return nullptr;
    return candidate;
  }

  ControlNode *FindSummaryControl(const StorageAccessInfo &info,
                                  ControlNode *parent_scope) const {
    ControlNode *control = nullptr;
    for (const TaskStorageAccess &access : info.accesses) {
      if (parent_scope && (!access.task->IsWithin(parent_scope) ||
                           access.task == parent_scope->HeaderTask()))
        continue;
      // Lexical If/assume scopes do not create a new iteration clock. Keep
      // their captured conditions while finding the outermost repeated child.
      auto *candidate = SummaryChild(access.task, parent_scope);
      if (!candidate)
        return nullptr;
      if (control == nullptr)
        control = candidate;
      else if (control != candidate)
        return nullptr;
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

    if (!LoopHasKnownProgress(control, false))
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
    if (child_result.promote_live_through_as_live_in &&
        parent_scope == nullptr) {
      // The storage spans the whole child control, so it cannot alias inside
      // that child. Only a non-periodic parent can reduce it to a finite
      // prefix ending at the child's last read. A loop parent needs the
      // incoming value again on its next iteration.
      parent_live_through = false;
      child_result.live_out = false;
    }
    std::vector<GenerationLifetime> generations;
    if (!parent_live_through) {
      auto guard_analyzer = CreateTileLangAnalyzer();
      auto control_signature =
          CollectConditionSignature(control->HeaderTask(), parent_scope);
      if (!control_signature)
        return std::nullopt;
      PrimExpr control_guard =
          ConditionPredicate(control_signature->conditions);
      bool control_guard_iteration_invariant =
          control_signature->iteration_invariant;

      ffi::StructuralEqual structural_equal;
      std::vector<GenerationLifetime> local_summaries;
      for (const GenerationLifetime &generation : child_result.generations) {
        PrimExpr generation_guard =
            generation.guard.defined() ? generation.guard : IntImm::Bool(true);
        PrimExpr promoted_guard =
            guard_analyzer->Simplify(control_guard && generation_guard);
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
    child_result.promote_live_through_as_live_in &= parent_scope != nullptr;
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
      auto *control = SummaryChild(access.task, parent_scope);
      if (control == nullptr)
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
  L0StorageGroups groups_;
  StorageAccessInfoMap storage_access_info_;
  std::vector<Var> storage_order_;
  std::map<ControlNode *, LifetimeSummaryMap> lifetime_summaries_;
  BufferAliasMap buffer_aliases_;
};

} // namespace buffer_alias_analysis_detail

class BufferAliasAnalyzer {
public:
  explicit BufferAliasAnalyzer(const BufferAliasAnalysisContext &context,
                               L0StorageGroups groups = L0StorageGroups())
      : impl_(context, std::move(groups)) {}

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
