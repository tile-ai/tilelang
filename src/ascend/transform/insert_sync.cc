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
 * \file insert_sync.cc
 * \brief Synchronization analysis, flag allocation, and insertion.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "./auto_schedule/dependency_analysis.h"
#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/multi_buffer.h"
#include "./auto_schedule/scheduled_tir.h"
#include "./auto_schedule/task_analysis.h"
#include "./auto_schedule/task_annotations.h"
#include "ascend/op/builtin.h"
#include "ascend/transform/attr.h"
#include "buffer_version.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;
using ffi::GetRef;

namespace {

// ============================================================================
// Synchronization site/edge model, collection, and lookup
// ============================================================================

struct SyncInsertionSite {
  IRStructure *path_begin;
  IRStructure *path_end;
  bool at_beginning;
  CoreMask core;
  std::string pipe;
  int domain_id;
  PrimExpr guard{Bool(true)};

  SyncInsertionSite(IRStructure *path_begin, TaskNode *task, bool at_beginning,
                    CoreMask core, const EpochDomain &domain)
      : path_begin(path_begin), path_end(nullptr), at_beginning(at_beginning),
        core(core), pipe(GetResourcePipeName(task->GetPipeMask())),
        domain_id(domain.id) {
    ICHECK(path_begin != nullptr);
    ICHECK(task != nullptr);
    ICHECK(IsConcreteCore(core));
    ICHECK_GT(domain_id, 0);
    if (domain.IsCounter()) {
      ICHECK(domain.owner != nullptr && path_begin != domain.owner &&
             path_begin->IsWithin(domain.owner))
          << "Counter-domain site must stay strictly inside its owner";
    } else {
      ICHECK_EQ(path_begin->GetParent(), domain.owner)
          << "Lexical-domain site must be a direct child of its owner";
    }
    PrimExpr active_guard = domain.active_guard;
    bool active_guard_is_enclosed = is_one(active_guard);
    bool path_begin_guard_matches_active = false;
    auto append = [&](PrimExpr term) {
      guard = is_one(guard) ? std::move(term) : guard && term;
    };
    std::vector<IRStructure *> path = task->PathFrom(path_begin);
    for (size_t path_index = 0; path_index < path.size(); ++path_index) {
      path_end = path[path_index];
      bool has_conditions = path_end->HasConditions();
      if (has_conditions) {
        // Only a guard scope's direct child may absorb its active guard.
        if (path_end != path_begin || is_one(active_guard) ||
            path_begin->GetParent() != domain.owner)
          break;
        PrimExpr condition_guard = path_end->GetConditionGuard();
        if (!active_guard.same_as(condition_guard) &&
            !ffi::StructuralEqual()(active_guard, condition_guard)) {
          break;
        }
        path_begin_guard_matches_active = true;
      }
      if (path_index + 1 == path.size())
        break;
      ICHECK(path_end->IsControl());
      auto *control = static_cast<ControlNode *>(path_end);
      if (control->BodyHasLoopBreak())
        break;
      PrimExpr nonempty = control->control->extent >
                          make_zero(control->control->extent.dtype());
      bool can_enter = control->task->outer_ctx.CanProve(nonempty);
      if (!can_enter)
        break;
      if (path_end == path_begin && has_conditions)
        active_guard_is_enclosed = path_begin_guard_matches_active;
      const ForNode *for_node = control->control.get();
      PrimExpr loop_var = for_node->loop_var;
      PrimExpr min = for_node->min;
      PrimExpr extent = for_node->extent;
      PrimExpr step = for_node->step.has_value() ? for_node->step.value()
                                                 : IntImm(DataType::Int(32), 1);
      if (at_beginning) {
        append(loop_var == min);
      } else {
        append(((loop_var - min) / step + 1) * step >= extent);
      }
    }
    ICHECK(path_end != nullptr) << "Failed to find a sync insertion node";
    // A crossed path_begin condition physically encloses the insertion. If it
    // was not crossed, keep the domain activation predicate on the emitted
    // synchronization itself.
    if (!active_guard_is_enclosed)
      append(active_guard);
  }

  int EpochDomainId() const { return domain_id; }

  bool operator<(const SyncInsertionSite &other) const {
    int parent_order = CompareIRStructure(path_begin->GetParent(),
                                          other.path_begin->GetParent());
    if (parent_order != 0)
      return parent_order < 0;

    int path_end_order = CompareIRStructure(path_end, other.path_end);
    if (path_end_order == -2)
      return at_beginning;
    if (path_end_order == 2)
      return !other.at_beginning;
    if (path_end_order != 0)
      return path_end_order < 0;
    if (at_beginning != other.at_beginning)
      return at_beginning;
    if (core != other.core)
      return core < other.core;
    if (pipe != other.pipe)
      return pipe < other.pipe;
    // Keep domains as the final key. For collected sites, equal parent,
    // path_end, direction, core, and pipe also imply equal path_begin, so all
    // variants of one physical position remain contiguous.
    return EpochDomainId() < other.EpochDomainId();
  }

  bool operator==(const SyncInsertionSite &other) const {
    return path_begin == other.path_begin && path_end == other.path_end &&
           at_beginning == other.at_beginning && core == other.core &&
           pipe == other.pipe && EpochDomainId() == other.EpochDomainId();
  }

  bool HasSamePhysicalPosition(const SyncInsertionSite &other) const {
    return path_begin == other.path_begin && path_end == other.path_end &&
           at_beginning == other.at_beginning && core == other.core &&
           pipe == other.pipe;
  }

  const PrimExpr &GetGuard() const { return guard; }
};

using SyncInsertionSiteList = std::vector<SyncInsertionSite>;

// Inputs needed to look up a collected insertion site without reconstructing
// it. Domain ids are pass-local and reconstructed from canonical guards.
using SyncInsertionSiteKey =
    std::tuple<IRStructure *, TaskNode *, bool, CoreMask, int>;

struct DepEdge {
  size_t src;
  size_t dst;
  int distance;
  bool strict;

  DepEdge(size_t src, size_t dst, int distance, bool strict)
      : src(src), dst(dst), distance(distance), strict(strict) {}

  bool operator<(const DepEdge &other) const {
    // Process tighter edges first: smaller distance, later producer, then
    // earlier consumer. Site IDs preserve SyncInsertionSite's total order.
    if (distance != other.distance)
      return distance < other.distance;
    if (src != other.src)
      return src > other.src;
    if (dst != other.dst)
      return dst < other.dst;
    return strict < other.strict;
  }
};

// Fields: counter group, producer core/pipe, consumer core/pipe, versions,
// distance.
// Counter event IDs always use the owner clock, but edge distance remains in
// the iteration units of the loop where the dependency was analyzed.
using CounterSyncSignature =
    std::tuple<int, CoreMask, std::string, CoreMask, std::string, int, int>;

// The ordinal is assigned once within each owner/signature sequence. Equal
// keys in different owners identify one shared physical counter channel.
using CounterChannelKey = std::pair<CounterSyncSignature, int>;

struct SyncPoint {
  DepEdge edge;
  int num_versions;
  PrimExpr flag_iteration;
  ControlNode *flag_loop;
  int counter_group_id;
  size_t allocation_index;
  std::optional<CounterChannelKey> counter_channel;

  SyncPoint(DepEdge edge, int num_versions, ControlNode *flag_loop)
      : edge(std::move(edge)), num_versions(num_versions),
        flag_iteration(CalculateIterationCount(flag_loop)),
        flag_loop(flag_loop), counter_group_id(0) {}

  SyncPoint(DepEdge edge, int num_versions, PrimExpr flag_iteration,
            ControlNode *counter_owner, int counter_group_id)
      : edge(std::move(edge)), num_versions(num_versions),
        flag_iteration(std::move(flag_iteration)), flag_loop(counter_owner),
        counter_group_id(counter_group_id) {
    ICHECK(this->flag_iteration.defined());
    ICHECK(counter_owner != nullptr);
    ICHECK_GT(counter_group_id, 0);
  }

  const CounterChannelKey &GetCounterChannel() const {
    ICHECK(counter_channel.has_value())
        << "Counter synchronization point has no channel identity";
    return counter_channel.value();
  }

  bool operator<(const SyncPoint &other) const {
    if (edge < other.edge)
      return true;
    if (other.edge < edge)
      return false;
    int loop_order = CompareIRStructure(flag_loop, other.flag_loop);
    if (loop_order != 0)
      return loop_order < 0;
    if (num_versions != other.num_versions)
      return num_versions < other.num_versions;
    return false;
  }
};

struct VisibleSyncPoint {
  size_t sync_point_id;
  std::vector<DepEdge> projected_edges;
};

class SyncSiteRegistry {
public:
  explicit SyncSiteRegistry(const EpochDomainRegistry &domains)
      : domains_(domains) {}

  void Collect(const std::vector<IRStructure *> &nodes) {
    sites_.clear();
    site_ids_.clear();
    parent_site_ids_.clear();

    std::map<SyncInsertionSite, std::map<int, SyncInsertionSite>> parent_sites;
    std::map<SyncInsertionSiteKey, SyncInsertionSite> keyed_sites;
    std::vector<TaskNode *> tasks;
    for (IRStructure *node : nodes)
      CollectAllTaskNodes(node, tasks);

    for (TaskNode *task : tasks) {
      if (task->GetCoreMask() == kCoreUnassigned)
        continue;
      for (IRStructure *path_begin : task->PathFrom()) {
        SyncInsertionSiteList inner_sites;
        IRStructure *parent = path_begin->GetParent();
        ICHECK(parent == nullptr || parent->IsControl());
        auto *scope =
            parent == nullptr ? nullptr : static_cast<ControlNode *>(parent);
        std::vector<int> domain_ids = SiteDomainsForTaskAtScope(task, scope);
        AddInsertionSites(path_begin, task, domain_ids, inner_sites,
                          keyed_sites);
        sites_.insert(sites_.end(), inner_sites.begin(), inner_sites.end());

        if (parent == nullptr)
          continue;
        auto *parent_scope =
            static_cast<ControlNode *>(parent)->GetParentControl();
        std::vector<int> parent_domain_ids =
            SiteDomainsForTaskAtScope(task, parent_scope);
        for (const SyncInsertionSite &inner_site : inner_sites) {
          for (int parent_domain_id : parent_domain_ids) {
            SyncInsertionSiteKey parent_key{parent, task,
                                            inner_site.at_beginning,
                                            inner_site.core, parent_domain_id};
            auto parent_it = keyed_sites.find(parent_key);
            ICHECK(parent_it != keyed_sites.end())
                << "Synchronization insertion site is missing its parent site";
            // One child endpoint may promote into several parent domains, but
            // each domain must select exactly one physical parent endpoint.
            auto [recorded_it, inserted] = parent_sites[inner_site].emplace(
                parent_domain_id, parent_it->second);
            ICHECK(inserted || recorded_it->second.HasSamePhysicalPosition(
                                   parent_it->second))
                << "One synchronization site has conflicting parent sites "
                   "in the same epoch domain";
          }
        }
      }
    }
    std::sort(sites_.begin(), sites_.end());
    sites_.erase(std::unique(sites_.begin(), sites_.end()), sites_.end());
    for (const auto &[key, site] : keyed_sites) {
      auto [_, inserted] = site_ids_.emplace(key, FindSiteId(site));
      ICHECK(inserted) << "Duplicate synchronization insertion-site key";
    }
    parent_site_ids_.resize(sites_.size());
    for (const auto &[site, parents] : parent_sites) {
      std::vector<size_t> &parent_ids = parent_site_ids_[FindSiteId(site)];
      for (const auto &[_, parent] : parents)
        parent_ids.push_back(FindSiteId(parent));
    }
  }

  const SyncInsertionSiteList &Sites() const { return sites_; }

  size_t Size() const { return sites_.size(); }

  const EpochDomain &Domain(const SyncInsertionSite &site) const {
    return domains_.Domain(site.EpochDomainId());
  }

  bool SameEpochDomain(const SyncInsertionSite &lhs,
                       const SyncInsertionSite &rhs) const {
    return lhs.EpochDomainId() == rhs.EpochDomainId();
  }

  template <typename Visitor>
  void ForEachPhysicalSitePair(const DepEdge &edge,
                               const Visitor &visit) const {
    ICHECK(SameEpochDomain(Src(edge), Dst(edge)))
        << "A dependency edge must have one active-epoch domain";
    std::vector<size_t> src_variants = PhysicalSiteVariants(edge.src);
    std::vector<size_t> dst_variants = PhysicalSiteVariants(edge.dst);
    ForEachMatchingDomainSitePair(
        src_variants, dst_variants, [&](size_t src, size_t dst) {
          visit(DepEdge(src, dst, edge.distance, edge.strict));
        });
  }

  template <typename Visitor>
  void ForEachParentSitePair(const DepEdge &edge, const Visitor &visit) const {
    ICHECK(SameEpochDomain(Src(edge), Dst(edge)))
        << "A dependency edge must have one active-epoch domain";
    ICHECK_LT(std::max(edge.src, edge.dst), parent_site_ids_.size());
    const std::vector<size_t> &src_parents = parent_site_ids_[edge.src];
    const std::vector<size_t> &dst_parents = parent_site_ids_[edge.dst];
    ForEachMatchingDomainSitePair(
        src_parents, dst_parents, [&](size_t src, size_t dst) {
          visit(DepEdge(src, dst, edge.distance, edge.strict));
        });
  }

  const SyncInsertionSite &Src(const DepEdge &edge) const {
    return sites_[edge.src];
  }

  const SyncInsertionSite &Dst(const DepEdge &edge) const {
    return sites_[edge.dst];
  }

  const SyncInsertionSite &Src(const SyncPoint &sync_point) const {
    return Src(sync_point.edge);
  }

  const SyncInsertionSite &Dst(const SyncPoint &sync_point) const {
    return Dst(sync_point.edge);
  }

  bool IsCrossCore(const SyncPoint &sync_point) const {
    return Src(sync_point).core != Dst(sync_point).core;
  }

  bool UsesFlag(const SyncPoint &sync_point) const {
    return IsCrossCore(sync_point) ||
           Src(sync_point).pipe != Dst(sync_point).pipe;
  }

  bool HasSameFlagKind(const SyncPoint &lhs, const SyncPoint &rhs) const {
    const SyncInsertionSite &lhs_src = Src(lhs);
    const SyncInsertionSite &lhs_dst = Dst(lhs);
    const SyncInsertionSite &rhs_src = Src(rhs);
    const SyncInsertionSite &rhs_dst = Dst(rhs);
    return lhs_src.core == rhs_src.core && lhs_src.pipe == rhs_src.pipe &&
           lhs_dst.core == rhs_dst.core && lhs_dst.pipe == rhs_dst.pipe;
  }

  bool HasCompatibleFlagKind(const SyncPoint &lhs, const SyncPoint &rhs) const {
    bool lhs_cross_core = IsCrossCore(lhs);
    bool rhs_cross_core = IsCrossCore(rhs);
    if (lhs_cross_core || rhs_cross_core)
      return lhs_cross_core && rhs_cross_core;
    return HasSameFlagKind(lhs, rhs);
  }

  CounterSyncSignature
  GetCounterSyncSignature(const SyncPoint &sync_point) const {
    ICHECK_GT(sync_point.counter_group_id, 0);
    const SyncInsertionSite &src = Src(sync_point);
    const SyncInsertionSite &dst = Dst(sync_point);
    return {sync_point.counter_group_id,
            src.core,
            src.pipe,
            dst.core,
            dst.pipe,
            sync_point.num_versions,
            sync_point.edge.distance};
  }

  size_t FindSiteId(const SyncInsertionSite &site) const {
    auto it = std::lower_bound(sites_.begin(), sites_.end(), site);
    ICHECK(it != sites_.end() && *it == site)
        << "Synchronization insertion site was not collected";
    return std::distance(sites_.begin(), it);
  }

  size_t FindSiteId(const SyncInsertionSiteKey &key) const {
    auto it = site_ids_.find(key);
    ICHECK(it != site_ids_.end())
        << "Synchronization insertion-site key was not collected";
    return it->second;
  }

  size_t FindBoundarySite(size_t site_id, bool at_beginning) const {
    SyncInsertionSite boundary = sites_[site_id];
    boundary.at_beginning = at_beginning;
    return FindSiteId(boundary);
  }

  std::vector<size_t>
  CollectSiteIds(const std::vector<IRStructure *> &nodes) const {
    std::vector<size_t> result;
    for (IRStructure *node : nodes) {
      std::vector<TaskNode *> tasks;
      CollectAllTaskNodes(node, tasks);
      IRStructure *parent = node->GetParent();
      ICHECK(parent == nullptr || parent->IsControl());
      auto *scope =
          parent == nullptr ? nullptr : static_cast<ControlNode *>(parent);
      for (TaskNode *task : tasks) {
        AddInsertionSiteIds(node, task, SiteDomainsForTaskAtScope(task, scope),
                            result);
      }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }

private:
  std::vector<int> SiteDomainsForTaskAtScope(const TaskNode *task,
                                             ControlNode *scope) const {
    std::vector<int> result = domains_.DomainsForTaskAtScope(task, scope);
    int lexical_domain = domains_.UnconditionalLexicalDomain(scope);
    if (std::find(result.begin(), result.end(), lexical_domain) == result.end())
      result.push_back(lexical_domain);
    return result;
  }

  std::vector<size_t> PhysicalSiteVariants(size_t site_id) const {
    ICHECK_LT(site_id, sites_.size());
    const SyncInsertionSite &site = sites_[site_id];
    size_t begin = site_id;
    while (begin > 0 && site.HasSamePhysicalPosition(sites_[begin - 1]))
      --begin;
    size_t end = site_id + 1;
    while (end < sites_.size() && site.HasSamePhysicalPosition(sites_[end]))
      ++end;
    std::vector<size_t> result;
    result.reserve(end - begin);
    for (size_t candidate_id = begin; candidate_id < end; ++candidate_id)
      result.push_back(candidate_id);
    return result;
  }

  template <typename Visitor>
  void ForEachMatchingDomainSitePair(const std::vector<size_t> &src_sites,
                                     const std::vector<size_t> &dst_sites,
                                     const Visitor &visit) const {
    size_t src_index = 0;
    size_t dst_index = 0;
    while (src_index < src_sites.size() && dst_index < dst_sites.size()) {
      size_t src = src_sites[src_index];
      size_t dst = dst_sites[dst_index];
      int src_domain = sites_[src].EpochDomainId();
      int dst_domain = sites_[dst].EpochDomainId();
      if (src_domain < dst_domain) {
        ++src_index;
        continue;
      }
      if (src_domain > dst_domain) {
        ++dst_index;
        continue;
      }
      visit(src, dst);
      ++src_index;
      ++dst_index;
    }
  }

  void AddInsertionSites(
      IRStructure *path_begin, TaskNode *task,
      const std::vector<int> &domain_ids, SyncInsertionSiteList &sites,
      std::map<SyncInsertionSiteKey, SyncInsertionSite> &keyed_sites) const {
    for (bool at_beginning : {false, true}) {
      for (CoreMask core : kConcreteCores) {
        if (!HasCore(task->GetCoreMask(), core))
          continue;
        auto add = [&](int domain_id) {
          const EpochDomain &domain = domains_.Domain(domain_id);
          sites.emplace_back(path_begin, task, at_beginning, core, domain);
          const SyncInsertionSite &site = sites.back();
          SyncInsertionSiteKey key{path_begin, task, at_beginning, core,
                                   site.EpochDomainId()};
          auto [it, inserted] = keyed_sites.emplace(std::move(key), site);
          ICHECK(inserted || it->second == site)
              << "Synchronization insertion-site key has inconsistent sites";
        };
        for (int domain_id : domain_ids)
          add(domain_id);
      }
    }
  }

  void AddInsertionSiteIds(IRStructure *path_begin, TaskNode *task,
                           const std::vector<int> &domain_ids,
                           std::vector<size_t> &site_ids) const {
    for (bool at_beginning : {false, true}) {
      for (CoreMask core : kConcreteCores) {
        if (!HasCore(task->GetCoreMask(), core))
          continue;
        for (int domain_id : domain_ids) {
          site_ids.push_back(FindSiteId(SyncInsertionSiteKey{
              path_begin, task, at_beginning, core, domain_id}));
        }
      }
    }
  }

  const EpochDomainRegistry &domains_;
  SyncInsertionSiteList sites_;
  std::map<SyncInsertionSiteKey, size_t> site_ids_;
  std::vector<std::vector<size_t>> parent_site_ids_;
};

// ============================================================================
// Dependency analysis model and closure optimization
// ============================================================================

class DepClosure {
public:
  DepClosure() = default;

  explicit DepClosure(size_t num_sites)
      : outgoing_(num_sites), incoming_(num_sites) {}

  bool Contains(const DepEdge &edge) const { return present_.count(edge); }

  bool HasEdge(size_t src, size_t dst, int distance) const {
    return Contains(DepEdge(src, dst, distance, false)) ||
           Contains(DepEdge(src, dst, distance, true));
  }

  int MinDistance(size_t src, size_t dst) const {
    int result = std::numeric_limits<int>::max();
    if (src >= outgoing_.size())
      return result;
    for (const DepEdge &edge : outgoing_[src]) {
      if (edge.dst == dst)
        result = std::min(result, edge.distance);
    }
    return result;
  }

  void Insert(const DepEdge &edge) {
    if (!present_.insert(edge).second)
      return;
    ICHECK_LT(std::max(edge.src, edge.dst), outgoing_.size());
    edges_.push_back(edge);
    outgoing_[edge.src].push_back(edge);
    incoming_[edge.dst].push_back(edge);
  }

  template <typename Projector>
  void Saturate(const DepEdge &start, int max_distance,
                const Projector &project) {
    if (start.distance > max_distance)
      return;
    std::vector<DepEdge> work;
    std::set<DepEdge> pending;
    auto enqueue_one = [&](const DepEdge &edge) {
      if (edge.distance <= max_distance && !Contains(edge) &&
          pending.insert(edge).second) {
        work.push_back(edge);
      }
    };
    auto enqueue = [&](const DepEdge &edge) { project(edge, enqueue_one); };
    enqueue(start);
    while (!work.empty()) {
      DepEdge edge = work.back();
      work.pop_back();
      if (Contains(edge)) {
        pending.erase(edge);
        continue;
      }
      for (const DepEdge &other : outgoing_[edge.dst]) {
        int distance = edge.distance + other.distance;
        if (distance <= max_distance) {
          enqueue(DepEdge(edge.src, other.dst, distance,
                          edge.strict || other.strict));
        }
      }
      for (const DepEdge &other : incoming_[edge.src]) {
        int distance = other.distance + edge.distance;
        if (distance <= max_distance) {
          enqueue(DepEdge(other.src, edge.dst, distance,
                          other.strict || edge.strict));
        }
      }
      pending.erase(edge);
      Insert(edge);
    }
  }

  DepClosure DistanceZero() const {
    DepClosure result(outgoing_.size());
    for (const DepEdge &edge : edges_) {
      if (edge.distance == 0)
        result.Insert(edge);
    }
    return result;
  }

  const std::vector<DepEdge> &Edges() const { return edges_; }

private:
  std::vector<DepEdge> edges_;
  std::set<DepEdge> present_;
  std::vector<std::vector<DepEdge>> outgoing_;
  std::vector<std::vector<DepEdge>> incoming_;
};

struct SyncAnalysisResult {
  DepClosure closure;
  std::vector<VisibleSyncPoint> visible_sync_points;
};

// Check if any operation in stmt has unit_flag_ctl/ctrl != 0. This walks the
// whole statement because a T.PerCoreTask leaf may hold a compound body.
bool HasUnitFlagCtrl(const Stmt &stmt) {
  class Detector : public StmtExprVisitor {
  public:
    bool found{false};

  private:
    void VisitExpr_(const CallNode *op) final {
      if (auto ann = op->annotations.Get("unit_flag_ctrl")) {
        if (const auto *imm = ann->as<IntImmNode>()) {
          found = imm->value != 0;
        } else {
          found = true;
        }
      }
      if (!found)
        StmtExprVisitor::VisitExpr_(op);
    }
  } detector;
  detector(stmt);
  return detector.found;
}

std::string GetHardEvent(const std::string &producer_pipe,
                         const std::string &consumer_pipe) {
  ICHECK(producer_pipe != "UNKNOWN" && consumer_pipe != "UNKNOWN");
  std::string event = producer_pipe + "_" + consumer_pipe;
  if (event == "MTE2_M")
    return "MTE2_MTE1";
  if (event == "M_MTE2")
    return "MTE1_MTE2";
  if (event == "MTE3_M")
    return "MTE3_MTE1";
  if (event == "M_MTE3")
    return "MTE1_MTE3";
  return event;
}

// ============================================================================
// Recursive dependency analysis and reuse planning
// ============================================================================
class SyncAnalyzer {
public:
  explicit SyncAnalyzer(const MultiBufferPlan &multi_buffer_plan,
                        const EpochDomainRegistry &domains,
                        BufferVersionMap manual = {},
                        ConflictHintList root_conflicts = {})
      : multi_buffer_plan_(multi_buffer_plan), domains_(domains),
        manual_(std::move(manual)), root_conflicts_(std::move(root_conflicts)),
        site_registry_(domains) {
    for (const MultiBufferInfo &info : multi_buffer_plan_.Infos()) {
      ICHECK(!info.owners.empty());
      std::vector<ControlNode *> owners;
      owners.reserve(info.owners.size());
      for (const MultiBufferOwnerInfo &owner : info.owners)
        owners.push_back(owner.loop);
      auto [_, inserted] =
          multi_buffer_owners_.emplace(info.storage, std::move(owners));
      ICHECK(inserted);
    }
  }

  SyncAnalyzer(const SyncAnalyzer &) = delete;
  SyncAnalyzer &operator=(const SyncAnalyzer &) = delete;

  void Analyze(const std::vector<std::shared_ptr<IRStructure>> &root) {
    std::vector<IRStructure *> nodes;
    nodes.reserve(root.size());
    for (const std::shared_ptr<IRStructure> &child : root)
      nodes.push_back(child.get());

    site_registry_.Collect(nodes);
    AnalyzeNodeList(nodes, /*loop=*/nullptr);
  }

  const SyncSiteRegistry &SiteRegistry() const { return site_registry_; }

  std::vector<SyncPoint> &SyncPoints() { return sync_points_; }

  const std::set<std::pair<size_t, size_t>> &ReusablePairs() const {
    return reusable_pairs_;
  }

private:
  SyncAnalysisResult AnalyzeNodeList(const std::vector<IRStructure *> &nodes,
                                     ControlNode *loop) {
    SyncAnalysisResult result{DepClosure(site_registry_.Size()), {}};
    std::vector<size_t> task_site_ids = site_registry_.CollectSiteIds(nodes);

    std::vector<std::vector<VisibleSyncPoint>> child_sync_points;
    for (IRStructure *node : nodes) {
      SyncAnalysisResult child_result = AnalyzeNode(node);
      AddPromotedChildClosure(result.closure, node, child_result.closure, loop);
      child_sync_points.push_back(std::move(child_result.visible_sync_points));
    }

    std::vector<SyncPoint> local_sync_points = CollectSyncPoints(nodes, loop);
    int max_distance = loop == nullptr ? 0 : 1;
    for (const SyncPoint &sync_point : local_sync_points)
      max_distance = std::max(max_distance, sync_point.edge.distance);

    AddSamePipeEdges(result.closure, task_site_ids, max_distance, loop);
    OptimizeSyncPoints(local_sync_points, result.closure, max_distance, loop);
    if (loop != nullptr)
      OptimizeFlagVersions(local_sync_points, result.closure, loop);

    std::vector<VisibleSyncPoint> local_points;
    local_points.reserve(local_sync_points.size());
    for (SyncPoint &sync_point : local_sync_points) {
      size_t sync_point_id = AppendSyncPoint(std::move(sync_point));
      local_points.push_back(
          {sync_point_id, {sync_points_[sync_point_id].edge}});
    }

    for (size_t i = 0; i < child_sync_points.size(); ++i) {
      for (size_t j = i + 1; j < child_sync_points.size(); ++j) {
        RecordReusablePairs(
            child_sync_points[i], child_sync_points[j], result.closure,
            /*allow_cross_iter=*/true, /*allow_mixed_iter=*/false);
      }
      RecordReusablePairs(child_sync_points[i], local_points, result.closure,
                          /*allow_cross_iter=*/false,
                          /*allow_mixed_iter=*/false);
      result.visible_sync_points.insert(result.visible_sync_points.end(),
                                        child_sync_points[i].begin(),
                                        child_sync_points[i].end());
    }

    // Mixed same/cross-iteration reuse is checked only at the collection level
    // where both points still have the same ring context.
    RecordReusablePairs(local_points, local_points, result.closure,
                        /*allow_cross_iter=*/false, /*allow_mixed_iter=*/true);
    result.visible_sync_points.insert(result.visible_sync_points.end(),
                                      local_points.begin(), local_points.end());
    if (loop != nullptr)
      loop_closures_.insert_or_assign(loop, result.closure);
    result.closure = result.closure.DistanceZero();
    return result;
  }

  SyncAnalysisResult AnalyzeNode(IRStructure *node) {
    if (node == nullptr)
      return {DepClosure(site_registry_.Size()), {}};

    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node);
      if (control->children.empty() || !control->control.defined())
        return {DepClosure(site_registry_.Size()), {}};

      std::vector<IRStructure *> nodes;
      for (const std::shared_ptr<IRStructure> &child : control->children)
        nodes.push_back(child.get());
      SyncAnalysisResult result = AnalyzeNodeList(nodes, control);

      if (!control->BodyHasLoopBreak()) {
        std::vector<VisibleSyncPoint> promoted_sync_points;
        for (VisibleSyncPoint &visible : result.visible_sync_points) {
          if (PromoteReuseSites(visible))
            promoted_sync_points.push_back(std::move(visible));
        }
        result.visible_sync_points = std::move(promoted_sync_points);
      } else {
        result.visible_sync_points.clear();
      }
      return result;
    }

    if (node->IsTask())
      return {DepClosure(site_registry_.Size()), {}};

    LOG(FATAL) << "Unknown IRStructure kind in SyncAnalyzer";
    return {DepClosure(site_registry_.Size()), {}};
  }

  size_t CounterAccessSite(const MultiBufferInfo &info,
                           const MultiBufferOwnerInfo &owner, TaskNode *task,
                           CoreMask core, bool at_beginning) const {
    int domain_id = domains_.DomainForStorage(info.storage, owner.loop);
    return site_registry_.FindSiteId(
        SyncInsertionSiteKey{task->GetChildOnPathFrom(owner.loop), task,
                             at_beginning, core, domain_id});
  }

  size_t AppendSyncPoint(SyncPoint sync_point) {
    if (sync_point.counter_group_id > 0) {
      // Local flag-version optimization has finished, so the channel key is
      // stable for both parent-level coverage and final flag allocation.
      CounterSyncSignature signature =
          site_registry_.GetCounterSyncSignature(sync_point);
      int ordinal =
          counter_channel_occurrences_[sync_point.flag_loop][signature]++;
      sync_point.counter_channel.emplace(std::move(signature), ordinal);
    }
    size_t index = sync_points_.size();
    sync_points_.push_back(std::move(sync_point));
    return index;
  }

  bool IsFullRingCounterSync(const SyncPoint &sync_point,
                             const MultiBufferInfo &info,
                             ControlNode *owner) const {
    if (!sync_point.counter_channel.has_value())
      return false;

    // AppendSyncPoint assigns a counter channel only after validating the
    // endpoints against their counter domain. Reassert that relation here
    // because sibling-owner coverage uses flag_loop as the endpoint owner.
    const SyncInsertionSite &src = site_registry_.Src(sync_point);
    const SyncInsertionSite &dst = site_registry_.Dst(sync_point);
    ICHECK_GT(sync_point.counter_group_id, 0);
    ICHECK_EQ(src.EpochDomainId(), dst.EpochDomainId());
    const EpochDomain &domain = site_registry_.Domain(src);
    ICHECK(domain.IsCounter());
    ICHECK_EQ(domain.owner, sync_point.flag_loop);

    return sync_point.flag_loop == owner &&
           sync_point.counter_group_id == info.counter_group_id &&
           sync_point.edge.distance == info.num_versions &&
           sync_point.num_versions == info.num_versions;
  }

  bool IsCounterDependencyCovered(const MultiBufferInfo &info,
                                  TaskNode *producer, CoreMask producer_core,
                                  TaskNode *consumer, CoreMask consumer_core,
                                  ControlNode *analysis_loop) const {
    ICHECK(info.UsesCounter());
    const MultiBufferOwnerInfo *producer_owner = info.FindOwner(producer);
    const MultiBufferOwnerInfo *consumer_owner = info.FindOwner(consumer);
    if (producer_owner == nullptr || consumer_owner == nullptr ||
        producer_owner == consumer_owner) {
      return false;
    }
    IRStructure *common_ancestor =
        producer_owner->loop->GetLowestCommonAncestor(consumer_owner->loop);
    if (common_ancestor == producer_owner->loop ||
        common_ancestor == consumer_owner->loop) {
      LOG(FATAL) << "Cannot resolve dependencies between nested multi-buffer "
                    "owners";
    }
    if (producer_owner->loop->BodyHasLoopBreak() ||
        consumer_owner->loop->BodyHasLoopBreak() ||
        common_ancestor != analysis_loop) {
      return false;
    }

    size_t producer_access = CounterAccessSite(
        info, *producer_owner, producer, producer_core, /*at_beginning=*/false);
    size_t consumer_access = CounterAccessSite(
        info, *consumer_owner, consumer, consumer_core, /*at_beginning=*/true);
    auto ordered = [&](ControlNode *owner, size_t src, size_t dst) {
      if (src == dst)
        return true;
      auto it = loop_closures_.find(owner);
      ICHECK(it != loop_closures_.end())
          << "Counter owner closure was not recorded";
      return it->second.HasEdge(src, dst, 0);
    };

    for (const SyncPoint &producer_sync : sync_points_) {
      if (!IsFullRingCounterSync(producer_sync, info, producer_owner->loop) ||
          !ordered(producer_owner->loop, producer_access,
                   producer_sync.edge.src)) {
        continue;
      }

      for (const SyncPoint &consumer_sync : sync_points_) {
        if (!IsFullRingCounterSync(consumer_sync, info, consumer_owner->loop) ||
            consumer_sync.GetCounterChannel() !=
                producer_sync.GetCounterChannel() ||
            !ordered(consumer_owner->loop, consumer_sync.edge.dst,
                     consumer_access)) {
          continue;
        }
        return true;
      }
    }
    return false;
  }

  int FindSharedDependencyDomain(TaskNode *producer, TaskNode *consumer,
                                 ControlNode *loop) const {
    std::vector<int> producer_domains =
        domains_.DomainsForTaskAtScope(producer, loop);
    std::vector<int> consumer_domains =
        domains_.DomainsForTaskAtScope(consumer, loop);
    int lexical_domain = domains_.UnconditionalLexicalDomain(loop);
    std::vector<int> candidates = producer_domains;
    candidates.insert(candidates.end(), consumer_domains.begin(),
                      consumer_domains.end());
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());

    // A guarded source domain is available at both endpoints only when both
    // tasks already use it. The unconditional lexical domain needs no guard
    // value and is therefore always a safe common fallback.
    auto has_site = [&](const std::vector<int> &task_domains, int candidate) {
      return candidate == lexical_domain ||
             std::find(task_domains.begin(), task_domains.end(), candidate) !=
                 task_domains.end();
    };
    auto projects_to_task = [&](int candidate,
                                const std::vector<int> &task_domains) {
      if (task_domains.empty())
        return candidate == lexical_domain;
      return std::any_of(
          task_domains.begin(), task_domains.end(), [&](int target) {
            return domains_.CanProjectDomain(candidate, target, loop);
          });
    };
    for (int candidate : candidates) {
      if (has_site(producer_domains, candidate) &&
          has_site(consumer_domains, candidate) &&
          projects_to_task(candidate, producer_domains) &&
          projects_to_task(candidate, consumer_domains)) {
        return candidate;
      }
    }
    ICHECK(projects_to_task(lexical_domain, producer_domains) &&
           projects_to_task(lexical_domain, consumer_domains))
        << "The unconditional lexical domain must project to every task at "
           "its scope";
    return lexical_domain;
  }

  std::vector<SyncPoint>
  CollectSyncPoints(const std::vector<IRStructure *> &nodes,
                    ControlNode *loop) {
    std::vector<DepInfo> deps =
        AnalyzeDependencies(nodes, loop, manual_, multi_buffer_owners_,
                            &dependency_cache_, root_conflicts_);

    auto expand_cores = [](CoreMask task_mask) {
      std::vector<CoreMask> result;
      for (CoreMask core : kConcreteCores) {
        if (HasCore(task_mask, core))
          result.push_back(core);
      }
      return result;
    };

    std::vector<SyncPoint> sync_points;
    for (const DepInfo &dep : deps) {
      if (dep.storage.has_value()) {
        std::string storage_scope = GetPtrStorageScope(dep.storage.value());
        if (storage_scope == "local" || storage_scope == "local.var" ||
            storage_scope == "local.fragment") {
          continue;
        }
      }

      int distance = dep.distance;
      int num_versions = 1;
      ControlNode *flag_loop = nullptr;
      const MultiBufferInfo *counter_info = nullptr;
      const MultiBufferOwnerInfo *counter_owner = nullptr;

      const MultiBufferInfo *info =
          dep.storage.has_value() ? multi_buffer_plan_.Find(dep.storage.value())
                                  : nullptr;
      int manual_versions = 0;
      if (dep.storage.has_value()) {
        if (auto versions = manual_.Get(dep.storage.value()))
          manual_versions = versions.value();
      }
      const MultiBufferOwnerInfo *enclosing_owner = nullptr;
      if (info != nullptr && loop != nullptr) {
        if (!info->UsesCounter())
          ICHECK_EQ(info->owners.size(), 1U);
        enclosing_owner = info->FindOwner(loop);
      }
      bool within_owner = enclosing_owner != nullptr;
      if (within_owner) {
        // Every dependency in an automatic ring owner's subtree uses that
        // owner's clock so descendant closure edges compose with owner-level
        // edges. Dependencies outside all owner subtrees remain ordinary
        // one-slot edges; sibling-owner obligations may instead be discharged
        // by matching complete counter protocols in both owner-local closures.
        num_versions = info->num_versions;
        flag_loop = enclosing_owner->loop;
        if (info->UsesCounter()) {
          counter_info = info;
          counter_owner = enclosing_owner;
        }
        // Only an unresolved dependency carried by the owner loop spans the
        // full physical ring. A descendant loop still carries by one of its
        // own iterations.
        if (loop == enclosing_owner->loop && distance < 0)
          distance = info->num_versions;
      } else if (info == nullptr) {
        if (manual_versions) {
          num_versions = dep.distance > 0 ? dep.distance : manual_versions;
          flag_loop = loop;
        }
      }
      if (distance < 0)
        distance = 1;
      ICHECK_EQ(counter_info != nullptr, counter_owner != nullptr);
      std::optional<int> dependency_domain_id;
      if (dep.storage.has_value()) {
        dependency_domain_id =
            domains_.DomainForStorage(dep.storage.value(), loop);
      }
      if (counter_info != nullptr) {
        ICHECK(dependency_domain_id.has_value());
        ICHECK_EQ(dependency_domain_id.value(),
                  domains_.DomainForStorage(counter_info->storage,
                                            counter_owner->loop));
      }
      if (info == nullptr && !manual_versions && loop != nullptr) {
        flag_loop = loop;
      }

      for (const auto &[producer, consumer] : dep.task_pairs) {
        int pair_domain_id =
            dependency_domain_id.has_value()
                ? dependency_domain_id.value()
                : FindSharedDependencyDomain(producer, consumer, loop);
        std::string producer_pipe =
            GetResourcePipeName(producer->GetPipeMask());
        std::string consumer_pipe =
            GetResourcePipeName(consumer->GetPipeMask());
        std::string hard_event = GetHardEvent(producer_pipe, consumer_pipe);
        bool fix_unit_flag = hard_event.find("FIX") != std::string::npos &&
                             (HasUnitFlagCtrl(producer->stmt) ||
                              HasUnitFlagCtrl(consumer->stmt));
        for (CoreMask producer_core : expand_cores(producer->GetCoreMask())) {
          for (CoreMask consumer_core : expand_cores(consumer->GetCoreMask())) {
            if (producer_core == consumer_core && fix_unit_flag)
              continue;
            if (info != nullptr && info->UsesCounter() && !within_owner &&
                IsCounterDependencyCovered(*info, producer, producer_core,
                                           consumer, consumer_core, loop)) {
              continue;
            }
            size_t producer_site = site_registry_.FindSiteId(
                SyncInsertionSiteKey{dep.prod_node, producer, false,
                                     producer_core, pair_domain_id});
            size_t consumer_site = site_registry_.FindSiteId(
                SyncInsertionSiteKey{dep.cons_node, consumer, true,
                                     consumer_core, pair_domain_id});
            DepEdge edge(producer_site, consumer_site, distance, true);
            if (counter_info != nullptr) {
              PrimExpr iteration = BufferLoad(counter_info->counter,
                                              {IntImm(DataType::Int(32), 0)});
              sync_points.emplace_back(std::move(edge), num_versions, iteration,
                                       counter_owner->loop,
                                       counter_info->counter_group_id);
            } else {
              sync_points.emplace_back(std::move(edge), num_versions,
                                       flag_loop);
            }
          }
        }
      }
    }
    return sync_points;
  }

  void AddPromotedChildClosure(DepClosure &out, IRStructure *child,
                               const DepClosure &child_closure,
                               ControlNode *projection_scope) const {
    if (child->IsControl() &&
        static_cast<ControlNode *>(child)->BodyHasLoopBreak()) {
      return;
    }

    for (const DepEdge &edge : child_closure.Edges()) {
      // A lexical domain ends at its loop. Re-express the same physical
      // endpoints in each compatible parent domain. A wider parent guard
      // cannot inherit this proof: an inactive child endpoint could otherwise
      // become a transitive bridge between unrelated parent-only accesses.
      site_registry_.ForEachParentSitePair(edge, [&](const DepEdge &parent) {
        if (site_registry_.Src(parent).path_end !=
                site_registry_.Src(edge).path_end ||
            site_registry_.Dst(parent).path_end !=
                site_registry_.Dst(edge).path_end) {
          return;
        }
        int source_domain = site_registry_.Src(edge).EpochDomainId();
        int parent_domain = site_registry_.Src(parent).EpochDomainId();
        if (!domains_.CanProjectDomain(source_domain, parent_domain,
                                       projection_scope)) {
          return;
        }
        SaturateDependencyEdge(out, parent,
                               /*max_distance=*/0, projection_scope);
      });
    }
  }

  void AddSamePipeEdges(DepClosure &closure,
                        const std::vector<size_t> &task_sites, int max_distance,
                        ControlNode *projection_scope) const {
    const SyncInsertionSiteList &sites = site_registry_.Sites();
    using PipeDomainKey = std::tuple<CoreMask, std::string, int>;
    std::map<PipeDomainKey, std::vector<size_t>> by_pipe;
    for (size_t site_id : task_sites) {
      const SyncInsertionSite &site = sites[site_id];
      by_pipe[{site.core, site.pipe, site.EpochDomainId()}].push_back(site_id);
    }

    auto level_stage = [](const SyncInsertionSite &site) {
      return site.path_begin->GetStage();
    };
    for (auto &[_, pipe_sites] : by_pipe) {
      // A single pipe issues sites in physical (iteration, position) order.
      // For a(k) before b(k+d), the minimum valid distance is
      // stage(a) - stage(b), plus one when a is physically after b.
      for (size_t lhs_id : pipe_sites) {
        for (size_t rhs_id : pipe_sites) {
          if (lhs_id == rhs_id)
            continue;
          const SyncInsertionSite &lhs = sites[lhs_id];
          const SyncInsertionSite &rhs = sites[rhs_id];
          bool lhs_before_rhs = lhs_id < rhs_id;
          int min_distance =
              level_stage(lhs) - level_stage(rhs) + (lhs_before_rhs ? 0 : 1);
          min_distance = std::max(min_distance, 0);
          for (int distance = min_distance; distance <= max_distance;
               ++distance) {
            SaturateDependencyEdge(closure,
                                   DepEdge(lhs_id, rhs_id, distance, false),
                                   max_distance, projection_scope);
          }
        }
      }
    }
  }

  void SaturateDependencyEdge(DepClosure &closure, const DepEdge &edge,
                              int max_distance,
                              ControlNode *projection_scope) const {
    closure.Saturate(
        edge, max_distance, [&](const DepEdge &candidate, const auto &visit) {
          if (candidate.distance != 0) {
            visit(candidate);
            return;
          }
          int source_domain = site_registry_.Src(candidate).EpochDomainId();
          site_registry_.ForEachPhysicalSitePair(
              candidate, [&](const DepEdge &projected) {
                int target_domain =
                    site_registry_.Src(projected).EpochDomainId();
                if (domains_.CanProjectDomain(source_domain, target_domain,
                                              projection_scope)) {
                  visit(projected);
                }
              });
        });
  }

  void OptimizeSyncPoints(std::vector<SyncPoint> &sync_points,
                          DepClosure &closure, int max_distance,
                          ControlNode *projection_scope) const {
    std::stable_sort(sync_points.begin(), sync_points.end());
    std::vector<SyncPoint> emit;
    for (SyncPoint &sync_point : sync_points) {
      if (closure.Contains(sync_point.edge))
        continue;
      SaturateDependencyEdge(closure, sync_point.edge, max_distance,
                             projection_scope);
      emit.push_back(std::move(sync_point));
    }
    sync_points = std::move(emit);
  }

  void OptimizeFlagVersions(std::vector<SyncPoint> &sync_points,
                            const DepClosure &closure,
                            ControlNode *loop) const {
    for (SyncPoint &sync_point : sync_points) {
      const SyncInsertionSite &src = site_registry_.Src(sync_point);
      const SyncInsertionSite &dst = site_registry_.Dst(sync_point);
      if (src.pipe == dst.pipe)
        continue;
      bool is_innermost = loop == sync_point.flag_loop;
      int innermost_num_versions = is_innermost ? sync_point.num_versions : 1;
      int min_distance =
          closure.MinDistance(sync_point.edge.dst, sync_point.edge.src);

      // If the loop trip count is no larger than the ring, no version is
      // reused.
      if (loop != nullptr) {
        const ForNode *inner = loop->control.get();
        PrimExpr step = inner->step.has_value()
                            ? inner->step.value()
                            : IntImm(inner->extent.dtype(), 1);
        // Bound only this loop's trip count; outer context may help prove its
        // symbolic extent without flattening any enclosing iterations.
        if (loop->task->outer_ctx.CanProve(
                inner->extent <=
                IntImm(inner->extent.dtype(), innermost_num_versions) * step)) {
          continue;
        }
      }
      if (sync_point.edge.distance > 0) {
        if (sync_point.edge.distance != innermost_num_versions ||
            min_distance > 0) {
          LOG(WARNING) << "cross-iteration flag " << src.pipe << "->"
                       << dst.pipe
                       << ": the synchronization ring is malformed.";
        }
        continue;
      }
      if (min_distance > innermost_num_versions) {
        LOG(WARNING) << "flag " << src.pipe << "->" << dst.pipe
                     << ": reverse ordering too loose to bound flag reuse.";
        continue;
      }
      if (min_distance <= 0) {
        LOG(WARNING) << "flag " << src.pipe << "->" << dst.pipe
                     << ": dependency cycle (reverse distance <= 0).";
        continue;
      }
      if (is_innermost)
        sync_point.num_versions = min_distance;
    }
  }

  bool CanReuse(const VisibleSyncPoint &lhs, const VisibleSyncPoint &rhs,
                const DepClosure &closure, bool allow_cross_iter,
                bool allow_mixed_iter) const {
    const SyncPoint &lhs_sync = sync_points_[lhs.sync_point_id];
    const SyncPoint &rhs_sync = sync_points_[rhs.sync_point_id];
    if (!site_registry_.UsesFlag(lhs_sync) ||
        !site_registry_.UsesFlag(rhs_sync) ||
        !site_registry_.HasCompatibleFlagKind(lhs_sync, rhs_sync)) {
      return false;
    }

    auto by_domain = [&](const VisibleSyncPoint &visible) {
      std::map<int, DepEdge> result;
      for (const DepEdge &edge : visible.projected_edges) {
        int src_domain = site_registry_.Src(edge).EpochDomainId();
        int dst_domain = site_registry_.Dst(edge).EpochDomainId();
        ICHECK_EQ(src_domain, dst_domain);
        auto [_, inserted] = result.emplace(src_domain, edge);
        ICHECK(inserted)
            << "A visible synchronization point has duplicate projections in "
               "one epoch domain";
      }
      return result;
    };
    std::map<int, DepEdge> lhs_edges = by_domain(lhs);
    std::map<int, DepEdge> rhs_edges = by_domain(rhs);
    if (lhs_edges.empty() || lhs_edges.size() != rhs_edges.size())
      return false;

    bool lhs_cross_iter = lhs_sync.edge.distance > 0;
    bool rhs_cross_iter = rhs_sync.edge.distance > 0;
    if (lhs_cross_iter != rhs_cross_iter) {
      // A mixed same/cross-iteration handshake alternates on one flag ring.
      if (!allow_mixed_iter || !site_registry_.IsCrossCore(lhs_sync) ||
          !site_registry_.IsCrossCore(rhs_sync) ||
          lhs_sync.num_versions != rhs_sync.num_versions ||
          lhs_sync.flag_loop != rhs_sync.flag_loop) {
        return false;
      }
      for (const auto &[domain_id, lhs_edge] : lhs_edges) {
        auto rhs_it = rhs_edges.find(domain_id);
        if (rhs_it == rhs_edges.end())
          return false;
        const DepEdge &rhs_edge = rhs_it->second;
        const DepEdge &cross_edge = lhs_cross_iter ? lhs_edge : rhs_edge;
        const DepEdge &same_edge = lhs_cross_iter ? rhs_edge : lhs_edge;
        if (!closure.HasEdge(cross_edge.dst, same_edge.src, 0) ||
            !closure.HasEdge(same_edge.dst, cross_edge.src, 0)) {
          return false;
        }
      }
      return true;
    }

    std::optional<bool> lhs_before;
    for (const auto &[domain_id, lhs_edge] : lhs_edges) {
      auto rhs_it = rhs_edges.find(domain_id);
      if (rhs_it == rhs_edges.end())
        return false;
      const DepEdge &rhs_edge = rhs_it->second;
      bool current_lhs_before;
      bool current_rhs_before;
      if (!lhs_cross_iter) {
        current_lhs_before = closure.HasEdge(lhs_edge.dst, rhs_edge.src, 0) &&
                             closure.HasEdge(rhs_edge.dst, lhs_edge.src, 1);
        current_rhs_before = closure.HasEdge(rhs_edge.dst, lhs_edge.src, 0) &&
                             closure.HasEdge(lhs_edge.dst, rhs_edge.src, 1);
      } else {
        if (!allow_cross_iter)
          return false;
        current_lhs_before = closure.HasEdge(lhs_edge.src, rhs_edge.dst, 0) &&
                             closure.HasEdge(rhs_edge.src, lhs_edge.dst, 1);
        current_rhs_before = closure.HasEdge(rhs_edge.src, lhs_edge.dst, 0) &&
                             closure.HasEdge(lhs_edge.src, rhs_edge.dst, 1);
      }
      if (current_lhs_before == current_rhs_before)
        return false;
      if (lhs_before.has_value() && lhs_before.value() != current_lhs_before) {
        return false;
      }
      lhs_before = current_lhs_before;
    }
    return lhs_before.has_value();
  }

  bool PromoteReuseSites(VisibleSyncPoint &visible) const {
    const SyncPoint &sync_point = sync_points_[visible.sync_point_id];
    bool cross_iter = sync_point.edge.distance > 0;
    std::set<DepEdge> promoted;
    for (const DepEdge &edge : visible.projected_edges) {
      site_registry_.ForEachParentSitePair(edge, [&](const DepEdge &parent) {
        size_t src = parent.src;
        size_t dst = parent.dst;
        if (site_registry_.Src(parent).path_end !=
            site_registry_.Src(edge).path_end) {
          src = site_registry_.FindBoundarySite(src, !cross_iter);
        }
        if (site_registry_.Dst(parent).path_end !=
            site_registry_.Dst(edge).path_end) {
          dst = site_registry_.FindBoundarySite(dst, cross_iter);
        }
        promoted.emplace(src, dst, edge.distance, edge.strict);
      });
    }
    if (promoted.empty())
      return false;
    visible.projected_edges.assign(promoted.begin(), promoted.end());
    return true;
  }

  void RecordReusablePairs(const std::vector<VisibleSyncPoint> &lhs_points,
                           const std::vector<VisibleSyncPoint> &rhs_points,
                           const DepClosure &closure, bool allow_cross_iter,
                           bool allow_mixed_iter) {
    for (const VisibleSyncPoint &lhs_visible : lhs_points) {
      for (const VisibleSyncPoint &rhs_visible : rhs_points) {
        size_t lhs = lhs_visible.sync_point_id;
        size_t rhs = rhs_visible.sync_point_id;
        if (lhs == rhs || !CanReuse(lhs_visible, rhs_visible, closure,
                                    allow_cross_iter, allow_mixed_iter)) {
          continue;
        }
        reusable_pairs_.emplace(std::min(lhs, rhs), std::max(lhs, rhs));
      }
    }
  }

  const MultiBufferPlan &multi_buffer_plan_;
  const EpochDomainRegistry &domains_;
  BufferVersionMap manual_;
  MultiBufferOwnerMap multi_buffer_owners_;
  ConflictHintList root_conflicts_;
  SyncSiteRegistry site_registry_;
  std::vector<SyncPoint> sync_points_;
  std::set<std::pair<size_t, size_t>> reusable_pairs_;
  DependencyCache dependency_cache_;
  std::map<ControlNode *, DepClosure> loop_closures_;
  std::unordered_map<ControlNode *, std::map<CounterSyncSignature, int>>
      counter_channel_occurrences_;
};

// ============================================================================
// Flag data model, explicit reservation, and physical allocation
// ============================================================================
struct FlagInfo {
  CoreMask core_mask;
  PrimExpr flag_var;
  std::string hardevent;
  int mode;
};

using FlagInfoList = std::vector<FlagInfo>;

struct CrossCoreFlagReservation {
  // Inclusive ranges of flag IDs used explicitly by the source kernel, for
  // every mode. Dynamic expressions contribute the ConstIntBound inferred
  // from their enclosing loops.
  std::vector<std::pair<int64_t, int64_t>> explicit_id_ranges;
  bool has_unbounded_id{false};
};

struct FlagAllocation {
  std::vector<size_t> sync_points;
  int num_versions;
  bool cross_iter;
  int base_id;

  FlagAllocation(size_t sync_point, int num_versions, bool cross_iter,
                 int base_id)
      : sync_points{sync_point}, num_versions(num_versions),
        cross_iter(cross_iter), base_id(base_id) {}
};

// Collect every CrossCore flag range used explicitly by the source kernel.
// Track enclosing loops, flat Bind declarations, branch guards, and
// assumptions so ConstIntBound can resolve dynamic flag expressions. Mode
// semantics are user-controlled; allocation conservatively reserves the ID
// range regardless of mode.
class ExplicitCrossCoreFlagCollector : public TaskAwareConstrVisitor {
public:
  CrossCoreFlagReservation Collect(const Stmt &stmt) {
    operator()(stmt);
    return std::move(reservation_);
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::ascend_cross_core_set_flag()) ||
        op->op.same_as(tl::ascend_cross_core_wait_flag())) {
      ICHECK_EQ(op->args.size(), 3)
          << "ascend_cross_core_set_flag/wait_flag expects exactly 3 "
             "arguments";

      const PrimExpr &flag_id = op->args[2];
      arith::Analyzer analyzer;
      GetConstrSet().Populate(analyzer);
      arith::ConstIntBound bound = analyzer.const_int_bound(flag_id);
      arith::Analyzer unconstrained_analyzer;
      Var unconstrained_flag_id("unconstrained_cross_core_flag_id",
                                flag_id.dtype());
      arith::ConstIntBound unconstrained_bound =
          unconstrained_analyzer.const_int_bound(unconstrained_flag_id);
      if (bound->min_value == arith::ConstIntBound::kNegInf ||
          bound->max_value == arith::ConstIntBound::kPosInf ||
          (bound->min_value == unconstrained_bound->min_value &&
           bound->max_value == unconstrained_bound->max_value)) {
        reservation_.has_unbounded_id = true;
      } else {
        ICHECK_LE(bound->min_value, bound->max_value)
            << "Ascend AutoSchedule inferred an invalid explicit cross-core "
               "flag range ["
            << bound->min_value << ", " << bound->max_value << "]";
        reservation_.explicit_id_ranges.emplace_back(bound->min_value,
                                                     bound->max_value);
      }
    }
    ConstrVisitor::VisitExpr_(op);
  }

  CrossCoreFlagReservation reservation_;
};

CrossCoreFlagReservation
CollectExplicitCrossCoreFlags(const Stmt &kernel_body) {
  ExplicitCrossCoreFlagCollector collector;
  return collector.Collect(kernel_body);
}

class FlagAllocator {
public:
  FlagAllocator(const SyncSiteRegistry &site_registry,
                std::vector<SyncPoint> &sync_points,
                const std::set<std::pair<size_t, size_t>> &reusable_pairs,
                const CrossCoreFlagReservation &cross_core_flag_reservation,
                int num_aiv_subcores)
      : site_registry_(site_registry), sync_points_(sync_points),
        reusable_pairs_(reusable_pairs),
        cross_core_flag_reservation_(cross_core_flag_reservation),
        num_aiv_subcores_(num_aiv_subcores) {}

  void Plan() {
    allocations_.clear();
    flag_infos_.clear();
    GroupCounterChannels();

    std::map<std::string, int> required_ids;
    std::map<std::string, int> next_intra_core_id;
    bool needs_cross_core_flag = false;
    for (size_t index = 0; index < sync_points_.size(); ++index) {
      const SyncPoint &sync_point = sync_points_[index];
      if (!site_registry_.UsesFlag(sync_point))
        continue;
      if (sync_point.counter_channel.has_value() &&
          counter_channel_representative_[index] != index) {
        continue;
      }
      std::string domain = FlagDomain(sync_point);
      required_ids[domain] += sync_point.num_versions;
      needs_cross_core_flag |= site_registry_.IsCrossCore(sync_point);
    }

    ICHECK(!needs_cross_core_flag ||
           !cross_core_flag_reservation_.has_unbounded_id)
        << "Ascend AutoSchedule cannot safely allocate cross-core flag IDs "
           "when an explicit CrossCoreSetFlag/WaitFlag uses a flag_id whose "
           "integer range cannot be bounded; constrain the flag_id with a "
           "constant-range loop or disable AutoSchedule";

    // All cross-core modes share the same 16 base slots. The Cube side may
    // additionally address slot + 16 for the second Vector core.
    std::array<bool, kCrossCoreFlagLimit> cross_core_slots{};
    for (const auto &[lo, hi] :
         cross_core_flag_reservation_.explicit_id_ranges) {
      ICHECK_LE(lo, hi);
      uint64_t span = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
      if (span >= static_cast<uint64_t>(kCrossCoreFlagLimit - 1)) {
        cross_core_slots.fill(true);
        continue;
      }
      for (uint64_t offset = 0; offset <= span; ++offset) {
        int64_t explicit_id = lo + static_cast<int64_t>(offset);
        int normalized_id = static_cast<int>(explicit_id % kCrossCoreFlagLimit);
        if (normalized_id < 0)
          normalized_id += kCrossCoreFlagLimit;
        cross_core_slots[normalized_id] = true;
      }
    }

    int explicit_cross_core_slots =
        std::count(cross_core_slots.begin(), cross_core_slots.end(), true);
    auto find_cross_core_range = [&](int width) {
      for (int base = 0; base + width <= kCrossCoreFlagLimit; ++base) {
        bool available = true;
        for (int offset = 0; offset < width; ++offset)
          available &= !cross_core_slots[base + offset];
        if (available)
          return base;
      }
      return -1;
    };

    for (bool cross_iter : {true, false}) {
      std::vector<size_t> candidates;
      for (size_t i = 0; i < sync_points_.size(); ++i) {
        if (site_registry_.UsesFlag(sync_points_[i]) &&
            (sync_points_[i].edge.distance > 0) == cross_iter) {
          candidates.push_back(i);
        }
      }
      std::stable_sort(candidates.begin(), candidates.end(),
                       [&](size_t lhs, size_t rhs) {
                         return sync_points_[lhs].num_versions >
                                sync_points_[rhs].num_versions;
                       });

      for (size_t index : candidates) {
        SyncPoint &sync_point = sync_points_[index];
        if (sync_point.counter_channel.has_value()) {
          size_t representative = counter_channel_representative_[index];
          if (representative != index) {
            size_t allocation_index =
                sync_points_[representative].allocation_index;
            ICHECK_LT(allocation_index, allocations_.size())
                << "Counter channel representative was not allocated first";
            allocations_[allocation_index].sync_points.push_back(index);
            sync_point.allocation_index = allocation_index;
            continue;
          }
        }
        bool compatible = false;
        std::string domain = FlagDomain(sync_point);
        bool cross_core = site_registry_.IsCrossCore(sync_point);
        int limit = cross_core ? kCrossCoreFlagLimit : kIntraCoreFlagLimit;
        int cross_core_base =
            cross_core ? find_cross_core_range(sync_point.num_versions) : -1;
        bool needs_reuse =
            cross_core ? required_ids[domain] + explicit_cross_core_slots >
                                 kCrossCoreFlagLimit ||
                             cross_core_base < 0
                       : required_ids[domain] > limit;

        // CanReuse proves the lifetime of individual sync points. This proof
        // covers the whole counter channel only when it has one owner.
        if (needs_reuse && !IsMultiOwnerCounterChannel(index)) {
          for (size_t allocation_index = 0;
               allocation_index < allocations_.size(); ++allocation_index) {
            FlagAllocation &allocation = allocations_[allocation_index];
            const SyncPoint &representative =
                sync_points_[allocation.sync_points.front()];
            if (!site_registry_.HasCompatibleFlagKind(sync_point,
                                                      representative)) {
              continue;
            }
            if (allocation.cross_iter != cross_iter &&
                (!cross_core || !site_registry_.IsCrossCore(representative))) {
              continue;
            }

            compatible = true;
            for (size_t member : allocation.sync_points) {
              if (IsMultiOwnerCounterChannel(member) ||
                  !reusable_pairs_.count(
                      {std::min(index, member), std::max(index, member)})) {
                compatible = false;
                break;
              }
            }
            if (compatible) {
              ICHECK_GE(allocation.num_versions, sync_point.num_versions);
              required_ids[domain] -= sync_point.num_versions;
              allocation.sync_points.push_back(index);
              sync_point.allocation_index = allocation_index;
              break;
            }
          }
        }
        if (compatible)
          continue;

        sync_point.allocation_index = allocations_.size();
        int base_id = 0;
        if (cross_core) {
          ICHECK_GE(cross_core_base, 0)
              << "Ascend AutoSchedule cannot allocate "
              << sync_point.num_versions
              << " contiguous cross-core flag slots without overlapping "
                 "explicit cross-core flags or safely reusable auto-scheduled "
                 "flags";
          base_id = cross_core_base;
          for (int offset = 0; offset < sync_point.num_versions; ++offset)
            cross_core_slots[base_id + offset] = true;
        } else {
          base_id = next_intra_core_id[domain];
        }
        allocations_.emplace_back(index, sync_point.num_versions, cross_iter,
                                  base_id);
        if (!cross_core)
          next_intra_core_id[domain] += sync_point.num_versions;
      }
    }

    for (const auto &[domain, total] : next_intra_core_id) {
      if (total <= kIntraCoreFlagLimit)
        continue;
      LOG(WARNING) << "Ascend auto-schedule allocated " << total
                   << " flag ids for " << domain << " after safe reuse, "
                   << "exceeding the dav-3510 limit of " << kIntraCoreFlagLimit
                   << "; the later RewriteFlagToBuf pass will spill the excess "
                      "intra-core flags to get_buf/rls_buf.";
    }

    for (const FlagAllocation &allocation : allocations_) {
      if (allocation.cross_iter)
        AppendCrossIterFlagInfo(allocation);
    }

    for (const FlagAllocation &allocation : allocations_) {
      auto multi_owner = std::find_if(
          allocation.sync_points.begin(), allocation.sync_points.end(),
          [&](size_t member) { return IsMultiOwnerCounterChannel(member); });
      if (multi_owner == allocation.sync_points.end())
        continue;
      size_t representative = counter_channel_representative_[*multi_owner];
      ICHECK_EQ(allocation.sync_points.size(),
                counter_channel_size_[representative]);
      for (size_t member : allocation.sync_points) {
        ICHECK_EQ(counter_channel_representative_[member], representative)
            << "A multi-owner counter channel allocation must remain "
               "exclusive";
      }
    }
  }

  const std::vector<FlagAllocation> &Allocations() const {
    return allocations_;
  }

  FlagInfoList TakeFlagInfos() { return std::move(flag_infos_); }

private:
  void GroupCounterChannels() {
    counter_channel_representative_.resize(sync_points_.size());
    std::map<CounterChannelKey, size_t> representatives;
    for (size_t index = 0; index < sync_points_.size(); ++index) {
      counter_channel_representative_[index] = index;
      const SyncPoint &sync_point = sync_points_[index];
      if (!sync_point.counter_channel.has_value() ||
          !site_registry_.UsesFlag(sync_point)) {
        continue;
      }
      auto [it, inserted] =
          representatives.emplace(sync_point.GetCounterChannel(), index);
      if (!inserted)
        counter_channel_representative_[index] = it->second;
    }

    counter_channel_size_.assign(sync_points_.size(), 0);
    for (size_t representative : counter_channel_representative_)
      ++counter_channel_size_[representative];
  }

  // GroupCounterChannels() must run first. It groups only flag-using sync
  // points, so non-flag events do not make a counter channel multi-owner.
  bool IsMultiOwnerCounterChannel(size_t index) const {
    if (!sync_points_[index].counter_channel.has_value())
      return false;
    size_t representative = counter_channel_representative_[index];
    return counter_channel_size_[representative] > 1;
  }

  std::string FlagDomain(const SyncPoint &sync_point) const {
    if (site_registry_.IsCrossCore(sync_point))
      return "Cube-Vector";
    return GetHardEvent(site_registry_.Src(sync_point).pipe,
                        site_registry_.Dst(sync_point).pipe);
  }

  void AppendCrossIterFlagInfo(const FlagAllocation &allocation) {
    const SyncPoint &sync_point = sync_points_[allocation.sync_points.front()];
    const SyncInsertionSite &src = site_registry_.Src(sync_point);
    const SyncInsertionSite &dst = site_registry_.Dst(sync_point);
    std::string hard_event = GetHardEvent(src.pipe, dst.pipe);
    size_t separator = hard_event.find('_');
    std::string producer_pipe = "PIPE_" + hard_event.substr(0, separator);
    std::string consumer_pipe = "PIPE_" + hard_event.substr(separator + 1);
    int base_id = allocation.base_id;

    if (!site_registry_.IsCrossCore(sync_point)) {
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos_.push_back(
            FlagInfo{src.core, Integer(base_id + version), hard_event, 0});
      }
      return;
    }

    if (src.core == kCoreVector) {
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos_.push_back(FlagInfo{kCoreVector, Integer(base_id + version),
                                       producer_pipe, 1});
      }
      for (int sid = 0; sid < num_aiv_subcores_; ++sid) {
        for (int version = 0; version < allocation.num_versions; ++version) {
          flag_infos_.push_back(FlagInfo{kCoreCube,
                                         Integer(base_id + sid * 16 + version),
                                         consumer_pipe, 2});
        }
      }
    } else {
      for (int sid = 0; sid < num_aiv_subcores_; ++sid) {
        for (int version = 0; version < allocation.num_versions; ++version) {
          flag_infos_.push_back(FlagInfo{kCoreCube,
                                         Integer(base_id + sid * 16 + version),
                                         producer_pipe, 1});
        }
      }
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos_.push_back(FlagInfo{kCoreVector, Integer(base_id + version),
                                       consumer_pipe, 2});
      }
    }
  }

  static constexpr int kIntraCoreFlagLimit = 8;
  static constexpr int kCrossCoreFlagLimit = 16;

  const SyncSiteRegistry &site_registry_;
  std::vector<SyncPoint> &sync_points_;
  const std::set<std::pair<size_t, size_t>> &reusable_pairs_;
  const CrossCoreFlagReservation &cross_core_flag_reservation_;
  int num_aiv_subcores_;
  std::vector<FlagAllocation> allocations_;
  FlagInfoList flag_infos_;
  std::vector<size_t> counter_channel_representative_;
  std::vector<size_t> counter_channel_size_;
};

// ============================================================================
// Synchronization statement construction and IR materialization
// ============================================================================
using SyncInsertionMap = std::map<CoreMask, std::vector<Stmt>>;

// InsertSync records synchronization here while dependency analysis still
// refers to the original tree. Build() materializes the lists as adjacent
// TaskNodes after all sync points have been planned and inserted. Same-core
// T.PerCoreTask synchronization is instead migrated beside its candidate tasks.
class SyncInsertionExtraInfo final : public IRExtraInfo {
public:
  std::unique_ptr<IRExtraInfo> Clone() const final {
    return std::make_unique<SyncInsertionExtraInfo>(*this);
  }

  SyncInsertionMap &Before() { return before_; }
  const SyncInsertionMap &Before() const { return before_; }
  SyncInsertionMap &After() { return after_; }
  const SyncInsertionMap &After() const { return after_; }

  void SubstituteVar(const Var &old_var, const Var &new_var) final {
    for (auto &[_, statements] : before_) {
      for (Stmt &statement : statements)
        statement = Substitute(statement, {{old_var, new_var}});
    }
    for (auto &[_, statements] : after_) {
      for (Stmt &statement : statements)
        statement = Substitute(statement, {{old_var, new_var}});
    }
  }

private:
  SyncInsertionMap before_;
  SyncInsertionMap after_;
};

SyncInsertionExtraInfo *GetSyncInsertionExtraInfo(IRStructure *node) {
  return node->GetExtraInfo<SyncInsertionExtraInfo>();
}

std::unique_ptr<IRExtraInfo> MakeSyncInsertionExtraInfo() {
  return std::make_unique<SyncInsertionExtraInfo>();
}

Stmt MakeSyncTask(const Stmt &body, CoreMask core_mask) {
  ICHECK(IsConcreteCore(core_mask));
  TaskMetadata metadata;
  metadata.core_mask = core_mask;
  return AttrStmt(MergeTaskMetadata(metadata, IntImm(DataType::Int(32), 0)),
                  attr::kAscendTask, Integer(1), body, body->span);
}

std::shared_ptr<TaskNode> MakeSyncTaskNode(const Stmt &body, CoreMask core_mask,
                                           int stage) {
  auto task = std::make_shared<TaskNode>();
  task->stmt = MakeSyncTask(body, core_mask);
  task->SetStage(stage);
  task->SetCoreMask(core_mask);
  task->SetPipeMask(static_cast<uint16_t>(ResourcePipe::kScalar));
  return task;
}

class PerCoreTaskSyncMigrator : public StmtMutator {
public:
  PerCoreTaskSyncMigrator(std::vector<Stmt> before, std::vector<Stmt> after,
                          CoreMask core_mask)
      : before_(std::move(before)), after_(std::move(after)),
        core_mask_(core_mask) {
    ICHECK(IsConcreteCore(core_mask_));
  }

  Stmt Rewrite(Stmt body) {
    Stmt result = VisitStmt(std::move(body));
    ICHECK_GT(num_candidates_, 0)
        << "Cannot find a T.Task marker in normalized T.PerCoreTask";
    return result;
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != attr::kAscendTask)
      return StmtMutator::VisitStmt_(op);
    Stmt candidate_body = VisitStmt(op->body);
    Stmt candidate =
        AttrStmt(op->node, op->attr_key, op->value, candidate_body, op->span);
    if (!HasTaskBufferAccess(candidate_body))
      return candidate;
    ++num_candidates_;
    Array<Stmt> sequence;
    sequence.reserve(before_.size() + 1 + after_.size());
    for (const Stmt &stmt : before_)
      sequence.push_back(MakeSyncTask(stmt, core_mask_));
    sequence.push_back(std::move(candidate));
    for (const Stmt &stmt : after_)
      sequence.push_back(MakeSyncTask(stmt, core_mask_));
    return SeqStmt::Flatten(sequence);
  }

  std::vector<Stmt> before_;
  std::vector<Stmt> after_;
  CoreMask core_mask_;
  int num_candidates_{0};
};

std::vector<Stmt> GetCoreStatements(const SyncInsertionMap &insertions,
                                    CoreMask core_mask) {
  auto it = insertions.find(core_mask);
  return it == insertions.end() ? std::vector<Stmt>() : it->second;
}

void AppendSyncTaskNodes(std::vector<std::shared_ptr<IRStructure>> *result,
                         const SyncInsertionMap &insertions, int stage,
                         std::optional<CoreMask> migrated_core = std::nullopt) {
  for (const auto &[core_mask, statements] : insertions) {
    if (migrated_core.has_value() && core_mask == migrated_core.value())
      continue;
    for (const Stmt &statement : statements)
      result->push_back(MakeSyncTaskNode(statement, core_mask, stage));
  }
}

inline Stmt MakeAscendSetFlag(const std::string &hard_event,
                              PrimExpr event_id) {
  return Evaluate(Call(DataType::Handle(), ascend_set_flag(),
                       {StringImm(hard_event), event_id}));
}

inline Stmt MakeAscendWaitFlag(const std::string &hard_event,
                               PrimExpr event_id) {
  return Evaluate(Call(DataType::Handle(), ascend_wait_flag(),
                       {StringImm(hard_event), event_id}));
}

inline Stmt MakeAscendPipeBarrier(const std::string &pipe) {
  return Evaluate(
      Call(DataType::Handle(), ascend_pipe_barrier(), {StringImm(pipe)}));
}

inline Stmt MakeAscendCrossCoreSetFlag(int mode_id, const std::string &pipe,
                                       PrimExpr flag_id) {
  return Evaluate(Call(DataType::Handle(), ascend_cross_core_set_flag(),
                       {Integer(mode_id), StringImm(pipe), flag_id}));
}

inline Stmt MakeAscendCrossCoreWaitFlag(int mode_id, const std::string &pipe,
                                        PrimExpr flag_id) {
  return Evaluate(Call(DataType::Handle(), ascend_cross_core_wait_flag(),
                       {Integer(mode_id), StringImm(pipe), flag_id}));
}

PrimExpr MakeFlagEventId(const PrimExpr &iteration, int num_versions,
                         int base_id) {
  DataType dtype = iteration.dtype();
  PrimExpr event_id =
      indexmod(iteration, IntImm(dtype, num_versions)) + IntImm(dtype, base_id);
  return dtype == DataType::Int(32) ? event_id
                                    : cast(DataType::Int(32), event_id);
}

bool ShouldElidePerCoreTaskSyncGuard(IRStructure *node, CoreMask core,
                                     const PrimExpr &guard) {
  if (!node->IsTask())
    return false;
  const auto *task = static_cast<const TaskNode *>(node);
  if (!task->IsPerCoreTask() || task->GetCoreMask() != core)
    return false;

  ConstrSet outer_ctx = task->GetContextBeforeConditionGuards();
  PrimExpr task_guard = task->GetConditionGuard();
  PrimExpr sync_guard = guard.defined() ? guard : Bool(true);
  ICHECK(GuardImplies(sync_guard, task_guard, outer_ctx))
      << "Cannot migrate synchronization into guarded T.PerCoreTask: "
         "the synchronization guard "
      << sync_guard << " does not imply the T.PerCoreTask guard " << task_guard;
  return GuardsEquivalent(sync_guard, task_guard, outer_ctx);
}

void InsertStatement(IRStructure *node, Stmt stmt, bool at_beginning,
                     CoreMask core, const PrimExpr &guard = PrimExpr()) {
  ICHECK(node != nullptr);
  ICHECK(IsConcreteCore(core));
  bool elide_guard = ShouldElidePerCoreTaskSyncGuard(node, core, guard);
  if (guard.defined() && !elide_guard)
    stmt = WrapWithGuard(std::move(stmt), guard);
  SyncInsertionExtraInfo *insertions = GetSyncInsertionExtraInfo(node);
  if (at_beginning) {
    std::vector<Stmt> &before = insertions->Before()[core];
    before.insert(before.begin(), std::move(stmt));
  } else {
    insertions->After()[core].push_back(std::move(stmt));
  }
}

class SyncMaterializer {
public:
  SyncMaterializer(const SyncSiteRegistry &site_registry,
                   const std::vector<SyncPoint> &sync_points,
                   const std::vector<FlagAllocation> &allocations,
                   int num_aiv_subcores)
      : site_registry_(site_registry), sync_points_(sync_points),
        allocations_(allocations), num_aiv_subcores_(num_aiv_subcores) {}

  void Run(std::vector<std::shared_ptr<IRStructure>> &root) {
    InsertSyncPoints();
    MaterializeInsertions(root, nullptr);
  }

private:
  void InsertSyncPoints() {
    for (const SyncPoint &sync_point : sync_points_) {
      const SyncInsertionSite &src = site_registry_.Src(sync_point);
      const SyncInsertionSite &dst = site_registry_.Dst(sync_point);
      std::string hard_event = GetHardEvent(src.pipe, dst.pipe);
      size_t separator = hard_event.find('_');
      std::string producer_pipe = "PIPE_" + hard_event.substr(0, separator);
      std::string consumer_pipe = "PIPE_" + hard_event.substr(separator + 1);
      PrimExpr producer_guard = src.GetGuard();
      PrimExpr consumer_guard = dst.GetGuard();

      if (!site_registry_.IsCrossCore(sync_point) &&
          producer_pipe == consumer_pipe) {
        if (producer_pipe == "PIPE_S" || producer_pipe == "PIPE_V" ||
            producer_pipe == "PIPE_M") {
          continue;
        }
        InsertStatement(dst.path_end, MakeAscendPipeBarrier(producer_pipe),
                        true, dst.core, consumer_guard);
        continue;
      }

      const FlagAllocation &allocation =
          allocations_[sync_point.allocation_index];
      int base_id = allocation.base_id;
      PrimExpr event_id = MakeFlagEventId(sync_point.flag_iteration,
                                          sync_point.num_versions, base_id);

      if (site_registry_.IsCrossCore(sync_point)) {
        if (src.core == kCoreVector) {
          InsertStatement(
              src.path_end,
              MakeAscendCrossCoreSetFlag(4, producer_pipe, event_id), false,
              src.core, producer_guard);
          for (int sid = 0; sid < num_aiv_subcores_; ++sid) {
            PrimExpr wait_id =
                MakeFlagEventId(sync_point.flag_iteration,
                                sync_point.num_versions, base_id + 16 * sid);
            InsertStatement(
                dst.path_end,
                MakeAscendCrossCoreWaitFlag(4, consumer_pipe, wait_id), true,
                dst.core, consumer_guard);
          }
        } else {
          for (int sid = 0; sid < num_aiv_subcores_; ++sid) {
            PrimExpr set_id =
                MakeFlagEventId(sync_point.flag_iteration,
                                sync_point.num_versions, base_id + 16 * sid);
            InsertStatement(
                src.path_end,
                MakeAscendCrossCoreSetFlag(4, producer_pipe, set_id), false,
                src.core, producer_guard);
          }
          InsertStatement(
              dst.path_end,
              MakeAscendCrossCoreWaitFlag(4, consumer_pipe, event_id), true,
              dst.core, consumer_guard);
        }
        continue;
      }

      InsertStatement(src.path_end, MakeAscendSetFlag(hard_event, event_id),
                      false, src.core, producer_guard);
      InsertStatement(dst.path_end, MakeAscendWaitFlag(hard_event, event_id),
                      true, dst.core, consumer_guard);
    }
  }

  void MaterializeInsertions(std::vector<std::shared_ptr<IRStructure>> &nodes,
                             ControlNode *parent) {
    std::vector<std::shared_ptr<IRStructure>> materialized;
    materialized.reserve(nodes.size());
    for (const std::shared_ptr<IRStructure> &node : nodes) {
      if (node->IsControl()) {
        auto *control = static_cast<ControlNode *>(node.get());
        MaterializeInsertions(control->children, control);
      }

      const SyncInsertionExtraInfo *insertions =
          GetSyncInsertionExtraInfo(node.get());
      const SyncInsertionMap &before_insertions = insertions->Before();
      const SyncInsertionMap &after_insertions = insertions->After();

      std::optional<CoreMask> migrated_core;
      if (node->IsTask()) {
        auto *task = static_cast<TaskNode *>(node.get());
        if (task->IsPerCoreTask()) {
          CoreMask core_mask = task->GetCoreMask();
          ICHECK(IsConcreteCore(core_mask))
              << "Scheduled T.PerCoreTask must belong to one concrete core";
          migrated_core = core_mask;
          std::vector<Stmt> before =
              GetCoreStatements(before_insertions, core_mask);
          std::vector<Stmt> after =
              GetCoreStatements(after_insertions, core_mask);
          if (!before.empty() || !after.empty()) {
            task->stmt = PerCoreTaskSyncMigrator(std::move(before),
                                                 std::move(after), core_mask)
                             .Rewrite(std::move(task->stmt));
          }
        }
      }

      AppendSyncTaskNodes(&materialized, before_insertions, node->GetStage(),
                          migrated_core);
      materialized.push_back(node);
      AppendSyncTaskNodes(&materialized, after_insertions, node->GetStage(),
                          migrated_core);
    }

    for (size_t i = 0; i < materialized.size(); ++i) {
      materialized[i]->SetIndex(i);
      materialized[i]->SetParent(parent);
    }
    nodes = std::move(materialized);
  }

  const SyncSiteRegistry &site_registry_;
  const std::vector<SyncPoint> &sync_points_;
  const std::vector<FlagAllocation> &allocations_;
  int num_aiv_subcores_;
};

// ============================================================================
// Kernel-level validation and scheduled-TIR rewrite
// ============================================================================
SBlock InsertKernelSync(const Stmt &kernel_body, ScheduledTIR scheduled_tir) {
  std::vector<std::shared_ptr<IRStructure>> &root = scheduled_tir.tree;
  int num_aiv_subcores = scheduled_tir.metadata.num_aiv_subcores.value_or(2);
  MultiBufferPlan multi_buffer_plan =
      ReadMultiBufferPlan(root, scheduled_tir.metadata.buffer_versions);
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(root, tasks);
  for (TaskNode *task : tasks) {
    ICHECK_NE(task->GetCoreMask(), kCoreUnassigned)
        << "InsertSync requires ResolveCore to assign every T.Task";
  }
  EpochDomainRegistry domains(root, multi_buffer_plan);
  SyncAnalyzer analyzer(multi_buffer_plan, domains,
                        scheduled_tir.metadata.manual_buffer_versions,
                        std::move(scheduled_tir.metadata.root_conflict_hints));
  scheduled_tir.metadata.root_conflict_hints = {};
  analyzer.Analyze(root);

  CrossCoreFlagReservation cross_core_flag_reservation =
      CollectExplicitCrossCoreFlags(kernel_body);
  FlagAllocator flag_allocator(analyzer.SiteRegistry(), analyzer.SyncPoints(),
                               analyzer.ReusablePairs(),
                               cross_core_flag_reservation, num_aiv_subcores);
  flag_allocator.Plan();
  SyncMaterializer(analyzer.SiteRegistry(), analyzer.SyncPoints(),
                   flag_allocator.Allocations(), num_aiv_subcores)
      .Run(root);
  FlagInfoList flag_infos = flag_allocator.TakeFlagInfos();

  std::vector<std::shared_ptr<IRStructure>> prefix;
  std::vector<std::shared_ptr<IRStructure>> suffix;
  for (const FlagInfo &info : flag_infos) {
    if (info.mode == 0) {
      prefix.push_back(MakeSyncTaskNode(
          MakeAscendSetFlag(info.hardevent, info.flag_var), info.core_mask, 0));
      suffix.push_back(
          MakeSyncTaskNode(MakeAscendWaitFlag(info.hardevent, info.flag_var),
                           info.core_mask, 0));
    } else if (info.mode == 1) {
      prefix.push_back(MakeSyncTaskNode(
          MakeAscendCrossCoreSetFlag(4, info.hardevent, info.flag_var),
          info.core_mask, 0));
    } else if (info.mode == 2) {
      suffix.push_back(MakeSyncTaskNode(
          MakeAscendCrossCoreWaitFlag(4, info.hardevent, info.flag_var),
          info.core_mask, 0));
    }
  }
  root.insert(root.begin(), prefix.begin(), prefix.end());
  root.insert(root.end(), suffix.begin(), suffix.end());
  for (size_t i = 0; i < root.size(); ++i) {
    root[i]->SetIndex(i);
    root[i]->SetParent(nullptr);
  }
  return EncodeScheduledTIR(std::move(scheduled_tir));
}

} // namespace

// ============================================================================
// Pass registration
// ============================================================================
tvm::transform::Pass InsertSync() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    return RewriteTilelangKernels(
        std::move(func), "InsertSync",
        [](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir = DecodeScheduledTIR(
              context.root, context.outer_ctx, MakeSyncInsertionExtraInfo);
          return InsertKernelSync(context.root->body, std::move(scheduled_tir));
        });
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InsertSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InsertSync", InsertSync);
}

} // namespace tl
} // namespace tvm
