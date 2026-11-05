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

// Shared storage-level multi-buffer protocol for Ascend scheduled TIR.

#pragma once

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../buffer_version.h"
#include "./ir_structure.h"
#include "./native_task_ir.h"
#include "./task_annotations.h"
#include "ascend/transform/attr.h"
#include "op/builtin.h"
#include "op/utils.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;
using namespace prim;

constexpr const char *kMultiBufferCounterMap = "tl.multi_buffer_counter_map";
constexpr const char *kStorageEpochGuardMap = "tl.storage_epoch_guard_map";
constexpr const char *kMultiBufferBroadcastFill =
    "tl.multi_buffer_broadcast_fill";

enum class BufferVersionMode {
  kAuto,
  kIteration,
  kCounter,
};

using MultiBufferCounterMap = ffi::Map<Var, TensorVar>;
using StorageEpochGuardMap = ffi::Map<Var, PrimExpr>;
using MultiBufferBroadcastFill = ffi::Array<Var>;

// Mode annotations use the same storage data-Var key as BufferVersionMap.
using BufferVersionModeTable =
    std::unordered_map<Var, BufferVersionMode, ObjectPtrHash, ObjectPtrEqual>;

// Only runtime assumption operands follow the physical version of a task.
// Native task metadata also contains Vars, but those fields name protocols
// rather than evaluating the named storage.
inline bool CanRewriteMultiBufferAttrNode(const ffi::String &attr_key,
                                          const ffi::Any &node) {
  bool is_assume = attr_key == "tilelang_assume" ||
                   attr_key == "tl.assume_requires_runtime_check";
  return is_assume && node.try_cast<PrimExpr>().has_value();
}

// Owner-external fills can initialize every physical version independently.
// Keep this predicate shared by the early owner planner and PrepareMultiBuffer
// so automatic annotation never accepts a task that coverage validation later
// rejects.
inline bool
CanBroadcastFillToStorage(const Stmt &stmt, const Var &storage,
                          const NativeStorageContext &context = {}) {
  class Detector : public StmtExprVisitor {
  public:
    using Result = ffi::Optional<ffi::VisitInterrupt>;
    Detector(Var storage, NativeStorageContext context)
        : storage_(std::move(storage)), context_(std::move(context)) {}

    bool found{false};
    bool valid{true};

  private:
    // Replaying a fill must not replay a read through an alias of its target.
    // Distinct external or opaque pointer roots are not a noalias proof.
    bool MayTouchStorage(const Expr &value) const {
      if (!value->ty.as<TensorTypeObj>() && !value->ty.as<PtrTypeObj>())
        return value.same_as(storage_);
      auto address = context_.Address(value);
      auto target = context_.Address(storage_);
      return !(address.exact && target.exact && address.fresh_allocation &&
               target.fresh_allocation && !address.root.same_as(target.root));
    }

    Result Visit_(const RegionStmtObj *op) final {
      if (auto attribute = ReadTaskAttribute(op)) {
        if (CanRewriteMultiBufferAttrNode(attribute->key, attribute->node))
          Visit(attribute->node.cast<PrimExpr>());
        return Visit(op->body);
      }
      return StmtExprVisitor::Visit_(op);
    }

    Result Visit_(const BindObj *op) final {
      Expr value = op->value;
      if (const auto *call = value.as<CallObj>(); IsScheduledBind(call))
        value = ReadScheduledBinding(call).value;
      return Visit(value);
    }

    Result Visit_(const VarObj *op) final {
      valid &= !MayTouchStorage(ffi::GetRef<Var>(op));
      return std::nullopt;
    }

    Result Visit_(const CallObj *op) final {
      static const Op &fill_op = Op::Get("tl.tileop.fill");
      if (op->op.same_as(fill_op)) {
        auto operands = DecodeTileOpArguments(op);
        TVM_FFI_CHECK_EQ(operands.operands.size(), 2U, ValueError);
        auto region =
            NormalizeToAccessRegion(operands.operands[0], kAccessWrite).region;
        auto tensor = RegionTensor(region);
        if (context_.Exact(tensor) &&
            context_.Identity(tensor).same_as(storage_)) {
          found = true;
          // Skip the write operand itself. Bounds, layout expressions, fill
          // values and dynamic annotations can still read the target.
          for (const Range &range : region->region) {
            Visit(range->begin);
            Visit(range->extent);
          }
          Visit(tensor->ty);
          for (size_t i = 1; i < op->args.size(); ++i)
            Visit(op->args[i]);
          return std::nullopt;
        }
      }
      return StmtExprVisitor::Visit_(op);
    }

    Var storage_;
    NativeStorageContext context_;
  };
  NativeStorageContext definitions = context;
  definitions.Collect(stmt);
  auto detector = ffi::make_object<Detector>(storage, std::move(definitions));
  detector->Visit(stmt);
  return detector->found && detector->valid;
}

struct MultiBufferOwnerInfo {
  ControlNode *loop{nullptr};
  PrimExpr active_guard{IntImm::Bool(true)};
};

// One scheduler-selected storage ring. A storage may have multiple disjoint
// owner loops. Compatible storages with the same ordered owners, owner-local
// guards, and schedule stages may share one physical counter group; iteration
// mode derives the slot lexically and therefore requires one owner.
//
// MultiBufferPlan contains every scheduler-selected storage, including a
// one-version storage. Every info has at least one owner. Physical buffer
// rewriting is needed only when num_versions > 1; version-one entries still
// use the same iteration/counter ownership and synchronization protocols.
//
// Entries stored in a plan have exactly one of these forms:
//   iteration: counter_group_id == 0, counter undefined,
//              exactly one owner whose active_guard is true;
//   counter:   counter_group_id > 0, counter defined,
//              every owner has an active_guard (possibly true).
// Compatible storages still have one info per storage; sharing a counter means
// those infos carry the same counter_group_id and counter.
struct MultiBufferInfo {
  int counter_group_id{0};
  Var storage{ffi::UnsafeInit{}};
  int num_versions{0};
  std::vector<MultiBufferOwnerInfo> owners;
  TensorVar counter{ffi::UnsafeInit{}};

  bool UsesCounter() const { return counter.defined(); }
  bool NeedsVersionDimension() const { return num_versions > 1; }

  const MultiBufferOwnerInfo *FindOwner(const IRStructure *node) const {
    ICHECK(node != nullptr) << "Cannot find an owner for a null IRStructure";
    const MultiBufferOwnerInfo *result = nullptr;
    for (const MultiBufferOwnerInfo &owner : owners) {
      if (!node->IsWithin(owner.loop) || node == owner.loop->HeaderTask())
        continue;
      ICHECK(result == nullptr)
          << "An IRStructure node belongs to nested multi-buffer owners for "
             "storage "
          << storage->name;
      result = &owner;
    }
    return result;
  }

  Var CounterStorage() const {
    ICHECK(UsesCounter() && !owners.empty());
    return owners.front().loop->StorageIdentity(counter);
  }
};

// Multi-buffer metadata shared by PrepareMultiBuffer, ResolveCore, InsertSync,
// and MaterializeMultiBuffer. Logical epoch domains are represented separately
// by loop annotations and EpochDomainRegistry below; they never participate in
// physical counter-group identity.
class MultiBufferPlan {
public:
  const std::vector<MultiBufferInfo> &Infos() const { return infos_; }

  // Null means that this storage has no selected automatic multi-buffer plan.
  // Storage access/dependency visitors may observe null; code handling a
  // selected plan must require a non-null result instead of inventing fallback
  // state.
  const MultiBufferInfo *Find(const Var &storage) const {
    auto it = storage_indices_.find(storage);
    return it == storage_indices_.end() ? nullptr : &infos_[it->second];
  }

  const MultiBufferInfo *Find(const TensorVar &tensor,
                              const NativeStorageContext &storage) const {
    return Find(storage.Identity(tensor));
  }

  // Pass-local builders append resolved storage records through this method.
  void AddInfo(MultiBufferInfo info) {
    ICHECK_GT(info.num_versions, 0);
    ICHECK(!info.owners.empty());
    for (size_t i = 0; i < info.owners.size(); ++i) {
      ICHECK(info.owners[i].loop != nullptr);
      ICHECK(info.owners[i].active_guard.ty().MatchesCode(kDLBool) &&
             info.owners[i].active_guard.ty().IsScalar());
      for (size_t j = 0; j < i; ++j)
        ICHECK(!info.owners[i].loop->IsWithin(info.owners[j].loop) &&
               !info.owners[j].loop->IsWithin(info.owners[i].loop))
            << "Multi-buffer storage has nested owner loops: " << info.storage;
    }
    if (info.UsesCounter()) {
      ICHECK_GT(info.counter_group_id, 0);
      auto [group_it, inserted] = counter_group_representatives_.emplace(
          info.counter_group_id, infos_.size());
      if (!inserted) {
        const MultiBufferInfo &existing = infos_[group_it->second];
        ICHECK(existing.UsesCounter() &&
               existing.counter.same_as(info.counter));
        ICHECK_EQ(existing.owners.size(), info.owners.size());
        for (size_t i = 0; i < info.owners.size(); ++i) {
          ICHECK_EQ(existing.owners[i].loop, info.owners[i].loop);
          ICHECK(ffi::StructuralEqual()(existing.owners[i].active_guard,
                                        info.owners[i].active_guard))
              << "Shared counter storages have inconsistent active guards";
        }
      }
    } else {
      ICHECK_EQ(info.counter_group_id, 0);
      ICHECK(!info.counter.defined());
      ICHECK_EQ(info.owners.size(), 1U);
      ICHECK(IsOne(info.owners.front().active_guard));
    }
    size_t index = infos_.size();
    auto [_, inserted] = storage_indices_.emplace(info.storage, index);
    ICHECK(inserted) << "Duplicate multi-buffer storage " << info.storage;
    infos_.push_back(std::move(info));
  }

private:
  std::vector<MultiBufferInfo> infos_;
  std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual>
      storage_indices_;
  std::unordered_map<int, size_t> counter_group_representatives_;
};

namespace multi_buffer_detail {

inline std::optional<NativeTaskAttribute> AsSemanticTask(const Stmt &stmt) {
  auto task = ReadTaskAttribute(stmt);
  if (!task || !IsSemanticTaskKey(task->key)) {
    return std::nullopt;
  }
  return task;
}

inline NativeTaskAttribute RequireSemanticTask(const Stmt &stmt) {
  auto task = AsSemanticTask(stmt);
  ICHECK(task.has_value())
      << "Multi-buffer task annotation requires T.Task/T.PerCoreTask";
  return task.value();
}

} // namespace multi_buffer_detail

inline MultiBufferBroadcastFill
GetMultiBufferBroadcastFills(const TaskNode *task) {
  ICHECK(task != nullptr);
  // Control headers also use TaskNode for access analysis, but only semantic
  // T.Task/T.PerCoreTask nodes can carry broadcast-fill metadata.
  auto semantic_task = multi_buffer_detail::AsSemanticTask(task->stmt);
  if (!semantic_task.has_value())
    return MultiBufferBroadcastFill();
  const auto *metadata = semantic_task->node.as<ffi::MapObj>();
  if (metadata == nullptr)
    return {};
  for (const auto &[key, value] : *metadata) {
    if (key.cast<ffi::String>() == kMultiBufferBroadcastFill)
      return value.cast<MultiBufferBroadcastFill>();
  }
  return {};
}

inline bool IsMultiBufferBroadcastFill(const TaskNode *task,
                                       const Var &storage) {
  for (const Var &candidate : GetMultiBufferBroadcastFills(task)) {
    if (candidate.same_as(storage))
      return true;
  }
  return false;
}

inline void AddMultiBufferBroadcastFill(TaskNode *task, const Var &storage) {
  MultiBufferBroadcastFill storages = GetMultiBufferBroadcastFills(task);
  for (const Var &candidate : storages) {
    if (candidate.same_as(storage))
      return;
  }
  storages.push_back(storage);
  auto semantic_task = multi_buffer_detail::RequireSemanticTask(task->stmt);
  ffi::Map<ffi::String, ffi::Any> metadata =
      task_metadata_detail::CopyTaskMetadata(semantic_task.node);
  metadata.Set(kMultiBufferBroadcastFill, std::move(storages));
  task->stmt = semantic_task.WithNode(std::move(metadata));
}

inline void ClearMultiBufferBroadcastFills(TaskNode *task) {
  auto semantic_task = multi_buffer_detail::RequireSemanticTask(task->stmt);
  ffi::Map<ffi::String, ffi::Any> metadata =
      task_metadata_detail::CopyTaskMetadata(semantic_task.node);
  metadata.erase(kMultiBufferBroadcastFill);
  task->stmt = semantic_task.WithNode(std::move(metadata));
}

inline MultiBufferOwnerMap
CollectMultiBufferOwners(const std::vector<std::shared_ptr<IRStructure>> &root,
                         const L0StorageGroups &groups) {
  MultiBufferOwnerMap owners;
  std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
    if (!node)
      return;
    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node);
      auto annotation =
          ReadLoopAnnotations(control->control.get()).Get(kMultiBufferEligible);
      if (annotation.has_value()) {
        std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen;
        for (const Var &storage : annotation.value().cast<Array<Var>>()) {
          ICHECK(seen.insert(storage).second)
              << "Duplicate multi-buffer owner storage " << storage;
          Array<Var> members = groups.Members(storage);
          if (std::any_of(members.begin(), members.end(),
                          [&](const Var &member) {
                            return control->BodyTouchesStorage(member);
                          }))
            owners[storage].push_back(control);
        }
      }
    }
    for (const auto *list : node->ChildLists())
      for (const auto &child : *list)
        visit(child.get());
  };
  for (const auto &node : root)
    visit(node.get());
  return owners;
}

// Recover the prepared plan in a later scheduled-TIR pass. This re-discovers
// the annotated owners but does not recompute mode or active guards.
inline MultiBufferPlan
ReadMultiBufferPlan(const std::vector<std::shared_ptr<IRStructure>> &root,
                    const BufferVersionMap &selected_versions,
                    const L0StorageGroups &groups) {
  MultiBufferPlan plan;
  MultiBufferOwnerMap owners_by_storage =
      CollectMultiBufferOwners(root, groups);
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> counter_groups;
  int next_counter_group_id = 1;
  for (const auto &[storage, num_versions] : selected_versions) {
    // Preserve scheduled order: owner position is part of the shared counter
    // protocol identity and later grouping compares owners pairwise.
    auto owners_it = owners_by_storage.find(storage);
    ICHECK(owners_it != owners_by_storage.end() && !owners_it->second.empty())
        << "Prepared multi-buffer storage " << storage->name
        << " has no owner loop";
    const std::vector<ControlNode *> &owners = owners_it->second;

    MultiBufferInfo info;
    info.storage = storage;
    info.num_versions = num_versions;
    bool uses_counter = false;
    for (ControlNode *owner_loop : owners) {
      MultiBufferOwnerInfo owner;
      owner.loop = owner_loop;
      auto annotations = ReadLoopAnnotations(owner_loop->control.get());
      auto counters = annotations.Get(kMultiBufferCounterMap);
      if (counters.has_value()) {
        MultiBufferCounterMap counter_map =
            counters.value().cast<MultiBufferCounterMap>();
        if (auto counter = counter_map.Get(storage)) {
          if (uses_counter) {
            ICHECK(info.counter.same_as(counter.value()))
                << "Disjoint multi-buffer owners must share one counter";
          } else {
            info.counter = counter.value();
            uses_counter = true;
          }
          auto guards = annotations.Get(kStorageEpochGuardMap);
          ICHECK(guards.has_value())
              << "Prepared counter storage is missing its active guard";
          StorageEpochGuardMap guard_map =
              guards.value().cast<StorageEpochGuardMap>();
          auto guard = guard_map.Get(storage);
          ICHECK(guard.has_value())
              << "Prepared counter storage is missing its active guard";
          owner.active_guard = guard.value();
        }
      }
      info.owners.push_back(std::move(owner));
    }
    if (uses_counter) {
      // Counter storage identity reconstructs the shared physical group. One
      // group may cover several disjoint owners; each owner is independently
      // bound to its active-epoch domain below.
      auto [group, inserted] =
          counter_groups.emplace(info.CounterStorage(), next_counter_group_id);
      if (inserted)
        ++next_counter_group_id;
      info.counter_group_id = group->second;
      for (const MultiBufferOwnerInfo &owner : info.owners) {
        auto counters = ReadLoopAnnotations(owner.loop->control.get())
                            .Get(kMultiBufferCounterMap);
        ICHECK(counters.has_value() &&
               counters.value().cast<MultiBufferCounterMap>().count(storage))
            << "Every disjoint owner must carry the shared counter protocol";
      }
    } else {
      ICHECK_EQ(info.owners.size(), 1U)
          << "Iteration multi-buffering requires exactly one owner";
    }
    plan.AddInfo(std::move(info));
  }
  return plan;
}

enum class EpochDomainKind {
  kLexical,
  kCounter,
};

// One logical active-epoch sequence reconstructed independently in each pass.
// A lexical domain is owned by the sibling list containing its path_begin. A
// counter domain is owned by one concrete multi-buffer owner and may contain
// sites anywhere strictly inside that owner.
struct EpochDomain {
  int id{0};
  EpochDomainKind kind{EpochDomainKind::kLexical};
  IRStructure *owner{nullptr};
  PrimExpr active_guard{IntImm::Bool(true)};
  TensorVar counter{ffi::UnsafeInit{}};
  // If and else have different immediate sibling lists even though their
  // parent node is the same. Only counter domains are always loop-owned.
  size_t child_list_index{0};

  bool IsCounter() const { return kind == EpochDomainKind::kCounter; }
};

// Reconstruct counter and lexical domains from the physical multi-buffer plan
// and per-loop storage guards. Domain ids are pass-local and are interned using
// structural equality only; PrepareMultiBuffer has already canonicalized
// proof-equivalent guards to one representative expression.
class EpochDomainRegistry {
  struct ScopeKey {
    const IRStructure *owner{nullptr};
    size_t child_list_index{0};

    bool operator<(const ScopeKey &other) const {
      if (owner != other.owner)
        return std::less<const IRStructure *>{}(owner, other.owner);
      return child_list_index < other.child_list_index;
    }
  };

public:
  EpochDomainRegistry(const std::vector<std::shared_ptr<IRStructure>> &root,
                      const MultiBufferPlan &multi_buffer_plan)
      : multi_buffer_plan_(multi_buffer_plan) {
    AddLexicalScope(nullptr);
    RegisterLexicalScopes(root);
    RegisterPreparedLexicalDomains(root);
    RegisterCounterDomains();
    ReorderDomainsForProjection();
  }

  const EpochDomain &Domain(int id) const {
    ICHECK_GT(id, 0);
    ICHECK_LE(static_cast<size_t>(id), domains_.size());
    const EpochDomain &domain = domains_[id - 1];
    ICHECK_EQ(domain.id, id) << "Missing prepared epoch domain " << id;
    return domain;
  }

  bool CanProjectDomain(int source_id, int target_id, const IRStructure *scope,
                        size_t child_list_index = 0) const {
    auto scope_it = projection_targets_.find({scope, child_list_index});
    ICHECK(scope_it != projection_targets_.end())
        << "Missing epoch-domain projection scope";
    auto source_it = scope_it->second.find(source_id);
    ICHECK(source_it != scope_it->second.end())
        << "Missing epoch-domain projection source " << source_id;
    return std::binary_search(source_it->second.begin(),
                              source_it->second.end(), target_id);
  }

  int DomainForStorage(const Var &storage, const IRStructure *scope,
                       size_t child_list_index = 0) const {
    if (scope != nullptr) {
      if (const MultiBufferInfo *info = multi_buffer_plan_.Find(storage)) {
        if (info->UsesCounter()) {
          if (const MultiBufferOwnerInfo *owner = info->FindOwner(scope)) {
            const int *domain_id = FindStorageBinding(
                counter_bindings_, storage, {owner->loop, 0});
            ICHECK(domain_id != nullptr)
                << "Missing counter epoch domain for storage " << storage->name;
            return *domain_id;
          }
        }
      }
    }
    return LexicalDomainId(storage, {scope, child_list_index});
  }

  int UnconditionalLexicalDomain(const IRStructure *scope,
                                 size_t child_list_index = 0) const {
    auto it = lexical_domain_ids_.find({scope, child_list_index});
    ICHECK(it != lexical_domain_ids_.end())
        << "Missing unconditional lexical epoch domain";
    return it->second;
  }

  std::vector<int> DomainsForTaskAtScope(const TaskNode *task,
                                         const IRStructure *scope,
                                         size_t child_list_index = 0) const {
    std::vector<int> result;
    std::unordered_set<int> domain_ids;
    for (const Var &storage : TaskStorages(task)) {
      int domain_id = DomainForStorage(storage, scope, child_list_index);
      if (domain_ids.insert(domain_id).second)
        result.push_back(domain_id);
    }
    return result;
  }

private:
  int InternDomain(EpochDomainKind kind, IRStructure *owner,
                   const PrimExpr &active_guard,
                   const TensorVar &counter = TensorVar(ffi::UnsafeInit{}),
                   size_t child_list_index = 0) {
    ICHECK(kind == EpochDomainKind::kLexical || owner != nullptr);
    ICHECK_EQ(kind == EpochDomainKind::kCounter, counter.defined());
    for (const EpochDomain &domain : domains_) {
      if (domain.kind != kind || domain.owner != owner ||
          domain.child_list_index != child_list_index)
        continue;
      if (counter.defined() && !domain.counter.same_as(counter))
        continue;
      if (!counter.defined() && domain.counter.defined())
        continue;
      if (domain.active_guard.same_as(active_guard) ||
          ffi::StructuralEqual()(domain.active_guard, active_guard)) {
        return domain.id;
      }
    }
    int domain_id = static_cast<int>(domains_.size()) + 1;
    domains_.push_back(EpochDomain{domain_id, kind, owner, active_guard,
                                   counter, child_list_index});
    return domain_id;
  }

  void
  RegisterLexicalScopes(const std::vector<std::shared_ptr<IRStructure>> &root) {
    std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
      if (!node)
        return;
      auto lists = node->ChildLists();
      for (size_t i = 0; i < lists.size(); ++i) {
        AddLexicalScope(node, i);
        for (const auto &child : *lists[i])
          visit(child.get());
      }
    };
    for (const auto &node : root)
      visit(node.get());
  }

  void RegisterPreparedLexicalDomains(
      const std::vector<std::shared_ptr<IRStructure>> &root) {
    std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
      if (!node)
        return;
      if (node->IsControl()) {
        auto *loop = static_cast<ControlNode *>(node);
        auto annotations = ReadLoopAnnotations(loop->control.get());
        auto guards = annotations.Get(kStorageEpochGuardMap);
        if (guards.has_value()) {
          StorageEpochGuardMap guard_map =
              guards.value().cast<StorageEpochGuardMap>();
          std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>
              counter_storages;
          if (auto counters = annotations.Get(kMultiBufferCounterMap)) {
            MultiBufferCounterMap counter_map =
                counters.value().cast<MultiBufferCounterMap>();
            for (const auto &[storage, _] : counter_map)
              counter_storages.insert(storage);
          }
          for (const auto &[storage, guard] : guard_map) {
            // Counter storages are bound once by RegisterCounterDomains. Their
            // entries remain in this shared annotation only to serialize the
            // canonical active guard alongside the physical counter binding.
            if (counter_storages.count(storage))
              continue;
            int domain_id =
                InternDomain(EpochDomainKind::kLexical, loop, guard);
            BindStorage(&lexical_bindings_, storage, {loop, 0}, domain_id);
          }
        }
      }
      for (const auto *list : node->ChildLists())
        for (const auto &child : *list)
          visit(child.get());
    };
    for (const auto &node : root)
      visit(node.get());
  }

  void RegisterCounterDomains() {
    for (const MultiBufferInfo &info : multi_buffer_plan_.Infos()) {
      if (!info.UsesCounter())
        continue;
      for (const MultiBufferOwnerInfo &owner : info.owners) {
        int domain_id = InternDomain(EpochDomainKind::kCounter, owner.loop,
                                     owner.active_guard, info.counter);
        BindStorage(&counter_bindings_, info.storage, {owner.loop, 0},
                    domain_id);
      }
    }
  }

  const ConstrSet &ContextForScope(ScopeKey scope,
                                   const ConstrSet &empty) const {
    if (scope.owner == nullptr)
      return empty;
    if (scope.owner->IsControl())
      return static_cast<const ControlNode *>(scope.owner)
          ->GetLoopBodyContext();
    auto lists = scope.owner->ChildLists();
    ICHECK_LT(scope.child_list_index, lists.size());
    // Decoder contexts capture an If condition once, with a fresh symbol for
    // mutable reads. Reuse that branch context rather than re-reading the
    // condition after intervening stores, or proving else facts in then.
    for (const auto &child : *lists[scope.child_list_index]) {
      if (child->IsTask())
        return static_cast<const TaskNode *>(child.get())->outer_ctx;
      if (const auto *header = child->HeaderTask())
        return header->outer_ctx;
    }
    if (const auto *header = scope.owner->HeaderTask())
      return header->outer_ctx;
    return empty;
  }

  void ReorderDomainsForProjection() {
    size_t num_domains = domains_.size();
    using ProjectionMatrix = std::vector<std::vector<bool>>;
    std::map<ScopeKey, ProjectionMatrix> projections_by_scope;
    std::vector<std::vector<bool>> can_project(
        num_domains, std::vector<bool>(num_domains, false));

    ConstrSet empty_ctx;
    for (const auto &[scope, _] : lexical_domain_ids_) {
      const ConstrSet &ctx = ContextForScope(scope, empty_ctx);
      ProjectionMatrix &scope_projection = projections_by_scope[scope];
      scope_projection.assign(num_domains,
                              std::vector<bool>(num_domains, false));
      for (size_t source = 0; source < num_domains; ++source) {
        for (size_t target = 0; target < num_domains; ++target) {
          scope_projection[source][target] =
              source == target ||
              GuardImplies(domains_[target].active_guard,
                           domains_[source].active_guard, ctx);
        }
      }
      for (size_t intermediate = 0; intermediate < num_domains;
           ++intermediate) {
        for (size_t source = 0; source < num_domains; ++source) {
          if (!scope_projection[source][intermediate])
            continue;
          for (size_t target = 0; target < num_domains; ++target) {
            scope_projection[source][target] =
                scope_projection[source][target] ||
                scope_projection[intermediate][target];
          }
        }
      }
      for (size_t source = 0; source < num_domains; ++source) {
        for (size_t target = 0; target < num_domains; ++target) {
          can_project[source][target] =
              can_project[source][target] || scope_projection[source][target];
        }
      }
    }
    for (size_t intermediate = 0; intermediate < num_domains; ++intermediate) {
      for (size_t source = 0; source < num_domains; ++source) {
        if (!can_project[source][intermediate])
          continue;
        for (size_t target = 0; target < num_domains; ++target) {
          can_project[source][target] =
              can_project[source][target] || can_project[intermediate][target];
        }
      }
    }

    std::vector<size_t> order;
    order.reserve(num_domains);
    for (size_t i = 0; i < num_domains; ++i)
      order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
      size_t lhs_targets =
          std::count(can_project[lhs].begin(), can_project[lhs].end(), true);
      size_t rhs_targets =
          std::count(can_project[rhs].begin(), can_project[rhs].end(), true);
      if (lhs_targets != rhs_targets)
        return lhs_targets < rhs_targets;
      return domains_[lhs].id < domains_[rhs].id;
    });

    std::vector<int> old_to_new(num_domains + 1, 0);
    std::vector<EpochDomain> reordered;
    reordered.reserve(num_domains);
    for (size_t old_index : order) {
      EpochDomain domain = domains_[old_index];
      int new_id = static_cast<int>(reordered.size()) + 1;
      old_to_new[domain.id] = new_id;
      domain.id = new_id;
      reordered.push_back(std::move(domain));
    }
    domains_ = std::move(reordered);

    auto remap = [&](int *domain_id) {
      ICHECK(domain_id != nullptr);
      ICHECK_GT(*domain_id, 0);
      ICHECK_LT(static_cast<size_t>(*domain_id), old_to_new.size());
      *domain_id = old_to_new[*domain_id];
    };
    for (auto &[_, domain_id] : lexical_domain_ids_)
      remap(&domain_id);
    for (auto &[_, by_scope] : lexical_bindings_) {
      for (auto &[_, domain_id] : by_scope)
        remap(&domain_id);
    }
    for (auto &[_, by_scope] : counter_bindings_) {
      for (auto &[_, domain_id] : by_scope)
        remap(&domain_id);
    }

    for (const auto &[scope, scope_projection] : projections_by_scope) {
      auto &targets_by_source = projection_targets_[scope];
      for (size_t old_source = 0; old_source < num_domains; ++old_source) {
        std::vector<int> &targets =
            targets_by_source[old_to_new[old_source + 1]];
        for (size_t old_target = 0; old_target < num_domains; ++old_target) {
          if (scope_projection[old_source][old_target])
            targets.push_back(old_to_new[old_target + 1]);
        }
        std::sort(targets.begin(), targets.end());
      }
    }
  }

  void BindStorage(std::unordered_map<Var, std::map<ScopeKey, int>,
                                      ObjectPtrHash, ObjectPtrEqual> *bindings,
                   const Var &storage, ScopeKey owner, int domain_id) {
    ICHECK(bindings != nullptr);
    auto [_, inserted] = (*bindings)[storage].emplace(owner, domain_id);
    ICHECK(inserted) << "Duplicate storage epoch binding for " << storage->name;
  }

  const int *FindStorageBinding(
      const std::unordered_map<Var, std::map<ScopeKey, int>, ObjectPtrHash,
                               ObjectPtrEqual> &bindings,
      const Var &storage, ScopeKey owner) const {
    auto storage_it = bindings.find(storage);
    if (storage_it == bindings.end())
      return nullptr;
    auto loop_it = storage_it->second.find(owner);
    return loop_it == storage_it->second.end() ? nullptr : &loop_it->second;
  }

  int LexicalDomainId(const Var &storage, ScopeKey owner) const {
    if (const int *binding =
            FindStorageBinding(lexical_bindings_, storage, owner)) {
      return *binding;
    }
    return UnconditionalLexicalDomain(owner.owner, owner.child_list_index);
  }

  void AddLexicalScope(IRStructure *scope, size_t child_list_index = 0) {
    int domain_id =
        InternDomain(EpochDomainKind::kLexical, scope, IntImm::Bool(true),
                     TensorVar(ffi::UnsafeInit{}), child_list_index);
    auto [it, inserted] = lexical_domain_ids_.emplace(
        ScopeKey{scope, child_list_index}, domain_id);
    ICHECK(inserted || it->second == domain_id)
        << "A lexical scope has inconsistent unconditional domains";
  }

  const std::vector<Var> &TaskStorages(const TaskNode *task) const {
    auto existing = task_storages_.find(task);
    if (existing != task_storages_.end())
      return existing->second;

    std::vector<Var> storages;
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen_storages;
    auto append = [&](const std::vector<TensorRegion> &regions) {
      for (const TensorRegion &region : regions) {
        Var storage = task->StorageIdentity(RegionTensor(region));
        if (!seen_storages.insert(storage).second)
          continue;
        if (IsRegisterRegion(region)) {
          continue;
        }
        if (const MultiBufferInfo *info = multi_buffer_plan_.Find(storage)) {
          if (!IsMultiBufferBroadcastFill(task, info->storage)) {
            ICHECK(info->FindOwner(task) != nullptr)
                << "Multi-buffer task is outside every owner for storage "
                << info->storage->name;
          }
        }
        storages.push_back(storage);
      }
    };
    append(task->GetReadRegions());
    append(task->GetWriteRegions());
    return task_storages_.emplace(task, std::move(storages)).first->second;
  }

  const MultiBufferPlan &multi_buffer_plan_;
  std::vector<EpochDomain> domains_;
  std::map<ScopeKey, int> lexical_domain_ids_;
  std::unordered_map<Var, std::map<ScopeKey, int>, ObjectPtrHash,
                     ObjectPtrEqual>
      lexical_bindings_;
  std::unordered_map<Var, std::map<ScopeKey, int>, ObjectPtrHash,
                     ObjectPtrEqual>
      counter_bindings_;
  std::map<ScopeKey, std::map<int, std::vector<int>>> projection_targets_;
  // Pass-local lazy cache. EpochDomainRegistry is owned and consumed by one
  // single-threaded kernel rewrite.
  mutable std::unordered_map<const TaskNode *, std::vector<Var>> task_storages_;
};

} // namespace ascend
} // namespace tl
} // namespace tvm
