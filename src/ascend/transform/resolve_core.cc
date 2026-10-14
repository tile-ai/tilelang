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
 * \file resolve_core.cc
 * \brief Narrow scheduled-task core candidates from their actual consumers.
 */

#include <algorithm>
#include <array>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/transform.h>

#include "./auto_schedule/ir_structure.h"
#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/multi_buffer.h"
#include "./auto_schedule/scheduled_tir.h"
#include "./auto_schedule/task_analysis.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;

namespace {

const char *CoreMaskName(CoreMask core_mask) {
  if (core_mask == kCoreVector)
    return "Vector";
  if (core_mask == kCoreCube)
    return "Cube";
  if (core_mask == kCoreBroadcast)
    return "Vector and Cube";
  return "no core";
}

// Resolve candidate masks to the cores that actually consume each scalar,
// register value, special register, control expression, prepared storage-epoch
// guard, and multi-buffer counter. Every update is a subset of the input
// candidate mask.
class CoreMaskResolver {
public:
  static void Resolve(const std::vector<std::shared_ptr<IRStructure>> &root,
                      const std::vector<TaskNode *> &all_tasks,
                      const MultiBufferPlan &plan,
                      const ffi::Optional<Var> &outer_sid) {
    CoreMaskResolver resolver(root, all_tasks, plan, outer_sid);
    resolver.Run();
  }

private:
  static constexpr int kSpecialRegisterBits =
      std::numeric_limits<uint16_t>::digits;
  using DemandMap =
      std::unordered_map<Var, CoreMask, ObjectPtrHash, ObjectPtrEqual>;
  using SpecialRegisterDemandMap =
      std::unordered_map<const TaskNode *, CoreMask>;
  using SpecialRegisterReadDemand = std::array<CoreMask, kSpecialRegisterBits>;
  using SpecialRegisterWriters =
      std::array<std::vector<TaskNode *>, kSpecialRegisterBits>;
  struct SpecialRegisterScopeDemand {
    SpecialRegisterReadDemand readers{};
    SpecialRegisterWriters writers;
  };
  struct PreparedEpochGuard {
    PrimExpr expression;
    TaskAccessInfo accesses;
  };

  CoreMaskResolver(const std::vector<std::shared_ptr<IRStructure>> &root,
                   const std::vector<TaskNode *> &all_tasks,
                   const MultiBufferPlan &plan,
                   const ffi::Optional<Var> &outer_sid)
      : root_(root), all_tasks_(all_tasks), plan_(plan),
        epoch_domains_(root, plan), availability_(all_tasks) {
    if (outer_sid.has_value())
      availability_.SetExternalVarCoreMask(outer_sid.value(), kCoreVector);
    RegisterStorageEpochGuards();
    RefineCandidateCoreMasks();
    for (TaskNode *task : all_tasks_) {
      CoreMask candidate = task->GetCoreMask();
      candidate_masks_[task] = candidate;
      execution_masks_[task] =
          IsConcreteCore(candidate) ? candidate : kCoreUnassigned;
    }
  }

  void Run() {
    PropagateToFixedPoint();
    AssignFallbackCores();
    PropagateToFixedPoint();
    for (TaskNode *task : all_tasks_) {
      CoreMask resolved = execution_masks_.at(task);
      ICHECK_NE(resolved, kCoreUnassigned)
          << "ResolveCore left a task without an execution core";
      task->SetCoreMask(resolved);
    }
  }

  CoreMask GetExecutionMask(const TaskNode *task) const {
    auto it = execution_masks_.find(task);
    ICHECK(it != execution_masks_.end());
    return it->second;
  }

  CoreMask GetRefinedCandidateCoreMask(TaskNode *task) const {
    CoreMask candidate = task->GetCoreMask();
    ICHECK_NE(candidate, kCoreUnassigned)
        << "ResolveCore requires AssignCore to run first";
    candidate &= availability_.GetTaskInputCoreMask(task);
    if (candidate == kCoreUnassigned) {
      LOG(FATAL) << "Cannot resolve core placement because a task's scalar "
                    "or control expression is unavailable on every legal "
                    "execution core: "
                 << task->stmt;
    }

    auto epoch_guards = task_epoch_guards_.find(task);
    if (epoch_guards != task_epoch_guards_.end()) {
      for (size_t guard_index : epoch_guards->second) {
        const PreparedEpochGuard &guard = epoch_guards_[guard_index];
        CoreMask available = availability_.GetAccessCoreMask(guard.accesses);
        if ((candidate & available) == kCoreUnassigned) {
          LOG(FATAL) << "Cannot resolve core placement because storage-epoch "
                        "guard "
                     << guard.expression
                     << " is unavailable on every legal execution core for "
                        "task: "
                     << task->stmt;
        }
        candidate &= available;
      }
    }

    for (const IRStructure *parent = task->GetParent(); parent != nullptr;
         parent = parent->GetParent()) {
      ICHECK(parent->IsControl());
      const auto *control = static_cast<const ControlNode *>(parent);
      CoreMask control_available =
          availability_.GetTaskReadCoreMask(control->task.get());
      if ((candidate & control_available) == kCoreUnassigned) {
        LOG(FATAL) << "Cannot evaluate a loop bound or control guard on any "
                      "legal execution core for task: "
                   << task->stmt << ". The enclosing control is available on "
                   << CoreMaskName(control_available) << ": "
                   << control->task->stmt;
      }
      candidate &= control_available;
    }
    return candidate;
  }

  void RefineCandidateCoreMasks() {
    // Candidate masks only lose core bits. Writing each refinement back lets
    // control-path restrictions propagate through scalar/register readers.
    bool changed = true;
    while (changed) {
      changed = false;
      for (TaskNode *task : all_tasks_) {
        CoreMask current = task->GetCoreMask();
        CoreMask refined = GetRefinedCandidateCoreMask(task);
        if (refined != current) {
          task->SetCoreMask(refined);
          changed = true;
        }
      }
    }
  }

  bool AddDemand(DemandMap *demands, const Var &value, CoreMask mask) {
    if (mask == kCoreUnassigned)
      return false;
    CoreMask &current = (*demands)[value];
    CoreMask updated = current | mask;
    if (updated == current)
      return false;
    current = updated;
    return true;
  }

  bool AddReadDemand(const TaskNode *task, CoreMask mask) {
    if (mask == kCoreUnassigned)
      return false;
    bool changed = false;
    for (const Var &var : task->GetReadVars())
      changed |= AddDemand(&var_demand_, var, mask);
    for (const BufferRegion &region : task->GetReadRegions()) {
      if (IsRegisterRegion(region))
        changed |= AddDemand(&register_demand_, region->buffer->data, mask);
    }
    return changed;
  }

  bool AddReadDemand(const TaskAccessInfo &accesses, CoreMask mask) {
    if (mask == kCoreUnassigned)
      return false;
    bool changed = false;
    for (const Var &var : accesses.read_vars)
      changed |= AddDemand(&var_demand_, var, mask);
    for (const BufferRegion &region : accesses.read_regions) {
      if (IsRegisterRegion(region))
        changed |= AddDemand(&register_demand_, region->buffer->data, mask);
    }
    return changed;
  }

  bool AddStorageEpochGuardDemand(const TaskNode *task, CoreMask mask) {
    auto it = task_epoch_guards_.find(task);
    if (it == task_epoch_guards_.end())
      return false;
    bool changed = false;
    for (size_t guard_index : it->second)
      changed |= AddReadDemand(epoch_guards_[guard_index].accesses, mask);
    return changed;
  }

  CoreMask GetWriteDemand(const TaskNode *task) const {
    CoreMask mask = kCoreUnassigned;
    auto special_it = special_writer_demand_.find(task);
    if (special_it != special_writer_demand_.end())
      mask |= special_it->second;
    for (const Var &var : task->GetWriteVars()) {
      auto it = var_demand_.find(var);
      if (it != var_demand_.end())
        mask |= it->second;
    }
    for (const BufferRegion &region : task->GetWriteRegions()) {
      if (!IsRegisterRegion(region))
        continue;
      auto it = register_demand_.find(region->buffer->data);
      if (it != register_demand_.end())
        mask |= it->second;
    }
    return mask;
  }

  bool RequireExecution(TaskNode *task, CoreMask demand) {
    if (demand == kCoreUnassigned)
      return false;
    CoreMask candidate = candidate_masks_.at(task);
    if ((demand & candidate) != demand) {
      LOG(FATAL) << "Cannot resolve core placement for task: consumers require "
                 << CoreMaskName(demand) << ", but the task can only run on "
                 << CoreMaskName(candidate)
                 << ". The producer cannot be broadcast to all required "
                    "cores because of its resource or core-local memory "
                    "accesses: "
                 << task->stmt;
    }
    CoreMask &current = execution_masks_.at(task);
    CoreMask updated = current | demand;
    if (updated == current)
      return false;
    ICHECK_EQ(updated & candidate, updated);
    current = updated;
    return true;
  }

  CoreMask GetSubtreeExecutionMask(const IRStructure *node) const {
    if (node->IsTask())
      return GetExecutionMask(static_cast<const TaskNode *>(node));
    CoreMask result = kCoreUnassigned;
    const auto *control = static_cast<const ControlNode *>(node);
    for (const auto &child : control->children)
      result |= GetSubtreeExecutionMask(child.get());
    return result;
  }

  bool SeedTreeReadDemand(IRStructure *node, CoreMask enclosing_coverage) {
    if (node->IsTask()) {
      auto *task = static_cast<TaskNode *>(node);
      bool changed = false;
      if (task->ContainsLoopBreak())
        changed |= RequireExecution(task, enclosing_coverage);
      CoreMask execution_mask = GetExecutionMask(task);
      changed |= AddReadDemand(task, execution_mask);
      changed |= AddStorageEpochGuardDemand(task, execution_mask);
      changed |= AddCounterDemand(task, execution_mask);
      return changed;
    }
    auto *control = static_cast<ControlNode *>(node);
    CoreMask coverage = GetSubtreeExecutionMask(node);
    bool changed = AddReadDemand(control->task.get(), coverage);
    changed |= AddCounterDemand(control->task.get(), coverage);
    CheckExpressionAvailability(control->task.get(), coverage);
    for (const auto &child : control->children)
      changed |= SeedTreeReadDemand(child.get(), coverage);
    return changed;
  }

  // Special registers have one independent instance per core. Return both
  // reader demand and concrete descendant writers so readers in an outer
  // sibling list can place writers nested under a control node. loop_break is
  // body-local and is therefore the only register not propagated outward.
  SpecialRegisterScopeDemand CollectSpecialRegisterDemand(
      const std::vector<std::shared_ptr<IRStructure>> &nodes) {
    SpecialRegisterScopeDemand scope;

    for (const auto &node : nodes) {
      if (node->IsControl()) {
        const auto *control = static_cast<const ControlNode *>(node.get());
        SpecialRegisterScopeDemand child_scope =
            CollectSpecialRegisterDemand(control->children);
        for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
          uint16_t bit = static_cast<uint16_t>(1U << bit_index);
          if (bit == static_cast<uint16_t>(SpecialRegister::kLoopControl))
            continue;
          scope.readers[bit_index] |= child_scope.readers[bit_index];
          scope.writers[bit_index].insert(
              scope.writers[bit_index].end(),
              child_scope.writers[bit_index].begin(),
              child_scope.writers[bit_index].end());
        }
      } else {
        const auto *task = static_cast<const TaskNode *>(node.get());
        CoreMask core_mask = GetExecutionMask(task);
        uint16_t read_mask = node->GetSpecialReadMask();
        for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
          uint16_t bit = static_cast<uint16_t>(1U << bit_index);
          if ((read_mask & bit) != 0)
            scope.readers[bit_index] |= core_mask;
        }
      }

      if (!node->IsTask())
        continue;
      auto *task = static_cast<TaskNode *>(node.get());
      // Cross-core bits only order their target hardware pipe. Their placement
      // was fixed by AssignCore rather than ordinary register demand.
      if (HasCrossCoreSync(task->stmt))
        continue;
      uint16_t write_mask = task->GetSpecialWriteMask();
      for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
        uint16_t bit = static_cast<uint16_t>(1U << bit_index);
        if ((write_mask & bit) != 0)
          scope.writers[bit_index].push_back(task);
      }
    }

    for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
      for (TaskNode *writer : scope.writers[bit_index])
        special_writer_demand_[writer] |= scope.readers[bit_index];
    }
    return scope;
  }

  bool AddCounterDemand(const TaskNode *task, CoreMask mask) {
    auto it = task_counter_groups_.find(task);
    if (it == task_counter_groups_.end())
      return false;
    bool changed = false;
    // MaterializeMultiBuffer will add one counter load for each returned
    // group, so its writers must exist on every core that executes this task.
    for (const MultiBufferInfo *info : it->second)
      changed |= AddDemand(&register_demand_, info->counter->data, mask);
    return changed;
  }

  void CheckExpressionAvailability(const TaskNode *task,
                                   CoreMask demand) const {
    if (demand == kCoreUnassigned)
      return;
    CoreMask available = availability_.GetTaskReadCoreMask(task);
    if ((available & demand) != demand) {
      LOG(FATAL) << "Cannot evaluate a loop bound or control guard on "
                 << CoreMaskName(demand)
                 << " because it reads core-local state available only on "
                 << CoreMaskName(available) << ": " << task->stmt;
    }
  }

  void RegisterStorageEpochGuards() {
    std::unordered_map<PrimExpr, size_t, ObjectPtrHash, ObjectPtrEqual>
        guard_indices;

    auto intern_guard = [&](const PrimExpr &guard) {
      auto [it, inserted] = guard_indices.emplace(guard, epoch_guards_.size());
      if (inserted) {
        epoch_guards_.push_back({guard, AnalyzeTaskAccesses(Evaluate(guard))});
      }
      return it->second;
    };
    auto add_task_guard = [&](TaskNode *task, const PrimExpr &guard) {
      if (is_one(guard))
        return;
      size_t guard_index = intern_guard(guard);
      std::vector<size_t> &task_guards = task_epoch_guards_[task];
      if (std::find(task_guards.begin(), task_guards.end(), guard_index) ==
          task_guards.end()) {
        task_guards.push_back(guard_index);
      }
    };

    for (TaskNode *task : all_tasks_) {
      auto add_scope_guards = [&](ControlNode *scope) {
        for (int domain_id :
             epoch_domains_.DomainsForTaskAtScope(task, scope)) {
          add_task_guard(task, epoch_domains_.Domain(domain_id).active_guard);
        }
      };
      add_scope_guards(nullptr);
      for (IRStructure *node : task->PathFrom()) {
        if (node->IsControl())
          add_scope_guards(static_cast<ControlNode *>(node));
      }
    }

    // A physical counter is replicated per concrete core, so every owner
    // advance guard must be evaluable on every core that uses any storage in
    // the group, including in a different disjoint owner. Model those guards
    // as implicit inputs of all group users before choosing their final cores.
    std::unordered_map<int, const MultiBufferInfo *> group_representatives;
    for (const MultiBufferInfo &info : plan_.Infos()) {
      if (info.UsesCounter())
        group_representatives.emplace(info.counter_group_id, &info);
    }
    for (TaskNode *task : all_tasks_) {
      std::vector<const MultiBufferInfo *> groups;
      std::unordered_set<int> seen_groups;
      auto append = [&](const std::vector<BufferRegion> &regions) {
        for (const BufferRegion &region : regions) {
          const MultiBufferInfo *info = plan_.Find(region->buffer);
          if (info == nullptr || !info->UsesCounter() ||
              IsMultiBufferBroadcastFill(task, info->storage) ||
              !seen_groups.insert(info->counter_group_id).second) {
            continue;
          }
          ICHECK(info->FindOwner(task) != nullptr)
              << "Counter-versioned task is outside every owner for storage "
              << info->storage->name_hint;
          groups.push_back(info);
        }
      };
      append(task->GetReadRegions());
      append(task->GetWriteRegions());
      if (!groups.empty())
        task_counter_groups_.emplace(task, groups);

      for (const MultiBufferInfo *info : groups) {
        const MultiBufferInfo *representative =
            group_representatives.at(info->counter_group_id);
        for (const MultiBufferOwnerInfo &owner : representative->owners)
          add_task_guard(task, owner.active_guard);
      }
    }
  }

  void PropagateToFixedPoint() {
    // Demand and execution masks only gain core bits, so rescanning the tree
    // converges even though special-register demand is rebuilt each round.
    bool changed = true;
    while (changed) {
      changed = false;
      special_writer_demand_.clear();
      CollectSpecialRegisterDemand(root_);
      for (const auto &node : root_)
        changed |= SeedTreeReadDemand(node.get(), kCoreBroadcast);
      for (TaskNode *task : all_tasks_)
        changed |= RequireExecution(task, GetWriteDemand(task));
    }
  }

  void AssignFallbackCores() {
    bool has_cube_core = false;
    for (TaskNode *task : all_tasks_) {
      if (HasCore(GetExecutionMask(task), kCoreCube))
        has_cube_core = true;
    }
    CoreMask preferred_core = has_cube_core ? kCoreCube : kCoreVector;
    for (TaskNode *task : all_tasks_) {
      if (GetExecutionMask(task) != kCoreUnassigned)
        continue;
      CoreMask available = candidate_masks_.at(task);
      ICHECK_NE(available, kCoreUnassigned)
          << "ResolveCore stored an empty candidate mask";
      CoreMask fallback =
          HasCore(available, preferred_core)
              ? preferred_core
              : (HasCore(available, kCoreVector) ? kCoreVector : kCoreCube);
      RequireExecution(task, fallback);
    }
  }

  const std::vector<std::shared_ptr<IRStructure>> &root_;
  const std::vector<TaskNode *> &all_tasks_;
  const MultiBufferPlan &plan_;
  EpochDomainRegistry epoch_domains_;
  CoreMaskAvailability availability_;
  std::unordered_map<const TaskNode *, CoreMask> candidate_masks_;
  std::unordered_map<const TaskNode *, CoreMask> execution_masks_;
  std::vector<PreparedEpochGuard> epoch_guards_;
  std::unordered_map<const TaskNode *, std::vector<size_t>> task_epoch_guards_;
  std::unordered_map<const TaskNode *, std::vector<const MultiBufferInfo *>>
      task_counter_groups_;
  DemandMap var_demand_;
  DemandMap register_demand_;
  SpecialRegisterDemandMap special_writer_demand_;
};

void ResolveCoreMasks(std::vector<std::shared_ptr<IRStructure>> &root,
                      const BufferVersionMap &buffer_versions,
                      const ffi::Optional<Var> &outer_sid,
                      const L0StorageGroups &groups) {
  std::vector<TaskNode *> all_tasks;
  CollectAllTaskNodes(root, all_tasks);
  // PrepareMultiBuffer persists the chosen counters and active guards on owner
  // loops; reconstruct exactly that plan instead of recomputing it here.
  MultiBufferPlan plan = ReadMultiBufferPlan(root, buffer_versions, groups);
  CoreMaskResolver::Resolve(root, all_tasks, plan, outer_sid);
}

} // namespace

tvm::transform::Pass ResolveCore() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    return RewriteTilelangKernels(
        std::move(func), "ResolveCore",
        [](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir =
              DecodeScheduledTIR(context.root, context.outer_ctx);
          ResolveCoreMasks(scheduled_tir.tree,
                           scheduled_tir.metadata.buffer_versions,
                           context.outer_sid,
                           L0StorageGroups(CollectL0SFBindings(context.root)));
          return EncodeScheduledTIR(std::move(scheduled_tir));
        });
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ResolveCore", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ResolveCore", ResolveCore);
}

} // namespace tl
} // namespace tvm
