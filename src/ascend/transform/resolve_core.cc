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
  if (core_mask == kCoreVector0)
    return "Vector0";
  if (core_mask == kCoreVector1)
    return "Vector1";
  if (core_mask == kCoreVector)
    return "Vector0 and Vector1";
  if (core_mask == (kCoreCube | kCoreVector1))
    return "Vector1 and Cube";
  if (core_mask == kCoreAll)
    return "Vector0, Vector1 and Cube";
  if (core_mask == kCoreCube)
    return "Cube";
  if (core_mask == (kCoreCube | kCoreVector0))
    return "Vector0 and Cube";
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
                      const ffi::Optional<Var> &outer_sid,
                      int num_aiv_subcores) {
    CoreMaskResolver resolver(root, all_tasks, plan, outer_sid,
                              num_aiv_subcores);
    resolver.Run();
  }

private:
  static constexpr int kSpecialRegisterBits =
      std::numeric_limits<uint16_t>::digits;
  // Cube is one physical core, but its work can belong to a shared stream or
  // either projected subcore. Preserve that domain through scalar/control
  // demand so a producer also retains the exits of its logical loop.
  static constexpr uint8_t kSharedCubeDomain = 1U;
  static uint8_t SubcoreCubeDomain(int sid) { return 1U << (sid + 1); }
  struct CoreDemand {
    CoreMask cores{kCoreUnassigned};
    uint8_t cube_domains{0};

    bool Merge(CoreDemand other) {
      CoreMask updated_cores = cores | other.cores;
      uint8_t updated_domains = cube_domains | other.cube_domains;
      bool changed = updated_cores != cores || updated_domains != cube_domains;
      cores = updated_cores;
      cube_domains = updated_domains;
      return changed;
    }

    CoreDemand Restrict(CoreMask mask) const {
      return {static_cast<CoreMask>(cores & mask),
              static_cast<uint8_t>((mask & kCoreCube) ? cube_domains : 0)};
    }
  };
  using DemandMap =
      std::unordered_map<Var, CoreDemand, ObjectPtrHash, ObjectPtrEqual>;
  using SpecialRegisterDemandMap =
      std::unordered_map<const TaskNode *, CoreDemand>;
  using SpecialRegisterReadDemand =
      std::array<CoreDemand, kSpecialRegisterBits>;
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
                   const ffi::Optional<Var> &outer_sid, int num_aiv_subcores)
      : root_(root), all_tasks_(all_tasks), plan_(plan),
        epoch_domains_(root, plan), availability_(all_tasks),
        active_core_mask_(kCoreCube | VectorCoreMask(num_aiv_subcores)) {
    if (outer_sid.has_value())
      availability_.SetExternalVarCoreMask(outer_sid.value(), kCoreVector);
    RegisterStorageEpochGuards();
    RefineCandidateCoreMasks();
    for (TaskNode *task : all_tasks_) {
      CoreMask candidate = task->GetCoreMask();
      candidate_masks_[task] = candidate;
      execution_demands_[task] = InitialDemand(
          task, IsCoreLocalMask(candidate) ? candidate : kCoreUnassigned);
    }
  }

  void Run() {
    PropagateToFixedPoint();
    AssignFallbackCores();
    PropagateToFixedPoint();
    for (TaskNode *task : all_tasks_) {
      CoreMask resolved = GetExecutionDemand(task).cores;
      ICHECK_NE(resolved, kCoreUnassigned)
          << "ResolveCore left a task without an execution core";
      task->SetCoreMask(resolved);
    }
  }

  CoreDemand GetExecutionDemand(const TaskNode *task) const {
    auto it = execution_demands_.find(task);
    ICHECK(it != execution_demands_.end());
    return it->second;
  }

  CoreDemand InitialDemand(const TaskNode *task, CoreMask cores) const {
    CoreDemand demand{cores};
    if (!(cores & kCoreCube))
      return demand;
    // Pure scalar/register producers and configuration writers inherit their
    // consumers' domains. In particular, an unused binding placed on Cube as
    // a fallback and a break's own condition must not seed shared Cube work.
    bool external_effect = false;
    PostOrderVisit(task->stmt, [&](const ObjectRef &node) {
      if (const auto *store = node.as<BufferStoreNode>())
        external_effect |=
            !IsRegisterRegion(BufferRegion::FullRegion(store->buffer));
      if (const auto *call = node.as<CallNode>()) {
        Call expression = ffi::GetRef<Call>(call);
        Stmt statement = Evaluate(expression);
        external_effect |=
            SideEffect(expression) > CallEffectKind::kReadState &&
            (AnalyzeSpecialRegisterWrites(statement) == 0 ||
             HasCrossCoreSync(statement));
      }
    });
    if (external_effect) {
      auto sid = GetTaskSubcore(task->stmt);
      demand.cube_domains = sid ? SubcoreCubeDomain(*sid) : kSharedCubeDomain;
    }
    return demand;
  }

  CoreMask GetRefinedCandidateCoreMask(TaskNode *task) const {
    CoreMask candidate = task->GetCoreMask();
    ICHECK_NE(candidate, kCoreUnassigned)
        << "ResolveCore requires AssignCore to run first";
    candidate &= active_core_mask_ & availability_.GetTaskInputCoreMask(task);
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

  bool AddDemand(DemandMap *demands, const Var &value, CoreDemand demand) {
    if (demand.cores == kCoreUnassigned)
      return false;
    return (*demands)[value].Merge(demand);
  }

  bool AddReadDemand(const TaskNode *task, CoreDemand demand) {
    if (demand.cores == kCoreUnassigned)
      return false;
    bool changed = false;
    for (const Var &var : task->GetReadVars())
      changed |= AddDemand(&var_demand_, var, demand);
    for (const BufferRegion &region : task->GetReadRegions()) {
      if (IsRegisterRegion(region))
        changed |= AddDemand(&register_demand_, region->buffer->data, demand);
    }
    return changed;
  }

  bool AddReadDemand(const TaskAccessInfo &accesses, CoreDemand demand) {
    if (demand.cores == kCoreUnassigned)
      return false;
    bool changed = false;
    for (const Var &var : accesses.read_vars)
      changed |= AddDemand(&var_demand_, var, demand);
    for (const BufferRegion &region : accesses.read_regions) {
      if (IsRegisterRegion(region))
        changed |= AddDemand(&register_demand_, region->buffer->data, demand);
    }
    return changed;
  }

  bool AddStorageEpochGuardDemand(const TaskNode *task, CoreDemand demand) {
    auto it = task_epoch_guards_.find(task);
    if (it == task_epoch_guards_.end())
      return false;
    bool changed = false;
    for (size_t guard_index : it->second) {
      ICHECK_EQ(
          availability_.GetAccessCoreMask(epoch_guards_[guard_index].accesses) &
              demand.cores,
          demand.cores)
          << "Cannot evaluate storage-epoch guard on "
          << CoreMaskName(demand.cores) << ": "
          << epoch_guards_[guard_index].expression;
      changed |= AddReadDemand(epoch_guards_[guard_index].accesses, demand);
    }
    return changed;
  }

  CoreDemand GetWriteDemand(const TaskNode *task) const {
    CoreDemand demand;
    auto special_it = special_writer_demand_.find(task);
    if (special_it != special_writer_demand_.end())
      demand.Merge(special_it->second);
    for (const Var &var : task->GetWriteVars()) {
      auto it = var_demand_.find(var);
      if (it != var_demand_.end())
        demand.Merge(it->second);
    }
    for (const BufferRegion &region : task->GetWriteRegions()) {
      if (!IsRegisterRegion(region))
        continue;
      auto it = register_demand_.find(region->buffer->data);
      if (it != register_demand_.end())
        demand.Merge(it->second);
    }
    return demand;
  }

  bool RequireExecution(TaskNode *task, CoreDemand demand) {
    if (demand.cores == kCoreUnassigned)
      return false;
    CoreMask candidate = candidate_masks_.at(task);
    if ((demand.cores & candidate) != demand.cores) {
      LOG(FATAL) << "Cannot resolve core placement for task: consumers require "
                 << CoreMaskName(demand.cores)
                 << ", but the task can only run on " << CoreMaskName(candidate)
                 << ". The producer cannot be broadcast to all required "
                    "cores because of its resource or core-local memory "
                    "accesses: "
                 << task->stmt;
    }
    if ((demand.cores & kCoreCube) &&
        !(GetExecutionDemand(task).cores & kCoreCube))
      demand.Merge(InitialDemand(task, kCoreCube));
    // A projected scalar may be consumed after its loop by shared Cube work.
    // Its computation still belongs to its own subcore's control domain.
    if (auto sid = GetTaskSubcore(task->stmt); sid && demand.cube_domains)
      demand.cube_domains = SubcoreCubeDomain(*sid);
    return execution_demands_.at(task).Merge(demand);
  }

  CoreDemand GetSubtreeExecutionDemand(const IRStructure *node) const {
    if (node->IsTask())
      return GetExecutionDemand(static_cast<const TaskNode *>(node));
    CoreDemand result;
    const auto *control = static_cast<const ControlNode *>(node);
    for (const auto &child : control->children)
      result.Merge(GetSubtreeExecutionDemand(child.get()));
    return result;
  }

  CoreDemand GetLoopExitDemand(const TaskNode *task,
                               CoreDemand enclosing_demand) const {
    auto sid = GetTaskSubcore(task->stmt);
    if (!sid)
      return enclosing_demand;
    CoreDemand demand = enclosing_demand.Restrict(VectorCore(*sid));
    uint8_t owners = enclosing_demand.cube_domains;
    // One physical Cube loop cannot exit independently for multiple logical
    // subcore streams, nor can a subcore-local break stop shared Cube work.
    ICHECK(!(owners & kSharedCubeDomain) && !((owners & SubcoreCubeDomain(0)) &&
                                              (owners & SubcoreCubeDomain(1))))
        << "Independent subcore loop_break with shared Cube work is not "
           "supported in one loop; put each subcore branch and its Cube work "
           "in a separate loop";
    if (owners & SubcoreCubeDomain(*sid))
      demand.Merge({kCoreCube, SubcoreCubeDomain(*sid)});
    return demand;
  }

  bool SeedTreeReadDemand(IRStructure *node, CoreDemand enclosing_demand) {
    if (node->IsTask()) {
      auto *task = static_cast<TaskNode *>(node);
      bool changed = false;
      if (task->ContainsLoopBreak())
        changed |=
            RequireExecution(task, GetLoopExitDemand(task, enclosing_demand));
      CoreDemand demand = GetExecutionDemand(task);
      changed |= AddReadDemand(task, demand);
      changed |= AddStorageEpochGuardDemand(task, demand);
      changed |= AddCounterDemand(task, demand);
      return changed;
    }
    auto *control = static_cast<ControlNode *>(node);
    CoreDemand demand = GetSubtreeExecutionDemand(node);
    bool changed = AddReadDemand(control->task.get(), demand);
    changed |= AddCounterDemand(control->task.get(), demand);
    CheckExpressionAvailability(control->task.get(), demand.cores);
    for (const auto &child : control->children)
      changed |= SeedTreeReadDemand(child.get(), demand);
    return changed;
  }

  // Special registers have one independent instance per core. Return both
  // reader demand and concrete descendant writers so readers in an outer
  // sibling list can place writers nested under a control node. loop_break
  // instead uses GetLoopExitDemand to retain its projected control domain.
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
          scope.readers[bit_index].Merge(child_scope.readers[bit_index]);
          scope.writers[bit_index].insert(
              scope.writers[bit_index].end(),
              child_scope.writers[bit_index].begin(),
              child_scope.writers[bit_index].end());
        }
      } else {
        const auto *task = static_cast<const TaskNode *>(node.get());
        CoreDemand demand = GetExecutionDemand(task);
        uint16_t read_mask = node->GetSpecialReadMask();
        for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
          uint16_t bit = static_cast<uint16_t>(1U << bit_index);
          if ((read_mask & bit) != 0)
            scope.readers[bit_index].Merge(demand);
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
      if ((1U << bit_index) ==
          static_cast<uint16_t>(SpecialRegister::kLoopControl))
        continue;
      for (TaskNode *writer : scope.writers[bit_index])
        special_writer_demand_[writer].Merge(
            scope.readers[bit_index].Restrict(candidate_masks_.at(writer)));
    }
    return scope;
  }

  bool AddCounterDemand(const TaskNode *task, CoreDemand demand) {
    auto it = task_counter_groups_.find(task);
    if (it == task_counter_groups_.end())
      return false;
    bool changed = false;
    // MaterializeMultiBuffer will add one counter load for each returned
    // group, so its writers must exist on every core that executes this task.
    for (const MultiBufferInfo *info : it->second)
      changed |= AddDemand(&register_demand_, info->counter->data, demand);
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
    // Physical cores and logical Cube domains only gain bits, so rescanning
    // converges even when a consumer outside a loop adds a new exit demand.
    bool changed = true;
    while (changed) {
      changed = false;
      special_writer_demand_.clear();
      CollectSpecialRegisterDemand(root_);
      for (const auto &node : root_)
        changed |= SeedTreeReadDemand(node.get(),
                                      GetSubtreeExecutionDemand(node.get()));
      for (TaskNode *task : all_tasks_)
        changed |= RequireExecution(task, GetWriteDemand(task));
    }
  }

  void AssignFallbackCores() {
    bool has_cube_core = false;
    for (TaskNode *task : all_tasks_) {
      if (HasCore(GetExecutionDemand(task).cores, kCoreCube))
        has_cube_core = true;
    }
    for (TaskNode *task : all_tasks_) {
      if (GetExecutionDemand(task).cores != kCoreUnassigned)
        continue;
      CoreMask available = candidate_masks_.at(task);
      ICHECK_NE(available, kCoreUnassigned)
          << "ResolveCore stored an empty candidate mask";
      CoreMask fallback = available & kCoreVector;
      if ((has_cube_core && HasCore(available, kCoreCube)) ||
          fallback == kCoreUnassigned)
        fallback = kCoreCube;
      if (auto sid = GetTaskSubcore(task->stmt);
          sid && HasCore(available, VectorCore(*sid)))
        fallback = VectorCore(*sid);
      RequireExecution(task, {fallback});
    }
  }

  const std::vector<std::shared_ptr<IRStructure>> &root_;
  const std::vector<TaskNode *> &all_tasks_;
  const MultiBufferPlan &plan_;
  EpochDomainRegistry epoch_domains_;
  CoreMaskAvailability availability_;
  CoreMask active_core_mask_;
  std::unordered_map<const TaskNode *, CoreMask> candidate_masks_;
  std::unordered_map<const TaskNode *, CoreDemand> execution_demands_;
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
                      const L0StorageGroups &groups, int num_aiv_subcores) {
  std::vector<TaskNode *> all_tasks;
  CollectAllTaskNodes(root, all_tasks);
  // PrepareMultiBuffer persists the chosen counters and active guards on owner
  // loops; reconstruct exactly that plan instead of recomputing it here.
  MultiBufferPlan plan = ReadMultiBufferPlan(root, buffer_versions, groups);
  CoreMaskResolver::Resolve(root, all_tasks, plan, outer_sid, num_aiv_subcores);
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
                           L0StorageGroups(CollectL0SFBindings(context.root)),
                           scheduled_tir.metadata.num_aiv_subcores.value_or(2));
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
