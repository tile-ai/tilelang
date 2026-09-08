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
 * \file assign_core.cc
 * \brief Assign scheduled tasks to Ascend vector and cube cores.
 */

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "./auto_schedule/ir_structure.h"
#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/scheduled_tir.h"
#include "./auto_schedule/task_analysis.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;

namespace {

static CoreMask GetCrossCorePipeCoreMask(const PrimExpr &pipe_arg) {
  switch (task_analysis_detail::SpecialRegisterForCrossCorePipe(pipe_arg)) {
  case SpecialRegister::kPipeVector:
  case SpecialRegister::kPipeMTE3:
    return kCoreVector;
  case SpecialRegister::kPipeCube:
  case SpecialRegister::kPipeMTE1:
  case SpecialRegister::kPipeFixpipe:
    return kCoreCube;
  case SpecialRegister::kPipeScalar:
  case SpecialRegister::kPipeMTE2:
    return kCoreUnassigned;
  default:
    LOG(FATAL) << "Unexpected special register for cross-core pipe";
    return kCoreUnassigned;
  }
}

static Optional<Call> GetCrossCoreCall(const Call &call) {
  if (!task_analysis_detail::IsCrossCoreSyncCall(call.get())) {
    return std::nullopt;
  }
  ICHECK_EQ(call->args.size(), 3)
      << "ascend_cross_core_set_flag/wait_flag expects exactly 3 arguments";
  const auto *mode = call->args[0].as<IntImmNode>();
  ICHECK(mode)
      << "Ascend AutoSchedule requires cross-core mode_id to be constant";
  return call;
}

static std::vector<Call> CollectCrossCoreCalls(const TaskNode *task) {
  class Collector : public StmtExprVisitor {
  public:
    std::vector<Call> calls;

  private:
    void VisitExpr_(const CallNode *op) final {
      Call call = GetRef<Call>(op);
      if (GetCrossCoreCall(call).defined())
        calls.push_back(call);
      StmtExprVisitor::VisitExpr_(op);
    }
  } collector;
  collector(task->stmt);
  return collector.calls;
}

using CrossCoreCallMap = std::unordered_map<TaskNode *, std::vector<Call>>;
struct CrossCorePlacement {
  CoreMask pipe_core_mask{kCoreUnassigned};
  CoreMask candidate_core_mask{kCoreBroadcast};
};
using CrossCorePlacementMap =
    std::unordered_map<PrimExpr, CrossCorePlacement, ffi::StructuralHash,
                       ffi::StructuralEqual>;

static void AssignCrossCoreTasksInList(
    const std::vector<std::shared_ptr<IRStructure>> &nodes,
    const CrossCoreCallMap &cross_core_calls, CoreMask kernel_core_mask,
    const CoreMaskAvailability &availability) {
  std::vector<TaskNode *> scoped_tasks;
  for (const auto &node : nodes) {
    if (!node->IsTask())
      continue;
    auto *task = static_cast<TaskNode *>(node.get());
    // PerCoreTask executes exactly one embedded candidate T.Task. Its
    // user-provided cross-core synchronization belongs to that local protocol,
    // not to outer placement inference, so keep the core inferred from the
    // candidate operation.
    if (task->IsPerCoreTask())
      continue;
    auto calls_it = cross_core_calls.find(task);
    if (calls_it == cross_core_calls.end())
      continue;
    scoped_tasks.push_back(task);
  }

  // Cross-core flag IDs share one hardware domain across all modes, so mode is
  // deliberately not part of the placement key.
  CrossCorePlacementMap placements;
  for (TaskNode *task : scoped_tasks) {
    // A cross-core call can be nested under a scheduled control node rather
    // than carrying the condition as its own task guard. Include every
    // ancestor header before resolving ambiguous PIPE_S/PIPE_MTE2 placement.
    CoreMask task_capability = task->GetCoreMask();
    for (const IRStructure *parent = task->GetParent(); parent != nullptr;
         parent = parent->GetParent()) {
      ICHECK(parent->IsControl());
      const auto *control = static_cast<const ControlNode *>(parent);
      task_capability &= availability.GetTaskReadCoreMask(control->task.get());
    }
    for (const Call &call : cross_core_calls.at(task)) {
      CrossCorePlacement &placement = placements[call->args[2]];
      placement.pipe_core_mask |= GetCrossCorePipeCoreMask(call->args[1]);
      placement.candidate_core_mask &= task_capability;
    }
  }

  std::unordered_map<TaskNode *, CoreMask> task_core_masks;
  for (TaskNode *task : scoped_tasks) {
    for (const Call &call : cross_core_calls.at(task)) {
      const PrimExpr &flag_id = call->args[2];
      const CrossCorePlacement &placement = placements.at(flag_id);
      CoreMask core_mask = placement.pipe_core_mask;
      CoreMask capability = placement.candidate_core_mask;
      if (capability == kCoreUnassigned) {
        LOG(FATAL) << "Cross-core set/wait calls with flag " << flag_id
                   << " have incompatible scalar inputs or guards in one "
                      "control scope";
      }
      if (core_mask == kCoreBroadcast) {
        LOG(FATAL) << "Cross-core set/wait calls with flag " << flag_id
                   << " in one control scope resolve to both Vector and Cube "
                      "cores; all matching calls must run on the same core";
      }
      if (IsConcreteCore(core_mask) && (capability & core_mask) != core_mask) {
        LOG(FATAL) << "Cross-core set/wait calls with flag " << flag_id
                   << " require "
                   << (core_mask == kCoreVector ? "Vector" : "Cube")
                   << ", but their scalar inputs and guards are unavailable "
                      "on that core";
      }
      if (core_mask == kCoreUnassigned && IsConcreteCore(capability)) {
        core_mask = capability;
      } else if (core_mask == kCoreUnassigned &&
                 IsConcreteCore(kernel_core_mask)) {
        // PIPE_S and PIPE_MTE2 exist on both core types. A pure kernel supplies
        // the fallback when neither the pipe nor task inputs pin placement.
        core_mask = kernel_core_mask;
      } else if (!IsConcreteCore(core_mask)) {
        LOG(FATAL)
            << "Cannot infer the core for cross-core set/wait calls with flag "
            << flag_id << " in one control scope"
            << ": PIPE_S and PIPE_MTE2 are ambiguous in a mixed or empty "
               "kernel. Use PIPE_V/PIPE_MTE3 for Vector or "
               "PIPE_M/PIPE_MTE1/PIPE_FIX for Cube on at least one call in "
               "that scope";
      }
      task_core_masks[task] |= core_mask;
    }
  }
  for (const auto &[task, core_mask] : task_core_masks) {
    if (core_mask == kCoreBroadcast) {
      LOG(FATAL) << "Cross-core synchronization calls in one task resolve to "
                    "both Vector and Cube cores: "
                 << task->stmt;
    }
    task->SetCoreMask(core_mask);
  }

  // A flag ID may be reused sequentially in another control scope. Recurse
  // into each ordered child list so only calls with the same immediate parent
  // influence the same core inference.
  for (const auto &node : nodes) {
    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node.get());
      AssignCrossCoreTasksInList(control->children, cross_core_calls,
                                 kernel_core_mask, availability);
    }
  }
}

static void
AssignCrossCoreTasks(const std::vector<std::shared_ptr<IRStructure>> &root,
                     const std::vector<TaskNode *> &all_tasks,
                     const CoreMaskAvailability &availability) {
  CoreMask kernel_core_mask = kCoreUnassigned;
  CrossCoreCallMap cross_core_calls;
  for (TaskNode *task : all_tasks) {
    CoreMask core_mask = task->GetCoreMask();
    if (task->IsPerCoreTask()) {
      if (IsConcreteCore(core_mask))
        kernel_core_mask |= core_mask;
      continue;
    }
    std::vector<Call> calls = CollectCrossCoreCalls(task);
    if (!calls.empty()) {
      cross_core_calls.emplace(task, std::move(calls));
      continue;
    }
    if (IsConcreteCore(core_mask))
      kernel_core_mask |= core_mask;
  }
  AssignCrossCoreTasksInList(root, cross_core_calls, kernel_core_mask,
                             availability);
}

// Assign the widest legal candidate mask to each scheduled task. The
// downstream ResolveCore pass narrows broadcast candidates after
// PrepareMultiBuffer has made its guard/counter decisions.
class CoreMaskAssigner {
public:
  static void Assign(const std::vector<std::shared_ptr<IRStructure>> &root,
                     const std::vector<TaskNode *> &all_tasks,
                     const ffi::Optional<Var> &outer_sid) {
    CoreMaskAssigner assigner(root, all_tasks, outer_sid);
    assigner.Run();
  }

private:
  CoreMaskAssigner(const std::vector<std::shared_ptr<IRStructure>> &root,
                   const std::vector<TaskNode *> &all_tasks,
                   const ffi::Optional<Var> &outer_sid)
      : root_(root), all_tasks_(all_tasks), availability_(all_tasks) {
    if (outer_sid.has_value())
      availability_.SetExternalVarCoreMask(outer_sid.value(), kCoreVector);
    for (TaskNode *task : all_tasks_) {
      ICHECK_EQ(task->GetCoreMask(), kCoreUnassigned)
          << "AssignCore expects unresolved scheduled TIR and must run once "
             "before PrepareMultiBuffer";
    }
  }

  void Run() {
    AssignCandidateCoreMasks();
    PropagateCandidateCapabilities();
    AssignCrossCoreTasks(root_, all_tasks_, availability_);
    PropagateCandidateCapabilities();
  }

  static CoreMask GetScalarMemoryMask(const TaskNode *task) {
    bool touches_vector_memory = false;
    bool touches_cube_memory = false;
    auto scan = [&](const std::vector<BufferRegion> &regions) {
      for (const BufferRegion &region : regions) {
        if (IsSharedBuffer(region->buffer)) {
          touches_vector_memory = true;
        } else if (IsL1Buffer(region->buffer) || IsL0ABuffer(region->buffer) ||
                   IsL0BBuffer(region->buffer) || IsL0CBuffer(region->buffer)) {
          touches_cube_memory = true;
        }
      }
    };
    scan(task->GetReadRegions());
    scan(task->GetWriteRegions());
    if (touches_vector_memory && touches_cube_memory) {
      LOG(FATAL) << "Cannot assign scalar task because it accesses "
                    "incompatible core-local memories on both Vector (UB) "
                    "and Cube (L1/L0) cores: "
                 << task->stmt;
    }
    if (touches_vector_memory)
      return kCoreVector;
    if (touches_cube_memory)
      return kCoreCube;
    return kCoreBroadcast;
  }

  static CoreMask GetMte2Mask(const TaskNode *task) {
    CoreMask result = kCoreUnassigned;
    for (const BufferRegion &region : task->GetWriteRegions()) {
      if (IsL1Buffer(region->buffer))
        result |= kCoreCube;
      else if (IsSharedBuffer(region->buffer))
        result |= kCoreVector;
    }
    return result == kCoreUnassigned ? kCoreVector : result;
  }

  static CoreMask GetCandidateCoreMask(const TaskNode *task) {
    uint16_t pipe_mask = task->GetPipeMask();
    CoreMask result = kCoreUnassigned;
    if (pipe_mask & static_cast<uint16_t>(ResourcePipe::kCube))
      result |= kCoreCube;
    if (pipe_mask & static_cast<uint16_t>(ResourcePipe::kVector))
      result |= kCoreVector;
    if (pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE1) ||
        pipe_mask & static_cast<uint16_t>(ResourcePipe::kFixpipe)) {
      result |= kCoreCube;
    }
    if (pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE3))
      result |= kCoreVector;
    if (pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE2))
      result |= GetMte2Mask(task);

    if (result == kCoreBroadcast) {
      LOG(FATAL) << "Cannot assign one atomic task to incompatible Vector and "
                    "Cube execution resources: "
                 << task->stmt;
    }
    if (result != kCoreUnassigned)
      return result;
    return GetScalarMemoryMask(task);
  }

  void AssignCandidateCoreMasks() {
    for (TaskNode *task : all_tasks_)
      task->SetCoreMask(GetCandidateCoreMask(task));
  }

  void PropagateCandidateCapabilities() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (TaskNode *task : all_tasks_) {
        CoreMask current = task->GetCoreMask();
        CoreMask narrowed = current & availability_.GetTaskInputCoreMask(task);
        if (narrowed == kCoreUnassigned) {
          LOG(FATAL) << "Cannot assign task because its scalar/register inputs "
                        "are unavailable on every legal execution core: "
                     << task->stmt;
        }
        if (narrowed != current) {
          task->SetCoreMask(narrowed);
          changed = true;
        }
      }
    }
  }

  const std::vector<std::shared_ptr<IRStructure>> &root_;
  const std::vector<TaskNode *> &all_tasks_;
  CoreMaskAvailability availability_;
};

void AssignCoreMasks(std::vector<std::shared_ptr<IRStructure>> &root,
                     const ffi::Optional<Var> &outer_sid) {
  std::vector<TaskNode *> all_tasks;
  CollectAllTaskNodes(root, all_tasks);
  CoreMaskAssigner::Assign(root, all_tasks, outer_sid);
}

} // namespace

tvm::transform::Pass AssignCore() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    return RewriteTilelangKernels(
        std::move(func), "AssignCore",
        [](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir =
              DecodeScheduledTIR(context.root, context.outer_ctx);
          AssignCoreMasks(scheduled_tir.tree, context.outer_sid);
          return EncodeScheduledTIR(std::move(scheduled_tir));
        });
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AssignCore", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AssignCore", AssignCore);
}

} // namespace tl
} // namespace tvm
