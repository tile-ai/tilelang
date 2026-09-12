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
 * \file auto_schedule.cc
 * \brief AutoSchedule pass and schedule builder.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "./auto_schedule/dependency_analysis.h"
#include "./auto_schedule/ir_structure.h"
#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/multi_buffer.h"
#include "./auto_schedule/scheduled_tir.h"
#include "./auto_schedule/task_analysis.h"
#include "./auto_schedule/task_annotations.h"
#include "ascend/op/utils.h"
#include "backend/common/target_utils.h"
#include "op/utils.h"
#include "runtime/thread_storage_scope.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/collector.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;

namespace {

// AutoSchedule target resource limits.

// Check whether a kernel body contains any SIMT_VF blocks. SIMT_VF
// (asc_vf_call) reserves 32 KB of the physical 248 KB Unified Buffer for
// thread context, so the usable shared memory depends on its presence.
class SimtVFDetector : public tirx::StmtVisitor {
public:
  bool found = false;

  void VisitStmt_(const tirx::SBlockNode *op) override {
    if (op->name_hint == "SIMT_VF") {
      found = true;
    }
    tirx::StmtVisitor::VisitStmt_(op);
  }
};

inline bool HasSimtVF(const tirx::Stmt &stmt) {
  SimtVFDetector detector;
  detector(stmt);
  return detector.found;
}

inline int64_t GetSharedMemoryLimit(const tirx::Stmt &kernel_body) {
  constexpr int64_t kPhysicalUnifiedBufferSize = 248 * 1024;
  constexpr int64_t kSimtVFThreadContextSize = 32 * 1024;
  return HasSimtVF(kernel_body)
             ? kPhysicalUnifiedBufferSize - kSimtVFThreadContextSize
             : kPhysicalUnifiedBufferSize;
}

inline int64_t GetL1MemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 512 * 1024; // 512 KB L1 cache on Ascend
  }
  return 0;
}

inline int64_t GetL0CMemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 256 * 1024; // 256 KB L0C buffer on Ascend
  }
  return 0;
}

inline int64_t GetL0AMemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 64 * 1024; // 64 KB L0A buffer on Ascend
  }
  return 0;
}

inline int64_t GetL0BMemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 64 * 1024; // 64 KB L0B buffer on Ascend
  }
  return 0;
}

// Scheduler.

std::unique_ptr<IRExtraInfo> MakeAutoScheduleExtraInfo();

// Check if two variables are the same
bool SameVar(const Var &a, const Var &b);

// Check if there is a variable dependency between two IRStructures
bool HasVarDependency(const IRStructure *a, const IRStructure *b);

// Check if two IRStructures have resource dependency (use same hardware
// resource)
bool HasResourceDependency(const IRStructure *a, const IRStructure *b,
                           uint16_t mask = UINT16_MAX);

// Builder that schedules TaskNode/ControlNode children in IRStructure
class ScheduleBuilder {
private:
  struct LetDependencyClosure {
    std::vector<bool> is_declaration;
    std::vector<std::set<size_t>> dependencies;
  };

  static LetDependencyClosure
  CollectLetDependencyClosure_(const std::vector<IRStructure *> &nodes);

public:
  // Recursive scheduling function
  // Directly schedules the entire IRStructure tree recursively in place
  void ScheduleRecursive(std::shared_ptr<IRStructure> &node);

  // Schedule an ordered child list in place (root list or a ControlNode body):
  // gather + recurse into each child, then Z3-schedule and write the reordered
  // list back.
  void ScheduleList(std::vector<std::shared_ptr<IRStructure>> &children);

  // Z3-based scheduler for a straight-line child list.
  std::vector<std::shared_ptr<IRStructure>>
  Z3SchedulePython(const std::vector<std::shared_ptr<IRStructure>> &children,
                   bool manual_schedule);

  // Distance-aware Z3 scheduler for a ControlNode loop body.
  void Z3SchedulePythonLoop(ControlNode *ctrl, bool manual_schedule);

  // Set memory limit for a given buffer scope (e.g. "shared", "shared.l1")
  void SetMemoryLimit(const std::string &scope, int64_t bytes) {
    memory_limits_[scope] = bytes;
  }

  // Set manual buffer version overrides (from annotate_buffer_versions).
  // Keys are buffer data variables, values are fixed num_versions.
  void SetBufferVersionOverrides(BufferVersionMap overrides) {
    buffer_version_overrides_ = std::move(overrides);
  }

  // Set the user-declared manual multi-buffer table (from
  // annotate_manual_multi_buffer), threaded into dependency analysis so
  // cross-iteration distances are solved per access pair.
  void SetManualBufferVersions(BufferVersionMap manual_buffer_versions) {
    manual_buffer_versions_ = std::move(manual_buffer_versions);
    dependency_cache_.clear();
  }

  void SetRootConflictHints(ConflictHintList root_conflict_hints) {
    root_conflict_hints_ = std::move(root_conflict_hints);
    dependency_cache_.clear();
  }

  const BufferVersionMap &GetSelectedBufferVersions() const {
    return selected_buffer_versions_;
  }

private:
  static bool HasRequestedStage_(
      const std::vector<std::shared_ptr<IRStructure>> &children) {
    bool has_unscheduled = false;
    bool has_requested = false;
    for (const auto &node : children) {
      int stage = node->GetStage();
      ICHECK_GE(stage, kUnscheduledStage)
          << "Schedule-unit stage must be -1 or non-negative, got " << stage;
      has_unscheduled |= stage == kUnscheduledStage;
      has_requested |= stage != kUnscheduledStage;
    }
    ICHECK(!(has_unscheduled && has_requested))
        << "AutoSchedule expects each child list to be either entirely "
           "unscheduled (stage=-1) or entirely manually staged (stage>=0); "
           "MaterializeScheduleUnits must normalize unannotated manual "
           "siblings to stage 0";
    return has_requested;
  }

  int RequestedStage_(const IRStructure *node) const {
    int stage = node->GetStage();
    if (stage == kUnscheduledStage)
      return 0;
    return stage;
  }

  void MergeSelectedBufferVersion(const Var &storage, int num_versions) {
    if (num_versions <= 1)
      return;
    auto existing = selected_buffer_versions_.find(storage);
    if (existing != selected_buffer_versions_.end()) {
      // Disjoint owners are scheduled independently but share one physical
      // storage ring, so keep enough versions for every owner. The per-owner
      // memory limits are intentionally not revalidated after this merge:
      // they are a coarse scheduling heuristic and do not model downstream
      // storage reuse, so a global sum here would not be authoritative. This
      // can produce a merged allocation above the physical limit; until a
      // lifetime-aware global check exists, this is an accepted limitation and
      // such a kernel may only be rejected by downstream/runtime validation.
      num_versions = std::max((*existing).second, num_versions);
    }
    selected_buffer_versions_.Set(storage, num_versions);
  }

  std::map<std::string, int64_t> memory_limits_;
  BufferVersionMap buffer_version_overrides_;
  BufferVersionMap manual_buffer_versions_;
  MultiBufferOwnerMap multi_buffer_owners_;
  ConflictHintList root_conflict_hints_;
  DependencyCache dependency_cache_;
  BufferVersionMap selected_buffer_versions_;
};

// Scratch timing state used only while AutoSchedule is solving and ordering an
// IRStructure tree. Task latency and II live directly on IRStructure because
// they are ordinary T.Task metadata.
class AutoScheduleExtraInfo final : public IRExtraInfo {
public:
  std::unique_ptr<IRExtraInfo> Clone() const final {
    return std::make_unique<AutoScheduleExtraInfo>(*this);
  }

  void SetStartTime(int64_t start_time) { start_time_ = start_time; }
  int64_t GetStartTime() const { return start_time_; }
  void SetIIperIter(int64_t ii_per_iter) { ii_per_iter_ = ii_per_iter; }
  int64_t GetIIperIter() const { return ii_per_iter_; }

private:
  int64_t start_time_{0};
  int64_t ii_per_iter_{0};
};

AutoScheduleExtraInfo *GetAutoScheduleExtraInfo(IRStructure *node) {
  return node->GetExtraInfo<AutoScheduleExtraInfo>();
}

int64_t GetTaskTimestamp(IRStructure *subtree_root, TaskNode *task,
                         bool producer_side) {
  std::vector<IRStructure *> path = task->PathFrom(subtree_root);
  int64_t timestamp = 0;
  for (size_t i = 0; i + 1 < path.size(); ++i) {
    ICHECK(path[i]->IsControl());
    auto *control = static_cast<ControlNode *>(path[i]);
    if (producer_side) {
      int64_t trip_count = control->GetTripCount();
      int64_t ii_per_iter = GetAutoScheduleExtraInfo(control)->GetIIperIter();
      timestamp += (trip_count > 0 ? trip_count - 1 : 0) * ii_per_iter;
    }
    timestamp += GetAutoScheduleExtraInfo(path[i + 1])->GetStartTime();
  }
  return timestamp;
}

int64_t GetDependencyLatency(const DepInfo &dependency) {
  int64_t latency = 0;
  for (const auto &[producer, consumer] : dependency.task_pairs) {
    int64_t producer_time =
        GetTaskTimestamp(dependency.prod_node, producer, true);
    int64_t consumer_time =
        GetTaskTimestamp(dependency.cons_node, consumer, false);
    latency = std::max(latency,
                       producer_time + producer->GetLatency() - consumer_time);
  }
  return latency;
}

std::unique_ptr<IRExtraInfo> MakeAutoScheduleExtraInfo() {
  return std::make_unique<AutoScheduleExtraInfo>();
}

bool SameVar(const Var &a, const Var &b) { return a.same_as(b); }

bool HasVarDependency(const IRStructure *a, const IRStructure *b) {
  for (const auto &write_var_a : a->GetWriteVars()) {
    for (const auto &read_var_b : b->GetReadVars()) {
      if (SameVar(write_var_a, read_var_b))
        return true;
    }
  }
  return false;
}

bool HasResourceDependency(const IRStructure *a, const IRStructure *b,
                           uint16_t mask) {
  if (((a->GetPipeMask() & b->GetPipeMask()) & mask) != 0)
    return true;
  return (a->GetHbmMask() & b->GetHbmMask()) != 0;
}

ScheduleBuilder::LetDependencyClosure
ScheduleBuilder::CollectLetDependencyClosure_(
    const std::vector<IRStructure *> &nodes) {
  LetDependencyClosure result;
  result.is_declaration.resize(nodes.size(), false);
  result.dependencies.resize(nodes.size());

  std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual>
      declaration_by_var;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const IRStructure *node = nodes[i];
    if (!node || !node->IsTask())
      continue;
    const auto *task = static_cast<const TaskNode *>(node);
    if (ffi::Optional<Bind> bind = GetFlatTaskBind(task->stmt);
        bind.defined()) {
      result.is_declaration[i] = true;
      declaration_by_var[bind.value()->var] = i;
    }
  }

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i])
      continue;
    std::vector<Var> pending = nodes[i]->GetReadVars();
    while (!pending.empty()) {
      Var var = std::move(pending.back());
      pending.pop_back();
      auto it = declaration_by_var.find(var);
      if (it == declaration_by_var.end())
        continue;
      size_t declaration = it->second;
      if (!result.dependencies[i].insert(declaration).second)
        continue;
      std::vector<Var> reads = nodes[declaration]->GetReadVars();
      pending.insert(pending.end(), reads.begin(), reads.end());
    }
  }
  return result;
}

/*
  Recursively schedule root node, after scheduling IRStructure satisfies the
following properties: 1) Each ordered child list (the root list or a
ControlNode body) is reordered by the Z3 scheduler and the children carry
scheduling metadata directly. 2) For each ControlNode, its `children` are the
Z3-scheduled loop body. Conditional execution is carried by each node's
`conditions` delta (no dedicated IfNode), so guarded tasks are ordinary
schedule candidates.
*/
void ScheduleBuilder::ScheduleList(
    std::vector<std::shared_ptr<IRStructure>> &children) {
  multi_buffer_owners_ = CollectMultiBufferOwners(children);
  dependency_cache_.clear();
  std::vector<std::shared_ptr<IRStructure>> origin_children = children;
  bool manual_schedule = HasRequestedStage_(origin_children);
  for (size_t i = 0; i < origin_children.size(); ++i) {
    origin_children[i]->SetIndex(i);
    origin_children[i]->SetParent(nullptr);
  }
  for (auto &child : origin_children) {
    ScheduleRecursive(child);
  }
  children = Z3SchedulePython(origin_children, manual_schedule);
  for (size_t i = 0; i < children.size(); ++i) {
    children[i]->SetIndex(i);
    children[i]->SetParent(nullptr);
  }
}

void ScheduleBuilder::ScheduleRecursive(std::shared_ptr<IRStructure> &node) {
  if (!node)
    return;

  if (node->IsTask()) {
    return;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children = ctrl->children;
    bool manual_schedule = HasRequestedStage_(origin_children);
    for (size_t i = 0; i < origin_children.size(); ++i) {
      origin_children[i]->SetIndex(i);
      origin_children[i]->SetParent(ctrl);
    }
    for (auto &child : origin_children) {
      ScheduleRecursive(child);
    }
    Z3SchedulePythonLoop(ctrl, manual_schedule);
    for (size_t i = 0; i < ctrl->children.size(); ++i) {
      ctrl->children[i]->SetIndex(i);
      ctrl->children[i]->SetParent(ctrl);
    }
    return;
  }

  LOG(FATAL) << "[ScheduleRecursive] Unknown IRStructure type" << node.get();
}

std::vector<std::shared_ptr<IRStructure>> ScheduleBuilder::Z3SchedulePython(
    const std::vector<std::shared_ptr<IRStructure>> &children,
    bool manual_schedule) {
  size_t n = children.size();
  std::vector<std::shared_ptr<IRStructure>> scheduled = children;
  for (const auto &node : scheduled) {
    ICHECK_EQ(RequestedStage_(node.get()), 0)
        << "T.Stage with a non-zero stage must be inside a serial loop "
           "annotated with enable_offset=True";
    // MaterializeScheduleUnits uses -1 to distinguish unscheduled units. A
    // straight-line list has no software-pipeline offset, so scheduling it
    // always publishes stage 0, including all fallback paths below.
    node->SetStage(0);
    GetAutoScheduleExtraInfo(node.get())->SetStartTime(0);
  }

  if (n <= 1) {
    return scheduled;
  }

  try {
    // Get the Python-registered function using ffi::Function::GetGlobal
    static std::optional<ffi::Function> z3_schedule_func =
        ffi::Function::GetGlobal("tl.transform.z3_schedule_python");
    if (!z3_schedule_func.has_value()) {
      ICHECK(!manual_schedule)
          << "Python Z3 scheduler is required for manual AutoSchedule";
      LOG(WARNING) << "Python Z3 scheduler not registered, falling back to "
                      "topological sort";
      return scheduled;
    }

    std::vector<IRStructure *> nodes;
    nodes.reserve(n);
    for (const auto &u : scheduled) {
      nodes.push_back(u.get());
    }

    // Prepare input data
    std::vector<int64_t> latencies;
    std::vector<int64_t> iis;
    std::vector<int64_t> resource_flags;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> data_deps;
    std::vector<std::pair<int64_t, int64_t>> resource_deps;
    std::vector<std::pair<int64_t, int64_t>> pipe_order_deps;

    latencies.reserve(n);
    iis.reserve(n);
    resource_flags.reserve(n);

    for (size_t i = 0; i < n; ++i) {
      const IRStructure *node = nodes[i];
      latencies.push_back(node->GetLatency());
      iis.push_back(node->GetII());

      // Pass pipe mask directly as resource flags
      resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
    }

    // Variable (Let-var write -> read) data dependencies
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        if (HasVarDependency(nodes[i], nodes[j])) {
          data_deps.emplace_back(i, j, latencies[i]);
        }
      }
    }

    // Special-register ordering: a node writing register R must be ordered
    // against every node reading R (RAW forward, WAR backward) and every
    // other writer of R (WAW), preserving program order. loop_break writes
    // kLoopControl, which every node reads, so it fences against all
    // siblings; hf32/pad match only their consumers.
    for (size_t i = 0; i < n; ++i) {
      uint16_t wmask = nodes[i]->GetSpecialWriteMask();
      if (wmask == 0)
        continue;
      for (size_t j = 0; j < n; ++j) {
        if (j == i || (wmask & nodes[j]->GetSpecialReadMask()) == 0)
          continue;
        if (i < j) {
          data_deps.emplace_back(i, j, latencies[i]);
        } else {
          data_deps.emplace_back(j, i, 0);
        }
      }
      for (size_t j = i + 1; j < n; ++j) {
        if ((wmask & nodes[j]->GetSpecialWriteMask()) != 0)
          data_deps.emplace_back(i, j, latencies[i]);
      }
    }

    // Buffer-region data dependencies
    auto deps = AnalyzeDependencies(
        nodes, /*loop=*/nullptr, manual_buffer_versions_, multi_buffer_owners_,
        &dependency_cache_, root_conflict_hints_);
    std::unordered_map<IRStructure *, size_t> node_idx;
    for (size_t i = 0; i < n; ++i) {
      node_idx[nodes[i]] = i;
    }
    for (const auto &dep : deps) {
      size_t i = node_idx[dep.prod_node];
      size_t j = node_idx[dep.cons_node];
      data_deps.emplace_back(i, j, GetDependencyLatency(dep));
    }

    // Collect resource dependencies
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        uint16_t shared_pipes =
            nodes[i]->GetPipeMask() & nodes[j]->GetPipeMask();
        if (HasResourceDependency(nodes[i], nodes[j])) {
          resource_deps.emplace_back(i, j);
        }
        if (manual_schedule && shared_pipes != 0) {
          pipe_order_deps.emplace_back(i, j);
        }
      }
    }

    // Convert vectors to TVM containers
    ffi::Array<int64_t> tvm_latencies;
    ffi::Array<int64_t> tvm_iis;
    ffi::Array<int64_t> tvm_resource_flags;
    ffi::Array<ffi::Array<int64_t>> tvm_data_deps;
    ffi::Array<ffi::Array<int64_t>> tvm_resource_deps;
    ffi::Array<ffi::Array<int64_t>> tvm_pipe_order_deps;

    for (auto val : latencies) {
      tvm_latencies.push_back(val);
    }
    for (auto val : iis) {
      tvm_iis.push_back(val);
    }
    for (auto val : resource_flags) {
      tvm_resource_flags.push_back(val);
    }
    for (const auto &dep : data_deps) {
      ffi::Array<int64_t> pair;
      pair.push_back(std::get<0>(dep));
      pair.push_back(std::get<1>(dep));
      pair.push_back(std::get<2>(dep));
      tvm_data_deps.push_back(pair);
    }
    for (const auto &dep : resource_deps) {
      ffi::Array<int64_t> pair;
      pair.push_back(dep.first);
      pair.push_back(dep.second);
      tvm_resource_deps.push_back(pair);
    }
    for (const auto &dep : pipe_order_deps) {
      ffi::Array<int64_t> pair;
      pair.push_back(dep.first);
      pair.push_back(dep.second);
      tvm_pipe_order_deps.push_back(pair);
    }

    // Extract results
    // Python function returns only start_times, C++ side will sort by
    // start_time
    auto start_times =
        z3_schedule_func
            .value()(tvm_latencies, tvm_iis, tvm_resource_flags, tvm_data_deps,
                     tvm_resource_deps, tvm_pipe_order_deps)
            .cast<ffi::Array<int64_t>>();

    if (start_times.size() != n) {
      ICHECK(!manual_schedule)
          << "Python Z3 scheduler returned invalid manual schedule results";
      LOG(WARNING) << "Python Z3 scheduler returned invalid results (size "
                      "mismatch), falling back to topological sort";
      return scheduled;
    }

    // Write start_times onto the scheduled nodes, then sort them in place by
    // start_time (stable: ties keep original program order).
    for (size_t i = 0; i < n; ++i) {
      GetAutoScheduleExtraInfo(nodes[i])->SetStartTime(start_times[i]);
    }

    std::stable_sort(
        scheduled.begin(), scheduled.end(),
        [](const std::shared_ptr<IRStructure> &a,
           const std::shared_ptr<IRStructure> &b) {
          return GetAutoScheduleExtraInfo(a.get())->GetStartTime() <
                 GetAutoScheduleExtraInfo(b.get())->GetStartTime();
        });
    return scheduled;

  } catch (const std::exception &e) {
    ICHECK(!manual_schedule)
        << "Manual per-pipe scheduling failed: " << e.what();
    LOG(WARNING) << "Python Z3 scheduler failed with exception: " << e.what()
                 << ", falling back to topological sort";
    return scheduled;
  } catch (...) {
    ICHECK(!manual_schedule)
        << "Manual per-pipe scheduling failed with unknown exception";
    LOG(WARNING) << "Python Z3 scheduler failed with unknown exception, "
                    "falling back to topological sort";
    return scheduled;
  }
}

// Z3-based scheduler for loops that calls Python implementation via FFI
// with distance-aware dependencies
void ScheduleBuilder::Z3SchedulePythonLoop(ControlNode *ctrl,
                                           bool manual_schedule) {
  if (ctrl->children.empty()) {
    LOG(WARNING)
        << "Z3SchedulePythonLoop called on a control node without children";
    return;
  }

  std::vector<IRStructure *> nodes;
  nodes.reserve(ctrl->children.size());
  for (const auto &child : ctrl->children) {
    nodes.push_back(child.get());
  }

  size_t n = nodes.size();

  std::vector<int64_t> manual_stages;
  manual_stages.reserve(n);
  int max_manual_stage = 0;
  for (const IRStructure *node : nodes) {
    int stage = RequestedStage_(node);
    manual_stages.push_back(stage);
    max_manual_stage = std::max(max_manual_stage, stage);
  }

  auto num_stages = 1;
  auto num_stages_val = ctrl->control.get()->annotations.Get("num_stages");
  if (num_stages_val.has_value()) {
    num_stages = num_stages_val.value().cast<IntImm>()->value;
  }

  // When false, the Z3 loop scheduler keeps each resource's tasks within a
  // single II window (max(start) - min(start) < II). Sourced from the loop's
  // "enable_offset" annotation; defaults to false (constraint applied).
  bool enable_offset = false;
  auto enable_offset_val =
      ctrl->control.get()->annotations.Get("enable_offset");
  if (enable_offset_val.has_value()) {
    enable_offset = enable_offset_val.value().cast<bool>();
  }

  if (manual_schedule) {
    ICHECK(
        !ctrl->control.get()->annotations.Get("tl_pipeline_order").has_value())
        << "Ascend manual AutoSchedule derives each pipe's issue order from "
           "source order; remove the T.Pipelined order argument";
    ICHECK(
        !ctrl->control.get()->annotations.Get("tl_pipeline_stage").has_value())
        << "Ascend manual AutoSchedule uses T.Stage scopes; remove the "
           "T.Pipelined stage argument";
    if (!enable_offset) {
      ICHECK_EQ(max_manual_stage, 0)
          << "Non-zero T.Stage values require enable_offset=True on the "
             "enclosing loop";
    }
    if (!num_stages_val.has_value()) {
      num_stages = max_manual_stage + 1;
    }
  }

  static std::optional<ffi::Function> z3_schedule_loop_func =
      ffi::Function::GetGlobal("tl.transform.z3_schedule_loop_python");
  if (!z3_schedule_loop_func.has_value()) {
    LOG(FATAL) << "Python Z3 loop scheduler not registered, falling back "
                  "to topological sort";
  }

  // Prepare input data
  std::vector<int64_t> latencies;
  std::vector<int64_t> iis;
  std::vector<int64_t> resource_flags;
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
      data_deps; // (i, j, distance, latency)
  std::vector<std::pair<int64_t, int64_t>> resource_deps;
  std::vector<std::pair<int64_t, int64_t>> pipe_order_deps;
  std::vector<std::pair<int64_t, int64_t>> stage_order_deps;

  latencies.reserve(n);
  iis.reserve(n);
  resource_flags.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    const IRStructure *node = nodes[i];
    latencies.push_back(node->GetLatency());
    iis.push_back(node->GetII());
    resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
  }

  // Keep one representative Buffer per storage for shape/scope accounting.
  // Version variables and dependency distances are keyed directly by storage.
  struct StorageResource {
    Var storage;
    Buffer representative;
  };
  std::vector<int64_t> storage_sizes;
  std::vector<StorageResource> storage_resources;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen_storages;
  std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual>
      storage_version_id;
  // Build memory groups from memory_limits_ map.
  // std::map iteration is ordered by key: "shared" then "shared.l1".
  std::map<std::string, int> scope_to_group;
  std::vector<int64_t> group_capacities;
  std::vector<std::string> group_scopes;
  std::vector<std::vector<int64_t>> group_indices;
  for (const auto &entry : memory_limits_) {
    if (entry.second > 0) {
      scope_to_group[entry.first] = (int)group_capacities.size();
      group_capacities.push_back(entry.second);
      group_scopes.push_back(entry.first);
      group_indices.push_back({});
    }
  }
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>
      annotated_storage_versions;
  auto get_buffer_size = [](const Buffer &buffer) -> int64_t {
    // Bits-based: dtype.bits()*lanes()/8 truncates sub-byte dtypes to 0
    // (fp4 -> 0). Accumulate elements, then ceil(elements*bits/8).
    int64_t elem_bits = buffer->dtype.bits() * buffer->dtype.lanes();
    PrimExpr elems = IntImm(DataType::Int(64), 1);
    for (const auto &dim : buffer->shape) {
      elems *= dim;
    }
    PrimExpr size = FloorDiv(elems * IntImm(DataType::Int(64), elem_bits) +
                                 IntImm(DataType::Int(64), 7),
                             IntImm(DataType::Int(64), 8));
    arith::Analyzer analyzer;
    auto size_val = analyzer.Simplify(size);
    if (const auto *int_imm = size_val.as<IntImmNode>()) {
      return int_imm->value;
    }
    return 0;
  };
  auto enumerate_buffers = [&](auto &&visit) {
    for (const auto &r : ctrl->GetReadRegions())
      visit(r->buffer);
    for (const auto &r : ctrl->GetWriteRegions())
      visit(r->buffer);
  };
  enumerate_buffers([&](const Buffer &buffer) {
    if (!IsSharedBuffer(buffer) && !IsL1Buffer(buffer) &&
        !IsL0CBuffer(buffer) && !IsL0ABuffer(buffer) && !IsL0BBuffer(buffer)) {
      return; // Only consider on-chip buffers for multi-buffer
    }
    if (seen_storages.insert(buffer->data).second)
      storage_resources.push_back({buffer->data, buffer});
  });

  for (const StorageResource &resource : storage_resources) {
    const Var &storage = resource.storage;
    const Buffer &buffer = resource.representative;
    std::string scope;
    if (IsL1Buffer(buffer)) {
      scope = "shared.l1";
    } else if (IsL0CBuffer(buffer)) {
      scope = "shared.l0c";
    } else if (IsL0ABuffer(buffer)) {
      scope = "shared.l0a";
    } else if (IsL0BBuffer(buffer)) {
      scope = "shared.l0b";
    } else {
      scope = "shared";
    }
    int group_id = scope_to_group[scope];
    if (!ctrl->IsMultiBufferEligible(storage)) {
      // These buffers cannot be multi-buffered, and cannot share memory with
      // other buffers either, so we directly reduce the capacity of the
      // memory group by their size
      group_capacities[group_id] -= get_buffer_size(buffer);
    } else {
      auto override_it = buffer_version_overrides_.find(storage);
      if (override_it != buffer_version_overrides_.end()) {
        int override_val = (*override_it).second;
        annotated_storage_versions[storage] = override_val;
        group_capacities[group_id] -= get_buffer_size(buffer) * override_val;
      } else {
        size_t id = storage_sizes.size();
        storage_version_id.emplace(storage, id);
        group_indices[group_id].push_back((int64_t)id);
        storage_sizes.push_back(get_buffer_size(buffer));
      }
    }
  }

  // Validate buffer capacities upfront: if the minimum requirement (one
  // version of each eligible buffer) already exceeds the remaining capacity,
  // Z3 will never find a feasible solution.
  for (size_t g = 0; g < group_capacities.size(); ++g) {
    int64_t capacity = group_capacities[g];
    const auto &idxs = group_indices[g];
    if (!idxs.empty()) {
      if (capacity < 0) {
        LOG(FATAL)
            << "Z3 loop scheduling failed: scope \"" << group_scopes[g]
            << "\" capacity is negative (" << capacity
            << "). Non-eligible and fixed-version buffers already exceed"
            << " the memory limit. Try using T.annotate_unlimit_memory(\""
            << group_scopes[g] << "\") or reduce buffer sizes.";
      }
      int64_t min_usage = 0;
      for (int64_t idx : idxs) {
        min_usage += storage_sizes[idx];
      }
      if (min_usage > capacity) {
        LOG(FATAL) << "Z3 loop scheduling failed: scope \"" << group_scopes[g]
                   << "\" capacity " << capacity
                   << " is insufficient for minimum buffer usage " << min_usage
                   << ". Try using T.annotate_unlimit_memory(\""
                   << group_scopes[g] << "\") or reduce buffer sizes.";
      }
    }
  }

  // Variable (Let-var write -> read) data dependencies
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (HasVarDependency(nodes[i], nodes[j])) {
        data_deps.emplace_back(i, j, 0, latencies[i]);
      }
    }
  }

  // Special-register ordering: a node writing register R must be ordered
  // against every node reading R (RAW forward, WAR backward) and every other
  // writer of R (WAW), preserving program order. loop_break writes
  // kLoopControl, which every node reads, so it fences against all siblings;
  // hf32/pad match only their consumers. The distance bit encodes intra-iter
  // (0) vs next-iter (1) direction.
  for (size_t i = 0; i < n; ++i) {
    uint16_t wmask = nodes[i]->GetSpecialWriteMask();
    if (wmask == 0)
      continue;
    for (size_t j = 0; j < n; ++j) {
      if (j == i || (wmask & nodes[j]->GetSpecialReadMask()) == 0)
        continue;
      if (i < j) {
        data_deps.emplace_back(i, j, 0, latencies[i]);
        data_deps.emplace_back(j, i, 1, 0);
      } else {
        data_deps.emplace_back(i, j, 1, latencies[i]);
        data_deps.emplace_back(j, i, 0, 0);
      }
    }
    for (size_t j = i + 1; j < n; ++j) {
      if ((wmask & nodes[j]->GetSpecialWriteMask()) == 0)
        continue;
      data_deps.emplace_back(i, j, 0, latencies[i]);
      data_deps.emplace_back(j, i, 1, latencies[j]);
    }
  }

  // Buffer-region data dependencies
  auto deps = AnalyzeDependencies(nodes, ctrl, manual_buffer_versions_,
                                  multi_buffer_owners_, &dependency_cache_,
                                  root_conflict_hints_);
  std::unordered_map<IRStructure *, size_t> node_idx;
  for (size_t i = 0; i < n; ++i) {
    node_idx[nodes[i]] = i;
  }
  for (const auto &dep : deps) {
    size_t i = node_idx[dep.prod_node];
    size_t j = node_idx[dep.cons_node];
    int64_t latency = GetDependencyLatency(dep);
    if (dep.distance >= 0) {
      data_deps.emplace_back(i, j, dep.distance, latency);
    } else {
      int64_t distance = 1;
      if (dep.storage.has_value()) {
        Var storage = dep.storage.value();
        auto it = storage_version_id.find(storage);
        if (it != storage_version_id.end()) {
          // Encode the storage-version id in the negative distance for the Z3
          // scheduler to recognize.
          distance = -(int64_t)it->second - 1;
        } else {
          auto ann_it = annotated_storage_versions.find(storage);
          if (ann_it != annotated_storage_versions.end()) {
            distance = ann_it->second;
          }
        }
      }
      data_deps.emplace_back(i, j, distance, latency);
    }
  }

  // Collect resource dependencies
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      uint16_t shared_pipes = nodes[i]->GetPipeMask() & nodes[j]->GetPipeMask();
      if (HasResourceDependency(nodes[i], nodes[j],
                                ~((uint16_t)ResourcePipe::kScalar))) {
        resource_deps.emplace_back(i, j);
      }
      if (manual_schedule && shared_pipes != 0) {
        pipe_order_deps.emplace_back(i, j);
      }
    }
  }

  // Stage-order constraints for copied Let variables.
  //
  // A Let task `a = buf[...]` only READS buffers (it writes a scalar var).
  // When `a` is consumed in a different pipeline stage, solve_conflict_var
  // clones the Let into that consumer's stage and prepends it to the loop
  // body. The clone re-reads `buf` at the consumer's iteration. If `buf` is
  // also WRITTEN inside the loop, that re-read must observe the value from
  // before the write — otherwise the clone reads a stale/overwritten buffer.
  //
  // AnalyzeDependencies already feeds the Let's own WAR edge to Z3, but only
  // constrains the Let's original stage, not the stages of nodes that will
  // receive a copy. Here we add: for every node U that transitively reads a
  // Let-defined var, and every WAR consumer W of that Let (a node that writes
  // the buffer the Let reads, same-iter or cross-iter), require k_U <= k_W in
  // the Z3 (k, r) model. Equality is allowed because the clone is placed at
  // the front of the loop body, so it reads before any write in that
  // iteration.
  if (enable_offset) {
    LetDependencyClosure let_closure = CollectLetDependencyClosure_(nodes);
    // Collect WAR consumers per Let node
    std::vector<std::set<size_t>> let_war_consumers(n);
    for (const auto &dep : deps) {
      size_t p = node_idx[dep.prod_node];
      if (let_closure.is_declaration[p]) {
        let_war_consumers[p].insert(node_idx[dep.cons_node]);
      }
    }
    for (size_t u = 0; u < n; ++u) {
      for (size_t l : let_closure.dependencies[u]) {
        for (size_t w : let_war_consumers[l]) {
          if (u == w)
            continue;
          stage_order_deps.emplace_back((int64_t)u, (int64_t)w);
        }
      }
    }
  }

  // Convert vectors to TVM containers
  ffi::Array<int64_t> tvm_latencies;
  ffi::Array<int64_t> tvm_iis;
  ffi::Array<int64_t> tvm_resource_flags;
  ffi::Array<ffi::Array<int64_t>>
      tvm_data_deps; // each element is [i, j, distance]
  ffi::Array<ffi::Array<int64_t>> tvm_resource_deps;
  ffi::Array<ffi::Array<int64_t>> tvm_pipe_order_deps;
  ffi::Array<int64_t> tvm_storage_sizes;
  ffi::Array<int64_t> tvm_manual_stages;

  for (auto val : latencies) {
    tvm_latencies.push_back(val);
  }
  for (auto val : iis) {
    tvm_iis.push_back(val);
  }
  for (auto val : resource_flags) {
    tvm_resource_flags.push_back(val);
  }
  for (const auto &dep : data_deps) {
    ffi::Array<int64_t> triple;
    triple.push_back(std::get<0>(dep));
    triple.push_back(std::get<1>(dep));
    triple.push_back(std::get<2>(dep));
    triple.push_back(std::get<3>(dep));
    tvm_data_deps.push_back(triple);
  }
  for (const auto &dep : resource_deps) {
    ffi::Array<int64_t> pair;
    pair.push_back(dep.first);
    pair.push_back(dep.second);
    tvm_resource_deps.push_back(pair);
  }
  for (const auto &dep : pipe_order_deps) {
    ffi::Array<int64_t> pair;
    pair.push_back(dep.first);
    pair.push_back(dep.second);
    tvm_pipe_order_deps.push_back(pair);
  }
  ffi::Array<ffi::Array<int64_t>> tvm_stage_order_deps;
  for (const auto &dep : stage_order_deps) {
    ffi::Array<int64_t> pair;
    pair.push_back(dep.first);
    pair.push_back(dep.second);
    tvm_stage_order_deps.push_back(pair);
  }
  for (int64_t size : storage_sizes) {
    tvm_storage_sizes.push_back(size);
  }
  for (int64_t stage : manual_stages) {
    tvm_manual_stages.push_back(stage);
  }

  // Build memory groups for FFI: each inner array is [capacity, idx0, idx1,
  // ...]
  ffi::Array<ffi::Array<int64_t>> tvm_memory_groups;
  for (size_t g = 0; g < group_capacities.size(); ++g) {
    ffi::Array<int64_t> group;
    group.push_back(group_capacities[g]);
    for (auto idx : group_indices[g]) {
      group.push_back(idx);
    }
    tvm_memory_groups.push_back(group);
  }

  // Extract results
  // Python function returns (start_times, storage_versions, best_ii) as a
  // 3-tuple. Stages are computed in C++ from start_times and II — see below.
  auto return_val =
      z3_schedule_loop_func
          .value()(num_stages, tvm_latencies, tvm_iis, tvm_resource_flags,
                   tvm_data_deps, tvm_resource_deps, tvm_storage_sizes,
                   tvm_memory_groups, tvm_stage_order_deps, enable_offset,
                   tvm_manual_stages, manual_schedule, tvm_pipe_order_deps)
          .cast<ffi::Tuple<ffi::Array<int64_t>, ffi::Array<int>, int64_t>>();

  ffi::Array<int64_t> start_times = return_val.get<0>();
  ffi::Array<int> storage_versions = return_val.get<1>();
  int64_t ii = return_val.get<2>();

  // Derive a per-task stage from (start_time / ii). Z3 schedules
  // `start = k * ii + r` so `start_time / ii` recovers k. Producers (low k,
  // earliest in z3's absolute time) get stage 0; later tasks get positive
  // stages.
  std::vector<int> raw_k(n, 0);
  int min_k_raw = std::numeric_limits<int>::max();
  int64_t min_start_time = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < n; ++i) {
    int k = (ii > 0) ? static_cast<int>(start_times[i] / ii) : 0;
    raw_k[i] = k;
    min_k_raw = std::min(min_k_raw, k);
    min_start_time = std::min(min_start_time, start_times[i]);
  }
  for (size_t i = 0; i < n; ++i) {
    // When offset is disabled, collapse every task into a single pipeline
    // stage so LowerScheduledTIR emits a flat (non-pipelined) loop body.
    nodes[i]->SetStage(manual_schedule
                           ? static_cast<int>(manual_stages[i])
                           : (enable_offset ? (raw_k[i] - min_k_raw) : 0));
    // Adjust start_time to be relative to the earliest start_time, so that
    // the first task starts at 0.
    GetAutoScheduleExtraInfo(nodes[i])->SetStartTime(start_times[i] -
                                                     min_start_time);
  }
  auto phys_time = [&](const std::shared_ptr<IRStructure> &c) -> int64_t {
    return GetAutoScheduleExtraInfo(c.get())->GetStartTime() -
           c->GetStage() * ii;
  };
  std::stable_sort(ctrl->children.begin(), ctrl->children.end(),
                   [&](const std::shared_ptr<IRStructure> &a,
                       const std::shared_ptr<IRStructure> &b) {
                     return phys_time(a) < phys_time(b);
                   });

  // Reorder & copy Let-defined variables. Recompute the closure after the
  // scheduled children have been reordered so declaration indices match the
  // final sibling list.
  std::vector<IRStructure *> scheduled_nodes;
  scheduled_nodes.reserve(n);
  for (const auto &child : ctrl->children)
    scheduled_nodes.push_back(child.get());
  LetDependencyClosure let_closure =
      CollectLetDependencyClosure_(scheduled_nodes);

  // Resolve Let-var stage conflicts
  // Later stages run earlier in a physical pipeline iteration, so prepend
  // their clones first.
  std::map<int, std::set<size_t>, std::greater<int>> stage_to_decls;
  std::map<int, int64_t> stage_base_start; // stage -> floored start time
  for (size_t u = 0; u < n; ++u) {
    int s = ctrl->children[u]->GetStage();
    for (size_t d : let_closure.dependencies[u]) {
      if (ctrl->children[d]->GetStage() == s)
        continue; // decl already in the consumer's stage: no clone needed
      stage_to_decls[s].insert(d);
      int64_t st =
          GetAutoScheduleExtraInfo(ctrl->children[u].get())->GetStartTime();
      auto bit = stage_base_start.find(s);
      if (bit == stage_base_start.end())
        stage_base_start[s] = st;
      else
        bit->second = std::min(bit->second, st);
    }
  }

  std::vector<std::shared_ptr<IRStructure>> new_clones;
  for (auto &[s, decls] : stage_to_decls) {
    int64_t base_start = (ii > 0) ? stage_base_start[s] / ii * ii : 0;
    std::vector<std::pair<Var, Var>> renames; // old var -> fresh clone var
    std::vector<std::shared_ptr<TaskNode>> stage_clones;
    for (size_t d : decls) {
      ICHECK(let_closure.is_declaration[d]);
      auto *decl_task = static_cast<TaskNode *>(ctrl->children[d].get());
      ffi::Optional<Bind> decl_bind = GetFlatTaskBind(decl_task->stmt);
      ICHECK(decl_bind.defined());
      auto new_var = decl_bind.value()->var.copy_with_suffix("");
      auto cloned_task = std::static_pointer_cast<TaskNode>(decl_task->Clone());
      cloned_task->stmt = ReplaceFlatTaskBind(
          decl_task->stmt,
          Bind(new_var, decl_bind.value()->value, decl_bind.value()->span));
      cloned_task->SetReadVars(decl_task->GetReadVars());
      auto write_vars = decl_task->GetWriteVars();
      for (auto &v : write_vars) {
        if (v.same_as(decl_bind.value()->var))
          v = new_var;
      }
      cloned_task->SetWriteVars(write_vars);
      cloned_task->SetStage(s);
      GetAutoScheduleExtraInfo(cloned_task.get())->SetStartTime(base_start);
      renames.emplace_back(decl_bind.value()->var, new_var);
      stage_clones.push_back(cloned_task);
    }
    // Rewire chained references among the clones
    for (auto &clone : stage_clones)
      for (auto &[old_var, new_var] : renames)
        clone->SubstituteVar(old_var, new_var);
    for (size_t u = 0; u < n; ++u) {
      if (ctrl->children[u]->GetStage() != s)
        continue;
      for (auto &[old_var, new_var] : renames)
        ctrl->children[u]->SubstituteVar(old_var, new_var);
    }
    // Emit the clones defs-before-uses
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> clone_writes;
    for (auto &c : stage_clones)
      for (auto &w : c->GetWriteVars())
        clone_writes.insert(w);
    std::vector<bool> emitted(stage_clones.size(), false);
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> available;
    size_t done = 0;
    while (done < stage_clones.size()) {
      bool progress = false;
      for (size_t ci = 0; ci < stage_clones.size(); ++ci) {
        if (emitted[ci])
          continue;
        bool ready = true;
        for (auto &rv : stage_clones[ci]->GetReadVars()) {
          if (clone_writes.count(rv) && !available.count(rv)) {
            ready = false;
            break;
          }
        }
        if (!ready)
          continue;
        for (auto &w : stage_clones[ci]->GetWriteVars())
          available.insert(w);
        new_clones.push_back(stage_clones[ci]);
        emitted[ci] = true;
        ++done;
        progress = true;
      }
      if (!progress) { // defensive: emit any remainder (no cycle expected)
        for (size_t ci = 0; ci < stage_clones.size(); ++ci) {
          if (!emitted[ci]) {
            new_clones.push_back(stage_clones[ci]);
            emitted[ci] = true;
            ++done;
          }
        }
      }
    }
  }
  ctrl->children.insert(ctrl->children.begin(), new_clones.begin(),
                        new_clones.end());

  // Estimate overall latency across loop iterations.
  int64_t overall_latency = 0;
  for (const auto &child : ctrl->children) {
    int64_t start_time = GetAutoScheduleExtraInfo(child.get())->GetStartTime();
    overall_latency =
        std::max(overall_latency, start_time + child->GetLatency());
  }
  overall_latency += ii * (ctrl->GetTripCount() - 1);

  ctrl->SetII(overall_latency);
  ctrl->SetLatency(overall_latency);
  GetAutoScheduleExtraInfo(ctrl)->SetIIperIter(ii);
  for (const StorageResource &resource : storage_resources) {
    const Var &storage = resource.storage;
    auto annotated = annotated_storage_versions.find(storage);
    if (annotated != annotated_storage_versions.end()) {
      MergeSelectedBufferVersion(storage, annotated->second);
      continue;
    }
    auto selected = storage_version_id.find(storage);
    if (selected != storage_version_id.end()) {
      MergeSelectedBufferVersion(storage, storage_versions[selected->second]);
    }
  }
}

// Pass orchestration.

static void WarnOnUnappliedBufferVersionOverrides(
    const BufferVersionMap &buffer_version_overrides,
    const BufferVersionMap &selected_buffer_versions) {
  for (const auto &[buffer_var, num_versions] : buffer_version_overrides) {
    if (num_versions > 1 && !selected_buffer_versions.count(buffer_var)) {
      LOG(WARNING) << "Ascend AutoSchedule did not enable multi-buffering for "
                      "buffer \""
                   << buffer_var->name_hint
                   << "\" although T.annotate_buffer_versions requested "
                   << num_versions
                   << " versions. The buffer must be eligible in a scheduled "
                      "serial loop and written before it is read.";
    }
  }
}

// Build and schedule one kernel segment, leaving core assignment/resolution
// and final lowering to the downstream passes.
static void ScheduleSingleKernel(const Stmt &kernel_body, Target target,
                                 ScheduledTIR *scheduled_tir) {
  auto &ir_structure = scheduled_tir->tree;
  ScheduledTIRMetadata &metadata = scheduled_tir->metadata;
  ICHECK(!ir_structure.empty()) << "IRStructure is empty (empty body?)";

  std::vector<TaskNode *> costed_tasks;
  CollectAllTaskNodes(ir_structure, costed_tasks);
  for (TaskNode *task : costed_tasks) {
    ICHECK_GT(task->GetII(), 0)
        << "AutoSchedule found a task without latency/II annotations: "
        << task->stmt
        << ". Run tl.transform.EstimateLatency before AutoSchedule.";
  }

  // Schedule IRStructure in place with the Z3 scheduler.
  ScheduleBuilder unit_builder;
  int64_t shared_mem_limit = GetSharedMemoryLimit(kernel_body);
  unit_builder.SetMemoryLimit("shared", shared_mem_limit);
  unit_builder.SetMemoryLimit("shared.l1", GetL1MemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l0c", GetL0CMemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l0a", GetL0AMemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l0b", GetL0BMemoryLimit(target));
  constexpr int64_t kUnlimitedMemory = 1LL << 40;
  for (const String &scope : metadata.unlimit_memory_scopes) {
    unit_builder.SetMemoryLimit(scope, kUnlimitedMemory);
  }
  unit_builder.SetBufferVersionOverrides(metadata.buffer_versions);
  unit_builder.SetManualBufferVersions(metadata.manual_buffer_versions);
  unit_builder.SetRootConflictHints(metadata.root_conflict_hints);
  unit_builder.ScheduleList(ir_structure);

  metadata.unlimit_memory_scopes = {};
  metadata.buffer_versions = unit_builder.GetSelectedBufferVersions();
}
} // namespace

// Build and schedule each kernel, serializing the tree into ordinary TIR
// markers for the downstream scheduled-TIR passes.
tvm::transform::Pass AutoSchedule() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    // Get target from PrimFunc attribute for GemmInst determination
    auto target_opt = func->GetAttr<Target>(tvm::attr::kTarget);
    Target target;
    if (target_opt.defined()) {
      target = target_opt.value();
    }
    return RewriteTilelangKernels(
        std::move(func), "AutoSchedule",
        [&](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir = DecodeScheduledTIR(
              context.root, context.outer_ctx, MakeAutoScheduleExtraInfo);
          BufferVersionMap buffer_version_overrides =
              scheduled_tir.metadata.buffer_versions;
          ScheduleSingleKernel(context.root->body, target, &scheduled_tir);
          WarnOnUnappliedBufferVersionOverrides(
              buffer_version_overrides, scheduled_tir.metadata.buffer_versions);
          return EncodeScheduledTIR(std::move(scheduled_tir));
        });
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm
