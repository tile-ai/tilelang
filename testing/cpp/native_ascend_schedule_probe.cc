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

// Test-only FFI access to the production scheduler codec and analyses.
#include "ascend/transform/auto_schedule/buffer_alias_analysis.h"
#include "ascend/transform/auto_schedule/dependency_analysis.h"
#include "ascend/transform/auto_schedule/multi_buffer.h"
#include "ascend/transform/auto_schedule/scheduled_tir.h"
#include "ascend/transform/auto_schedule/task_analysis.h"
#include "transform/common/native_layout_utils.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl::ascend {
namespace {

template <typename T> ffi::Array<T> ArrayOf(const std::vector<T> &values) {
  return ffi::Array<T>(values.begin(), values.end());
}

ffi::Array<ffi::Any> Prove(const ConstrSet &context,
                           const ffi::Array<PrimExpr> &queries) {
  auto analyzer = CreateTileLangAnalyzer();
  auto scope = context.Populate(analyzer);
  ffi::Array<ffi::Any> results;
  for (const auto &query : queries)
    results.push_back(analyzer->CanProve(query));
  return results;
}

ffi::Map<ffi::String, ffi::Any> AnalyzeSchedule(RegionStmt root,
                                                ffi::Array<PrimExpr> queries) {
  auto schedule = DecodeScheduledTIR(root, {});
  std::vector<TaskNode *> tasks;
  for (const auto &node : schedule.tree)
    CollectAllTaskNodes(node.get(), tasks);
  CoreMaskAvailability availability(tasks);
  ffi::Array<ffi::Any> task_rows, loop_rows, order;
  for (const TaskNode *task : tasks) {
    task_rows.push_back(ffi::Map<ffi::String, ffi::Any>{
        {"statement", task->stmt},
        {"reads", ArrayOf(task->GetReadRegions())},
        {"writes", ArrayOf(task->GetWriteRegions())},
        {"read_vars", ArrayOf(task->GetReadVars())},
        {"write_vars", ArrayOf(task->GetWriteVars())},
        {"core", static_cast<int>(task->GetCoreMask())},
        {"input_core",
         static_cast<int>(availability.GetTaskInputCoreMask(task))},
        {"stage", task->GetStage()},
        {"pipe", task->GetPipeMask()},
        {"proofs", Prove(task->outer_ctx, queries)}});
  }
  for (TaskNode *lhs : tasks) {
    ffi::Array<ffi::Any> row;
    for (TaskNode *rhs : tasks)
      row.push_back(CompareIRStructure(lhs, rhs));
    order.push_back(row);
  }
  std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node);
      loop_rows.push_back(ffi::Map<ffi::String, ffi::Any>{
          {"loop", control->control},
          {"trip_count", control->GetTripCount()},
          {"no_wrap", AnalyzeSerialInduction(control).no_wrap},
          {"known_progress",
           buffer_alias_analysis_detail::LoopHasKnownProgress(control, false)},
          {"single_trip",
           buffer_alias_analysis_detail::IsGuaranteedSingleTrip(control)},
          {"proofs", Prove(control->GetLoopBodyContext(), queries)}});
    }
    for (const auto *children : node->ChildLists())
      for (const auto &child : *children)
        visit(child.get());
  };
  for (const auto &node : schedule.tree)
    visit(node.get());
  return {{"tasks", task_rows}, {"loops", loop_rows}, {"order", order}};
}

ffi::Map<ffi::String, ffi::Any> AnalyzeMemory(Stmt body, Stmt definitions,
                                              bool physical) {
  NativeStorageContext storage;
  storage.Collect(definitions);
  auto detector = ffi::make_object<MemoryAccessDetector>(
      physical ? MemoryAccessDetector::AccessMode::kPhysical
               : MemoryAccessDetector::AccessMode::kLogical,
      LayoutMap{}, storage);
  detector->Analyze(body);
  return {{"reads", ArrayOf(detector->GetReadRegions())},
          {"writes", ArrayOf(detector->GetWriteRegions())},
          {"must_writes", ArrayOf(detector->GetMustWriteRegions())},
          {"opaque", ArrayOf(detector->GetOpaqueAccessVars())},
          {"read_vars", ArrayOf(detector->GetReadVars())},
          {"write_vars", ArrayOf(detector->GetWriteVars())}};
}

ffi::Array<ffi::Any> InspectStorage(Stmt definitions,
                                    ffi::Array<TensorVar> tensors,
                                    ffi::Array<ffi::Array<PrimExpr>> indices) {
  TVM_FFI_CHECK_EQ(tensors.size(), indices.size(), ValueError);
  NativeStorageContext context;
  context.Collect(definitions);
  auto analyzer = CreateTileLangAnalyzer();
  ffi::Array<ffi::Any> results;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto address = context.Address(tensors[i]);
    ffi::Map<ffi::String, ffi::Any> row{
        {"root", address.root},
        {"exact", address.exact},
        {"fresh_allocation", address.fresh_allocation},
        {"bytes", analyzer->Simplify(address.byte_offset)}};
    if (address.exact)
      row.Set("bits", analyzer->Simplify(
                          context.ElementBitOffset(tensors[i], indices[i])));
    results.push_back(row);
  }
  return results;
}

ffi::Map<ffi::String, ffi::Any>
InspectMultiBufferPlan(RegionStmt root, ffi::Array<Var> storages) {
  auto schedule = DecodeScheduledTIR(root, {});
  auto plan =
      ReadMultiBufferPlan(schedule.tree, schedule.metadata.buffer_versions,
                          L0StorageGroups(CollectL0SFBindings(root)));
  EpochDomainRegistry registry(schedule.tree, plan);
  ffi::Array<ffi::Any> infos, scopes;
  for (const auto &info : plan.Infos()) {
    ffi::Array<ffi::Any> owners;
    for (const auto &owner : info.owners)
      owners.push_back(ffi::Map<ffi::String, ffi::Any>{
          {"loop", owner.loop->control->var}, {"guard", owner.active_guard}});
    ffi::Map<ffi::String, ffi::Any> row{
        {"storage", info.storage},
        {"versions", info.num_versions},
        {"counter_group", info.counter_group_id},
        {"owners", owners},
        {"uses_counter", info.UsesCounter()}};
    if (info.UsesCounter()) {
      row.Set("counter", info.counter);
      row.Set("counter_storage", info.CounterStorage());
    }
    infos.push_back(row);
  }
  auto add_scope = [&](IRStructure *node, size_t child_list_index) {
    ffi::Array<ffi::Any> domains, projections;
    for (const Var &storage : storages) {
      int id = registry.DomainForStorage(storage, node, child_list_index);
      const auto &domain = registry.Domain(id);
      domains.push_back(
          ffi::Map<ffi::String, ffi::Any>{{"storage", storage},
                                          {"id", id},
                                          {"guard", domain.active_guard},
                                          {"counter", domain.IsCounter()}});
      ffi::Array<ffi::Any> targets;
      for (const Var &target : storages)
        targets.push_back(registry.CanProjectDomain(
            id, registry.DomainForStorage(target, node, child_list_index), node,
            child_list_index));
      projections.push_back(targets);
    }
    ffi::Map<ffi::String, ffi::Any> row{
        {"list", static_cast<int>(child_list_index)},
        {"domains", domains},
        {"projection", projections},
        {"unconditional",
         registry.UnconditionalLexicalDomain(node, child_list_index)}};
    if (node) {
      row.Set("owner", node->IsControl()
                           ? Stmt(static_cast<ControlNode *>(node)->control)
                           : static_cast<ScopeNode *>(node)->shell);
    }
    scopes.push_back(row);
  };
  add_scope(nullptr, 0);
  std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
    auto lists = node->ChildLists();
    for (size_t i = 0; i < lists.size(); ++i) {
      add_scope(node, i);
      for (const auto &child : *lists[i])
        visit(child.get());
    }
  };
  for (const auto &node : schedule.tree)
    visit(node.get());
  return {{"infos", infos}, {"scopes", scopes}};
}

ffi::Map<ffi::String, ffi::Any>
InspectDependencies(RegionStmt root, const BufferVersionMap &manual,
                    const ffi::Map<Var, ffi::Array<Var>> &owner_vars) {
  auto schedule = DecodeScheduledTIR(root, {});
  std::vector<TaskNode *> tasks;
  CollectTaskNodesAndHeaders(schedule.tree, tasks);
  std::unordered_map<TaskNode *, int64_t> ids;
  ffi::Array<ffi::Any> task_rows, scope_rows;
  std::unordered_map<Var, ControlNode *, ObjectPtrHash, ObjectPtrEqual> loops;
  std::function<void(IRStructure *)> collect_loops = [&](IRStructure *node) {
    if (node->IsControl()) {
      auto *loop = static_cast<ControlNode *>(node);
      loops.emplace(loop->control->var, loop);
    }
    for (const auto *list : node->ChildLists())
      for (const auto &child : *list)
        collect_loops(child.get());
  };
  for (const auto &node : schedule.tree)
    collect_loops(node.get());
  MultiBufferOwnerMap owners;
  for (const auto &[storage, vars] : owner_vars)
    for (const auto &var : vars) {
      TVM_FFI_CHECK(loops.count(var), ValueError)
          << "Unknown owner loop " << var;
      owners[storage].push_back(loops.at(var));
    }
  for (size_t i = 0; i < tasks.size(); ++i) {
    TaskNode *task = tasks[i];
    ids.emplace(task, i);
    auto *parent = task->GetParent();
    task_rows.push_back(ffi::Map<ffi::String, ffi::Any>{
        {"statement", task->stmt},
        {"id", static_cast<int64_t>(i)},
        {"header", parent && parent->HeaderTask() == task},
        {"reads", ArrayOf(task->GetReadRegions())},
        {"writes", ArrayOf(task->GetWriteRegions())},
        {"opaque", ArrayOf(task->GetOpaqueAccessVars())}});
  }
  ScopedConflictHintIndex scoped_hints(schedule.tree,
                                       schedule.metadata.outer_ctx);
  DependencyCache cache;
  std::function<void(const IRStructure::ChildList &, IRStructure *, int)>
      inspect = [&](const IRStructure::ChildList &list, IRStructure *owner,
                    int list_index) {
        ControlNode *clock =
            owner ? (owner->IsControl() ? static_cast<ControlNode *>(owner)
                                        : owner->GetParentControl())
                  : nullptr;
        std::vector<IRStructure *> nodes;
        for (const auto &node : list)
          nodes.push_back(node.get());
        ffi::Array<ffi::Any> dependencies;
        for (const auto &dep : AnalyzeDependencies(
                 nodes, clock, manual, owners, &cache,
                 schedule.metadata.root_conflict_hints, &scoped_hints)) {
          ffi::Array<ffi::Array<int64_t>> pairs;
          for (const auto &[producer, consumer] : dep.task_pairs)
            pairs.push_back({ids.at(producer), ids.at(consumer)});
          ffi::Map<ffi::String, ffi::Any> row{
              {"producer", static_cast<int64_t>(dep.prod_node->GetIndex())},
              {"consumer", static_cast<int64_t>(dep.cons_node->GetIndex())},
              {"distance", dep.distance},
              {"pairs", pairs},
              {"owner_exclusion", dep.kind == DependencyKind::kOwnerExclusion}};
          if (dep.storage)
            row.Set("storage", *dep.storage);
          dependencies.push_back(row);
        }
        ffi::Map<ffi::String, ffi::Any> row{{"dependencies", dependencies},
                                            {"list", list_index}};
        if (clock)
          row.Set("clock", clock->control->var);
        if (owner)
          row.Set("owner",
                  owner->IsControl()
                      ? Stmt(static_cast<ControlNode *>(owner)->control)
                      : static_cast<ScopeNode *>(owner)->shell);
        scope_rows.push_back(row);
        for (const auto &node : list) {
          auto children = node->ChildLists();
          for (size_t i = 0; i < children.size(); ++i)
            inspect(*children[i], node.get(), static_cast<int>(i));
        }
      };
  inspect(schedule.tree, nullptr, 0);
  return {{"tasks", task_rows},
          {"scopes", scope_rows},
          {"cache_entries", static_cast<int64_t>(cache.size())}};
}

bool InspectRegionConflict(Stmt definitions, TensorRegion lhs, TensorRegion rhs,
                           ffi::Array<PrimExpr> lhs_conditions,
                           ffi::Array<PrimExpr> rhs_conditions,
                           Optional<PrimExpr> lhs_pitch,
                           Optional<PrimExpr> rhs_pitch) {
  NativeStorageContext storage;
  storage.Collect(definitions);
  ConstrSet a_context, b_context;
  for (const auto &condition : lhs_conditions)
    a_context.AddConstr(condition);
  for (const auto &condition : rhs_conditions)
    b_context.AddConstr(condition);
  return RegionsMayConflict(a_context, lhs, b_context, rhs, nullptr, 0, 0, {},
                            lhs_pitch, rhs_pitch, storage, storage);
}

// Tests supply a graph of already-guaranteed completion/issue edges in one
// explicit clock. This fixture does not infer synchronization from lexical
// order or silently transfer distances between parent and child loops.
class ProbeLifetimeContext : public BufferAliasAnalysisContext {
public:
  ProbeLifetimeContext(const std::vector<TaskNode *> &tasks, ControlNode *clock,
                       const ffi::Array<ffi::Array<int64_t>> &edges,
                       ffi::Map<Var, int64_t> periods, bool implicit_slots,
                       bool recurrence)
      : clock_(clock), order_(tasks.size() * 2), periods_(std::move(periods)),
        implicit_slots_(implicit_slots), recurrence_(recurrence) {
    for (size_t i = 0; i < tasks.size(); ++i) {
      ids_.emplace(tasks[i], i);
      order_.AddEdge(2 * i, 2 * i + 1, 0);
      if (recurrence_)
        for (size_t endpoint = 0; endpoint < 2; ++endpoint)
          order_.AddEdge(2 * i + endpoint, 2 * i + endpoint, 1);
    }
    for (const auto &edge : edges) {
      TVM_FFI_CHECK_EQ(edge.size(), 4U, ValueError);
      TVM_FFI_CHECK(edge[0] >= 0 && size_t(edge[0]) < tasks.size() &&
                        edge[1] >= 0 && size_t(edge[1]) < tasks.size(),
                    ValueError);
      TVM_FFI_CHECK(
          order_.AddEdge(2 * edge[0] + 1, 2 * edge[1], edge[2], edge[3]),
          ValueError);
    }
  }
  std::vector<size_t> ResolveLifetimeEndpoints(const Var &, TaskNode *task,
                                               bool at_beginning,
                                               ControlNode *scope) const final {
    if (scope != clock_)
      return {};
    return {2 * ids_.at(task) + (at_beginning ? 0U : 1U)};
  }
  bool HappensBefore(ControlNode *scope, size_t src, size_t dst,
                     int64_t distance) const final {
    if (scope != clock_ || distance < 0)
      return false;
    auto minimum = order_.MinimumDistance(src, dst);
    return minimum &&
           (recurrence_ ? distance >= *minimum : distance == *minimum);
  }
  std::optional<int64_t> MinimumDistance(ControlNode *scope, size_t src,
                                         size_t dst) const final {
    return scope == clock_ && recurrence_ ? order_.MinimumDistance(src, dst)
                                          : std::nullopt;
  }
  std::optional<StorageIterationPeriod>
  StoragePeriod(const Var &storage, ControlNode *scope) const final {
    auto period = periods_.Get(storage);
    if (scope != clock_ || !period || *period <= 0)
      return std::nullopt;
    return StorageIterationPeriod{*period, implicit_slots_};
  }
  size_t PromoteLifetimeSite(const Var &, size_t site,
                             ControlNode *) const final {
    return site;
  }

private:
  ControlNode *clock_;
  IterationOrder order_;
  std::unordered_map<TaskNode *, size_t> ids_;
  ffi::Map<Var, int64_t> periods_;
  bool implicit_slots_;
  bool recurrence_;
};

BufferAliasMap InspectLifetimes(RegionStmt root, Optional<Var> clock_var,
                                ffi::Array<ffi::Array<int64_t>> edges,
                                ffi::Map<Var, int64_t> periods,
                                bool implicit_slots, bool recurrence) {
  auto schedule = DecodeScheduledTIR(root, {});
  std::vector<TaskNode *> tasks;
  CollectTaskNodesAndHeaders(schedule.tree, tasks);
  ControlNode *clock = nullptr;
  std::function<void(IRStructure *)> find_clock = [&](IRStructure *node) {
    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node);
      if (clock_var && control->control->var.same_as(*clock_var))
        clock = control;
    }
    for (const auto *list : node->ChildLists())
      for (const auto &child : *list)
        find_clock(child.get());
  };
  for (const auto &node : schedule.tree)
    find_clock(node.get());
  TVM_FFI_CHECK(!clock_var || clock, ValueError) << "Unknown lifetime clock";
  ProbeLifetimeContext context(tasks, clock, edges, std::move(periods),
                               implicit_slots, recurrence);
  BufferAliasAnalyzer analyzer(context,
                               L0StorageGroups(CollectL0SFBindings(root)));
  analyzer.Collect(schedule.tree);
  analyzer.Analyze(schedule.tree);
  analyzer.Validate();
  return analyzer.GetAliases();
}

ffi::Array<ffi::Any>
InspectIterationOrder(int64_t sites, ffi::Array<ffi::Array<int64_t>> edges,
                      ffi::Array<ffi::Array<int64_t>> queries) {
  TVM_FFI_CHECK_GE(sites, 0, ValueError);
  IterationOrder order(sites);
  for (const auto &edge : edges) {
    TVM_FFI_CHECK_EQ(edge.size(), 4U, ValueError);
    TVM_FFI_CHECK(order.AddEdge(edge[0], edge[1], edge[2], edge[3]),
                  ValueError);
  }
  ffi::Array<ffi::Any> result;
  for (const auto &query : queries) {
    TVM_FFI_CHECK_EQ(query.size(), 2U, ValueError);
    auto distance = order.MinimumDistance(query[0], query[1]);
    result.push_back(distance ? ffi::Any(*distance) : ffi::Any(nullptr));
  }
  return result;
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.testing.ReadLoopAnnotations",
           [](For loop) { return ReadLoopAnnotations(loop.get()); })
      .def("tl.testing.AppendLoopAnnotation",
           [](For loop, ffi::Map<ffi::String, ffi::Any> fields) {
             return AppendLoopAnnotation(loop, fields);
           })
      .def("tl.testing.RewriteLoopAnnotationCarriers",
           [](For loop, ffi::Function rewrite) {
             return RewriteLoopAnnotationCarriers(
                 loop, [&](const ffi::Map<ffi::String, ffi::Any> &fields) {
                   return rewrite(fields)
                       .cast<ffi::Map<ffi::String, ffi::Any>>();
                 });
           })
      .def("tl.testing.AnalyzeAscendSchedule", AnalyzeSchedule)
      .def("tl.testing.AnalyzeAscendDependencies", InspectDependencies)
      .def("tl.testing.AscendRegionsMayConflict", InspectRegionConflict)
      .def("tl.testing.AnalyzeAscendLifetimes", InspectLifetimes)
      .def("tl.testing.AscendIterationOrder", InspectIterationOrder)
      .def("tl.testing.AnalyzeAscendMemory", AnalyzeMemory)
      .def("tl.testing.InspectAscendStorage", InspectStorage)
      .def("tl.testing.InspectAscendMultiBufferPlan", InspectMultiBufferPlan)
      .def("tl.testing.CanBroadcastAscendFill",
           [](Stmt stmt, Var storage, Stmt definitions) {
             NativeStorageContext context;
             context.Collect(definitions);
             return CanBroadcastFillToStorage(stmt, storage, context);
           })
      .def("tl.testing.SetAscendBroadcastFill",
           [](Stmt stmt, Var storage, bool add) {
             TaskNode task;
             task.stmt = stmt;
             if (add)
               AddMultiBufferBroadcastFill(&task, storage);
             else
               ClearMultiBufferBroadcastFills(&task);
             return task.stmt;
           })
      .def("tl.testing.CompareAscendStorage",
           [](Stmt definitions, TensorVar lhs, TensorVar rhs) {
             NativeStorageContext context;
             context.Collect(definitions);
             return ffi::Map<ffi::String, ffi::Any>{
                 {"same_storage", context.SameStorage(lhs, rhs)},
                 {"disjoint", context.ProvenDisjoint(lhs, rhs)}};
           })
      .def("tl.testing.RoundTripAscendSchedule",
           [](RegionStmt root, bool clone) {
             auto schedule = DecodeScheduledTIR(root, {});
             if (clone)
               for (auto &node : schedule.tree)
                 node = node->Clone();
             return EncodeScheduledTIR(std::move(schedule));
           })
      .def("tl.testing.AscendRootConflictHintCodec",
           [](RegionStmt root, bool consume) {
             auto schedule = DecodeScheduledTIR(root, {});
             auto hints = schedule.metadata.root_conflict_hints;
             if (consume)
               schedule.metadata.root_conflict_hints = {};
             return ffi::Map<ffi::String, ffi::Any>{
                 {"hints", hints},
                 {"root", EncodeScheduledTIR(std::move(schedule))}};
           })
      .def("tl.testing.AscendScopedIndexAfterMutation",
           [](RegionStmt root, ffi::String mode, bool rebuild) {
             auto schedule = DecodeScheduledTIR(root, {});
             auto hints = std::make_unique<ScopedConflictHintIndex>(
                 schedule.tree, schedule.metadata.outer_ctx);
             DependencyCache cache;
             auto nodes = [&]() {
               std::vector<IRStructure *> result;
               for (const auto &node : schedule.tree)
                 result.push_back(node.get());
               return result;
             };
             auto before =
                 AnalyzeDependencies(nodes(), nullptr, {}, {}, &cache,
                                     schedule.metadata.root_conflict_hints,
                                     hints.get())
                     .size();
             if (mode == "clone" || mode == "foreign") {
               if (mode == "clone")
                 for (auto &node : schedule.tree)
                   node = node->Clone();
               else
                 schedule.tree = DecodeScheduledTIR(root, {}).tree;
             } else if (mode == "reparent") {
               ICHECK_GE(schedule.tree.size(), 2U);
               auto moved = schedule.tree.front();
               auto *parent = schedule.tree.back().get();
               auto lists = parent->ChildLists();
               ICHECK(!lists.empty());
               schedule.tree.erase(schedule.tree.begin());
               moved->SetParent(parent);
               moved->SetIndex(lists.front()->size());
               lists.front()->push_back(moved);
             } else if (mode == "remove_child" || mode == "header") {
               auto *scope =
                   static_cast<ScopeNode *>(schedule.tree.back().get());
               ICHECK(scope->IsScope());
               if (mode == "remove_child") {
                 ICHECK(!scope->children.empty());
                 scope->children.pop_back();
               } else {
                 ICHECK(scope->task);
                 scope->task =
                     std::static_pointer_cast<TaskNode>(scope->task->Clone());
                 scope->task->SetParent(scope);
               }
             } else if (mode == "statement") {
               bool changed = false;
               std::function<void(IRStructure *)> mutate =
                   [&](IRStructure *node) {
                     if (!changed && node->IsScope()) {
                       auto *scope = static_cast<ScopeNode *>(node);
                       if (IsTaskMetadataStatement(scope->shell)) {
                         scope->shell =
                             MakeNativeAnnotation({}, scope->shell->loc);
                         changed = true;
                       }
                     }
                     for (auto *list : node->ChildLists())
                       for (auto &child : *list)
                         mutate(child.get());
                   };
               for (auto &node : schedule.tree)
                 mutate(node.get());
               ICHECK(changed);
             } else if (mode == "core") {
               std::vector<TaskNode *> tasks;
               CollectAllTaskNodes(schedule.tree, tasks);
               ICHECK(!tasks.empty());
               tasks.front()->SetCoreMask(
                   tasks.front()->GetCoreMask() == kCoreCube ? kCoreAll
                                                             : kCoreCube);
             } else {
               TVM_FFI_THROW(ValueError)
                   << "Unknown scoped index mutation " << mode;
             }
             if (rebuild)
               hints = std::make_unique<ScopedConflictHintIndex>(
                   schedule.tree, schedule.metadata.outer_ctx);
             // Keep the old dependency cache deliberately. A fresh index must
             // also prevent cached answers from the previous hint state.
             auto after =
                 AnalyzeDependencies(nodes(), nullptr, {}, {}, &cache,
                                     schedule.metadata.root_conflict_hints,
                                     hints.get())
                     .size();
             return ffi::Map<ffi::String, ffi::Any>{
                 {"before", static_cast<int64_t>(before)},
                 {"after", static_cast<int64_t>(after)}};
           })
      .def("tl.testing.SubstituteAscendSchedule",
           [](RegionStmt root, Var old_var, Var new_var) {
             auto schedule = DecodeScheduledTIR(root, {});
             for (auto &node : schedule.tree) {
               node = node->Clone();
               node->SubstituteVar(old_var, new_var);
             }
             std::vector<TaskNode *> tasks;
             CollectAllTaskNodes(schedule.tree, tasks);
             ffi::Array<PrimExpr> offsets;
             for (const TaskNode *task : tasks)
               for (const TensorRegion &region : task->GetReadRegions())
                 offsets.push_back(task->StorageContext()
                                       .Address(RegionTensor(region))
                                       .byte_offset);
             return ffi::Map<ffi::String, ffi::Any>{
                 {"root", EncodeScheduledTIR(std::move(schedule))},
                 {"read_byte_offsets", offsets}};
           })
      .def("tl.testing.SetAscendScheduleMetadata",
           [](RegionStmt root, BufferVersionMap versions,
              BufferAliasMap aliases, bool has_aliases) {
             auto schedule = DecodeScheduledTIR(root, {});
             schedule.metadata.buffer_versions = std::move(versions);
             schedule.metadata.buffer_aliases = std::move(aliases);
             schedule.metadata.has_buffer_aliases = has_aliases;
             return EncodeScheduledTIR(std::move(schedule));
           })
      .def("tl.testing.AscendInductionAfterDroppingTail",
           [](RegionStmt root) {
             auto schedule = DecodeScheduledTIR(root, {});
             ffi::Array<ffi::Any> results;
             std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
               if (node->IsControl()) {
                 auto *control = static_cast<ControlNode *>(node);
                 bool before = AnalyzeSerialInduction(control).no_wrap;
                 ICHECK(!control->children.empty());
                 control->children.pop_back();
                 results.push_back(ffi::Map<ffi::String, ffi::Any>{
                     {"before", before},
                     {"after", AnalyzeSerialInduction(control).no_wrap}});
                 return;
               }
               for (const auto *children : node->ChildLists())
                 for (const auto &child : *children)
                   visit(child.get());
             };
             for (const auto &node : schedule.tree)
               visit(node.get());
             return results;
           })
      .def("tl.testing.AscendLoopOrdinal", [](RegionStmt root) {
        auto schedule = DecodeScheduledTIR(root, {});
        ffi::Array<PrimExpr> results;
        std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
          if (node->IsControl())
            results.push_back(
                CalculateIterationCount(static_cast<ControlNode *>(node)));
          for (const auto *children : node->ChildLists())
            for (const auto &child : *children)
              visit(child.get());
        };
        for (const auto &node : schedule.tree)
          visit(node.get());
        return results;
      });
}
} // namespace tvm::tl::ascend
