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

#pragma once

#include <array>
#include <unordered_map>
#include <unordered_set>

#include <tvm/tirx/analysis.h>

#include "layout/layout.h"
#include "scheduled_tir.h"
#include "support/check.h"
#include "task_analysis.h"
#include "task_annotations.h"

namespace tvm {
namespace tl {
namespace ascend {

// Project only sid-dependent task entries. Common control nodes and Cube
// tasks remain shared. Origin metadata retains the correspondence between
// siblings through scheduling and synchronization insertion.
// Distinct UB/scalar identities let the existing ownership, dependence and
// counter algorithms operate on each subcore without counter-specific rules.
class SubcoreProjector {
public:
  static RegionStmt Rewrite(const RegionStmt &root, const ConstrSet &outer_ctx,
                            const Var &sid) {
    ICHECK(!root->attrs->dict.count(kSplitSubcores));
    ScheduledTIR scheduled = DecodeScheduledTIR(root, outer_ctx);
    SubcoreProjector projector(sid,
                               scheduled.metadata.num_aiv_subcores.value_or(2));
    std::vector<TaskNode *> original_tasks;
    CollectAllTaskNodes(scheduled.tree, original_tasks);
    for (TaskNode *task : original_tasks)
      for (const Var &variable : task->GetWriteVars())
        projector.local_definitions_.insert(variable);
    projector.FindVaryingValues(scheduled.tree);
    if (!projector.NeedsProjection(scheduled.tree))
      return root;
    Stmt body = projector.RewriteList(scheduled.tree, -1);
    projector.VerifyProjectedFixpipeTasks(body);
    Array<Stmt> aliases;
    for (const auto &[tensor, source] : projector.external_aliases_) {
      Bind declaration(tensor, Call(tensor->ty, tirx::decl_tensor_op(),
                                    {source.data()}, {}, {tensor->ty}));
      auto task = MakeScheduleMarker(attr::kAscendTask, ffi::Map<String, Any>{},
                                     declaration);
      aliases.push_back(MakeScheduleMarker(
          attr::kScheduleUnit, ScheduleUnitMetadata{{kScheduleUnitStage, -1}},
          task));
    }
    aliases.push_back(body);
    body = SeqStmt(aliases);
    VarSet used;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>())
        used.insert(ffi::GetRef<Var>(var));
      if (const auto *load = node.as<TensorLoadNode>())
        used.insert(load->source.as_or_throw<TensorVar>());
      if (const auto *store = node.as<TensorStoreNode>())
        used.insert(store->dest.as_or_throw<TensorVar>());
    });
    auto active = [&](const Var &storage, int p) {
      auto it = projector.vars_[p].find(storage);
      return it != projector.vars_[p].end() && used.count(it->second);
    };
    RegionStmt result = root;
    RegionStmtNode *writer = result.CopyOnWrite();
    writer->body = SeqStmt(std::move(body));
    Map<String, Any> annotations;
    for (const auto &[key, value] : root->attrs->dict) {
      if (auto versions = value.try_cast<BufferVersionMap>()) {
        BufferVersionMap mapped;
        for (const auto &[storage, versions_count] : versions.value()) {
          if (projector.varying_.count(storage)) {
            for (int p = 0; p < projector.count_; ++p)
              if (active(storage, p))
                mapped.Set(projector.VarFor(storage, p), versions_count);
          } else {
            mapped.Set(storage, versions_count);
          }
        }
        annotations.Set(key, mapped);
      } else if (auto modes = value.try_cast<BufferVersionModeMap>()) {
        BufferVersionModeMap mapped;
        for (const auto &[storage, mode] : modes.value()) {
          if (projector.varying_.count(storage)) {
            for (int p = 0; p < projector.count_; ++p)
              if (active(storage, p))
                mapped.Set(projector.VarFor(storage, p), mode);
          } else {
            mapped.Set(storage, mode);
          }
        }
        annotations.Set(key, mapped);
      } else if (auto layouts = value.try_cast<Map<TensorVar, Layout>>()) {
        Map<TensorVar, Layout> mapped;
        for (const auto &[buffer, layout] : layouts.value()) {
          if (projector.Varies(buffer)) {
            for (int p = 0; p < projector.count_; ++p)
              if (active(buffer, p) || (!projector.varying_.count(buffer) &&
                                        projector.buffers_[p].count(buffer)))
                mapped.Set(projector.BufferFor(buffer, p), layout);
          } else {
            mapped.Set(buffer, layout);
          }
        }
        annotations.Set(key, mapped);
      } else {
        annotations.Set(key, projector.ExpandAnnotation(key, value));
      }
    }
    AivStorageMap ub_sids;
    for (const auto &[storage, p] : projector.ub_sids_)
      if (used.count(storage))
        ub_sids.Set(storage, p);
    annotations.Set(kAivStorageMap, ub_sids);
    annotations.Set(kSplitSubcores, IntImm::Bool(true));
    writer->attrs = DictAttrs(std::move(annotations));
    return result;
  }

private:
  using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
  using VarMap = std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual>;
  using BufferMap =
      std::unordered_map<TensorVar, TensorVar, ObjectPtrHash, ObjectPtrEqual>;

  SubcoreProjector(Var sid, int count) : sid_(std::move(sid)), count_(count) {
    ICHECK(count == 1 || count == 2);
    varying_.insert(sid_);
  }

  bool Varies(const TaskNode *task) const {
    for (const Var &var : task->GetReadVars())
      if (varying_.count(var))
        return true;
    for (const Var &var : task->GetWriteVars())
      if (varying_.count(var))
        return true;
    for (const auto &regions :
         {task->GetReadRegions(), task->GetWriteRegions()})
      for (const TensorRegion &region : regions)
        if (Varies(region->source.as_or_throw<TensorVar>()) ||
            (IsSharedBuffer(region->source.as_or_throw<TensorVar>()) &&
             !task->UsesPipe(ResourcePipe::kFixpipe)))
          return true;
    return task->UsesPipe(ResourcePipe::kVector) ||
           task->UsesPipe(ResourcePipe::kMTE3);
  }

  bool Varies(const PrimExpr &expr) const {
    bool varying = false;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>())
        varying |= varying_.count(ffi::GetRef<Var>(var));
      if (const auto *load = node.as<TensorLoadNode>())
        varying |= varying_.count(load->source.as_or_throw<TensorVar>());
    });
    return varying;
  }

  bool Varies(const TensorVar &buffer) const {
    if (varying_.count(buffer))
      return true;
    bool varying = false;
    for (const Var &variable : tvm::UndefinedVars(buffer->ty))
      varying |= varying_.count(variable);
    return varying;
  }

  bool NeedsProjection(
      const std::vector<std::shared_ptr<IRStructure>> &nodes) const {
    // Uniform vector work can keep one shared body, including sid-dependent
    // addresses in software dual copies. Split when participation differs or
    // Cube needs a subcore-local value or UB destination. SS accesses also
    // need concrete subcore indices to distinguish independent shared words.
    for (const auto &node : nodes) {
      const TaskNode *task =
          node->IsTask()
              ? static_cast<const TaskNode *>(node.get())
              : static_cast<const ControlNode *>(node.get())->task.get();
      bool varying_control = false;
      PostOrderVisit(
          WrapScheduledGuards(task->stmt, node->GetGuards()),
          [&](const ObjectRef &stmt) {
            if (const auto *region = stmt.as<RegionStmtNode>();
                region && ScheduleMarkerKey(region) == kScheduleCondition)
              varying_control |=
                  Varies(ScheduleMarkerPayload(region).as_or_throw<PrimExpr>());
            if (const auto *branch = stmt.as<IfNode>())
              varying_control |= Varies(branch->condition);
            if (const auto *loop = stmt.as<ForNode>())
              varying_control |= Varies(loop->begin) || Varies(loop->end) ||
                                 Varies(loop->step);
          });
      if (varying_control)
        return true;
      if (Varies(task) && !task->UsesPipe(ResourcePipe::kVector) &&
          !task->UsesPipe(ResourcePipe::kMTE3)) {
        if (task->UsesPipe(ResourcePipe::kCube) ||
            task->UsesPipe(ResourcePipe::kMTE1) ||
            task->UsesPipe(ResourcePipe::kFixpipe))
          return true;
        for (const auto &regions :
             {task->GetReadRegions(), task->GetWriteRegions()})
          for (const TensorRegion &region : regions)
            if (IsSSBuffer(region->source.as_or_throw<TensorVar>()) ||
                IsL1Buffer(region->source.as_or_throw<TensorVar>()) ||
                IsL0ABuffer(region->source.as_or_throw<TensorVar>()) ||
                IsL0BBuffer(region->source.as_or_throw<TensorVar>()) ||
                IsL0CBuffer(region->source.as_or_throw<TensorVar>()))
              return true;
      }
      if (node->IsControl() &&
          NeedsProjection(
              static_cast<const ControlNode *>(node.get())->children))
        return true;
    }
    return false;
  }

  void
  FindVaryingValues(const std::vector<std::shared_ptr<IRStructure>> &tree) {
    std::vector<TaskNode *> tasks;
    CollectAllTaskNodes(tree, tasks);
    for (TaskNode *task : tasks) {
      PostOrderVisit(task->stmt, [&](const ObjectRef &node) {
        const auto *call = node.as<CallNode>();
        if (!call || !IsAscendCopyCall(call))
          return;
        AscendCopy copy(call->args, AscendCallAnnotations(call));
        if (IsL0CBuffer(copy->src) && IsSharedBuffer(copy->dst) &&
            copy->dual_dst_ctl != 0) {
          // Hardware dual destinations share an offset and a generation.
          // Keep this UB allocation common to both address spaces and keep
          // the Fixpipe instruction intact (including partial fractals).
          dual_storages_.insert(task->StorageOf(copy->dst));
          dual_tasks_.insert(task);
        }
      });
    }
    for (TaskNode *task : tasks) {
      for (const auto &regions :
           {task->GetReadRegions(), task->GetWriteRegions()})
        for (const TensorRegion &region : regions) {
          auto tensor = region->source.as_or_throw<TensorVar>();
          Var storage = task->StorageOf(tensor);
          if (IsSharedBuffer(tensor) && !dual_storages_.count(storage)) {
            varying_.insert(tensor);
            varying_.insert(storage);
          }
        }
    }
    bool changed = true;
    while (changed) {
      changed = false;
      for (TaskNode *task : tasks) {
        bool varying = Varies(task);
        for (const IRStructure *parent = task->GetParent(); parent;
             parent = parent->GetParent())
          varying |=
              Varies(static_cast<const ControlNode *>(parent)->task.get());
        if (!varying)
          continue;
        for (const Var &var : task->GetWriteVars())
          changed |= varying_.insert(var).second;
        for (const TensorRegion &region : task->GetWriteRegions())
          if (IsRegisterRegion(region))
            changed |=
                varying_.insert(region->source.as_or_throw<TensorVar>()).second;
      }
    }
  }

  Var VarFor(const Var &variable, int subcore) {
    if (auto tensor = variable.as<TensorVar>())
      return BufferFor(tensor.value(), subcore);
    if (!varying_.count(variable))
      return variable;
    auto found = vars_[subcore].find(variable);
    if (found != vars_[subcore].end())
      return found->second;
    Var result(variable->name + "_aiv" + std::to_string(subcore), variable->ty,
               variable->loc);
    vars_[subcore].emplace(variable, result);
    vars_[subcore].emplace(result, result);
    return result;
  }

  TensorVar BufferFor(const TensorVar &buffer, int subcore) {
    auto found = buffers_[subcore].find(buffer);
    if (found != buffers_[subcore].end())
      return found->second;
    if (!Varies(buffer))
      return buffer;
    auto rewriter = ffi::make_object<Rewriter>(this, subcore);
    auto type = rewriter->Mutate(buffer->ty)
                    .ValueOrUnchanged(buffer->ty)
                    .as_or_throw<TensorType>();
    TensorVar result(buffer->name + "_aiv" + std::to_string(subcore), type,
                     buffer->loc);
    buffers_[subcore].emplace(buffer, result);
    buffers_[subcore].emplace(result, result);
    vars_[subcore].emplace(buffer, result);
    vars_[subcore].emplace(result, result);
    if (IsSharedBuffer(buffer) && varying_.count(buffer))
      ub_sids_.Set(result, subcore);
    if (!local_definitions_.count(buffer))
      external_aliases_.emplace_back(result, buffer);
    return result;
  }

  static void VerifyReplayableFixpipeTask(const Stmt &body) {
    bool external_effect = false;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *store = node.as<TensorStoreNode>())
        external_effect |=
            !(IsLocalBuffer(store->dest.as_or_throw<TensorVar>(), true) ||
              IsFragmentBuffer(store->dest.as_or_throw<TensorVar>()));
      if (const auto *call = node.as<CallNode>()) {
        if (IsAscendCopyCall(call)) {
          AscendCopy copy(call->args, AscendCallAnnotations(call));
          external_effect |=
              !IsL0CBuffer(copy->src) || !IsSharedBuffer(copy->dst);
        } else {
          external_effect |=
              SideEffect(ffi::GetRef<Expr>(call)) > CallEffectKind::kReadState;
        }
      }
    });
    ICHECK(!external_effect)
        << "Fixpipe routing in a projected T.Task with external side effects "
           "is not supported: projection may repeat those effects. Move the "
           "side effects into a separate T.Task and snapshot any dynamic "
           "route before the Fixpipe task";
  }

  void VerifyProjectedFixpipeTasks(const Stmt &body) const {
    std::unordered_map<int, std::vector<RegionStmt>> projections;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      const auto *attr = node.as<RegionStmtNode>();
      if (!attr || (ScheduleMarkerKey(attr) != tl::attr::kAscendTask &&
                    ScheduleMarkerKey(attr) != tl::attr::kAscendPerCoreTask))
        return;
      auto metadata =
          task_metadata_detail::CopyTaskMetadata(ScheduleMarkerPayload(attr));
      auto origin = metadata.Get("subcore_origin");
      if (origin &&
          replay_check_origins_.count(GetIntAnnotation(origin.value()).value()))
        projections[GetIntAnnotation(origin.value()).value()].push_back(
            ffi::GetRef<RegionStmt>(attr));
    });
    // An enclosing projected loop may rewrite a task once per recursive call.
    // Count surviving copies across the complete tree, after outer guards have
    // removed inactive branches, rather than within one RewriteList call.
    for (const auto &[origin, tasks] : projections) {
      if (tasks.size() > 1)
        for (const RegionStmt &task : tasks)
          VerifyReplayableFixpipeTask(task);
    }
  }

  class Rewriter : public NativeMutator {
  public:
    Rewriter(SubcoreProjector *owner, int p, bool allow_constants = true)
        : owner_(owner), p_(p), conditional_depth_(allow_constants ? 0 : 1) {}

    bool HasLocalCopyGuard() const { return has_local_copy_guard_; }

    bool CanReplayFixpipe() const {
      return has_varying_copy_route_ || has_local_copy_guard_;
    }

    Any RewriteMetadata(const Any &value) {
      if (auto buffer = value.try_cast<TensorVar>())
        return owner_->BufferFor(buffer.value(), p_);
      if (auto expression = value.as<Expr>())
        return VisitExpr(expression.value());
      if (auto array = value.try_cast<Array<Any>>()) {
        Array<Any> result;
        for (const Any &entry : array.value())
          result.push_back(RewriteMetadata(entry));
        return result;
      }
      if (auto map = value.try_cast<Map<Any, Any>>()) {
        Map<Any, Any> result;
        for (const auto &[key, entry] : map.value())
          result.Set(RewriteMetadata(key), RewriteMetadata(entry));
        return result;
      }
      return value;
    }

  private:
    struct PerCoreRouting {
      PrimExpr path{IntImm::Bool(true)};
      PrimExpr active{IntImm::Bool(false)};
      bool routed{false};
      bool replays_fixpipe{false};
    };

    PrimExpr GetPerCoreActiveGuard(const RegionStmt &task, PrimExpr active) {
      class MarkerUnwrapper : public NativeMutator {
        Stmt VisitStmt_(const RegionStmtNode *op) final {
          if (!ScheduleMarkerKey(op).empty())
            return VisitStmt(UnwrapScheduleMarker(op));
          return NativeMutator::VisitStmt_(op);
        }
      };
      Map<PrimVar, PrimExpr> bindings;
      VarSet locals;
      PostOrderVisit(
          MarkerUnwrapper().VisitStmt(task), [&](const ObjectRef &node) {
            if (const auto *loop = node.as<ForNode>())
              locals.insert(loop->var);
            if (const auto *bind = node.as<BindNode>()) {
              locals.insert(bind->var);
              auto scalar = bind->var.as<PrimVar>();
              auto scalar_value = bind->value.as<PrimExpr>();
              if (!scalar || !scalar_value)
                return;
              PrimExpr value = Substitute(scalar_value.value(), bindings);
              bool immutable = SideEffect(value) <= CallEffectKind::kReadState;
              PostOrderVisit(value, [&](const ObjectRef &input) {
                immutable &= input.as<TensorLoadNode>() == nullptr;
              });
              if (immutable)
                bindings.Set(scalar.value(), value);
            }
          });
      // Condition snapshots belong inside PerCoreTask. Expand only immutable
      // bindings when exposing the selected candidates to outer
      // synchronization.
      active = analyzer_->Simplify(Substitute(active, bindings));
      bool available = SideEffect(active) <= CallEffectKind::kReadState;
      PostOrderVisit(active, [&](const ObjectRef &node) {
        if (const auto *var = node.as<VarNode>())
          available &= !locals.count(ffi::GetRef<Var>(var));
        available &= node.as<TensorLoadNode>() == nullptr;
      });
      ICHECK(available)
          << "Cannot project PerCoreTask Fixpipe routing: candidate dispatch "
             "must use immutable values available before PerCoreTask. Move "
             "the dispatch and route computation outside PerCoreTask";
      return active;
    }

    Stmt RewritePerCoreCandidate(const RegionStmt &task) {
      auto metadata = task_metadata_detail::CopyTaskMetadata(
          ScheduleMarkerPayload(task.get()));
      auto origin = metadata.Get("subcore_origin");
      int origin_id;
      if (origin) {
        origin_id = GetIntAnnotation(origin.value()).value();
      } else {
        auto [it, inserted] =
            owner_->candidate_origins_.emplace(task, owner_->next_origin_);
        if (inserted)
          ++owner_->next_origin_;
        origin_id = it->second;
        metadata.Set("subcore_origin", origin_id);
      }
      RegionStmt marked = WithSchedulePayload(task.get(), metadata);
      // Each candidate is its own atomic task. Neither the routing target nor
      // its guard may leak into a sibling candidate's rewrite.
      auto rewriter =
          ffi::make_object<Rewriter>(owner_, p_, conditional_depth_ == 0);
      rewriter->local_vars_ = local_vars_;
      Stmt body = rewriter->VisitStmt(marked);
      local_vars_.insert(rewriter->local_vars_.begin(),
                         rewriter->local_vars_.end());
      PrimExpr active = analyzer_->Simplify(rewriter->active_);
      bool has_access = HasTaskBufferAccess(task->body);
      if (has_access) {
        per_core_->active = per_core_->active || (per_core_->path && active);
        per_core_->routed |= !(GetConstInt(active) == 1);
      }
      has_local_copy_guard_ |= rewriter->HasLocalCopyGuard();
      per_core_->replays_fixpipe |= rewriter->CanReplayFixpipe();
      // An enclosing loop can revisit an already projected candidate. Its
      // explicit route now names p_, while the original guard is outside it.
      // Ordinary AIV candidates execute on distinct physical subcores; only
      // Fixpipe candidates can replay their effects on the same AIC.
      if (rewriter->CanReplayFixpipe() ||
          (!origin && rewriter->copy_target_.has_value() &&
           (GetConstInt(active) == 1)))
        owner_->replay_check_origins_.insert(origin_id);
      if ((GetConstInt(active) == 0))
        return Evaluate(IntImm::Int32(0));
      return (GetConstInt(active) == 1)
                 ? body
                 : Stmt(MakeScheduleMarker(kScheduleCondition, active, body));
    }

    Stmt VisitStmt_(const ForNode *op) final {
      // Each projected loop needs its own binder, even with uniform bounds.
      owner_->varying_.insert(op->var);
      PrimVar loop_var = owner_->VarFor(op->var, p_).as_or_throw<PrimVar>();
      local_vars_.insert(loop_var);
      For loop = NativeMutator::VisitStmt_(op).as_or_throw<For>();
      if (tirx::IsNoOp(loop->body))
        return Evaluate(IntImm::Int32(0));
      loop.CopyOnWrite()->var = loop_var;
      loop.CopyOnWrite()->annotations =
          RewriteMetadata(op->annotations).cast<Map<String, Any>>();
      return loop;
    }

    Stmt VisitStmt_(const RegionStmtNode *op) final {
      auto key = ScheduleMarkerKey(op);
      if (key.empty()) {
        for (const Var &variable : op->body_params) {
          owner_->varying_.insert(variable);
          local_vars_.insert(owner_->VarFor(variable, p_));
        }
        for (const Var &variable : op->result_vars)
          owner_->varying_.insert(variable);
        return NativeMutator::VisitStmt_(op);
      }
      if (key == kScheduleCondition) {
        auto condition = analyzer_->Simplify(
            VisitExpr(ScheduleMarkerPayload(op).as_or_throw<PrimExpr>()));
        if (GetConstInt(condition) == 0)
          return Evaluate(IntImm::Int32(0));
        if (GetConstInt(condition) == 1)
          return VisitStmt(UnwrapScheduleMarker(op));
        PrimExpr path = per_core_ ? per_core_->path : IntImm::Bool(true);
        if (per_core_)
          per_core_->path = path && condition;
        ++conditional_depth_;
        Stmt body = VisitStmt(UnwrapScheduleMarker(op));
        --conditional_depth_;
        if (per_core_)
          per_core_->path = path;
        if (tirx::IsNoOp(body))
          return body;
        return MakeScheduleMarker(key, condition, body, op->loc,
                                  !op->result_vars.empty());
      }
      if (per_core_ && key == attr::kAscendTask)
        return RewritePerCoreCandidate(ffi::GetRef<RegionStmt>(op));
      Stmt body{ffi::UnsafeInit{}};
      if (key == attr::kAscendPerCoreTask) {
        ICHECK(per_core_ == nullptr);
        PerCoreRouting routing;
        per_core_ = &routing;
        body = VisitStmt(UnwrapScheduleMarker(op));
        per_core_ = nullptr;
        ICHECK(owner_->count_ == 1 || !routing.replays_fixpipe ||
               !HasCrossCoreSync(body))
            << "Subcore projection of T.PerCoreTask with explicit "
               "synchronization and "
               "subcore-dependent or task-local Fixpipe routing is not "
               "supported";
        if (routing.routed) {
          auto projected =
              MakeScheduleMarker(key, ScheduleMarkerPayload(op), body, op->loc);
          active_ = active_ && GetPerCoreActiveGuard(projected, routing.active);
        }
      } else {
        body = VisitStmt(UnwrapScheduleMarker(op));
      }
      if (tirx::IsNoOp(body))
        return body;
      Any payload = RewriteMetadata(ScheduleMarkerPayload(op));
      if (key == attr::kAscendTask || key == attr::kAscendPerCoreTask) {
        auto metadata = task_metadata_detail::CopyTaskMetadata(payload);
        metadata.Set("subcore", p_);
        payload = metadata;
      }
      return MakeScheduleMarker(key, payload, body, op->loc,
                                !op->result_vars.empty());
    }

    Expr VisitExpr_(const VarNode *op) final {
      Var variable = ffi::GetRef<Var>(op);
      if (variable.same_as(owner_->sid_))
        return MakeConst(variable.as_or_throw<PrimVar>().ty(), p_);
      Var mapped = owner_->VarFor(variable, p_);
      if (auto constant = owner_->constants_.Get(mapped))
        return constant.value();
      return mapped;
    }

    Stmt VisitStmt_(const BindNode *op) final {
      Var variable = owner_->VarFor(op->var, p_);
      Expr value = VisitExpr(op->value);
      if (auto scalar = value.as<PrimExpr>())
        value = analyzer_->Simplify(scalar.value());
      local_vars_.insert(variable);
      if (conditional_depth_ == 0 && value.as<IntImmNode>())
        owner_->constants_.Set(variable, value.as_or_throw<PrimExpr>());
      return Bind(variable, value, op->loc);
    }

    Stmt VisitStmt_(const SeqStmtNode *op) final {
      Array<Stmt> statements;
      for (const Stmt &statement : op->seq) {
        Stmt rewritten = VisitStmt(statement);
        if (!tirx::IsNoOp(rewritten))
          statements.push_back(std::move(rewritten));
      }
      return SeqStmt(statements);
    }

    Stmt VisitStmt_(const IfNode *op) final {
      PrimExpr condition = analyzer_->Simplify(VisitExpr(op->condition));
      if ((GetConstInt(condition) == 1))
        return VisitStmt(op->then_case);
      if ((GetConstInt(condition) == 0))
        return op->else_case.has_value() ? VisitStmt(op->else_case.value())
                                         : Evaluate(IntImm::Int32(0));
      PrimExpr path = per_core_ ? per_core_->path : IntImm::Bool(true);
      if (per_core_)
        per_core_->path = path && condition;
      ++conditional_depth_;
      Stmt then_case = VisitStmt(op->then_case);
      if (per_core_)
        per_core_->path = path && !condition;
      Optional<Stmt> else_case =
          op->else_case.has_value()
              ? Optional<Stmt>(VisitStmt(op->else_case.value()))
              : std::nullopt;
      --conditional_depth_;
      if (per_core_)
        per_core_->path = path;
      if (tirx::IsNoOp(then_case) &&
          (!else_case.has_value() || tirx::IsNoOp(else_case.value())))
        return Evaluate(IntImm::Int32(0));
      if (tirx::IsNoOp(then_case))
        return If(!condition, SeqStmt(else_case.value()), std::nullopt,
                  op->loc);
      if (else_case.has_value() && tirx::IsNoOp(else_case.value()))
        else_case = std::nullopt;
      return If(condition, SeqStmt(then_case),
                else_case ? Optional<SeqStmt>(SeqStmt(else_case.value()))
                          : std::nullopt,
                op->loc);
    }

    Stmt VisitStmt_(const EvaluateNode *op) final {
      Expr source_value = op->value;
      Evaluate statement =
          NativeMutator::VisitStmt_(op).as_or_throw<Evaluate>();
      const auto *call_node = statement->value.as<CallNode>();
      if (!call_node || !IsAscendCopyCall(call_node))
        return statement;
      Call call = ffi::GetRef<Call>(call_node);
      AscendCopy copy(call->args, AscendCallAnnotations(call.get()));
      if (!IsL0CBuffer(copy->src) || !IsSharedBuffer(copy->dst))
        return statement;
      // AIC executes this task, while its UB destination is owned by p.
      ICHECK_EQ(copy->dual_dst_ctl, 0);
      Map<String, Any> annotations = AscendCallAnnotations(call.get());
      PrimExpr target = IntImm::Int32(p_);
      if (auto explicit_sid = annotations.Get("sub_blockid"))
        target = copy->sub_blockid;
      // Keep a single routing target for each projected atomic task.
      if (copy_target_.has_value()) {
        ICHECK(analyzer_->CanProveEqual(target, copy_target_.value()))
            << "Projected Fixpipe copies in one T.Task must use the same "
               "sub_blockid; put copies with different destinations in "
               "separate T.Task scopes";
      }
      copy_target_ = target;
      annotations.Set("sub_blockid", IntImm::Int32(p_));
      call.CopyOnWrite()->attrs = DictAttrs(annotations);
      statement.CopyOnWrite()->value = call;
      Call source = source_value.as_or_throw<Call>();
      if (auto origin = owner_->copy_origins_.find(source);
          origin != owner_->copy_origins_.end())
        source = origin->second;
      // Classify the original route: projection can turn sid and its aliases
      // into constants even though multiple copies execute on the same AIC.
      auto source_route =
          AscendCallAnnotations(source.get()).Get("sub_blockid");
      auto source_route_expr =
          source_route ? source_route.value().as<PrimExpr>() : std::nullopt;
      has_varying_copy_route_ |=
          !source_route ||
          (source_route_expr && owner_->Varies(source_route_expr.value()));
      owner_->copy_origins_.emplace(call, std::move(source));

      PrimExpr condition = analyzer_->Simplify(target == p_);
      bool task_local = SideEffect(condition) > CallEffectKind::kReadState;
      PostOrderVisit(condition, [&](const ObjectRef &node) {
        if (const auto *var = node.as<VarNode>())
          task_local |= local_vars_.count(ffi::GetRef<Var>(var)) != 0;
        task_local |= node.as<TensorLoadNode>() != nullptr;
      });
      // Task-local definitions must dominate the route test. Mutable loads
      // also stay at the copy site so earlier task writes remain visible.
      if (task_local) {
        has_local_copy_guard_ = true;
        return If(condition, SeqStmt(statement), std::nullopt, op->loc);
      }
      active_ = active_ && condition;
      return statement;
    }

  public:
    PrimExpr active_{IntImm::Bool(true)};

  private:
    SubcoreProjector *owner_;
    int p_;
    int conditional_depth_;
    PerCoreRouting *per_core_{nullptr};
    VarSet local_vars_;
    Optional<PrimExpr> copy_target_;
    bool has_varying_copy_route_{false};
    bool has_local_copy_guard_{false};
    sym::Analyzer analyzer_;
  };

  bool Varies(const Any &value) const {
    if (auto buffer = value.try_cast<TensorVar>())
      return Varies(buffer.value());
    if (auto var = value.try_cast<Var>())
      return varying_.count(var.value());
    return false;
  }

  Any ExpandAnnotation(const String &key, const Any &value) {
    if (key == "multi_buffer_eligible") {
      Array<Any> result;
      for (const Any &entry : value.cast<Array<Any>>()) {
        if (!Varies(entry)) {
          result.push_back(entry);
          continue;
        }
        for (int p = 0; p < count_; ++p)
          result.push_back(
              ffi::make_object<Rewriter>(this, p)->RewriteMetadata(entry));
      }
      return result;
    }
    if (key == "conflict_hint" || key == kRootConflictHints) {
      Array<Any> result;
      for (const Any &entry : value.cast<Array<Any>>())
        for (int p = 0; p < count_; ++p)
          result.push_back(
              ffi::make_object<Rewriter>(this, p)->RewriteMetadata(entry));
      return result;
    }
    if (auto map = value.try_cast<Map<Any, Any>>()) {
      Map<Any, Any> result;
      for (const auto &[storage, entry] : map.value()) {
        if (!Varies(storage)) {
          result.Set(storage, entry);
          continue;
        }
        for (int p = 0; p < count_; ++p) {
          auto rewriter = ffi::make_object<Rewriter>(this, p);
          result.Set(rewriter->RewriteMetadata(storage),
                     rewriter->RewriteMetadata(entry));
        }
      }
      return result;
    }
    return value;
  }

  Stmt RewriteList(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                   int forced_sid, bool conditional_parent = false) {
    Array<Stmt> result;
    for (const auto &node : nodes) {
      const TaskNode *task =
          node->IsTask()
              ? static_cast<const TaskNode *>(node.get())
              : static_cast<const ControlNode *>(node.get())->task.get();
      bool project = forced_sid >= 0 || Varies(task);
      ICHECK(!project || !dual_tasks_.count(task))
          << "Hardware L0C dual_copy requires a subcore-uniform control path; "
             "use T.copy under if cond(sid) for independently selected AIV "
             "destinations";
      int begin = forced_sid >= 0 ? forced_sid : 0;
      int end = forced_sid >= 0 ? forced_sid + 1 : (project ? count_ : 1);
      auto [origin_it, inserted] = origins_.emplace(node.get(), next_origin_);
      if (inserted)
        ++next_origin_;
      int origin = origin_it->second;
      for (int p = begin; p < end; ++p) {
        Stmt body{ffi::UnsafeInit{}};
        if (node->IsTask()) {
          RegionStmt marker = task->stmt.as_or_throw<RegionStmt>();
          auto metadata = task_metadata_detail::CopyTaskMetadata(
              ScheduleMarkerPayload(marker.get()));
          if (project) {
            metadata.Set("subcore", p);
            metadata.Set("subcore_origin", origin);
          }
          body = WithSchedulePayload(marker.get(), metadata);
        } else {
          const auto *control = static_cast<const ControlNode *>(node.get());
          For loop = control->control;
          if (project)
            varying_.insert(loop->var);
          loop.CopyOnWrite()->body =
              SeqStmt(RewriteList(control->children, project ? p : -1,
                                  conditional_parent || node->HasConditions()));
          if (!project) {
            Map<String, Any> annotations;
            for (const auto &[key, value] : loop->annotations)
              annotations.Set(key, ExpandAnnotation(key, value));
            loop.CopyOnWrite()->annotations = std::move(annotations);
          }
          body = loop;
        }
        body = WrapScheduledGuards(body, node->GetGuards());
        if (project) {
          auto rewriter =
              ffi::make_object<Rewriter>(this, p, !conditional_parent);
          body = rewriter->VisitStmt(body);
          sym::Analyzer analyzer;
          PrimExpr active = analyzer->Simplify(rewriter->active_);
          if ((GetConstInt(active) == 0))
            continue;
          if (node->IsTask() && rewriter->CanReplayFixpipe())
            replay_check_origins_.insert(origin);
          if (!(GetConstInt(active) == 1))
            body = MakeScheduleMarker(kScheduleCondition, active, body);
        }
        if (tirx::IsNoOp(body))
          continue;
        result.push_back(MakeScheduleMarker(
            attr::kScheduleUnit,
            ScheduleUnitMetadata{{kScheduleUnitStage, node->GetStage()}},
            body));
      }
    }
    return SeqStmt(result);
  }

  Var sid_;
  int count_;
  int next_origin_{0};
  std::unordered_map<const IRStructure *, int> origins_;
  std::unordered_map<RegionStmt, int, ObjectPtrHash, ObjectPtrEqual>
      candidate_origins_;
  std::unordered_set<int> replay_check_origins_;
  std::unordered_map<Call, Call, ObjectPtrHash, ObjectPtrEqual> copy_origins_;
  VarSet varying_;
  VarSet dual_storages_;
  std::unordered_set<const TaskNode *> dual_tasks_;
  std::array<VarMap, 2> vars_;
  std::array<BufferMap, 2> buffers_;
  Map<Var, PrimExpr> constants_;
  AivStorageMap ub_sids_;
  VarSet local_definitions_;
  std::vector<std::pair<TensorVar, TensorVar>> external_aliases_;
};

} // namespace ascend
} // namespace tl
} // namespace tvm
