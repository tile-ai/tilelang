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
  static SBlock Rewrite(const SBlock &root, const ConstrSet &outer_ctx,
                        const Var &sid) {
    ICHECK(!root->annotations.count(kSplitSubcores));
    ScheduledTIR scheduled = DecodeScheduledTIR(root, outer_ctx);
    SubcoreProjector projector(sid,
                               scheduled.metadata.num_aiv_subcores.value_or(2));
    projector.FindVaryingValues(scheduled.tree);
    if (!projector.NeedsProjection(scheduled.tree))
      return root;
    Stmt body = projector.RewriteList(scheduled.tree, -1);
    projector.VerifyProjectedFixpipeTasks(body);
    VarSet used;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>())
        used.insert(ffi::GetRef<Var>(var));
      if (const auto *load = node.as<BufferLoadNode>())
        used.insert(load->buffer->data);
      if (const auto *store = node.as<BufferStoreNode>())
        used.insert(store->buffer->data);
    });
    auto active = [&](const Var &storage, int p) {
      auto it = projector.vars_[p].find(storage);
      return it != projector.vars_[p].end() && used.count(it->second);
    };
    SBlock result = root;
    SBlockNode *writer = result.CopyOnWrite();
    writer->body = std::move(body);
    Array<Buffer> allocations;
    for (const Buffer &buffer : root->alloc_buffers) {
      if (projector.varying_.count(buffer->data)) {
        for (int p = 0; p < projector.count_; ++p)
          if (active(buffer->data, p))
            allocations.push_back(projector.BufferFor(buffer, p));
      } else {
        allocations.push_back(buffer);
      }
    }
    writer->alloc_buffers = std::move(allocations);
    Map<String, Any> annotations;
    for (const auto &[key, value] : root->annotations) {
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
      } else if (auto layouts = value.try_cast<Map<Buffer, Layout>>()) {
        Map<Buffer, Layout> mapped;
        for (const auto &[buffer, layout] : layouts.value()) {
          if (projector.Varies(buffer)) {
            for (int p = 0; p < projector.count_; ++p)
              if (active(buffer->data, p) ||
                  (!projector.varying_.count(buffer->data) &&
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
    annotations.Set(kSplitSubcores, Bool(true));
    writer->annotations = std::move(annotations);
    return result;
  }

private:
  using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
  using VarMap = std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual>;
  using BufferMap =
      std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>;

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
      for (const BufferRegion &region : regions)
        if (Varies(region->buffer) || (IsSharedBuffer(region->buffer) &&
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
      if (const auto *load = node.as<BufferLoadNode>())
        varying |= varying_.count(load->buffer->data);
    });
    return varying;
  }

  bool Varies(const Buffer &buffer) const {
    if (varying_.count(buffer->data) || Varies(buffer->elem_offset))
      return true;
    for (const auto &expressions : {buffer->shape, buffer->strides})
      for (const PrimExpr &expr : expressions)
        if (Varies(expr))
          return true;
    return false;
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
      PostOrderVisit(WrapScheduledGuards(task->stmt, node->GetGuards()),
                     [&](const ObjectRef &stmt) {
                       if (const auto *branch = stmt.as<IfThenElseNode>())
                         varying_control |= Varies(branch->condition);
                       if (const auto *loop = stmt.as<ForNode>())
                         varying_control |= Varies(loop->min) ||
                                            Varies(loop->extent) ||
                                            (loop->step.has_value() &&
                                             Varies(loop->step.value()));
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
          for (const BufferRegion &region : regions)
            if (IsSSBuffer(region->buffer) || IsL1Buffer(region->buffer) ||
                IsL0ABuffer(region->buffer) || IsL0BBuffer(region->buffer) ||
                IsL0CBuffer(region->buffer))
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
        AscendCopy copy(call->args, call->annotations);
        if (IsL0CBuffer(copy->src) && IsSharedBuffer(copy->dst) &&
            copy->dual_dst_ctl != 0) {
          // Hardware dual destinations share an offset and a generation.
          // Keep this UB allocation common to both address spaces and keep
          // the Fixpipe instruction intact (including partial fractals).
          dual_storages_.insert(copy->dst->data);
          dual_tasks_.insert(task);
        }
      });
    }
    for (TaskNode *task : tasks) {
      for (const auto &regions :
           {task->GetReadRegions(), task->GetWriteRegions()})
        for (const BufferRegion &region : regions)
          if (IsSharedBuffer(region->buffer) &&
              !dual_storages_.count(region->buffer->data))
            varying_.insert(region->buffer->data);
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
        for (const BufferRegion &region : task->GetWriteRegions())
          if (IsRegisterRegion(region))
            changed |= varying_.insert(region->buffer->data).second;
      }
    }
  }

  Var VarFor(const Var &var, int p) {
    if (!varying_.count(var))
      return var;
    auto it = vars_[p].find(var);
    if (it != vars_[p].end())
      return it->second;
    Var result = var.copy_with_suffix("_aiv" + std::to_string(p));
    vars_[p].emplace(var, result);
    // Enclosing controls revisit projected children. Keep the same identity
    // when their loop binders are encountered again on this subcore.
    vars_[p].emplace(result, result);
    return result;
  }

  Buffer BufferFor(const Buffer &buffer, int p) {
    auto it = buffers_[p].find(buffer);
    if (it != buffers_[p].end())
      return it->second;
    if (!Varies(buffer))
      return buffer;
    // Views can depend on sid even when their underlying storage is common.
    // Use the same projection for definitions, accesses, and metadata keys.
    Rewriter rewriter(this, p);
    arith::Analyzer analyzer;
    auto rewrite = [&](const PrimExpr &expr) {
      return analyzer.Simplify(rewriter(expr));
    };
    Buffer result = buffer;
    auto *writer = result.CopyOnWrite();
    writer->data = VarFor(buffer->data, p);
    writer->name = buffer->name + "_aiv" + std::to_string(p);
    writer->shape = buffer->shape.Map(rewrite);
    writer->strides = buffer->strides.Map(rewrite);
    writer->elem_offset = rewrite(buffer->elem_offset);
    buffers_[p].emplace(buffer, result);
    // A projected view may still depend on a projected loop binder.
    buffers_[p].emplace(result, result);
    if (IsSharedBuffer(buffer) && varying_.count(buffer->data))
      ub_sids_.Set(result->data, Integer(p));
    return result;
  }

  static void VerifyReplayableFixpipeTask(const Stmt &body) {
    bool external_effect = false;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *store = node.as<BufferStoreNode>())
        external_effect |=
            !IsRegisterRegion(BufferRegion::FullRegion(store->buffer));
      if (const auto *call = node.as<CallNode>()) {
        if (IsAscendCopyCall(call)) {
          AscendCopy copy(call->args, call->annotations);
          external_effect |=
              !IsL0CBuffer(copy->src) || !IsSharedBuffer(copy->dst);
        } else {
          external_effect |= SideEffect(ffi::GetRef<PrimExpr>(call)) >
                             CallEffectKind::kReadState;
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
    std::unordered_map<int, std::vector<AttrStmt>> projections;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      const auto *attr = node.as<AttrStmtNode>();
      if (!attr || (attr->attr_key != tl::attr::kAscendTask &&
                    attr->attr_key != tl::attr::kAscendPerCoreTask))
        return;
      auto metadata = task_metadata_detail::CopyTaskMetadata(attr->node);
      auto origin = metadata.Get("subcore_origin");
      if (origin &&
          replay_check_origins_.count(origin.value().cast<Integer>()->value))
        projections[origin.value().cast<Integer>()->value].push_back(
            ffi::GetRef<AttrStmt>(attr));
    });
    // An enclosing projected loop may rewrite a task once per recursive call.
    // Count surviving copies across the complete tree, after outer guards have
    // removed inactive branches, rather than within one RewriteList call.
    for (const auto &[origin, tasks] : projections) {
      if (tasks.size() > 1)
        for (const AttrStmt &task : tasks)
          VerifyReplayableFixpipeTask(task);
    }
  }

  class Rewriter : public StmtExprMutator {
  public:
    Rewriter(SubcoreProjector *owner, int p, bool allow_constants = true)
        : owner_(owner), p_(p), conditional_depth_(allow_constants ? 0 : 1) {}

    bool HasLocalCopyGuard() const { return has_local_copy_guard_; }

    bool CanReplayFixpipe() const {
      return has_varying_copy_route_ || has_local_copy_guard_;
    }

    Any RewriteMetadata(const Any &value) {
      if (auto buffer = value.try_cast<Buffer>())
        return owner_->BufferFor(buffer.value(), p_);
      if (const auto *expr = value.as<PrimExprNode>())
        return VisitExpr(ffi::GetRef<PrimExpr>(expr));
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
      PrimExpr path{Bool(true)};
      PrimExpr active{Bool(false)};
      bool routed{false};
      bool replays_fixpipe{false};
    };

    PrimExpr GetPerCoreActiveGuard(const AttrStmt &task, PrimExpr active) {
      Map<Var, PrimExpr> bindings;
      VarSet locals;
      PostOrderVisit(task->body, [&](const ObjectRef &node) {
        if (const auto *loop = node.as<ForNode>())
          locals.insert(loop->loop_var);
        if (const auto *bind = node.as<BindNode>()) {
          locals.insert(bind->var);
          PrimExpr value = Substitute(bind->value, bindings);
          bool immutable = SideEffect(value) <= CallEffectKind::kReadState;
          PostOrderVisit(value, [&](const ObjectRef &input) {
            immutable &= input.as<BufferLoadNode>() == nullptr;
          });
          if (immutable)
            bindings.Set(bind->var, value);
        }
      });
      // Condition snapshots belong inside PerCoreTask. Expand only immutable
      // bindings when exposing the selected candidates to outer
      // synchronization.
      active = analyzer_.Simplify(Substitute(active, bindings));
      bool available = SideEffect(active) <= CallEffectKind::kReadState;
      PostOrderVisit(active, [&](const ObjectRef &node) {
        if (const auto *var = node.as<VarNode>())
          available &= !locals.count(ffi::GetRef<Var>(var));
        available &= node.as<BufferLoadNode>() == nullptr;
      });
      ICHECK(available)
          << "Cannot project PerCoreTask Fixpipe routing: candidate dispatch "
             "must use immutable values available before PerCoreTask. Move "
             "the dispatch and route computation outside PerCoreTask";
      return active;
    }

    Stmt RewritePerCoreCandidate(const AttrStmt &task) {
      auto metadata = task_metadata_detail::CopyTaskMetadata(task->node);
      auto origin = metadata.Get("subcore_origin");
      int origin_id;
      if (origin) {
        origin_id = origin.value().cast<Integer>()->value;
      } else {
        auto [it, inserted] =
            owner_->candidate_origins_.emplace(task, owner_->next_origin_);
        if (inserted)
          ++owner_->next_origin_;
        origin_id = it->second;
        metadata.Set("subcore_origin", Integer(origin_id));
      }
      AttrStmt marked = task;
      marked.CopyOnWrite()->node = std::move(metadata);
      // Each candidate is its own atomic task. Neither the routing target nor
      // its guard may leak into a sibling candidate's rewrite.
      Rewriter rewriter(owner_, p_, conditional_depth_ == 0);
      rewriter.local_vars_ = local_vars_;
      Stmt body = rewriter(marked);
      local_vars_.insert(rewriter.local_vars_.begin(),
                         rewriter.local_vars_.end());
      PrimExpr active = analyzer_.Simplify(rewriter.active_);
      bool has_access = HasTaskBufferAccess(task->body);
      if (has_access) {
        per_core_->active = per_core_->active || (per_core_->path && active);
        per_core_->routed |= !is_one(active);
      }
      has_local_copy_guard_ |= rewriter.HasLocalCopyGuard();
      per_core_->replays_fixpipe |= rewriter.CanReplayFixpipe();
      // An enclosing loop can revisit an already projected candidate. Its
      // explicit route now names p_, while the original guard is outside it.
      // Ordinary AIV candidates execute on distinct physical subcores; only
      // Fixpipe candidates can replay their effects on the same AIC.
      if (rewriter.CanReplayFixpipe() ||
          (!origin && rewriter.copy_target_.defined() && is_one(active)))
        owner_->replay_check_origins_.insert(origin_id);
      if (is_zero(active))
        return Evaluate(0);
      return is_one(active) ? body : IfThenElse(active, body);
    }

    Buffer VisitBufferUse(const Buffer &buffer) final {
      return owner_->BufferFor(buffer, p_);
    }

    Buffer VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
      return owner_->BufferFor(buffer, p_);
    }

    Stmt VisitStmt_(const SBlockNode *op) final {
      SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
      block.CopyOnWrite()->annotations =
          RewriteMetadata(op->annotations).cast<Map<String, Any>>();
      return block;
    }

    Stmt VisitStmt_(const ForNode *op) final {
      // Each projected loop needs its own binder, even with uniform bounds.
      owner_->varying_.insert(op->loop_var);
      Var loop_var = owner_->VarFor(op->loop_var, p_);
      local_vars_.insert(loop_var);
      For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
      if (is_no_op(loop->body))
        return Evaluate(0);
      loop.CopyOnWrite()->loop_var = loop_var;
      loop.CopyOnWrite()->annotations =
          RewriteMetadata(op->annotations).cast<Map<String, Any>>();
      return loop;
    }

    Stmt VisitStmt_(const AttrStmtNode *op) final {
      if (per_core_ && op->attr_key == tl::attr::kAscendTask)
        return RewritePerCoreCandidate(ffi::GetRef<AttrStmt>(op));
      AttrStmt attr;
      if (op->attr_key == tl::attr::kAscendPerCoreTask) {
        ICHECK(per_core_ == nullptr);
        PerCoreRouting routing;
        per_core_ = &routing;
        attr = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
        per_core_ = nullptr;
        // Subcore-dependent and task-local routes can execute the group more
        // than once on AIC. Its explicit synchronization is a single protocol
        // whose participation and execution counts must not follow that replay.
        ICHECK(owner_->count_ == 1 || !routing.replays_fixpipe ||
               !HasCrossCoreSync(attr->body))
            << "Subcore projection of T.PerCoreTask with explicit "
               "synchronization and subcore-dependent or task-local Fixpipe "
               "routing is not supported: projection may change the "
               "synchronization participation or execution counts. Move it "
               "outside T.PerCoreTask, or compute each explicit sub_blockid "
               "before T.PerCoreTask using subcore-uniform values";
        if (routing.routed)
          active_ = active_ && GetPerCoreActiveGuard(attr, routing.active);
      } else {
        attr = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
      }
      // Projection can remove a condition inside an assumption/attribute
      // guard. Do not leave a schedule unit whose guarded payload is empty.
      if (is_no_op(attr->body))
        return Evaluate(0);
      attr.CopyOnWrite()->node = RewriteMetadata(op->node);
      if (op->attr_key == tl::attr::kAscendTask ||
          op->attr_key == tl::attr::kAscendPerCoreTask) {
        auto metadata = task_metadata_detail::CopyTaskMetadata(attr->node);
        metadata.Set("subcore", Integer(p_));
        attr.CopyOnWrite()->node = std::move(metadata);
      }
      return attr;
    }
    PrimExpr VisitExpr_(const VarNode *op) final {
      Var var = ffi::GetRef<Var>(op);
      if (var.same_as(owner_->sid_))
        return make_const(var.dtype(), p_);
      Var mapped = owner_->VarFor(var, p_);
      if (auto constant = owner_->constants_.Get(mapped))
        return constant.value();
      return mapped;
    }

    PrimExpr VisitExpr_(const BufferLoadNode *op) final {
      BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
      load.CopyOnWrite()->buffer = owner_->BufferFor(op->buffer, p_);
      return load;
    }

    Stmt VisitStmt_(const BufferStoreNode *op) final {
      BufferStore store =
          Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
      store.CopyOnWrite()->buffer = owner_->BufferFor(op->buffer, p_);
      return store;
    }

    Stmt VisitStmt_(const BindNode *op) final {
      Var var = owner_->VarFor(op->var, p_);
      PrimExpr value = analyzer_.Simplify(VisitExpr(op->value));
      local_vars_.insert(var);
      if (conditional_depth_ == 0 && value.as<IntImmNode>())
        owner_->constants_.Set(var, value);
      return Bind(var, value, op->span);
    }

    Stmt VisitStmt_(const SeqStmtNode *op) final {
      Array<Stmt> statements;
      for (const Stmt &statement : op->seq) {
        Stmt rewritten = VisitStmt(statement);
        if (!is_no_op(rewritten))
          statements.push_back(std::move(rewritten));
      }
      return SeqStmt::Flatten(statements);
    }

    Stmt VisitStmt_(const IfThenElseNode *op) final {
      PrimExpr condition = analyzer_.Simplify(VisitExpr(op->condition));
      if (is_one(condition))
        return VisitStmt(op->then_case);
      if (is_zero(condition))
        return op->else_case.has_value() ? VisitStmt(op->else_case.value())
                                         : Evaluate(0);
      PrimExpr path = per_core_ ? per_core_->path : Bool(true);
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
      if (is_no_op(then_case) &&
          (!else_case.has_value() || is_no_op(else_case.value())))
        return Evaluate(0);
      if (is_no_op(then_case))
        return IfThenElse(!condition, else_case.value(), std::nullopt,
                          op->span);
      if (else_case.has_value() && is_no_op(else_case.value()))
        else_case = std::nullopt;
      return IfThenElse(condition, then_case, else_case, op->span);
    }

    Stmt VisitStmt_(const AllocBufferNode *op) final {
      return AllocBuffer(
          owner_->BufferFor(op->buffer, p_),
          RewriteMetadata(op->annotations).cast<Map<String, Any>>(), op->span);
    }

    PrimExpr VisitExpr_(const CallNode *op) final {
      Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
      Map<String, ObjectRef> annotations;
      for (const auto &[key, value] : call->annotations)
        annotations.Set(key,
                        value.as<PrimExprNode>()
                            ? ObjectRef(VisitExpr(Downcast<PrimExpr>(value)))
                            : value);
      call.CopyOnWrite()->annotations = annotations;
      return call;
    }

    Stmt VisitStmt_(const EvaluateNode *op) final {
      PrimExpr source_value = op->value;
      Evaluate statement = Downcast<Evaluate>(StmtExprMutator::VisitStmt_(op));
      const auto *call_node = statement->value.as<CallNode>();
      if (!call_node || !IsAscendCopyCall(call_node))
        return statement;
      Call call = ffi::GetRef<Call>(call_node);
      AscendCopy copy(call->args, call->annotations);
      if (!IsL0CBuffer(copy->src) || !IsSharedBuffer(copy->dst))
        return statement;
      // AIC executes this task, while its UB destination is owned by p.
      ICHECK_EQ(copy->dual_dst_ctl, 0);
      Map<String, ObjectRef> annotations = call->annotations;
      PrimExpr target = Integer(p_);
      if (auto explicit_sid = annotations.Get("sub_blockid"))
        target = Downcast<PrimExpr>(explicit_sid.value());
      // Keep a single routing target for each projected atomic task.
      if (copy_target_.defined()) {
        ICHECK(analyzer_.CanProveEqual(target, copy_target_.value()))
            << "Projected Fixpipe copies in one T.Task must use the same "
               "sub_blockid; put copies with different destinations in "
               "separate T.Task scopes";
      }
      copy_target_ = target;
      annotations.Set("sub_blockid", Integer(p_));
      call.CopyOnWrite()->annotations = annotations;
      statement.CopyOnWrite()->value = call;
      Call source = Downcast<Call>(source_value);
      if (auto origin = owner_->copy_origins_.find(source);
          origin != owner_->copy_origins_.end())
        source = origin->second;
      // Classify the original route: projection can turn sid and its aliases
      // into constants even though multiple copies execute on the same AIC.
      auto source_route = source->annotations.Get("sub_blockid");
      has_varying_copy_route_ |=
          !source_route ||
          owner_->Varies(Downcast<PrimExpr>(source_route.value()));
      owner_->copy_origins_.emplace(call, std::move(source));

      PrimExpr condition = analyzer_.Simplify(target == p_);
      bool task_local = SideEffect(condition) > CallEffectKind::kReadState;
      PostOrderVisit(condition, [&](const ObjectRef &node) {
        if (const auto *var = node.as<VarNode>())
          task_local |= local_vars_.count(ffi::GetRef<Var>(var)) != 0;
        task_local |= node.as<BufferLoadNode>() != nullptr;
      });
      // Task-local definitions must dominate the route test. Mutable loads
      // also stay at the copy site so earlier task writes remain visible.
      if (task_local) {
        has_local_copy_guard_ = true;
        return IfThenElse(condition, statement, std::nullopt, op->span);
      }
      active_ = active_ && condition;
      return statement;
    }

  public:
    PrimExpr active_{Bool(true)};

  private:
    SubcoreProjector *owner_;
    int p_;
    int conditional_depth_;
    PerCoreRouting *per_core_{nullptr};
    VarSet local_vars_;
    Optional<PrimExpr> copy_target_;
    bool has_varying_copy_route_{false};
    bool has_local_copy_guard_{false};
    arith::Analyzer analyzer_;
  };

  bool Varies(const Any &value) const {
    if (auto var = value.try_cast<Var>())
      return varying_.count(var.value());
    if (auto buffer = value.try_cast<Buffer>())
      return Varies(buffer.value());
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
          result.push_back(Rewriter(this, p).RewriteMetadata(entry));
      }
      return result;
    }
    if (key == "conflict_hint" || key == kRootConflictHints) {
      Array<Any> result;
      for (const Any &entry : value.cast<Array<Any>>())
        for (int p = 0; p < count_; ++p)
          result.push_back(Rewriter(this, p).RewriteMetadata(entry));
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
          Rewriter rewriter(this, p);
          result.Set(rewriter.RewriteMetadata(storage),
                     rewriter.RewriteMetadata(entry));
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
        Stmt body;
        if (node->IsTask()) {
          AttrStmt marker = Downcast<AttrStmt>(task->stmt);
          auto metadata = task_metadata_detail::CopyTaskMetadata(marker->node);
          if (project) {
            metadata.Set("subcore", Integer(p));
            metadata.Set("subcore_origin", Integer(origin));
          }
          body = AttrStmt(metadata, marker->attr_key, marker->value,
                          marker->body, marker->span);
        } else {
          const auto *control = static_cast<const ControlNode *>(node.get());
          For loop = control->control;
          if (project)
            varying_.insert(loop->loop_var);
          loop.CopyOnWrite()->body =
              RewriteList(control->children, project ? p : -1,
                          conditional_parent || node->HasConditions());
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
          Rewriter rewriter(this, p, !conditional_parent);
          body = rewriter(body);
          arith::Analyzer analyzer;
          PrimExpr active = analyzer.Simplify(rewriter.active_);
          if (is_zero(active))
            continue;
          if (node->IsTask() && rewriter.CanReplayFixpipe())
            replay_check_origins_.insert(origin);
          if (!is_one(active))
            body = IfThenElse(active, body);
        }
        if (const auto *eval = body.as<EvaluateNode>();
            eval && is_zero(eval->value))
          continue;
        result.push_back(
            AttrStmt(ScheduleUnitMetadata{{kScheduleUnitStage,
                                           Integer(node->GetStage())}},
                     attr::kScheduleUnit, Integer(1), body));
      }
    }
    return SeqStmt::Flatten(result);
  }

  Var sid_;
  int count_;
  int next_origin_{0};
  std::unordered_map<const IRStructure *, int> origins_;
  std::unordered_map<AttrStmt, int, ObjectPtrHash, ObjectPtrEqual>
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
};

} // namespace ascend
} // namespace tl
} // namespace tvm
