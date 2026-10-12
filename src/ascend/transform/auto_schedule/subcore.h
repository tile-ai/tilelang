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

#include "layout/layout.h"
#include "scheduled_tir.h"
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
    SBlock result = root;
    Stmt body = projector.RewriteList(scheduled.tree, -1);
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
          if (projector.varying_.count(buffer->data)) {
            for (int p = 0; p < projector.count_; ++p)
              if (active(buffer->data, p))
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
    return RepairUnitFlags(std::move(result), outer_ctx);
  }

private:
  using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
  using VarMap = std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual>;
  using BufferMap =
      std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>;

  // A dual Fixpipe instruction consumes each L0C unit flag once. After
  // projection, two reads of the same fractal must not both clear that flag.
  // Keep unit flags for exclusive branches and disjoint whole fractals. For
  // overlapping reads use the ordinary M/FIX dependency protocol instead.
  static SBlock RepairUnitFlags(SBlock root, const ConstrSet &outer_ctx) {
    ScheduledTIR scheduled = DecodeScheduledTIR(root, outer_ctx);
    std::vector<TaskNode *> tasks;
    CollectAllTaskNodes(scheduled.tree, tasks);
    struct Read {
      TaskNode *task;
      AscendCopy copy;
    };
    std::unordered_map<int, std::vector<Read>> reads;
    for (TaskNode *task : tasks) {
      auto metadata = task_metadata_detail::CopyTaskMetadata(
          Downcast<AttrStmt>(task->stmt)->node);
      auto origin = metadata.Get("subcore_origin");
      if (!origin)
        continue;
      PostOrderVisit(task->stmt, [&](const ObjectRef &node) {
        const auto *call = node.as<CallNode>();
        if (!call || !IsAscendCopyCall(call))
          return;
        AscendCopy copy(call->args, call->annotations);
        if (IsL0CBuffer(copy->src) && IsSharedBuffer(copy->dst) &&
            !is_zero(copy->unit_flag_ctl))
          reads[origin.value().cast<Integer>()->value].push_back({task, copy});
      });
    }
    VarSet disable;
    auto single_iteration_guard = [](TaskNode *task) {
      PrimExpr guard = task->GetConditionGuard();
      IRStructure *parent = task->GetParent();
      // Buffer ownership and conflict hints may retain a one-iteration loop
      // around the copy. Its guard still selects the same single flag read.
      while (parent && parent->IsControl()) {
        auto *control = static_cast<ControlNode *>(parent);
        if (control->GetTripCount() != 1)
          break;
        guard = guard && control->GetConditionGuard();
        parent = parent->GetParent();
      }
      return std::make_pair(parent, guard);
    };
    for (const auto &[origin, copies] : reads) {
      for (size_t i = 0; i < copies.size(); ++i) {
        for (size_t j = i + 1; j < copies.size(); ++j) {
          const Read &a = copies[i], &b = copies[j];
          auto [a_parent, a_guard] = single_iteration_guard(a.task);
          auto [b_parent, b_guard] = single_iteration_guard(b.task);
          ConstrSet context = b.task->outer_ctx;
          if (a_parent == b_parent && b_parent && b_parent->IsControl())
            context = static_cast<ControlNode *>(b_parent)
                          ->GetLoopBodyContext()
                          .Merge(context);
          arith::Analyzer analyzer;
          analyzer.z3_prover.SetRLimit(50000);
          context.Populate(analyzer);
          PrimExpr exclusive = !(a_guard && b_guard);
          if (a_parent == b_parent && (analyzer.CanProve(exclusive) ||
                                       analyzer.z3_prover.CanProve(exclusive)))
            continue;
          bool disjoint = false;
          if (a.copy->src_range.size() == b.copy->src_range.size()) {
            for (size_t dim = 0; dim < a.copy->src_range.size(); ++dim) {
              const Range &x = a.copy->src_range[dim];
              const Range &y = b.copy->src_range[dim];
              int fractal = dim + 2 >= a.copy->src_range.size() ? 16 : 1;
              disjoint |= analyzer.CanProve(
                  floordiv(x->min + x->extent + fractal - 1, fractal) <=
                      floordiv(y->min, fractal) ||
                  floordiv(y->min + y->extent + fractal - 1, fractal) <=
                      floordiv(x->min, fractal));
            }
          }
          if (!disjoint)
            disable.insert(a.copy->src->data);
        }
      }
    }
    if (disable.empty())
      return root;
    class Rewriter : public StmtExprMutator {
    public:
      explicit Rewriter(const VarSet &disable) : disable_(disable) {}

    private:
      PrimExpr VisitExpr_(const CallNode *op) final {
        Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
        if (!call->annotations.count("unit_flag_ctrl"))
          return call;
        bool affected = false;
        PostOrderVisit(call, [&](const ObjectRef &node) {
          if (const auto *load = node.as<BufferLoadNode>())
            affected |= disable_.count(load->buffer->data);
        });
        if (affected)
          call.CopyOnWrite()->annotations.Set("unit_flag_ctrl", Integer(0));
        return call;
      }
      const VarSet &disable_;
    } rewriter(disable);
    root.CopyOnWrite()->body = rewriter(root->body);
    return root;
  }

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
        if (varying_.count(region->buffer->data) ||
            (IsSharedBuffer(region->buffer) &&
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

  bool NeedsProjection(
      const std::vector<std::shared_ptr<IRStructure>> &nodes) const {
    // Uniform vector work can keep one shared body, including sid-dependent
    // addresses in software dual copies. Split when participation differs or
    // Cube needs a subcore-local value or UB destination.
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
            if (IsL1Buffer(region->buffer) || IsL0ABuffer(region->buffer) ||
                IsL0BBuffer(region->buffer) || IsL0CBuffer(region->buffer))
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
    auto [it, inserted] = vars_[p].emplace(var, Var());
    if (inserted)
      it->second = var.copy_with_suffix("_aiv" + std::to_string(p));
    return it->second;
  }

  Buffer BufferFor(const Buffer &buffer, int p) {
    if (!varying_.count(buffer->data))
      return buffer;
    auto it = buffers_[p].find(buffer);
    if (it != buffers_[p].end())
      return it->second;
    Buffer result = buffer;
    result.CopyOnWrite()->data = VarFor(buffer->data, p);
    result.CopyOnWrite()->name = buffer->name + "_aiv" + std::to_string(p);
    buffers_[p].emplace(buffer, result);
    if (IsSharedBuffer(buffer))
      ub_sids_.Set(result->data, Integer(p));
    return result;
  }

  class Rewriter : public StmtExprMutator {
  public:
    Rewriter(SubcoreProjector *owner, int p, bool allow_constants = true)
        : owner_(owner), p_(p), conditional_depth_(allow_constants ? 0 : 1) {}

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
      For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
      if (is_no_op(loop->body))
        return Evaluate(0);
      loop.CopyOnWrite()->loop_var = owner_->VarFor(op->loop_var, p_);
      loop.CopyOnWrite()->annotations =
          RewriteMetadata(op->annotations).cast<Map<String, Any>>();
      return loop;
    }

    Stmt VisitStmt_(const AttrStmtNode *op) final {
      AttrStmt attr = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
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
      if (conditional_depth_ == 0 && value.as<IntImmNode>())
        owner_->constants_.Set(var, value);
      return Bind(var, value, op->span);
    }

    Stmt VisitStmt_(const IfThenElseNode *op) final {
      PrimExpr condition = analyzer_.Simplify(VisitExpr(op->condition));
      if (is_one(condition))
        return VisitStmt(op->then_case);
      if (is_zero(condition))
        return op->else_case.has_value() ? VisitStmt(op->else_case.value())
                                         : Evaluate(0);
      ++conditional_depth_;
      Stmt then_case = VisitStmt(op->then_case);
      Optional<Stmt> else_case =
          op->else_case.has_value()
              ? Optional<Stmt>(VisitStmt(op->else_case.value()))
              : std::nullopt;
      --conditional_depth_;
      if (is_no_op(then_case) &&
          (!else_case.has_value() || is_no_op(else_case.value())))
        return Evaluate(0);
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
      if (!IsAscendCopyCall(call.get()))
        return call;
      AscendCopy copy(call->args, call->annotations);
      if (!IsL0CBuffer(copy->src) || !IsSharedBuffer(copy->dst))
        return call;
      // AIC executes this task, while its UB destination is owned by p.
      ICHECK_EQ(copy->dual_dst_ctl, 0);
      PrimExpr target = Integer(p_);
      if (auto explicit_sid = annotations.Get("sub_blockid"))
        target = Downcast<PrimExpr>(explicit_sid.value());
      // Routing guards the entire atomic task, so its copies must agree on
      // the destination instead of silently intersecting different targets.
      if (copy_target_.defined()) {
        ICHECK(analyzer_.CanProveEqual(target, copy_target_.value()))
            << "Projected Fixpipe copies in one T.Task must use the same "
               "sub_blockid; put copies with different destinations in "
               "separate T.Task scopes";
      }
      copy_target_ = target;
      active_ = active_ && (target == p_);
      annotations.Set("sub_blockid", Integer(p_));
      call.CopyOnWrite()->annotations = annotations;
      return call;
    }

  public:
    PrimExpr active_{Bool(true)};

  private:
    SubcoreProjector *owner_;
    int p_;
    int conditional_depth_;
    Optional<PrimExpr> copy_target_;
    arith::Analyzer analyzer_;
  };

  bool Varies(const Any &value) const {
    if (auto var = value.try_cast<Var>())
      return varying_.count(var.value());
    if (auto buffer = value.try_cast<Buffer>())
      return varying_.count(buffer.value()->data);
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
