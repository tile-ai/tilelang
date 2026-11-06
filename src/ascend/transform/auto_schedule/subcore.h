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

#include "ascend/transform/source_scopes.h"
#include "ir/annotations.h"
#include "ir/tensor_utils.h"
#include "transform/common/constr_visitor.h"
#include <array>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>
#include <unordered_map>
#include <unordered_set>

#include <tvm/tirx/analysis.h>

#include "layout/layout.h"
#include "scheduled_tir.h"
#include "support/check.h"
#include "task_analysis.h"
#include "task_annotations.h"

namespace tvm::tl::ascend {

// Project only subcore-dependent entries. Uniform controls retain one lexical
// body; a projected control owns fresh native definitions in each copy. The
// origin metadata preserves correspondence through scheduling and sync.
class SubcoreProjector {
public:
  static RegionStmt
  Rewrite(const RegionStmt &root, const ConstrSet &outer_ctx,
          const PrimVar &sid,
          const ScheduleUnitMetadata &kernel_annotations = {}) {
    auto shell = ReadSourceAttribute(root).value();
    auto values = shell.value.cast<ScheduleUnitMetadata>();
    auto annotations = values.Get("annotations")
                           .value_or(ScheduleUnitMetadata{})
                           .cast<ScheduleUnitMetadata>();
    ICHECK(!annotations.count(kSplitSubcores));
    for (const Stmt &stmt : root->body->seq)
      if (IsTaskMetadataStatement(stmt))
        ICHECK(!ReadAnnotations(stmt.as<EvaluateObj>()->value.as<CallObj>())
                    .count(kSplitSubcores));
    ScheduledTIR scheduled =
        DecodeScheduledTIR(root, outer_ctx, nullptr, kernel_annotations);
    SubcoreProjector projector(sid,
                               scheduled.metadata.num_aiv_subcores.value_or(2));
    projector.storage_.Collect(root->body);
    projector.FindVaryingValues(scheduled.tree);
    if (!projector.NeedsProjection(scheduled.tree))
      return root;
    SeqStmt body = projector.RewriteList(scheduled.tree, -1, root->body->loc);
    projector.VerifyProjectedFixpipeTasks(body);
    annotations = projector.ExpandFields(annotations);
    values.Set("annotations", annotations);
    auto statements = body->seq;
    statements.push_back(MakeAnnotation({{kSplitSubcores, true}}, root->loc));
    return MakeAttribute(shell.node, shell.key, values,
                         SeqStmt(statements, body->loc), root->loc);
  }

private:
  using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
  using VarMap = std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual>;
  using Nodes = IRStructure::ChildList;

  SubcoreProjector(PrimVar sid, int count)
      : sid_(std::move(sid)), count_(count) {
    ICHECK(count == 1 || count == 2);
    varying_.insert(sid_);
  }

  static const TaskNode *Header(const IRStructure *node) {
    return node->IsTask() ? static_cast<const TaskNode *>(node)
                          : node->HeaderTask();
  }

  bool Varies(ffi::AnyView value) const {
    bool result = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        value, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (auto var = node.as<Var>())
            result |= varying_.count(*var) != 0;
          return ffi::WalkResult::Advance();
        });
    return result;
  }

  bool Varies(const TaskNode *task) const {
    if (!task)
      return false;
    for (const Var &var : task->GetReadVars())
      if (Varies(var) || Varies(var->ty))
        return true;
    for (const Var &var : task->GetWriteVars())
      if (Varies(var) || Varies(var->ty))
        return true;
    for (const auto &regions :
         {task->GetReadRegions(), task->GetWriteRegions()})
      for (const TensorRegion &region : regions) {
        TensorVar tensor = RegionTensor(region);
        if (Varies(region) || Varies(tensor.ty()) ||
            varying_.count(storage_.Identity(tensor)) ||
            (IsSharedBuffer(tensor) && !task->UsesPipe(ResourcePipe::kFixpipe)))
          return true;
      }
    return task->UsesPipe(ResourcePipe::kVector) ||
           task->UsesPipe(ResourcePipe::kMTE3);
  }

  bool NeedsProjection(const Nodes &nodes) const {
    // Uniform vector work keeps one shared body, including software dual-copy
    // addresses. Split when participation differs or Cube reads an AIV-local
    // value/destination. Shared scalar words need concrete subcore indices.
    for (const auto &node : nodes) {
      const TaskNode *task = Header(node.get());
      if (task) {
        if (!node->IsTask() && Varies(task))
          return true;
        bool varying_control = false;
        ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
            WrapScheduledGuards(task->stmt, node->GetGuards()),
            [&](const ObjectRef &stmt) -> ffi::Expected<ffi::WalkResult> {
              if (const auto *branch = stmt.as<IfObj>())
                varying_control |= Varies(branch->condition);
              if (const auto *loop = stmt.as<ForObj>())
                varying_control |= Varies(loop->begin) || Varies(loop->end) ||
                                   Varies(loop->step);
              return ffi::WalkResult::Advance();
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
            for (const TensorRegion &region : regions) {
              TensorVar tensor = RegionTensor(region);
              if (IsSSBuffer(tensor) || IsL1Buffer(tensor) ||
                  IsL0ABuffer(tensor) || IsL0BBuffer(tensor) ||
                  IsL0CBuffer(tensor))
                return true;
            }
        }
      }
      for (const auto *children : node->ChildLists())
        if (NeedsProjection(*children))
          return true;
    }
    return false;
  }

  void FindVaryingValues(const Nodes &tree) {
    for (const auto &[var, _] : storage_.bindings)
      definitions_.insert(var);
    std::vector<TaskNode *> tasks;
    CollectAllTaskNodes(tree, tasks);
    for (TaskNode *task : tasks)
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          task->stmt,
          [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
            if (!IsAscendCopyCall(call))
              return ffi::WalkResult::Advance();

            auto decoded = DecodeTileOpArguments(call);
            AscendCopy copy(decoded.operands, decoded.annotations);
            if (IsL0CBuffer(copy->src) && IsSharedBuffer(copy->dst) &&
                copy->dual_dst_ctl != 0) {
              dual_storages_.insert(storage_.Identity(copy->dst));
              dual_tasks_.insert(task);
            }
            return ffi::WalkResult::Advance();
          });
    for (TaskNode *task : tasks)
      for (const auto &regions :
           {task->GetReadRegions(), task->GetWriteRegions()})
        for (const TensorRegion &region : regions) {
          TensorVar tensor = RegionTensor(region);
          Var storage = storage_.Identity(tensor);
          if (IsSharedBuffer(tensor) && !dual_storages_.count(storage)) {
            shared_roots_.insert(storage);
            varying_.insert(storage);
            varying_.insert(tensor);
          }
        }
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto &[var, value] : storage_.bindings)
        if (Varies(value) || Varies(var->ty))
          changed |= varying_.insert(var).second;
      for (TaskNode *task : tasks) {
        bool varying = Varies(task);
        for (const IRStructure *parent = task->GetParent(); parent;
             parent = parent->GetParent())
          varying |= Varies(parent->HeaderTask());
        if (!varying)
          continue;
        for (const Var &var : task->GetWriteVars())
          changed |= varying_.insert(var).second;
        for (const TensorRegion &region : task->GetWriteRegions())
          if (IsRegisterRegion(region))
            changed |=
                varying_.insert(storage_.Identity(RegionTensor(region))).second;
      }
    }
  }

  static int MetadataInteger(const Any &value, const char *name) {
    return static_cast<int>(
        task_metadata_detail::RequireConstantTaskMetadataInt(value, name)
            ->value.as<int64_t>()
            .value());
  }

  static bool Empty(const Stmt &stmt) {
    if (const auto *sequence = stmt.as<SeqStmtObj>()) {
      for (const Stmt &child : sequence->seq)
        if (!Empty(child))
          return false;
      return true;
    }
    return false;
  }

  static void Append(Array<Stmt> *statements, const Stmt &stmt) {
    if (const auto *sequence = stmt.as<SeqStmtObj>()) {
      for (const Stmt &child : sequence->seq)
        Append(statements, child);
    } else {
      statements->push_back(stmt);
    }
  }

  static void VerifyReplayableFixpipeTask(const Stmt &body) {
    bool external_effect = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        body, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *store = node.as<TensorStoreObj>())
            external_effect |= !IsRegisterRegion(
                FullTensorRegion(store->dest.as_or_throw<TensorVar>()));
          if (const auto *call = node.as<CallObj>()) {
            if (IsAscendCopyCall(call)) {
              auto decoded = DecodeTileOpArguments(call);
              AscendCopy copy(decoded.operands, decoded.annotations);
              external_effect |=
                  !IsL0CBuffer(copy->src) || !IsSharedBuffer(copy->dst);
            } else if (!IsScheduledBind(call) && call->op.as<OpObj>() &&
                       !IsAnnotationCall(call)) {
              external_effect |= SideEffect(ffi::GetRef<Expr>(call)) >
                                 CallEffectKind::kReadState;
            }
          }
          return ffi::WalkResult::Advance();
        });
    ICHECK(!external_effect)
        << "Fixpipe routing in a projected T.Task with external side effects "
           "is not supported: projection may repeat those effects. Move the "
           "side effects into a separate T.Task and snapshot any dynamic "
           "route before the Fixpipe task";
  }

  void VerifyProjectedFixpipeTasks(const Stmt &body) const {
    std::unordered_map<int, std::vector<RegionStmt>> projections;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        body, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          const auto *task = node.as<RegionStmtObj>();
          if (!IsSemanticTask(task))
            return ffi::WalkResult::Advance();
          auto metadata = TaskAnnotations(task);
          if (auto origin = metadata.Get("subcore_origin")) {
            int id = MetadataInteger(*origin, "subcore_origin");
            if (replay_check_origins_.count(id))
              projections[id].push_back(ffi::GetRef<RegionStmt>(task));
          }
          return ffi::WalkResult::Advance();
        });
    // Count copies after all enclosing guards have selected surviving paths.
    for (const auto &[_, tasks] : projections)
      if (tasks.size() > 1)
        for (const RegionStmt &task : tasks)
          VerifyReplayableFixpipeTask(task);
  }

  class Rewriter : public StmtExprMutator {
  public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    Rewriter(SubcoreProjector *owner, int p, bool allow_constants = true)
        : owner_(owner), p_(p), conditional_depth_(allow_constants ? 0 : 1) {}

    Stmt Rewrite(const Stmt &value) {
      return Mutate(value).ValueOrUnchanged(value);
    }
    Expr RewriteExpr(const Expr &value) {
      Expr result = Mutate(value).ValueOrUnchanged(value);
      if (auto primitive = result.as<PrimExpr>())
        return analyzer_->Simplify(*primitive);
      return result;
    }
    PrimExpr Primitive(const PrimExpr &value) {
      return RewriteExpr(value).as_or_throw<PrimExpr>();
    }
    Any RewriteMetadata(const Any &value) {
      return Mutate(ffi::AnyView(value)).ValueOrUnchanged(value);
    }
    Type RewriteType(const Type &type) {
      return Mutate(type)
          .as_or_throw<ffi::UnchangedOr<Type>>()
          .ValueOrUnchanged(type);
    }
    bool CanReplayFixpipe() const {
      return has_varying_copy_route_ || has_local_copy_guard_;
    }
    PrimExpr active_{IntImm::Bool(true)};

  private:
    struct PerCoreRouting {
      PrimExpr path{IntImm::Bool(true)};
      PrimExpr active{IntImm::Bool(false)};
      bool routed{false};
      bool replays_fixpipe{false};
    };

    PrimExpr GetPerCoreActiveGuard(const RegionStmt &task, PrimExpr active) {
      Map<Var, PrimExpr> bindings;
      VarSet locals;
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          task->body,
          [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
            if (const auto *loop = node.as<ForObj>())
              locals.insert(loop->var);
            if (const auto *region = node.as<RegionStmtObj>()) {
              locals.insert(region->body_params.begin(),
                            region->body_params.end());
              locals.insert(region->result_vars.begin(),
                            region->result_vars.end());
            }
            if (const auto *bind = node.as<BindObj>()) {
              locals.insert(bind->var);
              Expr source = bind->value;
              if (const auto *call = source.as<CallObj>();
                  IsScheduledBind(call))
                source = ReadScheduledBinding(call).value;
              auto primitive = source.as<PrimExpr>();
              if (!primitive)
                return ffi::WalkResult::Advance();
              PrimExpr value = SubstituteConstraintExpr(*primitive, bindings);
              bool immutable = SideEffect(value) <= CallEffectKind::kReadState;
              ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
                  value,
                  [&](const ObjectRef &input)
                      -> ffi::Expected<ffi::WalkResult> {
                    immutable &= input.as<TensorLoadObj>() == nullptr;
                    return ffi::WalkResult::Advance();
                  });
              if (immutable)
                bindings.Set(bind->var, value);
            }
            return ffi::WalkResult::Advance();
          });
      active = analyzer_->Simplify(SubstituteConstraintExpr(active, bindings));
      bool available = SideEffect(active) <= CallEffectKind::kReadState;
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          active, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
            if (auto var = node.as<Var>())
              available &= !locals.count(*var);
            available &= node.as<TensorLoadObj>() == nullptr;
            return ffi::WalkResult::Advance();
          });
      ICHECK(available)
          << "Cannot project PerCoreTask Fixpipe routing: candidate dispatch "
             "must use immutable values available before PerCoreTask. Move "
             "the dispatch and route computation outside PerCoreTask";
      return active;
    }

    Stmt RewritePerCoreCandidate(const RegionStmt &task) {
      auto metadata = TaskAnnotations(task.get());
      auto origin = metadata.Get("subcore_origin");
      int origin_id;
      if (origin) {
        origin_id = MetadataInteger(*origin, "subcore_origin");
      } else {
        auto [it, inserted] =
            owner_->candidate_origins_.emplace(task, owner_->next_origin_);
        if (inserted)
          ++owner_->next_origin_;
        origin_id = it->second;
        metadata.Set("subcore_origin", IntImm::Int32(origin_id));
      }
      auto rewriter =
          ffi::make_object<Rewriter>(owner_, p_, conditional_depth_ == 0);
      rewriter->local_vars_ = local_vars_;
      Stmt body = rewriter->Rewrite(WithTaskAnnotations(task, metadata));
      local_vars_.insert(rewriter->local_vars_.begin(),
                         rewriter->local_vars_.end());
      PrimExpr active = analyzer_->Simplify(rewriter->active_);
      if (HasTaskBufferAccess(task->body)) {
        per_core_->active = per_core_->active || (per_core_->path && active);
        per_core_->routed |= !IsOne(active);
      }
      has_local_copy_guard_ |= rewriter->has_local_copy_guard_;
      per_core_->replays_fixpipe |= rewriter->CanReplayFixpipe();
      if (rewriter->CanReplayFixpipe() ||
          (!origin && rewriter->copy_target_.has_value() && IsOne(active)))
        owner_->replay_check_origins_.insert(origin_id);
      if (IsZero(active))
        return SeqStmt({}, task->loc);
      return IsOne(active) ? body
                           : If(active, SeqStmt(body), std::nullopt, body->loc);
    }

    ffi::UnchangedOr<Expr> Mutate_(const VarObj *op, InplaceMode) final {
      Var var = ffi::GetRef<Var>(op);
      if (var.same_as(owner_->sid_))
        return MakeConst(owner_->sid_.ty(), p_);
      Var mapped = owner_->VarFor(var, p_);
      if (auto constant = owner_->constants_.Get(mapped))
        return *constant;
      return mapped;
    }

    ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode) final {
      owner_->definitions_.insert(op->var);
      owner_->varying_.insert(op->var);
      Expr value = op->value;
      std::optional<ScheduledBinding> schedule;
      if (const auto *call = value.as<CallObj>(); IsScheduledBind(call)) {
        schedule = ReadScheduledBinding(call);
        value = schedule->value;
      }
      value = RewriteExpr(value);
      Var var = owner_->VarFor(op->var, p_);
      if (!var->ty.same_as(value->ty)) {
        var = var.CopyWithType(value->ty);
        owner_->SetVar(op->var, var, p_);
      }
      local_vars_.insert(var);
      if (conditional_depth_ == 0 && value.as<IntImmObj>())
        owner_->constants_.Set(var, value.as_or_throw<PrimExpr>());
      Stmt result = Bind(var, value, op->loc);
      if (schedule) {
        auto metadata =
            RewriteMetadata(schedule->task).cast<ScheduleUnitMetadata>();
        metadata.Set("subcore", IntImm::Int32(p_));
        result = MakeScheduledBinding(result.as_or_throw<Bind>(),
                                      schedule->stage, metadata);
      }
      // Native allocation ownership is a lexical operand-bearing directive,
      // emitted after its Bind; it cannot be hoisted to root arguments.
      if (owner_->shared_roots_.count(op->var) &&
          !owner_->projected_.count(op->var))
        return SeqStmt({result, MakeAnnotation({{kAivStorageMap,
                                                 AivStorageMap{{var, p_}}}},
                                               op->loc)},
                       op->loc);
      return result;
    }

    ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode) final {
      owner_->definitions_.insert(op->var);
      owner_->varying_.insert(op->var);
      PrimVar variable = owner_->VarFor(op->var, p_).as_or_throw<PrimVar>();
      local_vars_.insert(variable);
      PrimExpr begin = Primitive(op->begin), end = Primitive(op->end),
               step = Primitive(op->step);
      Stmt body = Rewrite(op->body);
      if (Empty(body))
        return SeqStmt({}, op->loc);
      For result = ffi::GetRef<For>(op);
      auto *writer = result.CopyOnWrite();
      writer->var = variable;
      writer->begin = begin;
      writer->end = end;
      writer->step = step;
      writer->body = SeqStmt(body);
      // Static annotations remain static; dynamic loop fields already live in
      // the native tl.loop.annotations region and are rewritten with its body.
      return result;
    }

    ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                   InplaceMode mode) final {
      if (IsAscendBlock(op)) {
        Stmt body = Rewrite(op->body);
        if (Empty(body))
          return SeqStmt({}, op->loc);
        return MakeAscendBlock(AscendBlockName(op),
                               RewriteMetadata(AscendBlockAnnotations(op))
                                   .cast<ScheduleUnitMetadata>(),
                               SeqStmt(body), op->loc);
      }
      if (per_core_ && IsTask(op))
        return RewritePerCoreCandidate(ffi::GetRef<RegionStmt>(op));
      if (IsSemanticTask(op)) {
        Stmt body{ffi::UnsafeInit{}};
        if (IsPerCoreTask(op)) {
          ICHECK(per_core_ == nullptr);
          PerCoreRouting routing;
          per_core_ = &routing;
          body = Rewrite(op->body);
          per_core_ = nullptr;
          ICHECK(owner_->count_ == 1 || !routing.replays_fixpipe ||
                 !HasCrossCoreSync(body))
              << "Subcore projection of T.PerCoreTask with explicit "
                 "synchronization and subcore-dependent or task-local Fixpipe "
                 "routing is not supported: projection may change the "
                 "synchronization participation or execution counts. Move it "
                 "outside T.PerCoreTask, or compute each explicit sub_blockid "
                 "before T.PerCoreTask using subcore-uniform values";
          if (routing.routed) {
            RegionStmt routed = ffi::GetRef<RegionStmt>(op);
            routed.CopyOnWrite()->body = SeqStmt(body);
            active_ = active_ && GetPerCoreActiveGuard(routed, routing.active);
          }
        } else {
          body = Rewrite(op->body);
        }
        if (Empty(body))
          return SeqStmt({}, op->loc);
        auto metadata =
            RewriteMetadata(TaskAnnotations(op)).cast<ScheduleUnitMetadata>();
        metadata.Set("subcore", IntImm::Int32(p_));
        RegionStmt result =
            WithTaskAnnotations(ffi::GetRef<RegionStmt>(op), metadata);
        result.CopyOnWrite()->body = SeqStmt(body);
        return result;
      }
      if (auto source = ReadSourceAttribute(op)) {
        Stmt body = Rewrite(op->body);
        if (Empty(body))
          return SeqStmt({}, op->loc);
        return MakeAttribute(RewriteMetadata(source->node), source->key,
                             RewriteMetadata(source->value), SeqStmt(body),
                             op->loc);
      }
      // Native region body parameters and exported results use the same
      // projection as their body uses.
      for (const auto &variables : {op->body_params, op->result_vars})
        for (const Var &var : variables) {
          owner_->definitions_.insert(var);
          owner_->varying_.insert(var);
          local_vars_.insert(owner_->VarFor(var, p_));
        }
      return StmtExprMutator::Mutate_(op, mode);
    }

    ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode) final {
      Array<Stmt> statements;
      for (const Stmt &stmt : op->seq)
        Append(&statements, Rewrite(stmt));
      return SeqStmt(statements, op->loc);
    }

    ffi::UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode) final {
      PrimExpr condition = Primitive(op->condition);
      if (IsOne(condition))
        return Rewrite(op->then_case);
      if (IsZero(condition))
        return op->else_case ? Rewrite(*op->else_case) : SeqStmt({}, op->loc);
      PrimExpr path = per_core_ ? per_core_->path : IntImm::Bool(true);
      if (per_core_)
        per_core_->path = path && condition;
      ++conditional_depth_;
      Stmt then_case = Rewrite(op->then_case);
      if (per_core_)
        per_core_->path = path && !condition;
      Optional<SeqStmt> else_case;
      if (op->else_case)
        else_case = SeqStmt(Rewrite(*op->else_case));
      --conditional_depth_;
      if (per_core_)
        per_core_->path = path;
      if (Empty(then_case) && (!else_case || Empty(*else_case)))
        return SeqStmt({}, op->loc);
      if (Empty(then_case))
        return If(!condition, *else_case, std::nullopt, op->loc);
      if (else_case && Empty(*else_case))
        else_case = std::nullopt;
      return If(condition, SeqStmt(then_case), else_case, op->loc);
    }

    ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj *op,
                                   InplaceMode mode) final {
      Stmt rewritten = StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(
          ffi::GetRef<Stmt>(op));
      const auto *statement = rewritten.as<EvaluateObj>();
      const auto *call_node = statement->value.as<CallObj>();
      if (!call_node || !IsAscendCopyCall(call_node))
        return rewritten;
      Call call = ffi::GetRef<Call>(call_node);
      auto decoded = DecodeTileOpArguments(call_node);
      AscendCopy copy(decoded.operands, decoded.annotations);
      if (!IsL0CBuffer(copy->src) || !IsSharedBuffer(copy->dst))
        return rewritten;
      ICHECK_EQ(copy->dual_dst_ctl, 0);
      PrimExpr target = IntImm::Int32(p_);
      if (auto explicit_sid = decoded.annotations.Get("sub_blockid"))
        target = explicit_sid->cast<PrimExpr>();
      if (copy_target_)
        ICHECK(analyzer_->CanProveEqual(target, *copy_target_))
            << "Projected Fixpipe copies in one T.Task must use the same "
               "sub_blockid; put copies with different destinations in "
               "separate T.Task scopes";
      copy_target_ = target;
      decoded.annotations.Set("sub_blockid", IntImm::Int32(p_));
      Call replacement =
          MakeAnnotatedCall(call->ty, call->op.as_or_throw<Op>(),
                            decoded.operands, decoded.annotations, call->loc);
      Stmt result = Evaluate(replacement, op->loc);
      Call source = op->value.as_or_throw<Call>();
      if (auto origin = owner_->copy_origins_.find(source);
          origin != owner_->copy_origins_.end())
        source = origin->second;
      auto source_route =
          DecodeTileOpArguments(source.get()).annotations.Get("sub_blockid");
      has_varying_copy_route_ |= !source_route || owner_->Varies(*source_route);
      owner_->copy_origins_.emplace(replacement, source);
      PrimExpr condition = analyzer_->Simplify(target == p_);
      bool task_local = SideEffect(condition) > CallEffectKind::kReadState;
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          condition,
          [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
            if (auto var = node.as<Var>())
              task_local |= local_vars_.count(*var) != 0;
            task_local |= node.as<TensorLoadObj>() != nullptr;
            return ffi::WalkResult::Advance();
          });
      if (task_local) {
        has_local_copy_guard_ = true;
        return If(condition, SeqStmt({result}, op->loc), std::nullopt, op->loc);
      }
      active_ = active_ && condition;
      return result;
    }

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

  void SetVar(const Var &original, const Var &replacement, int p) {
    vars_[p].insert_or_assign(original, replacement);
    vars_[p].insert_or_assign(replacement, replacement);
    projected_.insert(replacement);
  }

  Var VarFor(const Var &var, int p) {
    if (auto it = vars_[p].find(var); it != vars_[p].end())
      return it->second;
    if (!definitions_.count(var) || !varying_.count(var))
      return var;
    auto rewriter = ffi::make_object<Rewriter>(this, p);
    Type type = rewriter->RewriteType(var->ty);
    Var result(var->name + "_aiv" + std::to_string(p), type, var->loc);
    SetVar(var, result, p);
    return result;
  }

  // A common metadata carrier remains at its original lexical position. Expand
  // maps/lists over projected definitions, without inventing undeclared keys.
  bool Available(const Any &value, int p) const {
    bool available = true;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        value, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (auto var = node.as<Var>();
              var && definitions_.count(*var) && varying_.count(*var))
            available &= vars_[p].count(*var) != 0;
          return ffi::WalkResult::Advance();
        });
    return available;
  }

  Any ExpandAnnotation(const String &key, const Any &value) {
    if (auto map = value.try_cast<Map<Any, Any>>()) {
      Map<Any, Any> result;
      for (const auto &[entry_key, entry] : *map) {
        if (!Varies(entry_key)) {
          result.Set(entry_key, ExpandAnnotation(key, entry));
          continue;
        }
        for (int p = 0; p < count_; ++p) {
          if (!Available(entry_key, p) || !Available(entry, p))
            continue;
          auto rewriter = ffi::make_object<Rewriter>(this, p);
          result.Set(rewriter->RewriteMetadata(entry_key),
                     rewriter->RewriteMetadata(entry));
        }
      }
      return result;
    }
    if (auto array = value.try_cast<Array<Any>>()) {
      Array<Any> result;
      for (const Any &entry : *array) {
        if (!Varies(entry)) {
          result.push_back(entry);
          continue;
        }
        for (int p = 0; p < count_; ++p)
          if (Available(entry, p))
            result.push_back(
                ffi::make_object<Rewriter>(this, p)->RewriteMetadata(entry));
      }
      return result;
    }
    return value;
  }

  ScheduleUnitMetadata ExpandFields(const ScheduleUnitMetadata &fields) {
    ScheduleUnitMetadata result;
    for (const auto &[key, value] : fields)
      result.Set(key, ExpandAnnotation(key, value));
    return result;
  }

  SeqStmt RewriteList(const Nodes &nodes, int forced_sid, Location loc,
                      bool conditional_parent = false) {
    Array<Stmt> result;
    for (const auto &node : nodes) {
      const TaskNode *task = Header(node.get());
      bool project = forced_sid >= 0 || Varies(task);
      ICHECK(!project || !dual_tasks_.count(task))
          << "Hardware L0C dual_copy requires a subcore-uniform control path; "
             "use T.copy under if cond(sid) for independently selected AIV "
             "destinations";
      auto [origin_it, inserted] = origins_.emplace(node.get(), next_origin_);
      if (inserted)
        ++next_origin_;
      int begin = forced_sid >= 0 ? forced_sid : 0;
      int end = forced_sid >= 0 ? forced_sid + 1 : (project ? count_ : 1);
      for (int p = begin; p < end; ++p) {
        Stmt body{ffi::UnsafeInit{}};
        if (node->IsTask()) {
          if (const auto *bind = task->stmt.as<BindObj>();
              IsScheduledBind(bind)) {
            auto marker = ReadScheduledBinding(bind->value.as<CallObj>());
            if (project) {
              marker.task.Set("subcore", IntImm::Int32(p));
              marker.task.Set("subcore_origin",
                              IntImm::Int32(origin_it->second));
            }
            body = WithScheduledBindingMetadata(ffi::GetRef<Bind>(bind),
                                                node->GetStage(), marker.task);
          } else {
            auto marker = SingleStatement(task->stmt).as_or_throw<RegionStmt>();
            auto metadata = TaskAnnotations(marker.get());
            if (project) {
              metadata.Set("subcore", IntImm::Int32(p));
              metadata.Set("subcore_origin", IntImm::Int32(origin_it->second));
            }
            body = WithTaskAnnotations(marker, metadata);
          }
        } else if (node->IsControl()) {
          const auto *control = static_cast<const ControlNode *>(node.get());
          For loop = control->control;
          if (project) {
            definitions_.insert(loop->var);
            varying_.insert(loop->var);
            VarFor(loop->var, p);
          }
          loop.CopyOnWrite()->body =
              RewriteList(control->children, project ? p : -1, loop->body->loc,
                          conditional_parent || node->HasConditions());
          body = loop;
        } else {
          const auto *scope = static_cast<const ScopeNode *>(node.get());
          if (const auto *branch = scope->shell.as<IfObj>()) {
            auto children = RewriteList(scope->children, project ? p : -1,
                                        branch->then_case->loc, true);
            Optional<SeqStmt> otherwise;
            if (scope->has_else)
              otherwise = RewriteList(scope->else_children, project ? p : -1,
                                      branch->else_case.value()->loc, true);
            body = If(branch->condition, children, otherwise, branch->loc);
          } else if (auto attribute = ReadSourceAttribute(scope->shell)) {
            body = attribute->WithBody(
                RewriteList(scope->children, project ? p : -1,
                            attribute->region->body->loc, conditional_parent));
            if (!project)
              body = MakeAttribute(
                  attribute->node, attribute->key,
                  ExpandAnnotation(attribute->key, attribute->value),
                  body.as_or_throw<RegionStmt>()->body, body->loc);
          } else {
            auto fields = ReadAnnotations(
                scope->shell.as<EvaluateObj>()->value.as<CallObj>());
            body = MakeAnnotation(project ? fields : ExpandFields(fields),
                                  scope->shell->loc);
          }
        }
        body = WrapScheduledGuards(body, node->GetGuards());
        if (project) {
          auto rewriter =
              ffi::make_object<Rewriter>(this, p, !conditional_parent);
          body = rewriter->Rewrite(body);
          sym::Analyzer analyzer;
          PrimExpr active = analyzer->Simplify(rewriter->active_);
          if (IsZero(active))
            continue;
          if (node->IsTask() && rewriter->CanReplayFixpipe())
            replay_check_origins_.insert(origin_it->second);
          if (!IsOne(active))
            body = If(active, SeqStmt(body), std::nullopt, body->loc);
        }
        if (Empty(body))
          continue;
        if (node->IsScope() || (task && task->stmt.as<BindObj>())) {
          Append(&result, body);
        } else {
          // Guards are native lexical scopes outside the schedule unit. A
          // conditional can select a whole unit but cannot hide Bind exports.
          if (auto branch = body.as<If>()) {
            If guarded = *branch;
            guarded.CopyOnWrite()->then_case = SeqStmt(
                {MakeScheduleUnit(guarded->then_case, node->GetStage())},
                body->loc);
            result.push_back(guarded);
          } else {
            result.push_back(MakeScheduleUnit(body, node->GetStage()));
          }
        }
      }
    }
    return SeqStmt(result, loc);
  }

  PrimVar sid_;
  int count_;
  int next_origin_{0};
  StorageProvenance storage_ = MakeAscendStorageProvenance();
  std::unordered_map<const IRStructure *, int> origins_;
  std::unordered_map<RegionStmt, int, ObjectPtrHash, ObjectPtrEqual>
      candidate_origins_;
  std::unordered_set<int> replay_check_origins_;
  std::unordered_map<Call, Call, ObjectPtrHash, ObjectPtrEqual> copy_origins_;
  VarSet varying_, definitions_, projected_, dual_storages_, shared_roots_;
  std::unordered_set<const TaskNode *> dual_tasks_;
  std::array<VarMap, 2> vars_;
  Map<Var, PrimExpr> constants_;
};

} // namespace tvm::tl::ascend
