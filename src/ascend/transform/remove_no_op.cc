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
 * \file remove_no_op.cc
 * \brief Ascend no-op elimination over native tensor definitions and regions.
 */

#include "ir/annotations.h"
#include "ir/stmt_functor.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_set>
#include <utility>

#include "ascend/ir/task.h"
#include "ascend/transform/source_scopes.h"
#include "ir/analyzer.h"

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ffi;
using namespace ascend;

namespace {
struct AscendRemoveNoOpConfigObj : public ffi::Object {
  bool use_dataflow_analysis;
  int64_t max_simplification_steps;

  static void RegisterReflection() {
    reflection::ObjectDef<AscendRemoveNoOpConfigObj>()
        .def_ro("use_dataflow_analysis",
                &AscendRemoveNoOpConfigObj::use_dataflow_analysis,
                "Propagate known stores and remove overwritten writes within "
                "native lexical sequences.",
                reflection::DefaultValue(false))
        .def_ro("max_simplification_steps",
                &AscendRemoveNoOpConfigObj::max_simplification_steps,
                "Maximum rewrite steps; zero means unlimited.",
                reflection::DefaultValue(0));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.transform.AscendRemoveNoOpConfig",
                                    AscendRemoveNoOpConfigObj, ffi::Object);
};
class AscendRemoveNoOpConfig : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AscendRemoveNoOpConfig,
                                                ffi::ObjectRef,
                                                AscendRemoveNoOpConfigObj);
};
TVM_FFI_STATIC_INIT_BLOCK() {
  AscendRemoveNoOpConfigObj::RegisterReflection();
  tvm::transform::PassContext::RegisterConfigOption<AscendRemoveNoOpConfig>(
      "tl.AscendRemoveNoOp");
}

using VarSet = std::unordered_set<const VarObj *>;

class AllocationUsageCollector : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  using Result = ffi::Optional<ffi::VisitInterrupt>;
  VarSet allocated, used;
  bool has_volatile{false};

  Result Visit_(const BindObj *op) final {
    if (const auto *call = op->value.as<CallObj>()) {
      if (call->op.same_as(alloc_tensor_op()) ||
          call->op.same_as(decl_tensor_op()))
        allocated.insert(op->var.get());
      if (call->op.same_as(alloc_tensor_op()) && call->attrs) {
        auto attrs = call->attrs.value().as_or_throw<DictAttrs>()->dict;
        has_volatile |= attrs.count("tirx.volatile") || attrs.count("volatile");
      }
    }
    // Definitions are not reads; shape/type operands of the value still are.
    return Visit(op->value);
  }
  Result Visit_(const VarObj *op) final {
    used.insert(op);
    return StmtExprVisitor::Visit_(op);
  }
  Result Visit_(const ForObj *op) final {
    // Loop metadata and referenced Var types can keep allocations live too.
    return StmtExprVisitor::Visit_(op);
  }
  Result Visit_(const RegionStmtObj *op) final {
    if (auto attr = ReadSourceAttribute(op))
      has_volatile |= attr->key == "volatile_scope";
    return StmtExprVisitor::Visit_(op);
  }
};

bool HasRuntimeEffect(const Expr &expr, bool has_volatile) {
  auto effect = SideEffect(expr);
  return effect > CallEffectKind::kReadState ||
         (has_volatile && effect >= CallEffectKind::kReadState);
}

class AscendNoOpRemover : public StmtExprMutatorWithAnalyzer {
public:
  using Parent = StmtExprMutatorWithAnalyzer;
  using Parent::Mutate;
  using Parent::Mutate_;
  AscendNoOpRemover(const sym::Analyzer &analyzer, bool dataflow,
                    bool has_volatile)
      : Parent(analyzer), dataflow_(dataflow), has_volatile_(has_volatile) {}

private:
  Stmt MakeEvaluate(const Expr &value) const {
    return HasRuntimeEffect(value, has_volatile_)
               ? Stmt(Evaluate(value, value->loc))
               : Stmt(Evaluate(0));
  }
  bool SameIndices(const Array<PrimExpr> &a, const Array<PrimExpr> &b) const {
    if (a.size() != b.size())
      return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (SideEffect(a[i]) > CallEffectKind::kPure ||
          SideEffect(b[i]) > CallEffectKind::kPure)
        return false;
      if (!ffi::StructuralEqual()(a[i], b[i]) &&
          !GetSyncedAnalyzer()->CanProveEqual(a[i], b[i]))
        return false;
    }
    return true;
  }
  bool SameAddress(const TensorStoreObj *a, const TensorStoreObj *b) const {
    // Distinct native tensor identities may alias with different offset/dtype/
    // layout. Only an identical view and equal indices establish equality.
    return a->dest.same_as(b->dest) && SameIndices(a->indices, b->indices);
  }
  bool PureStoreValue(const TensorStoreObj *store) const {
    if (SideEffect(store->value) > CallEffectKind::kPure)
      return false;
    return std::all_of(store->indices.begin(), store->indices.end(),
                       [](const PrimExpr &index) {
                         return SideEffect(index) <= CallEffectKind::kPure;
                       });
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode) final {
    if (const auto *call = op->value.as<CallObj>(); IsAnnotationCall(call))
      return Unchanged();
    return MakeEvaluate(op->value);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op, InplaceMode mode) final {
    TensorStore store = Parent::Mutate_(op, mode)
                            .ValueOrUnchanged(GetRef<Stmt>(op))
                            .as_or_throw<TensorStore>();
    const auto *load = store->value.as<TensorLoadObj>();
    if (!has_volatile_ && load && load->source.same_as(store->dest) &&
        SameIndices(load->indices, store->indices))
      return Evaluate(0);
    return store;
  }

  UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode mode) final {
    Stmt stmt = Parent::Mutate_(op, mode).ValueOrUnchanged(GetRef<Stmt>(op));
    const auto *branch = stmt.as<IfObj>();
    if (!branch)
      return stmt;
    bool then_empty = tirx::IsNoOp(branch->then_case);
    bool else_empty = !branch->else_case || tirx::IsNoOp(*branch->else_case);
    if (then_empty && else_empty)
      return MakeEvaluate(branch->condition);
    if (else_empty)
      return If(branch->condition, branch->then_case, std::nullopt,
                branch->loc);
    if (then_empty)
      return If(Not(branch->condition), *branch->else_case, std::nullopt,
                branch->loc);
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    // Native For always tests var < end, independently of the step sign.
    // Evaluate-on-entry extraction belongs to NormalizeControlFlowForSchedule.
    bool pure_bounds = SideEffect(op->begin) <= CallEffectKind::kPure &&
                       SideEffect(op->end) <= CallEffectKind::kPure &&
                       SideEffect(op->step) <= CallEffectKind::kPure;
    if (pure_bounds && GetSyncedAnalyzer()->CanProve(op->begin >= op->end))
      return Evaluate(0);
    Stmt stmt{ffi::UnsafeInit{}};
    if (pure_bounds) {
      stmt = WithTraversalConstraints(
          this, {sym::PredicateConstraint(op->var < op->end)}, [&] {
            return Parent::Mutate_(op, mode).ValueOrUnchanged(GetRef<Stmt>(op));
          });
    } else {
      stmt = Parent::Mutate_(op, mode).ValueOrUnchanged(GetRef<Stmt>(op));
    }
    const auto *loop = stmt.as<ForObj>();
    // Do not erase a potentially non-terminating zero/negative-step loop just
    // because its body is empty, or discard effectful bound evaluations.
    if (pure_bounds && tirx::IsNoOp(loop->body) &&
        GetSyncedAnalyzer()->CanProve(loop->step > 0))
      return Evaluate(0);
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    auto attribute = ReadSourceAttribute(op);
    if (attribute && attribute->key == "pragma_debug_skip_region")
      return Evaluate(0);
    if (attribute && attribute->key == "async_wait_queue_scope") {
      auto inner = ReadSourceAttribute(attribute->region->body);
      TVM_FFI_CHECK(inner && inner->key == "async_wait_inflight_count",
                    ValueError)
          << "async wait requires the inflight-count region";
      PrimExpr count = inner->value.cast<PrimExpr>();
      if (SideEffect(count) <= CallEffectKind::kPure &&
          GetSyncedAnalyzer()->CanProve(count < 0))
        return Mutate(inner->region->body)
            .ValueOrUnchanged(inner->region->body);
    }
    // Unknown regions have their own result/body-parameter semantics. Keep
    // their carrier even when the body is empty.
    return Parent::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode) final {
    Array<Stmt> result;
    Optional<TensorStore> known;
    std::optional<size_t> pending_write;
    auto append = [&](const Stmt &stmt) {
      if (tirx::IsNoOp(stmt))
        return;
      const auto *store = stmt.as<TensorStoreObj>();
      if (dataflow_ && !has_volatile_ && store && PureStoreValue(store)) {
        if (known && SameAddress(known.value().get(), store)) {
          bool same_value = ffi::StructuralEqual()(known->value, store->value);
          if (!same_value && store->value.ty().IsScalar() &&
              (store->value.ty().MatchesCode(kDLInt) ||
               store->value.ty().MatchesCode(kDLUInt)))
            same_value =
                GetSyncedAnalyzer()->CanProveEqual(known->value, store->value);
          if (same_value)
            return;
          if (pending_write)
            result.Set(*pending_write, Evaluate(0));
        }
        // Any other store may alias this view, even when parameter identities
        // differ. Forget the old value instead of assuming noalias.
        known = GetRef<TensorStore>(store);
        pending_write = result.size();
      } else if (const auto *bind = stmt.as<BindObj>();
                 bind && SideEffect(bind->value) <= CallEffectKind::kPure) {
        // Immutable scalar definitions do not change memory. A pointer/tensor
        // view definition is safe as well, but its later stores invalidate.
      } else {
        known = std::nullopt;
        pending_write = std::nullopt;
      }
      result.push_back(stmt);
    };
    for (const Stmt &stmt : op->seq) {
      Stmt rewritten = Mutate(stmt).ValueOrUnchanged(stmt);
      if (const auto *seq = rewritten.as<SeqStmtObj>()) {
        for (const Stmt &child : seq->seq)
          append(child);
      } else {
        append(rewritten);
      }
    }
    Array<Stmt> compact;
    for (const Stmt &stmt : result)
      if (!tirx::IsNoOp(stmt))
        compact.push_back(stmt);
    return SeqStmt(std::move(compact), op->loc);
  }

  bool dataflow_, has_volatile_;
};

class DeadTensorBindingRemover : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit DeadTensorBindingRemover(VarSet dead) : dead_(std::move(dead)) {}
  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (!dead_.count(op->var.get()))
      return StmtExprMutator::Mutate_(op, mode);
    const auto *call = op->value.as<CallObj>();
    Array<Stmt> effects;
    for (const Expr &arg : call->args)
      if (SideEffect(arg) > CallEffectKind::kReadState)
        effects.push_back(Evaluate(arg, arg->loc));
    if (auto tensor = op->var.as<TensorVar>())
      for (const PrimExpr &extent : tensor.value().shape())
        if (SideEffect(extent) > CallEffectKind::kReadState)
          effects.push_back(Evaluate(extent, extent->loc));
    return SeqStmt(std::move(effects), op->loc);
  }

private:
  VarSet dead_;
};
} // namespace

tvm::transform::Pass AscendRemoveNoOp() {
  auto pass = [](tirx::Function f, const IRModule &,
                 const tvm::transform::PassContext &ctx) {
    if (f->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return f;
    auto config =
        ctx->GetConfigOrDefault<AscendRemoveNoOpConfig>("tl.AscendRemoveNoOp");
    auto analyzer = CreateTileLangAnalyzer();
    analyzer->rewrite_simplify.SetMaximumRewriteSteps(
        config->max_simplification_steps);
    SeqStmt body = f->body;
    for (;;) {
      auto usage = make_object<AllocationUsageCollector>();
      usage->Visit(body);
      auto remover = make_object<AscendNoOpRemover>(
          analyzer, config->use_dataflow_analysis, usage->has_volatile);
      body = remover->Mutate(body).ValueOrUnchanged(body);
      usage = make_object<AllocationUsageCollector>();
      usage->Visit(body);
      VarSet dead;
      for (const VarObj *var : usage->allocated)
        if (!usage->used.count(var))
          dead.insert(var);
      if (dead.empty())
        break;
      auto dce = make_object<DeadTensorBindingRemover>(std::move(dead));
      body = dce->Mutate(body).ValueOrUnchanged(body);
    }
    f.CopyOnWrite()->body = std::move(body);
    return f;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.AscendRemoveNoOp");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.AscendRemoveNoOp",
                              AscendRemoveNoOp);
}
} // namespace tvm::tl
