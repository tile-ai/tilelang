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
 * \brief Ascend-specific no-op removal.
 */

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "ascend/transform/native_visitor.h"
#include "compat/expr.h"

namespace tvm::tl {
namespace {

struct AscendRemoveNoOpConfigNode : public ffi::Object {
  bool use_dataflow_analysis{false};
  int64_t max_simplification_steps{0};

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<AscendRemoveNoOpConfigNode>()
        .def_ro("use_dataflow_analysis",
                &AscendRemoveNoOpConfigNode::use_dataflow_analysis,
                ffi::reflection::DefaultValue(false))
        .def_ro("max_simplification_steps",
                &AscendRemoveNoOpConfigNode::max_simplification_steps,
                ffi::reflection::DefaultValue(0));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.transform.AscendRemoveNoOpConfig",
                                    AscendRemoveNoOpConfigNode, ffi::Object);
};

class AscendRemoveNoOpConfig : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AscendRemoveNoOpConfig,
                                                ffi::ObjectRef,
                                                AscendRemoveNoOpConfigNode);
};

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

class AllocationUsageCollector : public tirx::StmtExprVisitor {
public:
  VarSet Dead() {
    VarSet live = used_;
    std::vector<Var> work(live.begin(), live.end());
    for (size_t index = 0; index < work.size(); ++index) {
      if (auto found = parents_.find(work[index]); found != parents_.end()) {
        for (const Var &parent : found->second) {
          if (live.insert(parent).second)
            work.push_back(parent);
        }
      }
    }
    VarSet dead;
    for (const Var &variable : allocated_) {
      if (!live.count(variable))
        dead.insert(variable);
    }
    return dead;
  }

protected:
  ffi::Optional<ffi::VisitInterrupt> Visit_(const BindNode *node) final {
    auto call = node->value.as<CallNode>();
    if (node->var.as<tirx::TensorVar>() && call &&
        (call->op.same_as(tirx::alloc_tensor_op()) ||
         call->op.same_as(tirx::decl_tensor_op()))) {
      allocated_.insert(node->var);
      if (call->op.same_as(tirx::decl_tensor_op()))
        parents_[node->var] = tvm::UndefinedVars(call->args[0]);
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const VarNode *node) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone)
      used_.insert(ffi::GetRef<Var>(node));
    return StmtExprVisitor::Visit_(node);
  }

private:
  VarSet allocated_;
  VarSet used_;
  std::unordered_map<Var, ffi::Array<Var>, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      parents_;
};

class AscendNoOpRemover : public ascend::AnalyzerMutator {
public:
  AscendNoOpRemover(const sym::Analyzer &analyzer, bool dataflow, VarSet dead)
      : ascend::AnalyzerMutator(analyzer), dataflow_(dataflow),
        dead_(std::move(dead)) {}

private:
  using Parent = ascend::AnalyzerMutator;

  static bool Empty(const Stmt &statement) {
    if (auto sequence = statement.as<SeqStmtNode>()) {
      for (const auto &child : sequence->seq)
        if (!Empty(child))
          return false;
      return true;
    }
    if (auto evaluate = statement.as<EvaluateNode>())
      return SideEffect(evaluate->value) <= CallEffectKind::kReadState;
    return false;
  }

  static Stmt Effects(const ffi::Array<Expr> &expressions) {
    ffi::Array<Stmt> result;
    for (const auto &expression : expressions) {
      if (SideEffect(expression) > CallEffectKind::kReadState)
        result.push_back(Evaluate(expression));
    }
    return SeqStmt(result);
  }

  Stmt VisitStmt_(const BindNode *node) final {
    if (dead_.count(node->var))
      return SeqStmt({});
    return Parent::VisitStmt_(node);
  }

  Stmt VisitStmt_(const SeqStmtNode *node) final {
    known_.clear();
    Stmt result = Parent::VisitStmt_(node);
    known_.clear();
    return result;
  }

  Stmt VisitStmt_(const EvaluateNode *node) final {
    Expr value = VisitExpr(node->value);
    return Effects({value});
  }

  Stmt VisitStmt_(const RegionStmtNode *node) final {
    if (node->attrs->dict.count("pragma_debug_skip_region"))
      return SeqStmt({});
    known_.clear();
    Stmt result = Parent::VisitStmt_(node);
    known_.clear();
    return result;
  }

  Stmt VisitStmt_(const IfNode *node) final {
    known_.clear();
    Stmt result = Parent::VisitStmt_(node);
    known_.clear();
    auto conditional = result.as<IfNode>();
    if (!conditional)
      return result;
    bool empty_then = Empty(conditional->then_case);
    bool empty_else =
        !conditional->else_case || Empty(conditional->else_case.value());
    if (empty_then && empty_else)
      return Effects({conditional->condition});
    if (empty_else)
      return If(conditional->condition, conditional->then_case, std::nullopt,
                conditional->loc);
    if (empty_then)
      return If(!conditional->condition, conditional->else_case.value(),
                std::nullopt, conditional->loc);
    return result;
  }

  Stmt VisitStmt_(const ForNode *node) final {
    known_.clear();
    PrimExpr enters = (node->step > 0 && node->begin < node->end) ||
                      (node->step < 0 && node->begin > node->end);
    if (analyzer_->CanProve(!enters))
      return Effects({node->begin, node->end, node->step});
    Stmt result = Parent::VisitStmt_(node);
    known_.clear();
    auto loop = result.as<ForNode>();
    if (loop && Empty(loop->body) &&
        SideEffect(loop->begin) == CallEffectKind::kPure &&
        SideEffect(loop->end) == CallEffectKind::kPure &&
        SideEffect(loop->step) == CallEffectKind::kPure) {
      return SeqStmt({});
    }
    return result;
  }

  Stmt VisitStmt_(const WhileNode *node) final {
    known_.clear();
    Stmt result = Parent::VisitStmt_(node);
    known_.clear();
    return result;
  }

  Stmt VisitStmt_(const TensorStoreNode *node) final {
    auto store = Parent::VisitStmt_(node).as_or_throw<TensorStore>();
    TensorLoad destination(store->dest, store->indices);
    bool pure_indices = true;
    for (const PrimExpr &index : store->indices)
      pure_indices &= SideEffect(index) == CallEffectKind::kPure;
    bool redundant =
        pure_indices && ffi::StructuralEqual()(destination, store->value);
    auto previous = known_.find(destination);
    if (pure_indices && dataflow_ && previous != known_.end()) {
      redundant |= ffi::StructuralEqual()(previous->second, store->value);
    }
    if (redundant)
      return SeqStmt({});
    known_.clear();
    if (dataflow_ && pure_indices &&
        SideEffect(store->value) == CallEffectKind::kPure) {
      known_.emplace(destination, store->value);
    }
    return store;
  }

  PrimExpr VisitExpr_(const TensorLoadNode *node) final {
    PrimExpr result = Parent::VisitExpr_(node);
    if (dataflow_) {
      if (auto found = known_.find(result); found != known_.end())
        return found->second;
    }
    return result;
  }

  Expr VisitExpr_(const CallNode *node) final {
    if (SideEffect(ffi::GetRef<Expr>(node)) > CallEffectKind::kReadState)
      known_.clear();
    return Parent::VisitExpr_(node);
  }

  bool dataflow_;
  VarSet dead_;
  std::unordered_map<PrimExpr, PrimExpr, ffi::StructuralHash,
                     ffi::StructuralEqual>
      known_;
};

} // namespace

tvm::transform::Pass AscendRemoveNoOp() {
  auto pass =
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> ffi::Optional<tirx::Function> {
    if (!function->body)
      return function;
    auto config = context->GetConfigOrDefault<AscendRemoveNoOpConfig>(
        "tl.AscendRemoveNoOp");
    sym::Analyzer analyzer;
    analyzer->rewrite_simplify.SetMaximumRewriteSteps(
        config->max_simplification_steps);
    auto collector = ffi::make_object<AllocationUsageCollector>();
    collector->Visit(function->body.value());
    auto rewriter = ffi::make_object<AscendNoOpRemover>(
        analyzer, config->use_dataflow_analysis, collector->Dead());
    function.CopyOnWrite()->body = rewriter->VisitStmt(function->body.value());
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.AscendRemoveNoOp");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  AscendRemoveNoOpConfigNode::RegisterReflection();
  tvm::transform::PassContext::RegisterConfigOption<AscendRemoveNoOpConfig>(
      "tl.AscendRemoveNoOp");
  ffi::reflection::GlobalDef().def("tl.transform.AscendRemoveNoOp",
                                   AscendRemoveNoOp);
}
} // namespace tvm::tl
