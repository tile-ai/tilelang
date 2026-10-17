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

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "arith/const_fold.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/control_flow_graph.h"
#include "tirx/transform/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

struct AscendRemoveNoOpConfigNode
    : public AttrsNodeReflAdapter<AscendRemoveNoOpConfigNode> {
  bool use_dataflow_analysis;
  int64_t max_simplification_steps;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AscendRemoveNoOpConfigNode>()
        .def_ro("use_dataflow_analysis",
                &AscendRemoveNoOpConfigNode::use_dataflow_analysis,
                "If true, known buffer values are propagated and used to "
                "statically prove statements as no-ops.",
                refl::DefaultValue(false))
        .def_ro("max_simplification_steps",
                &AscendRemoveNoOpConfigNode::max_simplification_steps,
                "If non-zero, RewriteSimplifier throws an error after this "
                "many steps. Intended for debugging and testing.",
                refl::DefaultValue(0));
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.transform.AscendRemoveNoOpConfig",
                                    AscendRemoveNoOpConfigNode, BaseAttrsNode);
};

class AscendRemoveNoOpConfig : public Attrs {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AscendRemoveNoOpConfig, Attrs,
                                                AscendRemoveNoOpConfigNode);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  AscendRemoveNoOpConfigNode::RegisterReflection();
}

TVM_REGISTER_PASS_CONFIG_OPTION("tl.AscendRemoveNoOp", AscendRemoveNoOpConfig);

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

// Track storage Vars that are allocated and actually used. Allocation and
// declaration metadata do not by themselves make the storage live.
class AllocationUsageCollector : public StmtExprVisitor {
public:
  static VarSet FindDeadAllocations(const Stmt &stmt) {
    AllocationUsageCollector collector;
    collector(stmt);

    VarSet dead;
    for (const Var &var : collector.allocated_) {
      if (!collector.used_.count(var)) {
        dead.insert(var);
      }
    }
    return dead;
  }

private:
  void VisitStmt_(const AllocBufferNode *op) final {
    allocated_.insert(op->buffer->data);
  }

  void VisitStmt_(const DeclBufferNode *) final {}

  void VisitExpr_(const VarNode *op) final {
    used_.insert(ffi::GetRef<Var>(op));
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    used_.insert(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    used_.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() == 5) {
      if (const auto *var = op->args[1].as<VarNode>()) {
        used_.insert(ffi::GetRef<Var>(var));
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  VarSet allocated_;
  VarSet used_;
};

class AscendNoOpRemover : public arith::IRMutatorWithAnalyzer {
public:
  static Stmt Apply(Stmt stmt, arith::Analyzer *analyzer,
                    std::optional<ControlFlowGraph> touch_pattern,
                    const StmtNode *context) {
    VarSet dead_allocations =
        AllocationUsageCollector::FindDeadAllocations(stmt);
    AscendNoOpRemover visitor(analyzer, std::move(touch_pattern), context,
                              std::move(dead_allocations));
    return visitor(std::move(stmt));
  }

private:
  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  AscendNoOpRemover(arith::Analyzer *analyzer,
                    std::optional<ControlFlowGraph> touch_pattern,
                    const StmtNode *context, VarSet dead_allocations)
      : Parent(analyzer), touch_pattern_(std::move(touch_pattern)),
        context_(context), dead_allocations_(std::move(dead_allocations)) {}

  Stmt VisitStmt_(const BindNode *op) final {
    // Unused Bind elimination requires a separate two-pass analysis.
    return Parent::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "pragma_debug_skip_region") {
      return MakeEvaluate(0);
    }
    if (op->attr_key == s_tir::attr::async_wait_queue_scope) {
      std::pair<PrimExpr, PrimExpr> wait_attrs = GetAsyncWaitAttributes(op);
      PrimExpr wait_count = wait_attrs.second;
      arith::Analyzer analyzer;
      if (analyzer.CanProve(wait_count < 0)) {
        // Unrolling can make a loop-dependent wait count negative. Such waits
        // are no-ops.
        const auto *inner = op->body.as<AttrStmtNode>();
        TVM_FFI_ICHECK(inner);
        return Parent::VisitStmt(inner->body);
      }
    }

    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<AttrStmtNode>();
    return is_no_op(op->body) ? MakeEvaluate(op->value) : stmt;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<IfThenElseNode>();
    if (!op) {
      return stmt;
    }

    if (op->else_case.defined()) {
      bool no_op_else = is_no_op(op->else_case.value());
      bool no_op_then = is_no_op(op->then_case);
      if (no_op_else && no_op_then) {
        return MakeEvaluate(op->condition);
      }
      if (no_op_else) {
        return IfThenElse(op->condition, op->then_case);
      }
      if (no_op_then) {
        return IfThenElse(!op->condition, op->else_case.value());
      }
      return stmt;
    }

    return is_no_op(op->then_case) ? MakeEvaluate(op->condition) : stmt;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    arith::IntSet extent_range = arith::EvalSet(op->extent, var_range_map_);
    if (!arith::is_neg_inf(extent_range.max()) &&
        !arith::is_pos_inf(extent_range.max()) &&
        analyzer_->CanProve(extent_range.max() <= 0)) {
      return Evaluate(0);
    }

    var_range_map_[op->loop_var.get()] =
        arith::IntSet::FromMinExtent(op->min, op->extent);
    Stmt stmt = Parent::VisitStmt_(op);
    var_range_map_.erase(op->loop_var.get());
    op = stmt.as<ForNode>();
    if (is_zero(op->extent)) {
      return Evaluate(0);
    }
    return is_no_op(op->body) ? MakeEvaluate({op->min, op->extent}) : stmt;
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    if (dead_allocations_.count(op->buffer->data)) {
      return Evaluate(0);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    return HasSideEffect(op->value) ? ffi::GetRef<Stmt>(op) : Evaluate(0);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = ffi::GetRef<BufferStore>(op);

    auto only_side_effects = [&]() {
      ffi::Array<Stmt> statements;
      statements.push_back(MakeEvaluate(store->value));
      for (const PrimExpr &index : store->indices) {
        statements.push_back(MakeEvaluate(index));
      }
      return VisitStmt(SeqStmt(statements));
    };

    if (touch_pattern_.has_value()) {
      Stmt context = context_ ? ffi::GetRef<Stmt>(context_) : store;
      if (touch_pattern_->IsOverwrittenWithoutEffect(store, context)) {
        touch_pattern_->RemoveStore(store);
        return only_side_effects();
      }
    }

    PrimExpr stores_existing_value =
        store->value ==
        BufferLoad(store->buffer, store->indices, store->predicate);
    if (touch_pattern_.has_value()) {
      Stmt context = context_ ? ffi::GetRef<Stmt>(context_) : Stmt(store);
      stores_existing_value = touch_pattern_->SimplifyInContext(
          stores_existing_value, context, analyzer_);
    } else {
      stores_existing_value = analyzer_->Simplify(stores_existing_value);
    }
    if (is_one(stores_existing_value)) {
      return only_side_effects();
    }

    if (const auto *load = store->value.as<BufferLoadNode>()) {
      if (load->buffer->data.same_as(store->buffer->data) &&
          analyzer_->CanProveEqual(load->buffer->elem_offset,
                                   store->buffer->elem_offset) &&
          ArrayValueEqual(load->buffer->shape, store->buffer->shape) &&
          ArrayValueEqual(load->buffer->strides, store->buffer->strides) &&
          ArrayValueEqual(load->indices, store->indices)) {
        return only_side_effects();
      }
    }

    return store;
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    if (dead_allocations_.count(op->buffer->data)) {
      return Evaluate(0);
    }
    return StmtMutator::VisitStmt_(op);
  }

  bool ArrayValueEqual(const ffi::Array<PrimExpr> &lhs,
                       const ffi::Array<PrimExpr> &rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!analyzer_->CanProveEqual(lhs[i], rhs[i])) {
        return false;
      }
    }
    return true;
  }

  bool HasSideEffect(const PrimExpr &value) const {
    return SideEffect(value) > CallEffectKind::kReadState;
  }

  Stmt MakeEvaluate(PrimExpr value) {
    return HasSideEffect(value) ? Evaluate(value) : Evaluate(0);
  }

  Stmt MakeEvaluate(const ffi::Array<PrimExpr> &values) {
    ffi::Array<Stmt> statements;
    for (const PrimExpr &value : values) {
      if (HasSideEffect(value)) {
        statements.push_back(Evaluate(value));
      }
    }

    if (statements.empty()) {
      return Evaluate(0);
    }
    if (statements.size() == 1) {
      return statements[0];
    }
    return SeqStmt(statements);
  }

  std::unordered_map<const VarNode *, arith::IntSet> var_range_map_;
  std::optional<ControlFlowGraph> touch_pattern_;
  const StmtNode *context_;
  VarSet dead_allocations_;
};

tvm::transform::Pass AscendRemoveNoOp() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const PassContext &ctx) {
    std::optional<ControlFlowGraph> touch_pattern;
    AscendRemoveNoOpConfig config =
        ctx->GetConfig<AscendRemoveNoOpConfig>("tl.AscendRemoveNoOp")
            .value_or(AttrsWithDefaultValues<AscendRemoveNoOpConfig>());

    if (config->use_dataflow_analysis) {
      touch_pattern.emplace(func->body, config->max_simplification_steps);
    }

    arith::Analyzer analyzer;
    analyzer.rewrite_simplify.SetMaximumRewriteSteps(
        config->max_simplification_steps);
    PrimFuncNode *write_ptr = func.CopyOnWrite();
    write_ptr->body =
        AscendNoOpRemover::Apply(std::move(write_ptr->body), &analyzer,
                                 std::move(touch_pattern), nullptr);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendRemoveNoOp", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendRemoveNoOp", AscendRemoveNoOp);
}

} // namespace tl
} // namespace tvm
