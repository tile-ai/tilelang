/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file frontend_legalize.cc
 * \brief Legalize the program from frontend
 */

#include "support/check.h"
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

/*!
 * \brief Whether \p value may be substituted at every use site.
 *
 * A let-bound value is duplicated and can move into a branch or a loop body.
 * Calls that are not `kPure` have to stay where they are: a warp shuffle,
 * ballot or `activemask` is convergent, so moving it under divergent control
 * flow changes which lanes participate, and duplicating a side effect changes
 * how often it runs. `launch_thread_idx` is opaque but reads a per-thread
 * constant, so it is safe to duplicate.
 */
bool IsSafeToInline(const PrimExpr &value) {
  static const auto effect_map =
      Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
  bool safe = true;
  PostOrderVisit(value, [&](const ffi::ObjectRef &node) {
    if (!safe) {
      return;
    }
    const auto *call = node.as<CallNode>();
    if (call == nullptr) {
      return;
    }
    auto opt_op = call->op.as<Op>();
    if (!opt_op.has_value()) {
      safe = false;
      return;
    }
    const Op &call_op = opt_op.value();
    if (call_op.same_as(tl::launch_thread_idx())) {
      return;
    }
    if (!effect_map.count(call_op) ||
        effect_map[call_op]->value > static_cast<int>(CallEffectKind::kPure)) {
      safe = false;
    }
  });
  return safe;
}

} // namespace

class LetInliner : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LetInliner substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const ForNode *node) final {
    if (node->kind == ForKind::kParallel) {
      parallel_for_scope_++;
    }
    auto n = StmtExprMutator::VisitStmt_(node);
    if (node->kind == ForKind::kParallel) {
      parallel_for_scope_--;
    }
    return n;
  }

  PrimExpr VisitExpr_(const VarNode *node) final {
    if (let_bindings_.count(node)) {
      return arith::IRMutatorWithAnalyzer::VisitExpr(let_bindings_[node]);
    } else {
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    }
  }

  Stmt VisitStmt_(const BindNode *node) final {
    PrimExpr value = VisitExpr(node->value);
    if (!IsSafeToInline(value)) {
      // Keep the binding: inlining would duplicate the value and move it to
      // every use, which is wrong for convergent or side-effecting calls.
      return Bind(node->var, value, node->span);
    }
    let_bindings_[node->var.get()] = std::move(value);
    return Evaluate(Integer(0));
  }

  PrimExpr VisitExpr_(const LetNode *node) final {
    PrimExpr value = VisitExpr(node->value);
    if (!IsSafeToInline(value)) {
      return Let(node->var, value, VisitExpr(node->body), node->span);
    }
    let_bindings_[node->var.get()] = std::move(value);
    return VisitExpr(node->body);
  }

  int parallel_for_scope_ = 0;
  std::unordered_map<const VarNode *, PrimExpr> let_bindings_;
};

using namespace tirx::transform;

Pass LetInline() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LetInliner::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LetInline", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LetInline", LetInline);
}

} // namespace tl
} // namespace tvm
