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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <vector>

#include "arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tir;

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

  Stmt VisitStmt_(const LetStmtNode *node) final {
    VisitExpr(node->value);
    return Evaluate(0);
  }

  Stmt VisitStmt_(const SeqStmtNode *node) final {
    struct SavedBinding {
      const VarNode *var;
      bool had_prev;
      PrimExpr prev_value;
    };

    std::vector<SavedBinding> saved_bindings;
    ffi::Array<Stmt> seq;
    for (const Stmt &stmt : node->seq) {
      if (const auto *let = stmt.as<LetStmtNode>()) {
        PrimExpr value = VisitExpr(let->value);
        auto it = let_bindings_.find(let->var.get());
        SavedBinding saved{let->var.get(), it != let_bindings_.end(),
                           PrimExpr()};
        if (saved.had_prev) {
          saved.prev_value = it->second;
        }
        saved_bindings.push_back(saved);
        let_bindings_[let->var.get()] = value;
      } else {
        seq.push_back(VisitStmt(stmt));
      }
    }
    for (auto it = saved_bindings.rbegin(); it != saved_bindings.rend(); ++it) {
      if (it->had_prev) {
        let_bindings_[it->var] = it->prev_value;
      } else {
        let_bindings_.erase(it->var);
      }
    }
    return SeqStmt::Flatten(seq);
  }

  PrimExpr VisitExpr_(const LetNode *node) final {
    let_bindings_[node->var.get()] = node->value;
    return arith::IRMutatorWithAnalyzer::VisitExpr(node->body);
  }

  int parallel_for_scope_ = 0;
  std::unordered_map<const VarNode *, PrimExpr> let_bindings_;
};

using namespace tir::transform;

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
