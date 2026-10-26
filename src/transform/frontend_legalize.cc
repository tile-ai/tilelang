#include "compat/metadata.h"
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

#include "compat/visit.h"

#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

bool ContainsVariable(const ffi::ObjectRef &root, const Var &variable) {
  bool found = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      root, [&](const Var &candidate) -> ffi::Expected<ffi::WalkResult> {
        found |= candidate.same_as(variable);
        return ffi::WalkResult::Advance();
      });
  return found;
}

bool IsReplayableRead(const PrimExpr &expression) {
  if (SideEffect(expression) > CallEffectKind::kReadState)
    return false;
  bool replayable = true;
  PostOrderVisit(expression, [&](const ffi::ObjectRef &object) {
    if (auto call = object.as<CallNode>()) {
      replayable &=
          Op::GetAttrMap(tvm::op_attr::kCallEffectKind)
              .get(call->op, CallEffectKind::kOpaque) == CallEffectKind::kPure;
    }
  });
  return replayable;
}

class LetInliner : public tvm::tl::StmtExprMutator {
private:
  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node,
                                 ffi::InplaceMode mode) final {
    ffi::Array<Stmt> statements;
    for (size_t index = 0; index < node->seq.size(); ++index) {
      bool previous = allow_read_binding_;
      allow_read_binding_ = false;
      if (auto binding = node->seq[index].as<BindNode>();
          binding && index + 1 < node->seq.size()) {
        if (auto store = node->seq[index + 1].as<TensorStoreNode>()) {
          allow_read_binding_ = IsReplayableRead(store->value) &&
                                ContainsVariable(store->value, binding->var);
          for (const PrimExpr &coordinate : store->indices) {
            allow_read_binding_ &=
                SideEffect(coordinate) <= CallEffectKind::kPure;
          }
          for (size_t later = index + 2; later < node->seq.size(); ++later) {
            allow_read_binding_ &=
                !ContainsVariable(node->seq[later], binding->var);
          }
        }
      }
      Stmt statement =
          Mutate(node->seq[index], mode).ValueOrUnchanged(node->seq[index]);
      allow_read_binding_ = previous;
      if (auto sequence = statement.as<SeqStmtNode>();
          sequence && sequence->seq.empty())
        continue;
      statements.push_back(statement);
    }
    return SeqStmt(statements, node->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    Expr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    auto scalar = value.as<PrimExpr>();
    if (node->var.as<PrimVar>() && scalar &&
        (SideEffect(scalar.value()) <= CallEffectKind::kPure ||
         (allow_read_binding_ && IsReplayableRead(scalar.value())))) {
      VarRemapSet(node->var, value);
      return SeqStmt(ffi::Array<Stmt>{}, node->loc);
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LetNode *node,
                                     ffi::InplaceMode mode) final {
    PrimExpr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    if (SideEffect(value) > CallEffectKind::kPure &&
        !(IsReplayableRead(value) && IsReplayableRead(node->body))) {
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    }
    ffi::Any previous = VarRemapGet(node->var);
    VarRemapSet(node->var, value);
    try {
      PrimExpr body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
      VarRemapSet(node->var,
                  previous == nullptr ? ffi::Any(node->var) : previous);
      return body;
    } catch (...) {
      VarRemapSet(node->var,
                  previous == nullptr ? ffi::Any(node->var) : previous);
      throw;
    }
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                 ffi::InplaceMode mode) final {
    Call call = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(node))
                    .as_or_throw<Call>();
    if (call->attrs) {
      auto attrs = Mutate(call->attrs.value(), mode)
                       .as_or_throw<ffi::UnchangedOr<Attrs>>();
      if (!attrs.UnchangedOrSameAs(call->attrs.value())) {
        call.CopyOnWrite()->attrs = std::move(attrs).ValueUnchecked();
      }
    }
    return call;
  }

  bool allow_read_binding_{false};
};

} // namespace

tvm::transform::Pass LetInline() {
  auto transform = [](tirx::Function function, const IRModule &,
                      const tvm::transform::PassContext &) {
    if (function->body) {
      function.CopyOnWrite()->body =
          ffi::make_object<LetInliner>()
              ->Mutate(function->body, ffi::InplaceMode::kDisallow)
              .ValueOrUnchanged(function->body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(transform, 0, "tl.LetInline");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LetInline", LetInline);
}

} // namespace tvm::tl
