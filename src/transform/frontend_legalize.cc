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
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

class LetInliner : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    Expr value = Mutate(op->value, mode).ValueOrUnchanged(op->value);
    // A tensor declaration/allocation supplies a stable typed identity to the
    // layout and tile-op passes. A memory read or effect must keep its original
    // evaluation point. Only pure scalar expressions are safe to duplicate.
    if (value.as<PrimExpr>() && SideEffect(value) <= CallEffectKind::kPure) {
      VarRemapSet(op->var, value);
      return SeqStmt(ffi::Array<Stmt>{}, op->loc);
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LetObj *op,
                                     InplaceMode mode) final {
    PrimExpr value = Mutate(op->value, mode).ValueOrUnchanged(op->value);
    if (SideEffect(value) > CallEffectKind::kPure) {
      return StmtExprMutator::Mutate_(op, mode);
    }
    VarRemapSet(op->var, value);
    return Mutate(op->body, mode).ValueOrUnchanged(op->body);
  }
};

} // namespace

tvm::transform::Pass LetInline() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    auto inliner = ffi::make_object<LetInliner>();
    SeqStmt body =
        inliner->Mutate(function->body).ValueOrUnchanged(function->body);
    if (!body.same_as(function->body))
      function.CopyOnWrite()->body = body;
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LetInline");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LetInline", LetInline);
}

} // namespace tvm::tl
