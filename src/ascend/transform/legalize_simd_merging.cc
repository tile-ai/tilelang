#include "compat/metadata.h"
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
 * \file legalize_simd_merging.cc
 * \brief Legalize MODE_MERGING SIMD expressions into explicit read-modify-write
 *        statements.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "compat/expr.h"
#include "compat/visit.h"
#include "op/builtin.h"

namespace tvm::tl {
namespace {

bool IsSimdMergingCall(const CallNode *call) {
  if (call->args.empty())
    return false;
  auto mode = call->args.back().as<StringImmNode>();
  auto operation = call->op.as<OpNode>();
  return mode && mode->value == "MODE_MERGING" && operation &&
         std::string(operation->name).starts_with("tl.simd.");
}

class SimdMergingLegalizer : public tvm::tl::StmtExprMutator {
protected:
  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto store = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                     .ValueOrUnchanged(original)
                     .as_or_throw<TensorStore>();
    auto call = store->value.as<CallNode>();
    if (!call || !IsSimdMergingCall(call))
      return store;
    TensorLoad destination(store->dest, store->indices, store->loc);
    auto index_type =
        store->indices.empty() ? PrimType::Int(32) : store->indices[0].ty();
    Call pointer(std::nullopt, access_ptr(),
                 {destination, IntImm(index_type, 1), IntImm::Int32(3)});
    ffi::Array<Expr> arguments{pointer};
    arguments.insert(arguments.end(), call->args.begin(), call->args.end());
    return Evaluate(Call(PrimType::Void(), call->op, arguments, call->attrs,
                         call->ty_args, call->loc),
                    store->loc);
  }
};

void VerifySimdMerging(const Stmt &statement) {
  PostOrderVisit(statement, [](const ffi::ObjectRef &object) {
    auto call = object.as<CallNode>();
    if (!call || !IsSimdMergingCall(call))
      return;
    auto primitive = call->ty.as<PrimType>();
    auto tuple = call->ty.as<TupleTypeNode>();
    TVM_FFI_CHECK(primitive ? primitive->IsVoid()
                            : tuple && tuple->fields.empty(),
                  ValueError)
        << "MODE_MERGING requires assignment to a mutable SIMD register";
    auto pointer = call->args[0].as<CallNode>();
    TVM_FFI_CHECK(pointer && pointer->op.same_as(access_ptr()) &&
                      pointer->args.size() == 3,
                  ValueError)
        << "MODE_MERGING requires an explicit tl.access_ptr destination";
    TVM_FFI_CHECK(GetConstInt(pointer->args[2]) == 3, ValueError)
        << "MODE_MERGING destination must be marked read-write";
  });
}
} // namespace

tvm::transform::Pass LegalizeSimdMerging() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (!function->body)
      return function;
    auto rewriter = ffi::make_object<SimdMergingLegalizer>();
    SeqStmt original = function->body.value();
    function.CopyOnWrite()->body =
        rewriter->Mutate(original).ValueOrUnchanged(original);
    VerifySimdMerging(function->body.value());
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LegalizeSimdMerging");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LegalizeSimdMerging",
                                   LegalizeSimdMerging);
}
} // namespace tvm::tl
