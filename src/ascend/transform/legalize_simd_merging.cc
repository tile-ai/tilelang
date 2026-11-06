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
 * \brief Make the inactive-lane destination of SIMD merging explicit.
 */
#include "ir/annotations.h"
#include "op/builtin.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

bool IsSimdMergingCall(const CallObj *call) {
  const auto *op = call->op.as<OpObj>();
  if (!op || !std::string(op->name).starts_with("tl.simd."))
    return false;
  auto args = DecodeCallOperands(call);
  const auto *mode = args.empty() ? nullptr : args.back().as<StringImmObj>();
  return mode && mode->value == "MODE_MERGING";
}

class SimdMergingVerifier : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  Optional<VisitInterrupt> Visit_(const CallObj *call) final {
    if (IsSimdMergingCall(call)) {
      const auto result = call->ty.as<PrimType>();
      // Preserve the public diagnostic class for an unassigned merging call.
      TVM_FFI_CHECK(result && result->IsVoid(), RuntimeError)
          << "MODE_MERGING is a read-modify-write operation and requires "
             "assignment to a mutable SIMD register, for example "
             "`dst[i] = T.simd.vadds(...)`.";
      auto args = DecodeCallOperands(call);
      const auto *destination = args[0].as<CallObj>();
      TVM_FFI_CHECK(destination && destination->op.same_as(tl::access_ptr()) &&
                        destination->args.size() == 3 &&
                        destination->args[0].as<TensorLoadObj>(),
                    ValueError)
          << "Legalized MODE_MERGING requires an explicit tl.access_ptr "
             "destination";
      const auto *extent = destination->args[1].as<IntImmObj>();
      const auto *mask = destination->args[2].as<IntImmObj>();
      TVM_FFI_CHECK(extent && extent->value == 1 && mask && mask->value == 3,
                    ValueError)
          << "MODE_MERGING destination must be one read-write SIMD register";
    }
    return StmtExprVisitor::Visit_(call);
  }
};

class SimdMergingLegalizer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op, InplaceMode mode) final {
    auto original = GetRef<Stmt>(op);
    auto store = StmtExprMutator::Mutate_(op, mode)
                     .ValueOrUnchanged(original)
                     .as_or_throw<TensorStore>();
    const auto *call = store->value.as<CallObj>();
    if (!call || !IsSimdMergingCall(call))
      return store;
    auto type = store->dest->ty.as_or_throw<TensorType>();
    TensorLoad load(store->dest, store->indices, store->loc);
    auto index_type =
        store->indices.empty() ? PrimType::Int(32) : store->indices[0].ty();
    Call destination(
        type->DataPtrType(), tl::access_ptr(),
        {load, IntImm(index_type, 1), IntImm(PrimType::Int(32), 3)}, {}, {},
        store->loc);
    // Preserve the final annotation Tuple and every static attribute. Only
    // the runtime operand count changes; return type arguments must become
    // void along with Call.ty rather than retaining the old vector result.
    auto args = call->args;
    args.insert(args.begin(), destination);
    auto attrs = call->attrs;
    if (attrs && attrs.value().as<DictAttrsObj>()) {
      auto dict = attrs.value().as_or_throw<DictAttrs>();
      if (auto count = dict.GetAttr<int64_t>("operand_count")) {
        auto fields = dict->dict;
        fields.Set("operand_count", *count + 1);
        attrs = DictAttrs(std::move(fields));
      }
    }
    return Evaluate(Call(PrimType::Void(), call->op, std::move(args),
                         std::move(attrs), {}, call->loc),
                    store->loc);
  }
};
} // namespace

tvm::transform::Pass LegalizeSimdMerging() {
  auto pass = [](tirx::Function function, const IRModule &,
                 const tvm::transform::PassContext &) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto rewriter = ffi::make_object<SimdMergingLegalizer>();
    auto body =
        rewriter->Mutate(function->body).ValueOrUnchanged(function->body);
    ffi::make_object<SimdMergingVerifier>()->Visit(body);
    function.CopyOnWrite()->body = std::move(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LegalizeSimdMerging");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LegalizeSimdMerging",
                                   LegalizeSimdMerging);
}
} // namespace tvm::tl
