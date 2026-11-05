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

#include "tensor_descriptor.h"

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include <unordered_map>

namespace tvm::tl {

const Op &tensor_descriptor_ptr_op() {
  static const Op op = Op::Get("tl.tensor_descriptor_ptr");
  return op;
}

const Op &input_tensor_descriptor_op() {
  static const Op op = Op::Get("tl.input_tensor_descriptor");
  return op;
}

int InputTensorDescriptorIndex(const CallObj *call,
                               const ffi::Array<Var> &params) {
  TVM_FFI_ICHECK(call->op.same_as(input_tensor_descriptor_op()));
  auto index = call->args[0].as_or_throw<IntImm>()->value.as<int64_t>();
  TVM_FFI_CHECK(index && *index >= 0 &&
                    static_cast<uint64_t>(*index) < params.size(),
                ValueError)
      << "tl.input_tensor_descriptor requires a valid packed argument index";
  TVM_FFI_CHECK(params[*index].as<tirx::TensorVar>().has_value(), TypeError)
      << "tl.input_tensor_descriptor requires a tensor parameter at index "
      << *index;
  return static_cast<int>(*index);
}

namespace {

class DescriptorParameterLowerer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit DescriptorParameterLowerer(const ffi::Array<Var> &params)
      : params_(params) {
    for (size_t i = 0; i < params.size(); ++i)
      indices_.emplace(params[i].get(), static_cast<int>(i));
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    if (call->op.same_as(input_tensor_descriptor_op())) {
      InputTensorDescriptorIndex(call, params_);
      return ffi::Unchanged();
    }
    if (!call->op.same_as(tensor_descriptor_ptr_op()))
      return StmtExprMutator::Mutate_(call, mode);
    auto tensor = call->args[0].as_or_throw<tirx::TensorVar>();
    auto found = indices_.find(tensor.get());
    TVM_FFI_CHECK(found != indices_.end(), ValueError)
        << "tl.tensor_descriptor_ptr requires an original packed tensor "
           "parameter; local allocations and views have no input descriptor: "
        << tensor->name;
    return Call(PtrType::VoidPtrType(), input_tensor_descriptor_op(),
                {IntImm::Int32(found->second)}, {}, {}, call->loc);
  }

private:
  ffi::Array<Var> params_;
  std::unordered_map<const VarObj *, int> indices_;
};

class DescriptorIndexRemapper : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  DescriptorIndexRemapper(const ffi::Array<Var> &old_params,
                          const ffi::Array<Var> &new_params)
      : old_params_(old_params) {
    for (size_t i = 0; i < new_params.size(); ++i)
      new_indices_.emplace(new_params[i].get(), static_cast<int>(i));
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    if (!call->op.same_as(input_tensor_descriptor_op()))
      return StmtExprMutator::Mutate_(call, mode);
    int old_index = InputTensorDescriptorIndex(call, old_params_);
    auto found = new_indices_.find(old_params_[old_index].get());
    TVM_FFI_CHECK(found != new_indices_.end(), ValueError)
        << "Cannot remove a parameter whose input descriptor is used";
    if (old_index == found->second)
      return ffi::Unchanged();
    return Call(call->ty, call->op, {IntImm::Int32(found->second)}, {}, {},
                call->loc);
  }

private:
  ffi::Array<Var> old_params_;
  std::unordered_map<const VarObj *, int> new_indices_;
};

} // namespace

tirx::Function LowerTensorDescriptorParameters(tirx::Function function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  auto body = ffi::make_object<DescriptorParameterLowerer>(function->params)
                  ->Mutate(function->body)
                  .ValueOrUnchanged(function->body);
  if (!body.same_as(function->body))
    function.CopyOnWrite()->body = body;
  return function;
}

void CollectInputTensorDescriptorUses(
    const SeqStmt &body, const ffi::Array<Var> &params,
    std::unordered_set<const VarObj *> *uses) {
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        if (call->op.same_as(input_tensor_descriptor_op())) {
          int index = InputTensorDescriptorIndex(call, params);
          uses->insert(params[index].get());
        }
        return ffi::WalkResult::Advance();
      });
}

SeqStmt RemapInputTensorDescriptors(const tirx::Function &function,
                                    const ffi::Array<Var> &new_params) {
  return ffi::make_object<DescriptorIndexRemapper>(function->params, new_params)
      ->Mutate(function->body)
      .ValueOrUnchanged(function->body);
}

tvm::transform::Pass LowerTensorDescriptors() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        return LowerTensorDescriptorParameters(std::move(function));
      },
      0, "tl.LowerTensorDescriptors");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.tensor_descriptor_ptr")
      .signature(sig::arg<tirx::TensorVar>("tensor"))
      .set_attr(tvm::op_attr::kFixedReturnType, PtrType::VoidPtrType())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure);
  OpDef("tl.input_tensor_descriptor")
      .signature(sig::arg<IntImm>("parameter_index"))
      .set_attr(tvm::op_attr::kFixedReturnType, PtrType::VoidPtrType())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure);
  ffi::reflection::GlobalDef().def("tl.transform.LowerTensorDescriptors",
                                   LowerTensorDescriptors);
}

} // namespace tvm::tl
