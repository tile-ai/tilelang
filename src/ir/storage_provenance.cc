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

#include "storage_provenance.h"
#include "tensor_utils.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl {

std::optional<runtime::MemoryScope> StorageScopeOf(const Type &type) {
  if (const auto *tensor = type.as<TensorTypeObj>())
    return tensor->scope;
  if (const auto *pointer = type.as<PtrTypeObj>())
    return pointer->scope;
  return std::nullopt;
}

bool ProvenDisjointStorage(const StorageAddress &a, const StorageAddress &b) {
  if (!a.exact || !b.exact || a.root.same_as(b.root))
    return false;
  return (a.parameter_alias_contract &&
          a.parameter_alias_contract == b.parameter_alias_contract) ||
         (a.fresh_allocation && b.fresh_allocation);
}

StorageProvenance::StorageProvenance(ScopeCompatibility compatible,
                                     InitializerResolver initializer)
    : compatible_(std::move(compatible)), initializer_(std::move(initializer)) {
  if (!compatible_)
    compatible_ = [](const Type &a, const Type &b) {
      auto lhs = StorageScopeOf(a), rhs = StorageScopeOf(b);
      return lhs && rhs && *lhs == *rhs;
    };
}

void StorageProvenance::SetParameterAliasContract(
    ParameterAliasContract contract) {
  parameter_alias_contract_ = std::move(contract);
}

void StorageProvenance::Collect(const Stmt &body) {
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      body, [&](const BindObj *bind) -> ffi::Expected<ffi::WalkResult> {
        bindings.Set(bind->var,
                     initializer_ ? initializer_(bind->value) : bind->value);
        return ffi::WalkResult::Advance();
      });
}

StorageAddress StorageProvenance::Address(const Expr &value) const {
  std::unordered_set<const VarObj *> active;
  auto result = Resolve(value, &active);
  if (result.exact && parameter_alias_contract_)
    if (auto root = result.root.as<Var>();
        root && parameter_alias_contract_->count(*root))
      result.parameter_alias_contract = parameter_alias_contract_;
  return result;
}

StorageProvenance StorageProvenance::Substitute(const Var &old_var,
                                                const Var &new_var) const {
  StorageProvenance result(compatible_, initializer_);
  result.parameter_alias_contract_ = parameter_alias_contract_;
  auto mutator = ffi::make_object<tvm::ExprMutator>();
  mutator->VarRemapSet(old_var, new_var);
  for (const auto &[variable, value] : bindings) {
    result.bindings.Set(variable.same_as(old_var) ? new_var : variable,
                        mutator->Mutate(value).ValueOrUnchanged(value));
  }
  return result;
}

Var StorageProvenance::Identity(const TensorVar &tensor) const {
  auto address = Address(tensor);
  if (auto variable = address.root.as<Var>())
    return variable.value();
  // No allocation identity can be proved for this pointer-producing call.
  // Keep a stable name for diagnostics; Exact(tensor) remains false.
  return tensor;
}

bool StorageProvenance::Exact(const TensorVar &tensor) const {
  return Address(tensor).exact;
}

bool StorageProvenance::SameStorage(const TensorVar &a,
                                    const TensorVar &b) const {
  auto lhs = Address(a), rhs = Address(b);
  return lhs.exact && rhs.exact && lhs.root.same_as(rhs.root);
}

PrimExpr
StorageProvenance::ElementBitOffset(const TensorVar &tensor,
                                    const ffi::Array<PrimExpr> &indices) const {
  auto address = Address(tensor);
  TVM_FFI_CHECK(address.exact, ValueError)
      << "Cannot derive a physical footprint for an opaque pointer "
      << address.root;
  int bits = tensor.dtype().bits() * tensor.dtype().lanes();
  TVM_FFI_CHECK_GT(bits, 0, ValueError);
  return cast(address.byte_offset, PrimType::Int(64)) * IntImm::Int64(8) +
         cast(TensorElementOffset(tensor, indices), PrimType::Int(64)) *
             IntImm::Int64(bits);
}

StorageAddress StorageProvenance::PreserveScope(StorageAddress address,
                                                const Type &type) const {
  address.exact &= compatible_(address.root->ty, type);
  return address;
}

StorageAddress
StorageProvenance::Resolve(const Expr &value,
                           std::unordered_set<const VarObj *> *active) const {
  if (auto variable = value.as<Var>()) {
    auto binding = bindings.Get(variable.value());
    if (!binding) {
      const bool address_root =
          value->ty.as<PtrTypeObj>() || value->ty.as<TensorTypeObj>();
      return {value, IntImm::Int64(0), address_root, false};
    }
    TVM_FFI_CHECK(active->insert(variable.value().get()).second, ValueError)
        << "Cyclic storage definition for " << variable.value();
    auto result = [&]() -> StorageAddress {
      if (auto tensor = variable->as<TensorVar>()) {
        if (binding.value().as<TensorVar>())
          return Resolve(binding.value(), active);
        const auto *call = binding.value().as<CallObj>();
        if (call && call->op.same_as(tirx::alloc_tensor_op()))
          return PreserveScope({value, IntImm::Int64(0), true, true}, call->ty);
        if (call && call->op.same_as(tirx::decl_tensor_op()))
          return Resolve(call->args[0], active);
        return {value, IntImm::Int64(0), false};
      }
      if (value->ty.as<PtrTypeObj>())
        return Resolve(binding.value(), active);
      return {value, IntImm::Int64(0), false};
    }();
    active->erase(variable.value().get());
    return PreserveScope(std::move(result), value->ty);
  }
  if (const auto *call = value.as<CallObj>()) {
    if (call->op.same_as(tirx::tensor_data_ptr_op()) ||
        (call->op.same_as(tirx::reinterpret_op()) && call->ty.as<PtrTypeObj>()))
      return PreserveScope(Resolve(call->args[0], active), call->ty);
    if (call->op.same_as(tirx::ptr_byte_offset_op())) {
      auto address = Resolve(call->args[0], active);
      PrimExpr offset = call->args[1].as_or_throw<PrimExpr>();
      address.byte_offset =
          address.byte_offset + cast(offset, PrimType::Int(64));
      address.exact &= SideEffect(offset) <= CallEffectKind::kPure;
      return PreserveScope(std::move(address), call->ty);
    }
  }
  return {value, IntImm::Int64(0), false};
}
ParameterAliasContract
CollectParameterAliasContract(const tirx::Function &function,
                              const StorageProvenance &storage,
                              const ffi::Array<Expr> &additional_exceptions) {
  if (!function->GetAttr<bool>("tirx.noalias").value_or(false))
    return {};
  auto restricted = std::make_shared<RestrictedParameters>();
  for (const Var &parameter : function->params) {
    auto address = storage.Address(parameter);
    if (address.exact)
      if (auto root = address.root.as<Var>())
        restricted->insert(*root);
  }
  bool unknown = false;
  auto exclude = [&](const ffi::Any &values) {
    for (const ffi::Any &value : values.cast<ffi::Array<ffi::Any>>()) {
      auto address = storage.Address(value.cast<Expr>());
      auto root = address.root.as<Var>();
      if (address.exact && root)
        restricted->erase(*root);
      else
        unknown = true;
    }
  };
  if (auto values =
          function->GetAttr<ffi::Array<ffi::Any>>("tl.non_restrict_params"))
    exclude(*values);
  if (auto indices =
          function->GetAttr<ffi::Array<int64_t>>("tl.may_alias_param_indices"))
    for (int64_t index : *indices) {
      TVM_FFI_CHECK(index >= 0 &&
                        static_cast<size_t>(index) < function->params.size(),
                    ValueError)
          << "Invalid may-alias parameter index " << index;
      exclude(ffi::Array<ffi::Any>{function->params[index]});
    }
  for (const Expr &exception : additional_exceptions)
    exclude(ffi::Array<ffi::Any>{exception});
  return unknown ? ParameterAliasContract{} : std::move(restricted);
}

} // namespace tvm::tl
