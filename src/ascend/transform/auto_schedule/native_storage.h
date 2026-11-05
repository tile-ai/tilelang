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

#pragma once

#include "native_task_ir.h"
#include "transform/common/constr_visitor.h"

#include <tvm/ir/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/type.h>

#include <memory>
#include <unordered_set>

namespace tvm::tl::ascend {

using namespace prim;

using NativeRestrictedParameters =
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using NativeParameterAliasContract =
    std::shared_ptr<const NativeRestrictedParameters>;

// Storage scopes are extensible, and different names need not denote disjoint
// hardware. Keep the Ascend spaces explicit; static/dynamic allocation policy
// and scalar register spelling do not create a new address space.
enum class NativeStorageSpace {
  kUnknown,
  kGlobal,
  kLocal,
  kUB,
  kL1,
  kL0A,
  kL0B,
  kL0ASF,
  kL0BSF,
  kL0C,
  kSS,
};

inline NativeStorageSpace
NativeStorageSpaceOf(const runtime::MemoryScope &scope) {
  const auto &name = scope.name();
  if (name == "global")
    return NativeStorageSpace::kGlobal;
  if (name == "local" || name == "local.var")
    return NativeStorageSpace::kLocal;
  if (name == "shared" || name == "shared.dyn")
    return NativeStorageSpace::kUB;
  if (name == "shared.l1" || name == "shared.l1.dyn")
    return NativeStorageSpace::kL1;
  if (name == "shared.l0a" || name == "shared.l0a.dyn")
    return NativeStorageSpace::kL0A;
  if (name == "shared.l0b" || name == "shared.l0b.dyn")
    return NativeStorageSpace::kL0B;
  if (name == "shared.l0a.sf")
    return NativeStorageSpace::kL0ASF;
  if (name == "shared.l0b.sf")
    return NativeStorageSpace::kL0BSF;
  if (name == "shared.l0c" || name == "shared.l0c.dyn")
    return NativeStorageSpace::kL0C;
  if (name == "shared.ss" || name == "shared.ss.dyn")
    return NativeStorageSpace::kSS;
  return NativeStorageSpace::kUnknown;
}

inline std::optional<runtime::MemoryScope>
NativeStorageScopeOf(const Type &type) {
  if (const auto *tensor = type.as<TensorTypeObj>())
    return tensor->scope;
  if (const auto *pointer = type.as<PtrTypeObj>())
    return pointer->scope;
  return std::nullopt;
}

inline bool CompatibleNativeStorageScopes(const Type &a, const Type &b) {
  auto lhs = NativeStorageScopeOf(a), rhs = NativeStorageScopeOf(b);
  if (!lhs || !rhs)
    return false;
  if (*lhs == *rhs)
    return true;
  auto space = NativeStorageSpaceOf(*lhs);
  return space != NativeStorageSpace::kUnknown &&
         space == NativeStorageSpaceOf(*rhs);
}

// Analysis-only provenance of a native pointer. Offsets are in bytes, while
// tensor layouts remain in logical elements (including packed FP4 elements).
// `exact` means that the address is expressible relative to a pointer/tensor
// root in the same hardware space. It is not a noalias promise: distinct
// external roots may overlap. The root type retains the originating scope;
// a scope-changing cast cannot borrow the destination's storage provenance.
// Unknown pointer expressions retain their root expression and can never
// establish a disjointness or storage-reuse proof.
struct NativeStorageAddress {
  Expr root{ffi::UnsafeInit{}};
  PrimExpr byte_offset{IntImm::Int64(0)};
  bool exact{false};
  bool fresh_allocation{false};
  NativeParameterAliasContract parameter_alias_contract;
};

inline bool ProvenDisjointNativeStorage(const NativeStorageAddress &a,
                                        const NativeStorageAddress &b) {
  if (!a.exact || !b.exact || a.root.same_as(b.root))
    return false;
  if (a.parameter_alias_contract &&
      a.parameter_alias_contract == b.parameter_alias_contract)
    return true;
  if (a.fresh_allocation && b.fresh_allocation)
    return true;
  auto global_and_fresh_on_chip = [](const NativeStorageAddress &global,
                                     const NativeStorageAddress &local) {
    auto global_scope = NativeStorageScopeOf(global.root->ty);
    auto local_scope = NativeStorageScopeOf(local.root->ty);
    if (!local.fresh_allocation || !global_scope || !local_scope ||
        NativeStorageSpaceOf(*global_scope) != NativeStorageSpace::kGlobal)
      return false;
    auto space = NativeStorageSpaceOf(*local_scope);
    return space != NativeStorageSpace::kUnknown &&
           space != NativeStorageSpace::kGlobal;
  };
  return global_and_fresh_on_chip(a, b) || global_and_fresh_on_chip(b, a);
}

class NativeStorageContext {
public:
  ffi::Map<Var, Expr> bindings;

  void SetParameterAliasContract(NativeParameterAliasContract contract) {
    parameter_alias_contract_ = std::move(contract);
  }

  void Collect(const Stmt &body) {
    VisitNativePostOrder(body, [&](const ffi::ObjectRef &object) {
      if (const auto *bind = object.as<BindObj>()) {
        Expr value = bind->value;
        if (const auto *call = value.as<CallObj>(); IsScheduledBind(call))
          value = ReadScheduledBinding(call).value;
        bindings.Set(bind->var, value);
      }
    });
  }

  NativeStorageAddress Address(const Expr &value) const {
    std::unordered_set<const VarObj *> active;
    auto result = Resolve(value, &active);
    if (result.exact && parameter_alias_contract_)
      if (auto root = result.root.as<Var>();
          root && parameter_alias_contract_->count(*root))
        result.parameter_alias_contract = parameter_alias_contract_;
    return result;
  }

  NativeStorageContext Substitute(const Var &old_var,
                                  const Var &new_var) const {
    NativeStorageContext result;
    result.parameter_alias_contract_ = parameter_alias_contract_;
    auto mutator = ffi::make_object<tvm::ExprMutator>();
    mutator->VarRemapSet(old_var, new_var);
    for (const auto &[variable, value] : bindings) {
      result.bindings.Set(variable.same_as(old_var) ? new_var : variable,
                          mutator->Mutate(value).ValueOrUnchanged(value));
    }
    return result;
  }

  Var Identity(const TensorVar &tensor) const {
    auto address = Address(tensor);
    if (auto variable = address.root.as<Var>())
      return variable.value();
    // No allocation identity can be proved for this pointer-producing call.
    // Keep a stable name for diagnostics; Exact(tensor) remains false.
    return tensor;
  }

  bool Exact(const TensorVar &tensor) const { return Address(tensor).exact; }

  bool SameStorage(const TensorVar &a, const TensorVar &b) const {
    auto lhs = Address(a), rhs = Address(b);
    return lhs.exact && rhs.exact && lhs.root.same_as(rhs.root);
  }

  // Independent allocations and global-vs-fresh-on-chip storage cannot
  // overlap. Distinct external parameters still need an explicit noalias
  // contract. Integer-derived pointers and scope-changing aliases remain
  // conservative even against a fresh allocation. False is not proof of
  // overlap.
  bool ProvenDisjoint(const TensorVar &a, const TensorVar &b) const {
    return ProvenDisjointNativeStorage(Address(a), Address(b));
  }

  PrimExpr ElementBitOffset(const TensorVar &tensor,
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

private:
  NativeParameterAliasContract parameter_alias_contract_;

  static NativeStorageAddress PreserveScope(NativeStorageAddress address,
                                            const Type &type) {
    address.exact &= CompatibleNativeStorageScopes(address.root->ty, type);
    return address;
  }

  NativeStorageAddress
  Resolve(const Expr &value, std::unordered_set<const VarObj *> *active) const {
    if (auto variable = value.as<Var>()) {
      auto binding = bindings.Get(variable.value());
      if (!binding) {
        const bool address_root =
            value->ty.as<PtrTypeObj>() || value->ty.as<TensorTypeObj>();
        return {value, IntImm::Int64(0), address_root, false};
      }
      TVM_FFI_CHECK(active->insert(variable.value().get()).second, ValueError)
          << "Cyclic native storage definition for " << variable.value();
      auto result = [&]() -> NativeStorageAddress {
        if (auto tensor = variable->as<TensorVar>()) {
          if (binding.value().as<TensorVar>())
            return Resolve(binding.value(), active);
          const auto *call = binding.value().as<CallObj>();
          if (call && call->op.same_as(tirx::alloc_tensor_op()))
            return PreserveScope({value, IntImm::Int64(0), true, true},
                                 call->ty);
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
          (call->op.same_as(tirx::reinterpret_op()) &&
           call->ty.as<PtrTypeObj>()))
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
};

// Only an existing function contract can distinguish external parameters.
// A shared token keeps proofs within one contract, including renamed task
// contexts. Non-restrict aliases remove their originating parameter; an
// opaque exception disables the contract because its origin is unknown.
inline NativeParameterAliasContract CollectNativeParameterAliasContract(
    const tirx::Function &function,
    const ScheduleUnitMetadata &kernel_annotations = {}) {
  if (!function->GetAttr<bool>("tirx.noalias").value_or(false))
    return {};
  NativeStorageContext storage;
  if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    storage.Collect(function->body);
  auto restricted = std::make_shared<NativeRestrictedParameters>();
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
  auto read = [&](const ScheduleUnitMetadata &fields) {
    if (auto values = fields.Get("tl.non_restrict_params"))
      exclude(*values);
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
  read(kernel_annotations);
  if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    VisitNativePostOrder(function->body, [&](const ffi::ObjectRef &object) {
      if (const auto *call = object.as<CallObj>();
          call && call->op.same_as(Op::Get("tl.annotation")))
        read(ReadNativeAnnotations(call));
      if (const auto *region = object.as<RegionStmtObj>())
        if (auto attribute = ReadTaskAttribute(region)) {
          if (attribute->key == "tl.non_restrict_params")
            exclude(attribute->value);
          else if (attribute->key == "tl.kernel")
            read(attribute->value.cast<ScheduleUnitMetadata>());
        }
    });
  return unknown ? NativeParameterAliasContract{} : std::move(restricted);
}

inline TensorRegion RemapTaskRegion(const TensorRegion &region,
                                    const ffi::Map<Var, Expr> &values) {
  auto mutator = ffi::make_object<tvm::ExprMutator>();
  for (const auto &[var, replacement] : values)
    mutator->VarRemapSet(var, replacement);
  Expr expression = region;
  return mutator->Mutate(expression)
      .ValueOrUnchanged(expression)
      .as_or_throw<TensorRegion>();
}

inline Stmt RemapTaskStatement(const Stmt &statement,
                               const ffi::Map<Var, Expr> &values) {
  auto mutator = ffi::make_object<tirx::StmtExprMutator>();
  for (const auto &[var, replacement] : values)
    mutator->VarRemapSet(var, replacement);
  return mutator->Mutate(statement).ValueOrUnchanged(statement);
}

inline ffi::Array<Range>
SubstituteTaskRanges(const ffi::Array<Range> &ranges,
                     const ffi::Map<Var, PrimExpr> &values) {
  ffi::Array<Range> result;
  for (const auto &range : ranges)
    result.push_back(
        Range::FromMinExtent(SubstituteConstraintExpr(range->begin, values),
                             SubstituteConstraintExpr(range->extent, values)));
  return result;
}

} // namespace tvm::tl::ascend
