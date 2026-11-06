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

#ifndef TILELANG_IR_STORAGE_PROVENANCE_H_
#define TILELANG_IR_STORAGE_PROVENANCE_H_

#include <functional>
#include <memory>
#include <optional>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/type.h>
#include <unordered_set>

namespace tvm::tl {
using namespace prim;
using namespace tirx;

using RestrictedParameters =
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using ParameterAliasContract = std::shared_ptr<const RestrictedParameters>;

// Exact addresses retain their originating pointer/tensor root and byte
// offset. Exactness is independent of noalias: external roots may overlap.
// Unknown expressions retain a diagnostic root but cannot establish a proof.
struct StorageAddress {
  Expr root{ffi::UnsafeInit{}};
  PrimExpr byte_offset{IntImm::Int64(0)};
  bool exact{false};
  bool fresh_allocation{false};
  ParameterAliasContract parameter_alias_contract;
};

TVM_TIRX_DLL std::optional<runtime::MemoryScope>
StorageScopeOf(const Type &type);
TVM_TIRX_DLL bool ProvenDisjointStorage(const StorageAddress &a,
                                        const StorageAddress &b);

// Backends may recognize equivalent spellings of the same hardware scope.
// The default accepts only equal scopes. Initializer resolution is explicit:
// generic provenance has no knowledge of a backend's scheduling markers.
class TVM_TIRX_DLL StorageProvenance {
public:
  using ScopeCompatibility = std::function<bool(const Type &, const Type &)>;
  using InitializerResolver = std::function<Expr(const Expr &)>;

  explicit StorageProvenance(ScopeCompatibility compatible = {},
                             InitializerResolver initializer = {});
  ffi::Map<Var, Expr> bindings;

  void SetParameterAliasContract(ParameterAliasContract contract);
  void Collect(const Stmt &body);
  StorageAddress Address(const Expr &value) const;
  StorageProvenance Substitute(const Var &old_var, const Var &new_var) const;
  Var Identity(const TensorVar &tensor) const;
  bool Exact(const TensorVar &tensor) const;
  bool SameStorage(const TensorVar &a, const TensorVar &b) const;
  PrimExpr ElementBitOffset(const TensorVar &tensor,
                            const ffi::Array<PrimExpr> &indices) const;

private:
  ParameterAliasContract parameter_alias_contract_;
  ScopeCompatibility compatible_;
  InitializerResolver initializer_;
  StorageAddress PreserveScope(StorageAddress address, const Type &type) const;
  StorageAddress Resolve(const Expr &value,
                         std::unordered_set<const VarObj *> *active) const;
};

// Only an existing function noalias contract can separate external roots.
// A caller may supply dialect-specific alias exceptions. Unknown origins
// disable the contract; copies/substitutions retain the same shared token.
TVM_TIRX_DLL ParameterAliasContract CollectParameterAliasContract(
    const tirx::Function &function, const StorageProvenance &storage,
    const ffi::Array<Expr> &additional_exceptions = {});

} // namespace tvm::tl
#endif // TILELANG_IR_STORAGE_PROVENANCE_H_
