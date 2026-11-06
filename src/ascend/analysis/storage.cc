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

#include "storage.h"
#include "ascend/ir/block.h"
#include "ascend/ir/task.h"
#include "ascend/transform/source_scopes.h"
#include "ir/annotations.h"
#include <tvm/ffi/extra/structural_visit.h>

namespace tvm::tl::ascend {

StorageSpace StorageSpaceOf(const runtime::MemoryScope &scope) {
  const auto &name = scope.name();
  if (name == "global")
    return StorageSpace::kGlobal;
  if (name == "local" || name == "local.var")
    return StorageSpace::kLocal;
  if (name == "shared" || name == "shared.dyn")
    return StorageSpace::kUB;
  if (name == "shared.l1" || name == "shared.l1.dyn")
    return StorageSpace::kL1;
  if (name == "shared.l0a" || name == "shared.l0a.dyn")
    return StorageSpace::kL0A;
  if (name == "shared.l0b" || name == "shared.l0b.dyn")
    return StorageSpace::kL0B;
  if (name == "shared.l0a.sf")
    return StorageSpace::kL0ASF;
  if (name == "shared.l0b.sf")
    return StorageSpace::kL0BSF;
  if (name == "shared.l0c" || name == "shared.l0c.dyn")
    return StorageSpace::kL0C;
  if (name == "shared.ss" || name == "shared.ss.dyn")
    return StorageSpace::kSS;
  return StorageSpace::kUnknown;
}

bool CompatibleStorageScopes(const Type &a, const Type &b) {
  auto lhs = StorageScopeOf(a), rhs = StorageScopeOf(b);
  if (!lhs || !rhs)
    return false;
  if (*lhs == *rhs)
    return true;
  auto space = StorageSpaceOf(*lhs);
  return space != StorageSpace::kUnknown && space == StorageSpaceOf(*rhs);
}

bool ProvenDisjointAscendStorage(const StorageAddress &a,
                                 const StorageAddress &b) {
  if (!a.exact || !b.exact || a.root.same_as(b.root))
    return false;
  if (ProvenDisjointStorage(a, b))
    return true;
  auto global_and_fresh_on_chip = [](const StorageAddress &global,
                                     const StorageAddress &local) {
    auto global_scope = StorageScopeOf(global.root->ty);
    auto local_scope = StorageScopeOf(local.root->ty);
    if (!local.fresh_allocation || !global_scope || !local_scope ||
        StorageSpaceOf(*global_scope) != StorageSpace::kGlobal)
      return false;
    auto space = StorageSpaceOf(*local_scope);
    return space != StorageSpace::kUnknown && space != StorageSpace::kGlobal;
  };
  return global_and_fresh_on_chip(a, b) || global_and_fresh_on_chip(b, a);
}

StorageProvenance MakeAscendStorageProvenance() {
  return StorageProvenance(CompatibleStorageScopes, [](const Expr &value) {
    const auto *call = value.as<CallObj>();
    return IsScheduledBind(call) ? ReadScheduledBinding(call).value : value;
  });
}

ParameterAliasContract CollectAscendParameterAliasContract(
    const tirx::Function &function,
    const ffi::Map<ffi::String, ffi::Any> &kernel_annotations) {
  if (!function->GetAttr<bool>("tirx.noalias").value_or(false))
    return {};
  auto storage = MakeAscendStorageProvenance();
  if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    storage.Collect(function->body);
  ffi::Array<Expr> exceptions;
  auto exclude = [&](const ffi::Any &values) {
    for (const ffi::Any &value : values.cast<ffi::Array<ffi::Any>>())
      exceptions.push_back(value.cast<Expr>());
  };
  auto read = [&](const ffi::Map<ffi::String, ffi::Any> &fields) {
    if (auto values = fields.Get("tl.non_restrict_params"))
      exclude(*values);
  };
  read(kernel_annotations);
  if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const ffi::ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *call = object.as<CallObj>(); IsAnnotationCall(call))
            read(ReadAnnotations(call));
          if (const auto *region = object.as<RegionStmtObj>()) {
            if (IsAscendBlock(region)) {
              read(AscendBlockAnnotations(region));
            } else if (auto source = ReadSourceAttribute(region)) {
              if (source->key == "tl.non_restrict_params")
                exclude(source->value);
              else if (source->key == "tl.kernel")
                read(source->value.cast<ffi::Map<ffi::String, ffi::Any>>());
            }
          }
          return ffi::WalkResult::Advance();
        });
  return CollectParameterAliasContract(function, storage, exceptions);
}

} // namespace tvm::tl::ascend
