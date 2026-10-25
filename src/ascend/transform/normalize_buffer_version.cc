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
 * \file tl/ascend/transform/normalize_buffer_version.cc
 * \brief Lower staged buffer-version metadata to a scoped internal AttrStmt.
 */

#include "buffer_version.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
class BufferVersionNormalizer : public tirx::StmtExprMutator {
protected:
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto region = StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(original)
                      .as_or_throw<RegionStmt>();
    TVM_FFI_CHECK(!region->attrs->dict.count(attr::kBufferVersion), ValueError)
        << attr::kBufferVersion
        << " is compiler-internal and must not be provided by the frontend";
    auto staged = region->attrs->dict.Get(kManualMultiBuffer);
    if (!staged)
      return region;
    auto versions = staged->cast<BufferVersionMap>();
    for (const auto &[storage, count] : versions) {
      TVM_FFI_CHECK(count > 0, ValueError)
          << "Manual buffer version counts must be positive: " << storage;
    }
    auto annotations = region->attrs->dict;
    annotations.erase(kManualMultiBuffer);
    if (!versions.empty())
      annotations.Set(attr::kBufferVersion, versions);
    region.CopyOnWrite()->attrs = DictAttrs(annotations);
    return region;
  }
};
} // namespace

tvm::transform::Pass NormalizeBufferVersion() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    TVM_FFI_CHECK(!function->attrs->dict.count(attr::kBufferVersion),
                  ValueError)
        << attr::kBufferVersion
        << " is compiler-internal and must not be provided by the frontend";
    if (!function->body)
      return function;
    auto rewriter = ffi::make_object<BufferVersionNormalizer>();
    SeqStmt original = function->body.value();
    function.CopyOnWrite()->body =
        rewriter->Mutate(original).ValueOrUnchanged(original);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.NormalizeBufferVersion");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.NormalizeBufferVersion",
                                   NormalizeBufferVersion);
}
} // namespace tvm::tl
