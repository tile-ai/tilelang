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
#include <tvm/ir/attrs.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

namespace {

class BufferVersionNormalizer : public StmtMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    ICHECK(!func->attrs.defined() ||
           !func->attrs->dict.count(tl::attr::kBufferVersion))
        << "'" << tl::attr::kBufferVersion
        << "' is compiler-internal and must not be provided by the frontend";

    BufferVersionNormalizer normalizer;
    PrimFuncNode *writer = func.CopyOnWrite();
    writer->body = normalizer(std::move(writer->body));
    return func;
  }

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(StmtMutator::VisitStmt_(op));
    ICHECK(!block->annotations.count(tl::attr::kBufferVersion))
        << "'" << tl::attr::kBufferVersion
        << "' is compiler-internal and must not be provided by the frontend";

    auto staged = block->annotations.Get(kManualMultiBuffer);
    if (!staged.has_value()) {
      return block;
    }

    ffi::Map<ffi::String, ffi::Any> annotations = block->annotations;
    annotations.erase(kManualMultiBuffer);

    BufferVersionMap staged_versions = staged.value().cast<BufferVersionMap>();
    for (const auto &[_, version] : staged_versions) {
      ICHECK_GT(version, 0) << "'" << kManualMultiBuffer
                            << "' values must be positive version counts";
    }

    SBlockNode *writer = block.CopyOnWrite();
    writer->annotations = std::move(annotations);
    if (!staged_versions.empty()) {
      writer->body =
          AttrStmt(std::move(staged_versions), tl::attr::kBufferVersion,
                   Integer(1), std::move(writer->body));
    }
    return block;
  }
};

} // namespace

tvm::transform::Pass NormalizeBufferVersion() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) {
    return BufferVersionNormalizer::Rewrite(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizeBufferVersion", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.NormalizeBufferVersion",
                        NormalizeBufferVersion);
}

} // namespace tl
} // namespace tvm
