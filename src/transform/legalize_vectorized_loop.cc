#include "compat/analyzer.h"
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
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include "transform/common/loop_aware_mutator.h"

#include "loop_vectorize.h"
#include "support/check.h"

#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <optional>

namespace tvm::tl {

class LoopVectorizedLegalizer : public ::tvm::tl::LoopAwareMutator {
public:
  explicit LoopVectorizedLegalizer(sym::Analyzer analyzer, bool cpu)
      : LoopAwareMutator(analyzer), analyzer_context_(std::move(analyzer)),
        cpu_(cpu) {}

private:
  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    For loop = LoopAwareMutator::Mutate_(node, mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(node))
                   .as_or_throw<For>();
    bool explicit_vectorization = loop->kind == ForKind::kVectorized;
    if (!explicit_vectorization && !(cpu_ && loop->kind == ForKind::kParallel))
      return loop;
    loop.CopyOnWrite()->kind = ForKind::kDefault;
    int width = GetVectorizeSize(loop, analyzer_context_);
    auto extent =
        GetConstInt(tl::arith::Simplify(analyzer_, loop->end - loop->begin));
    if (explicit_vectorization && width == 1 &&
        (!extent || extent.value() > 1)) {
      LOG(WARNING) << "T.vectorized loop over `" << loop->var
                   << "` with extent " << loop->end - loop->begin
                   << " is lowered as a serial loop because no legal "
                      "vectorization plan exists";
    }
    return VectorizeLoop(loop, analyzer_context_, {}, width);
  }

  sym::Analyzer analyzer_context_;
  bool cpu_;
};

tvm::transform::Pass LegalizeVectorizedLoop() {
  auto transform = [](tirx::Function function, const IRModule &,
                      const tvm::transform::PassContext &) {
    auto target = function->attrs.GetAttr<Target>(tvm::attr::kTarget);
    std::optional<With<Target>> target_scope;
    if (target)
      target_scope.emplace(target.value());
    function.CopyOnWrite()->body =
        ffi::make_object<LoopVectorizedLegalizer>(
            sym::Analyzer(), target && (target.value()->kind->name == "c" ||
                                        target.value()->kind->name == "llvm"))
            ->Mutate(function->body, ffi::InplaceMode::kDisallow)
            .ValueOrUnchanged(function->body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(transform, 0,
                                             "tl.LegalizeVectorizedLoop");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LegalizeVectorizedLoop",
                                   LegalizeVectorizedLoop);
}

} // namespace tvm::tl
