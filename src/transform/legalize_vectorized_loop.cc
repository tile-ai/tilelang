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
 * \file legalize_vectorized_loop.cc
 * \brief Plan vector widths on native loop domains before vector IR expansion.
 */
#include "../ir/analyzer.h"
#include "ir/stmt_functor.h"
#include "loop_vectorize.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
class LoopVectorizedLegalizer : public tl::StmtExprMutatorWithAnalyzer {
public:
  LoopVectorizedLegalizer(const sym::Analyzer &analyzer,
                          ffi::Optional<Target> target)
      : StmtExprMutatorWithAnalyzer(analyzer), target_(std::move(target)) {}
  using StmtExprMutatorWithAnalyzer::Mutate;
  using StmtExprMutatorWithAnalyzer::Mutate_;

  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    For loop = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                   .as_or_throw<For>();
    if (loop->kind != ForKind::kVectorized)
      return loop;
    loop.CopyOnWrite()->kind = ForKind::kDefault;
    int width = GetVectorizeSize(loop, GetSyncedAnalyzer(), {}, target_);
    if (width <= 1 &&
        !GetSyncedAnalyzer()->CanProve(loop->end - loop->begin <= 1)) {
      LOG(WARNING)
          << "T.vectorized loop over `" << loop->var
          << "` remains serial because no valid vector access plan was found";
    }
    return VectorizeLoop(loop, GetSyncedAnalyzer(), {}, width, target_);
  }

private:
  ffi::Optional<Target> target_;
};
} // namespace

tvm::transform::Pass LegalizeVectorizedLoop() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    VectorizeMemoryScope memory_scope(function);
    auto analyzer = CreateTileLangAnalyzer();
    // The Function's bound target is authoritative even when the pass is run
    // without an ambient context, or inside a context for another target.
    // Unbound direct callers retain the helper's explicit-context fallback.
    auto mutator = ffi::make_object<LoopVectorizedLegalizer>(
        analyzer, function->GetAttr<Target>(tvm::attr::kTarget));
    auto original = function->body;
    function.CopyOnWrite()->body =
        mutator->Mutate(original).ValueOrUnchanged(original);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.LegalizeVectorizedLoop");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LegalizeVectorizedLoop",
                                   LegalizeVectorizedLoop);
}
} // namespace tvm::tl
