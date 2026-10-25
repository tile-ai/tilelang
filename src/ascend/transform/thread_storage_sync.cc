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
 * \file thread_storage_sync.cc
 */
#include "ascend/target_utils.h"
#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/native_visitor.h"
#include "op/builtin.h"
#include "transform/common/attr.h"
#include "transform/common/thread_sync.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

class SimtThreadSynchronizer : public ascend::NativeMutator {
public:
  SimtThreadSynchronizer(tirx::Function function, ffi::String scope)
      : function_(std::move(function)), scope_(std::move(scope)) {}

  Stmt VisitStmt_(const RegionStmtNode *node) final {
    if (ascend::KernelRegionName(node) != "SIMT_VF")
      return ascend::NativeMutator::VisitStmt_(node);
    auto region = ffi::GetRef<RegionStmt>(node);
    region.CopyOnWrite()->body =
        RewriteThreadSync(function_, node->body, scope_, false);
    return region;
  }

private:
  tirx::Function function_;
  ffi::String scope_;
};

} // namespace

namespace transform {

tvm::transform::Pass AscendThreadSync(const ffi::String &storage_scope) {
  TVM_FFI_CHECK(storage_scope == "global" || storage_scope == "shared" ||
                    storage_scope == "shared.dyn",
                ValueError)
      << "Unsupported synchronization scope: " << storage_scope;
  auto pass = [storage_scope](tirx::Function function, IRModule,
                              tvm::transform::PassContext context)
      -> ffi::Optional<tirx::Function> {
    if (!function->body || storage_scope == "global" ||
        context->GetConfig<bool>(kDisableThreadStorageSync).value_or(false))
      return function;
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    if (target && TargetIsAscend(target.value())) {
      auto synchronizer =
          ffi::make_object<SimtThreadSynchronizer>(function, storage_scope);
      function.CopyOnWrite()->body =
          SeqStmt(synchronizer->VisitStmt(function->body.value()));
    } else {
      auto body =
          RewriteThreadSync(function, function->body.value(), storage_scope);
      function.CopyOnWrite()->body = body;
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.AscendThreadSync");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendThreadSync",
                                   AscendThreadSync);
}

} // namespace transform
} // namespace tvm::tl
