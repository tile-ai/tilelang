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
#include "transform/thread_storage_sync.h"
#include "ascend/ir/task.h"
#include "ascend/target_utils.h"
#include "ascend/transform/source_scopes.h"
#include "op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

// Each SIMT_VF is an independent thread execution domain. The enclosing
// function supplies allocation and alias definitions, while the body carries
// the actual launch_thread regions used by the shared native planner.
class SimtVFThreadSync : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  SimtVFThreadSync(tirx::Function function, ffi::String scope, int warp_size)
      : function_(std::move(function)), scope_(std::move(scope)),
        warp_size_(warp_size) {}

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    auto name = ascend::BlockNameAtSourceBoundary(op);
    if (!name || *name != "SIMT_VF")
      return StmtExprMutator::Mutate_(op, mode);
    auto body = PlanThreadStorageSync(function_, op->body, scope_, warp_size_,
                                      false, false);
    if (body.same_as(op->body))
      return ffi::Unchanged{};
    RegionStmt result = ffi::GetRef<RegionStmt>(op);
    result.CopyOnWrite()->body = std::move(body);
    return result;
  }

private:
  tirx::Function function_;
  ffi::String scope_;
  int warp_size_;
};

} // namespace

namespace transform {

tvm::transform::Pass AscendThreadSync(const ffi::String &storage_scope) {
  auto pass = [storage_scope](tirx::Function function, const IRModule &,
                              const tvm::transform::PassContext &context) {
    if (context->GetConfig<bool>(kDisableThreadStorageSync).value_or(false) ||
        function->GetAttr(tvm::attr::kDeclaration).value_or(false) ||
        storage_scope == "global")
      return function;
    bool is_ascend = false;
    int warp_size = 32;
    if (auto target = function->attrs.GetAttr<Target>(tvm::attr::kTarget)) {
      warp_size =
          target.value()->GetAttr<int64_t>("thread_warp_size").value_or(32);
      is_ascend = TargetIsAscend(target.value());
    }
    const SeqStmt &original = function->body;
    SeqStmt body = original;
    if (is_ascend) {
      auto rewrite = ffi::make_object<SimtVFThreadSync>(function, storage_scope,
                                                        warp_size);
      body = rewrite->Mutate(original).ValueOrUnchanged(original);
    } else {
      body = PlanThreadStorageSync(function, original, storage_scope, warp_size,
                                   true, true);
    }
    if (!body.same_as(original))
      function.CopyOnWrite()->body = std::move(body);
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
