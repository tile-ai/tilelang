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

#include "common/attr.h"
#include "common/thread_sync.h"
#include "op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace transform {
tvm::transform::Pass ThreadSync(const ffi::String &storage_scope) {
  TVM_FFI_CHECK(storage_scope == "global" || storage_scope == "shared" ||
                    storage_scope == "shared.dyn",
                ValueError)
      << "Unsupported synchronization scope: " << storage_scope;
  auto pass = [storage_scope](tirx::Function function, IRModule,
                              tvm::transform::PassContext context)
      -> ffi::Optional<tirx::Function> {
    if (storage_scope != "global" && function->body &&
        !context->GetConfig<bool>(kDisableThreadStorageSync).value_or(false)) {
      SeqStmt body =
          RewriteThreadSync(function, *function->body, storage_scope);
      function.CopyOnWrite()->body = body;
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.ThreadSync");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.ThreadSync", ThreadSync);
}
} // namespace transform
} // namespace tvm::tl
