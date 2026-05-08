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
 * \brief Hoist global buffer allocations to the top of the block (host side).
 * \file hoist_global_buffer_allocations.cc
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/var.h>

#include "../op/utils.h"
#include "common/attr.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

class GlobalBufferAllocationsHoister : public StmtMutator {
public:
  Stmt VisitStmt_(const BlockNode *op) final {
    auto node = Downcast<Block>(StmtMutator::VisitStmt_(op));

    if (IsHostMainBlock(op)) {
      for (const auto &buf : global_buffers_) {
        node.CopyOnWrite()->alloc_buffers.push_back(buf);
      }
    } else {
      ffi::Array<Buffer> new_alloc_buffers;
      for (const auto &buf : op->alloc_buffers) {
        if (IsGlobalBuffer(buf)) {
          global_buffers_.push_back(buf);
        } else {
          new_alloc_buffers.push_back(buf);
        }
      }
      node.CopyOnWrite()->alloc_buffers = std::move(new_alloc_buffers);
    }

    return node;
  }

  ffi::Array<Buffer> global_buffers_;
};

PrimFunc HoistGlobalBufferAllocations(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  GlobalBufferAllocationsHoister hoister;
  fptr->body = hoister(fptr->body);
  return func;
}

namespace transform {

Pass HoistGlobalBufferAllocations() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return ::tvm::tl::HoistGlobalBufferAllocations(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.HoistGlobalBufferAllocations",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.HoistGlobalBufferAllocations",
                        HoistGlobalBufferAllocations);
}

} // namespace transform

} // namespace tl
} // namespace tvm
