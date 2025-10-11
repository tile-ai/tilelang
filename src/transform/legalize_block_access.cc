/*
 * Copyright (c) 2024 TileLang
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file legalize_block_access.cc
 * \brief Populate block read/write regions ahead of pipeline passes.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

class BlockAccessLegalizer : public StmtExprMutator {
public:
  explicit BlockAccessLegalizer(Map<Var, Buffer> buffer_data_to_buffer)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)) {}

  Stmt VisitStmt_(const BlockNode *op) final {
    for (const Buffer &alloc : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(alloc->data, alloc);
    }

    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

    Array<Array<BufferRegion>> access =
        GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    BlockNode *n = block.CopyOnWrite();
    n->reads = access[0];
    n->writes = access[1];

    for (const Buffer &alloc : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(alloc->data);
    }

    return block;
  }

private:
  Map<Var, Buffer> buffer_data_to_buffer_;
};

namespace transform {

tvm::transform::Pass LegalizeBlockAccess() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &) {
    Map<Var, Buffer> buffer_data_to_buffer;
    for (const auto &[_, buffer] : f->buffer_map) {
      buffer_data_to_buffer.Set(buffer->data, buffer);
    }
    BlockAccessLegalizer legalizer(buffer_data_to_buffer);
    f.CopyOnWrite()->body = legalizer(f->body);
    return f;
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0,
                                                 "tl.LegalizeBlockAccess", {});
}

} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeBlockAccess",
                        transform::LegalizeBlockAccess);
});

} // namespace tl
} // namespace tvm
