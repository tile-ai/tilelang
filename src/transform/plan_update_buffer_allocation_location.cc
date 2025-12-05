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
 * \brief Planning where buffers to be allocated and update the AST.
 * \file plan_update_buffer_allocation_location.cc
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/var.h>

#include "tir/transforms/ir_utils.h"

// Forward-declare tir's var-level LCA helper which has no public header.
namespace tvm {
namespace tir {
ffi::Map<Var, ffi::Optional<Stmt>>
DetectBufferVarAccessLCA(const PrimFunc &func);
}
} // namespace tvm

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

// Use TVM's tir analysis API for LCA detection.

class CollectManagedAllocations : public StmtExprVisitor {
public:
  void VisitStmt_(const BlockNode *op) final {
    for (const auto &buf : op->alloc_buffers) {
      managed_allocations.insert(buf->data.get());
    }
    for (const auto &buf : op->match_buffers) {
      managed_allocations.insert(buf->buffer->data.get());
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief Buffers that are allocated outside of the BlockNode, and should not
   * be moved by BufferAllocationLocator. */
  std::unordered_set<const VarNode *> managed_allocations;
};

/*! \brief Collect the allocate buffer order. */
class BufferAllocateOrderCollector : public StmtExprVisitor {
public:
  static ffi::Array<Buffer> Collect(const PrimFunc &func) {
    BufferAllocateOrderCollector collector;
    for (const auto &kv : func->buffer_map) {
      collector.buffer_alloc_recorder_.push_back(kv.second);
    }
    collector(func->body);
    return std::move(collector.buffer_alloc_recorder_);
  }

private:
  bool find(const Buffer &buf) {
    return std::find(buffer_alloc_recorder_.begin(),
                     buffer_alloc_recorder_.end(),
                     buf) != buffer_alloc_recorder_.end();
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_alloc_recorder_.push_back(buffer);
    }
    // Also visit match_buffers to collect constant buffers associated with
    // AllocateConst nodes. These buffers only appear in read and match_buffer
    // regions.
    for (const auto &region : op->match_buffers) {
      if (!find(region->source->buffer)) {
        buffer_alloc_recorder_.push_back(region->source->buffer);
      }
    }

    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (!find(op->buffer)) {
      buffer_alloc_recorder_.push_back(op->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (!find(op->buffer)) {
      buffer_alloc_recorder_.push_back(op->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief The buffer allocated order recorder. */
  ffi::Array<Buffer> buffer_alloc_recorder_;
};

class BufferAllocationLocator : public StmtExprMutator {
public:
  explicit BufferAllocationLocator(const PrimFunc &func) {
    // Use TVM's tir LCA detection implementation
    ffi::Map<Buffer, ffi::Optional<Stmt>> buffer_lca =
        tir::DetectBufferAccessLCA(func);
    ffi::Map<Var, ffi::Optional<Stmt>> var_lca =
        tir::DetectBufferVarAccessLCA(func);

    // The buffer_alloc_recorder Array is used to keep the buffer allocation
    // order since the buffer_lca Map is unordered.
    ffi::Array<Buffer> buffer_alloc_recorder =
        BufferAllocateOrderCollector::Collect(func);
    std::unordered_set<const VarNode *> arg_buffer_vars;
    CollectManagedAllocations collector;
    collector(func->body);
    managed_allocations_ = collector.managed_allocations;

    for (const auto &kv : func->buffer_map) {
      const Buffer &buffer = kv.second;
      arg_buffer_vars.emplace(buffer->data.get());
      PushBinding(buffer->data, buffer);
    }
    // create buffers to be allocated at each stmts
    for (const auto &buffer : buffer_alloc_recorder) {
      // Prefer the LCA derived from the underlying data var. If missing, fall
      // back to Buffer LCA.
      const StmtNode *stmt = nullptr;
      auto vit = var_lca.find(buffer->data);
      if (vit != var_lca.end()) {
        stmt = (*vit).second.get();
      } else {
        auto bit = buffer_lca.find(buffer);
        if (bit != buffer_lca.end()) {
          stmt = (*bit).second.get();
        }
      }
      if (stmt != nullptr || vit != var_lca.end()) {
        if (arg_buffer_vars.count(buffer->data.get())) {
          continue;
        }
        if (managed_allocations_.count(buffer->data.get())) {
          alloc_buffers_[stmt].push_back(buffer);
        }
        // Do not push binding here. Bindings should reflect scope accurately,
        // and will be pushed/popped when visiting the owning stmt.
      }
    }
  }

private:
  // Maintain a stack of Buffers per data var to correctly handle cases
  // where multiple Buffer objects share the same underlying data Var.
  void PushBinding(const Var &v, const Buffer &buf) {
    ffi::Array<Buffer> arr;
    auto it = buffer_data_to_buffers_.find(v);
    if (it != buffer_data_to_buffers_.end()) {
      arr = (*it).second;
    }
    arr.push_back(buf);
    buffer_data_to_buffers_.Set(v, arr);
  }

  void PopBinding(const Var &v) {
    auto it = buffer_data_to_buffers_.find(v);
    if (it == buffer_data_to_buffers_.end())
      return;
    ffi::Array<Buffer> arr = (*it).second;
    if (!arr.empty()) {
      // erase last element
      std::vector<Buffer> tmp;
      tmp.reserve(arr.size() - 1);
      for (size_t i = 0; i + 1 < arr.size(); ++i)
        tmp.push_back(arr[i]);
      arr = ffi::Array<Buffer>(tmp);
    }
    if (arr.empty()) {
      buffer_data_to_buffers_.erase(v);
    } else {
      buffer_data_to_buffers_.Set(v, arr);
    }
  }

  bool HasBinding(const Var &v) const {
    auto it = buffer_data_to_buffers_.find(v);
    return it != buffer_data_to_buffers_.end() && !(*it).second.empty();
  }

  // Snapshot the current top binding per Var for APIs that require
  // a single Buffer per data Var (e.g. GetBlockReadWriteRegion).
  ffi::Map<Var, Buffer> SnapshotVarMap() const {
    ffi::Map<Var, Buffer> out;
    for (const auto &kv : buffer_data_to_buffers_) {
      const Var &v = kv.first;
      const ffi::Array<Buffer> &arr = kv.second;
      if (!arr.empty()) {
        out.Set(v, arr[arr.size() - 1]);
      }
    }
    return out;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto it = alloc_buffers_.find(op);
    if (it == alloc_buffers_.end()) {
      return StmtMutator::VisitStmt_(op);
    }
    for (const Buffer &buf : it->second) {
      PushBinding(buf->data, buf);
    }
    auto node = Downcast<For>(StmtMutator::VisitStmt_(op));
    ffi::Array<Buffer> new_block_alloc_bufs;
    for (const Buffer &buf : it->second) {
      if (managed_allocations_.count(buf->data.get())) {
        PopBinding(buf->data);
        new_block_alloc_bufs.push_back(buf);
      }
    }

    if (!new_block_alloc_bufs.empty()) {
      node.CopyOnWrite()->body =
          InjectOpaqueBlock(node->body, new_block_alloc_bufs);
    }

    return node;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    ICHECK(!op->init.defined());
    ffi::Array<Buffer> alloc_buffers;
    auto it = alloc_buffers_.find(op);
    if (it != alloc_buffers_.end()) {
      alloc_buffers = it->second;
      for (const Buffer &buf : it->second) {
        PushBinding(buf->data, buf);
      }
    }
    for (const MatchBufferRegion match_buffer : op->match_buffers) {
      const Var &target_var = match_buffer->buffer->data;
      const Var &source_var = match_buffer->source->buffer->data;
      ICHECK(HasBinding(source_var));
      PushBinding(target_var, match_buffer->buffer);
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);

    // No longer consider buffers created by match_buffer inside the block when
    // updating access region.
    for (const MatchBufferRegion match_buffer : op->match_buffers) {
      const Var &target_var = match_buffer->buffer->data;
      PopBinding(target_var);
    }
    // No longer consider buffers allocated inside the block when updating
    // access region.
    if (it != alloc_buffers_.end()) {
      for (const Buffer &buf : it->second) {
        PopBinding(buf->data);
      }
    }

    ObjectPtr<BlockNode> n = CopyOnWrite(op);
    n->alloc_buffers = std::move(alloc_buffers);
    // Erase buffer allocated inside the block from access region.
    n->reads = RemoveRedundantBufferRegion(n->reads);
    n->writes = RemoveRedundantBufferRegion(n->writes);
    return Stmt(n);
  }

  Stmt VisitStmt_(const BufferRealizeNode *op) final {
    ICHECK(false)
        << "Internal Error: BufferRealizeNode is not allowed in TensorIR.";
    throw;
  }

  Stmt InjectOpaqueBlock(Stmt body, const ffi::Array<Buffer> &alloc_buffers) {
    ICHECK(!alloc_buffers.empty());
    Block opaque_block(/*iter_vars=*/{},
                       /*reads=*/{},
                       /*writes=*/{},
                       /*name_hint=*/"",
                       /*body=*/std::move(body),
                       /*init=*/std::nullopt,
                       /*alloc_buffers=*/alloc_buffers);
    ObjectPtr<BlockNode> n = CopyOnWrite(opaque_block.get());
    // Snapshot to a Var->Buffer map using the innermost binding for each Var.
    ffi::Map<Var, Buffer> var_map = SnapshotVarMap();
    ffi::Array<ffi::Array<BufferRegion>> access =
        GetBlockReadWriteRegion(opaque_block, var_map);
    n->reads = access[0];
    n->writes = access[1];
    BlockRealize realize({}, Bool(true), Block(n));
    return realize;
  }

  ffi::Array<BufferRegion>
  RemoveRedundantBufferRegion(const ffi::Array<BufferRegion> &region) const {
    ffi::Array<BufferRegion> result;
    for (const BufferRegion &buffer_region : region) {
      if (HasBinding(buffer_region->buffer->data)) {
        result.push_back(buffer_region);
      }
    }
    return result;
  }

  /*! \brief The map from stmt to the buffers to be allocated under it. */
  std::unordered_map<const StmtNode *, ffi::Array<Buffer>> alloc_buffers_;
  /*! \brief Stack of buffers per data var for scoping correctness. */
  ffi::Map<Var, ffi::Array<Buffer>> buffer_data_to_buffers_;
  /*! \brief Buffers that are allocated within a BlockNode, and may be moved. */
  std::unordered_set<const VarNode *> managed_allocations_;
};

PrimFunc PlanAndUpdateBufferAllocationLocation(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  BufferAllocationLocator locator(func);
  fptr->body = locator(fptr->body);
  return func;
}

namespace transform {

Pass PlanAndUpdateBufferAllocationLocation() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return ::tvm::tl::PlanAndUpdateBufferAllocationLocation(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.PlanAndUpdateBufferAllocationLocation", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PlanAndUpdateBufferAllocationLocation",
                        PlanAndUpdateBufferAllocationLocation);
}

} // namespace transform

} // namespace tl
} // namespace tvm
