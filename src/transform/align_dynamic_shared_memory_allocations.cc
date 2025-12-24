/*!
 * \file align_dynamic_shared_memory_allocations.cc
 * \brief align dynamic shared memory allocations
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

class TileLangAlignDynamicSharedMemoryAllocations : public StmtExprMutator {
public:
  explicit TileLangAlignDynamicSharedMemoryAllocations(int align_bytes)
      : align_bytes_(align_bytes) {}

  static Stmt Substitute(int align_bytes, const Stmt &stmt) {
    TileLangAlignDynamicSharedMemoryAllocations smem_rewriter(align_bytes);
    return smem_rewriter.VisitStmt(stmt);
  }

  Stmt VisitStmt_(const AllocateNode *op) final {
    auto storage_scope =
        runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));
    if (storage_scope.rank == runtime::StorageRank::kShared &&
        storage_scope.tag == ".dyn") {
      auto new_extents =
          MakeRoundRobinAlignment(op->extents, align_bytes_, op->dtype.bytes());
      if (!new_extents.same_as(op->extents)) {
        auto new_allocate = Allocate(op->buffer_var, op->dtype, new_extents,
                                     op->condition, op->body, op->annotations);
        return StmtExprMutator::VisitStmt(new_allocate);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = tvm::ffi::GetRef<Block>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;
    alloc_buffers.MutateByApply([this](Buffer buf) {
      auto storage_scope =
          runtime::StorageScope::Create(GetPtrStorageScope(buf->data));
      if (storage_scope.rank == runtime::StorageRank::kShared &&
          storage_scope.tag == ".dyn") {
        auto new_shape = MakeRoundRobinAlignment(buf->shape, align_bytes_,
                                                 buf->dtype.bytes());
        if (!new_shape.same_as(buf->shape)) {
          ObjectPtr<BufferNode> new_buffer =
              tvm::ffi::make_object<BufferNode>(*(buf.get()));
          new_buffer->shape = std::move(new_shape);
          buffer_remap_.Set(buf, Buffer(new_buffer));
          return Buffer(new_buffer);
        }
      }
      return buf;
    });
    if (!alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    }
    return StmtExprMutator::VisitStmt_(block.get());
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store_node = tvm::ffi::GetRef<BufferStore>(op);
    Buffer buf = op->buffer;
    if (buffer_remap_.count(buf)) {
      buf = buffer_remap_[buf];
      return BufferStore(buf, op->value, op->indices);
    }
    return StmtExprMutator::VisitStmt_(store_node.get());
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load_node = tvm::ffi::GetRef<BufferLoad>(op);
    Buffer buf = op->buffer;
    if (buffer_remap_.count(buf)) {
      buf = buffer_remap_[buf];
      return BufferLoad(buf, op->indices);
    }
    return StmtExprMutator::VisitExpr_(load_node.get());
  }

private:
  static Array<PrimExpr> MakeRoundRobinAlignment(Array<PrimExpr> extents,
                                                 int align_bytes,
                                                 int dtype_bytes) {
    if (extents.empty())
      return extents;
    // Calculate total number of elements
    PrimExpr total_elems = make_const(extents[0].dtype(), 1);
    for (auto extent : extents) {
      total_elems = total_elems * extent;
    }
    // Calculate total bytes
    PrimExpr total_bytes = total_elems * dtype_bytes;
    // Check if already aligned
    PrimExpr remainder = indexmod(total_bytes, align_bytes);
    if (is_zero(remainder)) {
      return extents;
    }
    // Need to pad the last dimension
    Array<PrimExpr> adjusted;
    for (size_t i = 0; i < extents.size(); ++i) {
      adjusted.push_back(extents[i]);
    }
    // Calculate padded last dimension
    // pad = ceil(total_bytes / align_bytes) * align_bytes
    PrimExpr last_extent = extents.back();
    PrimExpr other_elems = make_const(extents[0].dtype(), 1);
    for (size_t i = 0; i < extents.size() - 1; ++i) {
      other_elems = other_elems * extents[i];
    }
    // new_last_extent = ceil(total_bytes / align_bytes) * align_bytes /
    // (other_elems * dtype_bytes)
    PrimExpr padded_total_bytes =
        floordiv(total_bytes + align_bytes - 1, align_bytes) * align_bytes;
    PrimExpr new_last_extent =
        floordiv(padded_total_bytes, other_elems * dtype_bytes);
    adjusted.Set(adjusted.size() - 1, new_last_extent);
    return adjusted;
  }

  int align_bytes_;
  Map<Buffer, Buffer> buffer_remap_;
};

tvm::transform::Pass AlignDynamicSharedMemoryAllocations(int align_bytes) {
  using namespace tir::transform;
  auto pass_func = [align_bytes](PrimFunc f, const IRModule &m,
                                 const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    n->body = TileLangAlignDynamicSharedMemoryAllocations::Substitute(
        align_bytes, n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.AlignDynamicSharedMemoryAllocations", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AlignDynamicSharedMemoryAllocations",
                        AlignDynamicSharedMemoryAllocations);
}

} // namespace tl
} // namespace tvm
