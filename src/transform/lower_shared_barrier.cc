/*!
 *  \file lower_shared_barrier.cc
 *  \brief Convert shared.barrier buffers to plain shared + ptx init.
 */
#include "tvm/ir/type.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

class SharedBarrierRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body) {
    SharedBarrierRewriter rewriter;
    return rewriter(body);
  }

private:
  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = GetRef<Block>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;

    // Record the mapping from buffer data var to buffer for later lookup
    for (auto buffer : alloc_buffers) {
      buffer_map_.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer->data, match_buffer->buffer});
    }

    Array<Buffer> barrier_buffers;

    for (auto [data, buffer] : buffer_map_) {
      const auto *ptr_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      auto storage_scope = ptr_type->storage_scope;
      ICHECK(ptr_type) << "Buffer Var's type annotation must be of PointerType";
      if (storage_scope == "shared.barrier") {
        barrier_buffers.push_back(buffer);
      }
    }

    if (barrier_buffers.size() == 0) {
      return StmtExprMutator::VisitStmt_(op);
    }

    ICHECK(thread_var_.defined()) << "thread_var_ is not defined";

    for (auto buffer : barrier_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    /*
    Transform the barrier buffers to new allocations
    transform:
        data_is_ready = T.alloc_buffer((128,), "uint64", scope="shared.barrier")
        compute_is_done = T.alloc_buffer((128,), "uint64",
    scope="shared.barrier")

    into:
        data_is_ready = T.alloc_buffer((1,), "uint64", scope="shared")
        compute_is_done = T.alloc_buffer((1,), "uint64", scope="shared")

        if tx == 0:
          T.ptx_init_barrier_thread_count(data_is_ready[0], 128)
          T.ptx_init_barrier_thread_count(compute_is_done[0], 128)
    */
    // 1. create new data vars
    Array<Var> new_data_vars;
    for (auto buffer : barrier_buffers) {
      auto data = buffer->data;
      auto ptr_type = data->type_annotation.as<PointerTypeNode>();
      auto new_data =
          Var(data->name_hint, PointerType(ptr_type->element_type, "shared"));
      var_remap_.Set(data, new_data);
      new_data_vars.push_back(new_data);
    }

    // 2. create new buffers
    Array<Buffer> new_buffers;
    for (auto buffer : barrier_buffers) {
      auto data = buffer->data;
      ICHECK(var_remap_.find(data) != var_remap_.end())
          << "data not found in var_remap_";
      auto new_data = var_remap_.at(data);
      auto new_buffer = Buffer(new_data, buffer->dtype, Array<PrimExpr>({1}),
                               Array<PrimExpr>({1}), PrimExpr(0), buffer->name,
                               buffer->data_alignment, buffer->offset_factor,
                               buffer->buffer_type);
      new_buffers.push_back(new_buffer);
      buffer_remap_.Set(buffer, new_buffer);
    }

    // remove the barrier buffers
    alloc_buffers.MutateByApply([this](Buffer buf) {
      if (buffer_remap_.find(buf) != buffer_remap_.end()) {
        return buffer_remap_.at(buf);
      }
      return buf;
    });
    if (!alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }

    // 3. create init calls for new buffers
    Array<Stmt> init_mbarrier_calls_;
    for (auto buffer : barrier_buffers) {
      auto data = buffer->data;
      auto old_buffer = buffer_data_to_buffer_.at(data);
      auto new_buffer = buffer_remap_.at(old_buffer);
      auto count = old_buffer->shape[0];

      auto call =
          Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
               {BufferLoad(new_buffer, {0}), PrimExpr(count)});
      init_mbarrier_calls_.push_back(Evaluate(call));
    }

    Array<Stmt> new_body;
    new_body.push_back(IfThenElse(EQ(thread_var_->var, 0),
                                  SeqStmt(init_mbarrier_calls_), Stmt()));
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                      {StringImm("shared")})));
    new_body.push_back(block->body);

    block.CopyOnWrite()->body = SeqStmt(new_body);

    return StmtExprMutator::VisitStmt_(block.get());
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto buffer = load->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[load->buffer];
      return BufferLoad(new_buffer, load->indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], buffer->dtype, buffer->shape,
          buffer->strides, buffer->elem_offset, buffer->name,
          buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
      return BufferLoad(new_buffer, load->indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto buffer = store->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[store->buffer];
      return BufferStore(new_buffer, store->value, store->indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], buffer->dtype, buffer->shape,
          buffer->strides, buffer->elem_offset, buffer->name,
          buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
      return BufferStore(new_buffer, store->value, store->indices);
    }
    return store;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // This is a workaround for cpu backend,
  // we need to define a thread_var for the serial loop.
  IterVar thread_var_;
  Map<Var, Var> var_remap_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  // Mapping from data Var of a Buffer to Buffer, for lookup
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
};

PrimFunc LowerSharedBarrier(PrimFunc f) {
  SharedBarrierRewriter rewriter;
  f.CopyOnWrite()->body = rewriter.Rewrite(f->body);
  return f;
}

namespace transform {
using namespace tir::transform;

tvm::transform::Pass LowerSharedBarrier() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return tl::LowerSharedBarrier(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerSharedBarrier", {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerSharedBarrier", LowerSharedBarrier);
});

} // namespace transform
} // namespace tl
} // namespace tvm
