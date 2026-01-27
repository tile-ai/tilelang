/*!
 *  \file lower_shared_barrier.cc
 *  \brief Convert shared.barrier buffers to plain shared + ptx init.
 */
#include "../op/builtin.h"
#include "tvm/ir/type.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

namespace tvm {
namespace tl {

namespace attr {
// BlockAttr, Recording the arrive counts for each barrier allocation
constexpr const char *kBarrierInit = "barrier_init";
} // namespace attr

using namespace tir;

class SharedBarrierRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body, bool disable_shuffle_elect = false) {
    SharedBarrierRewriter rewriter(disable_shuffle_elect);
    return rewriter(std::move(body));
  }

private:
  SharedBarrierRewriter(bool disable_shuffle_elect)
      : disable_shuffle_elect_(disable_shuffle_elect) {}

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = tvm::ffi::GetRef<Block>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;

    // Record the mapping from buffer data var to buffer for later lookup
    for (auto buffer : alloc_buffers) {
      buffer_map_.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer->data, match_buffer->buffer});
    }

    // Only check buffers allocated in THIS block, not accumulated from parent
    // blocks
    Array<Buffer> barrier_buffers;
    for (auto buffer : alloc_buffers) {
      const auto *ptr_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      if (!ptr_type)
        continue;
      auto storage_scope = ptr_type->storage_scope;
      if (storage_scope == "shared.barrier") {
        barrier_buffers.push_back(buffer);
      }
    }

    if (barrier_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    ICHECK(thread_var_.defined()) << "thread_var_ is not defined";

    for (auto buffer : barrier_buffers) {
      ICHECK(buffer->name != "mbarrier")
          << "Shared barrier's name 'mbarrier' is reserved";
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    /*
    Transform:
        mbarrier_list = T.alloc_barrier(arrive_counts: list[int], "handle",
    scope="shared.barrier")

    into:
        # This is emitted by the definition of T.alloc_barrier
        mbarrier_list = T.alloc_buffer(len(arrive_counts), "handle",
    scope="shared.barrier")

        # This is emitted by this pass
        if tx == 0:
          for i in range(len(arrive_counts)):
            T.ptx_init_barrier_thread_count(mbarrier_list[i], arrive_counts[i])
    */

    // Extract the arrive counts from the block attr "barrier_init"
    // The attr is a Map<Var, Array<PrimExpr>> where key is buffer.data and
    // value is arrive counts
    ICHECK(op->annotations.count(attr::kBarrierInit))
        << "barrier_init is not defined";
    auto barrier_init_map = op->annotations.Get(attr::kBarrierInit)
                                ->as<Map<Var, Array<PrimExpr>>>()
                                .value();

    // Create init calls for each barrier buffer
    // Initialize each barrier element with its respective arrive count
    Array<Stmt> init_mbarrier_calls_;
    for (auto buffer : barrier_buffers) {
      auto data = buffer->data;
      ICHECK(barrier_init_map.count(data))
          << "Barrier buffer " << buffer->name
          << " not found in barrier_init annotation";
      auto arrive_counts = barrier_init_map.at(data);
      ICHECK(arrive_counts.size() ==
             static_cast<size_t>(buffer->shape[0].as<IntImmNode>()->value))
          << "The number of arrive counts (" << arrive_counts.size()
          << ") must match the barrier buffer size (" << buffer->shape[0]
          << ") for buffer " << buffer->name;

      for (size_t i = 0; i < arrive_counts.size(); i++) {
        auto call =
            Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
                 {BufferLoad(buffer,
                             {IntImm(DataType::Int(32), static_cast<int>(i))}),
                  arrive_counts[i]});
        init_mbarrier_calls_.push_back(Evaluate(call));
      }
    }
    if (init_mbarrier_calls_.empty())
      return block;

    Array<Stmt> new_body;
    PrimExpr condition;
    if (!disable_shuffle_elect_) {
      condition = Call(DataType::Bool(), tl_shuffle_elect(), {0});
    } else {
      condition = EQ(thread_var_->var, 0);
    }
    new_body.push_back(IfThenElse(condition,
                                  init_mbarrier_calls_.size() == 1
                                      ? init_mbarrier_calls_.back()
                                      : SeqStmt(init_mbarrier_calls_),
                                  Stmt()));

    new_body.push_back(
        Evaluate(Call(DataType::Handle(), ptx_fence_barrier_init(), {})));
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
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto buffer = store->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[store->buffer];
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
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  // Mapping from data Var of a Buffer to Buffer, for lookup
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  // Disable shuffle elect for the warp specialized kernel
  bool disable_shuffle_elect_;
};

PrimFunc LowerSharedBarrier(PrimFunc f, bool disable_shuffle_elect) {
  f.CopyOnWrite()->body =
      SharedBarrierRewriter::Rewrite(f->body, disable_shuffle_elect);
  return f;
}

namespace transform {
using namespace tir::transform;

tvm::transform::Pass LowerSharedBarrier() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool disable_shuffle_elect =
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value();
    return tl::LowerSharedBarrier(std::move(f), disable_shuffle_elect);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerSharedBarrier", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerSharedBarrier", LowerSharedBarrier);
}

} // namespace transform
} // namespace tl
} // namespace tvm
