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

namespace tvm {
namespace tl {

using namespace tir;

class SharedBarrierRewriter : public StmtExprMutator {
public:
  /**
   * @brief Rewrite shared.barrier allocations and inserts PTX barrier initialization.
   *
   * Constructs a SharedBarrierRewriter with the given flag and applies it to the provided
   * statement body. Barrier buffers in `body` with storage scope "shared.barrier" are
   * replaced by one-element shared buffers and an initialization sequence (PTX `init_barrier`
   * calls plus `tvm_storage_sync("shared")`) is prepended to the corresponding block(s).
   *
   * @param body The statement (IR) to transform.
   * @param disable_shuffle_elect If false (default), the inserted initialization is gated
   *        by `tl_shuffle_elect()`; if true, the gating falls back to checking the
   *        thread index equals zero (uses recorded `threadIdx.x` IterVar).
   * @return Stmt The transformed statement with barrier buffers rewritten and
   *         initialization logic inserted.
   */
  static Stmt Rewrite(Stmt body, bool disable_shuffle_elect = false) {
    SharedBarrierRewriter rewriter(disable_shuffle_elect);
    return rewriter(body);
  }

private:
  /**
       * @brief Construct a SharedBarrierRewriter.
       *
       * @param disable_shuffle_elect When true, disables the shuffle-elect heuristic and
       *                              uses a fallback (thread == 0) to gate barrier initialization.
       */
      SharedBarrierRewriter(bool disable_shuffle_elect)
      : disable_shuffle_elect_(disable_shuffle_elect) {}

  /**
   * @brief Rewrite a Block that contains "shared.barrier" buffers.
   *
   * Scans the block's alloc_buffers and match_buffers for buffers whose
   * pointer type has storage_scope == "shared.barrier". For each such barrier
   * buffer this method:
   * - records the original Buffer in buffer_data_to_buffer_,
   * - creates a replacement one-element Buffer with scope "shared" and
   *   records the mapping in buffer_remap_,
   * - replaces the block's alloc_buffers with the remapped buffers,
   * - inserts initialization calls `ptx_init_barrier_thread_count(new_buf[0], original_count)`
   *   for each barrier buffer, guarded by either `tl_shuffle_elect()` (when
   *   disable_shuffle_elect_ is false) or `thread_var_.var == 0` (when true),
   * - inserts a `tvm_storage_sync("shared")` call, and
   * - prepends the original block body after these setup statements.
   *
   * If no "shared.barrier" buffers are found this method delegates to the
   * base StmtExprMutator visitor and returns that result unchanged.
   *
   * Notes:
   * - Mutates the BlockNode in-place (alloc_buffers and body) when replacements occur.
   * - Requires that thread_var_ is defined; an internal check will fail if it is not.
   *
   * @return The transformed statement (or the result of the base visitor if no changes).
   */
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

    // 2. create new buffers
    Array<Buffer> new_buffers;
    for (auto buffer : barrier_buffers) {
      auto data = buffer->data;
      auto new_buffer = Buffer(data, buffer->dtype, Array<PrimExpr>({1}),
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
    PrimExpr condition;
    if (!disable_shuffle_elect_) {
      condition = Call(DataType::Bool(), tl_shuffle_elect(), {0});
    } else {
      condition = EQ(thread_var_->var, 0);
    }
    new_body.push_back(
        IfThenElse(condition, SeqStmt(init_mbarrier_calls_), Stmt()));
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                      {StringImm("shared")})));
    new_body.push_back(block->body);

    block.CopyOnWrite()->body = SeqStmt(new_body);

    return StmtExprMutator::VisitStmt_(block.get());
  }

  /**
   * @brief Rewrite buffer loads to use a remapped buffer when applicable.
   *
   * If the loaded buffer has an entry in `buffer_remap_`, returns a new BufferLoad
   * that targets the remapped Buffer while preserving the original indices.
   * Otherwise returns the original BufferLoad unchanged.
   *
   * @param op The BufferLoad AST node being visited.
   * @return PrimExpr A BufferLoad expression referencing the remapped buffer when present,
   *         or the original load expression.
   */
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto buffer = load->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[load->buffer];
      return BufferLoad(new_buffer, load->indices);
    }
    return load;
  }

  /**
   * @brief Visit and rewrite buffer store statements, remapping barrier buffers.
   *
   * If the store's target buffer was replaced during rewriting (present in
   * buffer_remap_), returns a new BufferStore that writes to the remapped buffer
   * with the same value and indices. Otherwise returns the (possibly mutated)
   * original store.
   *
   * @return Stmt The rewritten BufferStore or the original store statement.
   */
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

/**
 * @brief Lower shared.barrier buffers in a PrimFunc by rewriting them and inserting PTX
 * initialization calls.
 *
 * Rewrites the function body to replace `shared.barrier` buffers with plain shared buffers
 * and inserts the corresponding barrier initialization sequence. The transformed body is
 * written back into the given PrimFunc (the input is modified in place) and the same
 * PrimFunc is returned.
 *
 * @param f The PrimFunc whose body will be transformed.
 * @param disable_shuffle_elect When false (default behavior), the pass may use a
 *        shuffle-elect heuristic to select the initializing thread; when true, the
 *        initializer is gated on `threadIdx.x == 0`.
 * @return PrimFunc The same PrimFunc with its body replaced by the lowered form.
 */
PrimFunc LowerSharedBarrier(PrimFunc f, bool disable_shuffle_elect) {
  f.CopyOnWrite()->body =
      SharedBarrierRewriter::Rewrite(f->body, disable_shuffle_elect);
  return f;
}

namespace transform {
using namespace tir::transform;

/**
 * @brief Create a transform pass that lowers shared.barrier buffers in PrimFunc bodies.
 *
 * The pass reads the PassContext boolean config option `kDisableShuffleElect` (default false)
 * and invokes tl::LowerSharedBarrier on each PrimFunc, forwarding the flag to control whether
 * barrier initialization uses the shuffle-elect heuristic or a thread-0 fallback.
 *
 * @return tvm::transform::Pass A PrimFunc pass named "tl.LowerSharedBarrier" (opt level 0).
 */
tvm::transform::Pass LowerSharedBarrier() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    bool disable_shuffle_elect =
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value();
    return tl::LowerSharedBarrier(std::move(f), disable_shuffle_elect);
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
