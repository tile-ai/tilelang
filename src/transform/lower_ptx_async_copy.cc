/*!
 * \brief Lower eligible global->shared copies into PTX cp.async
 * \file lower_ptx_async_copy.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "../op/builtin.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "tir/ir/buffer_common.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {

using namespace tir;

class PTXAsyncCopyInjector : public StmtMutator {
public:
  explicit PTXAsyncCopyInjector(bool enable_auto_async_copy)
      : enable_auto_async_copy_(enable_auto_async_copy) {}

  Stmt Finalize(Stmt body) {
    if (!pending_sync_copies_) {
      return body;
    }

    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(body);
    if (uncommitted_sync_copies_) {
      seq.push_back(MakeCommitGroupStmt());
    }
    seq.push_back(MakeWaitGroupStmt(0));
    pending_sync_copies_ = false;
    uncommitted_sync_copies_ = false;
    return SeqStmt(seq);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Track nested vectorized loop extents so we can rewrite element-wise
    // copies (e.g. float16 stores) into `tir.ptx_cp_async` with element bytes,
    // relying on the later `tl.VectorizeLoop` pass to widen:
    //   for v in T.vectorized(k): ptx_cp_async(dst, src, elem_bytes)
    // => ptx_cp_async(dst_base, src_base, elem_bytes * k)
    //
    // This mirrors the logic in `CPAsyncStoreRewriter` used by `T.copy`
    // lowering, and avoids duplicating vectorize-loop collapse here.
    int previous_vectorized_lanes = current_vectorized_lanes_;
    if (op->kind == ForKind::kVectorized) {
      if (const auto *extent_imm = op->extent.as<IntImmNode>()) {
        int lanes = static_cast<int>(extent_imm->value);
        if (lanes > 1 && current_vectorized_lanes_ <=
                             std::numeric_limits<int>::max() / lanes) {
          current_vectorized_lanes_ *= lanes;
        }
      }
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    current_vectorized_lanes_ = previous_vectorized_lanes;
    return stmt;
  }

  Optional<Stmt> TryInjectPTX(const BufferLoadNode *load,
                              const BufferStoreNode *store,
                              bool predicated = false,
                              const PrimExpr &predicate_value = PrimExpr()) {
    if (!IsGlobalBuffer(load->buffer)) {
      return Optional<Stmt>();
    }

    Optional<PrimExpr> src_index_opt =
        FlattenToLinearOffset(load->buffer, load->indices);
    Optional<PrimExpr> dst_index_opt =
        FlattenToLinearOffset(store->buffer, store->indices);
    if (!src_index_opt.defined() || !dst_index_opt.defined()) {
      return Optional<Stmt>();
    }
    PrimExpr src_index = src_index_opt.value();
    PrimExpr dst_index = dst_index_opt.value();

    if (src_index->dtype.lanes() != dst_index->dtype.lanes()) {
      // Not a straightforward vectorized copy; skip.
      return Optional<Stmt>();
    }

    const int index_lanes = src_index->dtype.lanes();
    const int value_lanes = load->dtype.lanes();
    if (value_lanes > 1 && index_lanes > 1 && value_lanes != index_lanes) {
      // Mismatched vector lane representations; be conservative.
      return Optional<Stmt>();
    }
    const int lanes = std::max(value_lanes, index_lanes);
    const int elem_bytes = lanes * load->dtype.bytes();
    const int total_bytes = static_cast<int>(elem_bytes) *
                            static_cast<int>(current_vectorized_lanes_);
    if (total_bytes != 4 && total_bytes != 8 && total_bytes != 16) {
      return Optional<Stmt>();
    }
    const int bytes = elem_bytes;

    auto dst_elem_type = GetPointerType(store->buffer->data->type_annotation);
    auto src_elem_type = GetPointerType(load->buffer->data->type_annotation);
    if (!dst_elem_type.has_value() || !src_elem_type.has_value()) {
      // Be conservative: if pointer metadata is missing, skip injection.
      return Optional<Stmt>();
    }

    auto make_cp_async_stmt =
        [&](const PrimExpr &dst_offset,
            const PrimExpr &src_offset) -> Optional<Stmt> {
      int dst_elem_count = bytes / dst_elem_type->bytes();
      int src_elem_count = bytes / src_elem_type->bytes();
      if (dst_elem_count <= 0 || src_elem_count <= 0) {
        return Optional<Stmt>();
      }

      PrimExpr dst_access_ptr = store->buffer.access_ptr(
          2, DataType::Handle(), 1, dst_offset, PrimExpr(dst_elem_count));
      PrimExpr src_access_ptr = load->buffer.access_ptr(
          1, DataType::Handle(), 1, src_offset, PrimExpr(src_elem_count));

      ffi::Array<PrimExpr> cp_async_args;
      if (predicated) {
        cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes),
                         predicate_value};
      } else {
        cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes)};
      }
      return Evaluate(Call(store->buffer->dtype,
                           tvm::tir::builtin::ptx_cp_async(), cp_async_args));
    };

    if (index_lanes == 1) {
      return make_cp_async_stmt(/*dst_offset=*/dst_index,
                                /*src_offset=*/src_index);
    }

    auto vector_base = [](const PrimExpr &e) -> PrimExpr {
      if (const auto *r = e.as<RampNode>()) {
        return r->base;
      }
      if (const auto *add = e.as<AddNode>()) {
        // Common pattern after flattening a vectorized N-D buffer access:
        //   (broadcast(base_offset) + ramp(vec_base, 1, lanes))
        // or its commuted form:
        //   (ramp(vec_base, 1, lanes) + broadcast(base_offset))
        const PrimExpr &a = add->a;
        const PrimExpr &b = add->b;
        if (const auto *ra = a.as<RampNode>()) {
          if (const auto *bb = b.as<BroadcastNode>()) {
            return tir::Add(ra->base, bb->value);
          }
        }
        if (const auto *rb = b.as<RampNode>()) {
          if (const auto *ba = a.as<BroadcastNode>()) {
            return tir::Add(rb->base, ba->value);
          }
        }
      }
      return PrimExpr();
    };

    if (lanes != index_lanes) {
      // Vector indices must cover the full transfer width.
      return Optional<Stmt>();
    }

    PrimExpr src_offset = vector_base(src_index);
    PrimExpr dst_offset = vector_base(dst_index);

    if (!src_offset.defined() || !dst_offset.defined()) {
      // If we can't extract offsets from vectorized indices, fall back.
      if (predicated) {
        LOG(WARNING)
            << "Cannot extract offsets from vectorized indices for predicated "
               "cp.async; falling back to regular buffer store/load";
      }
      return Optional<Stmt>();
    }
    return make_cp_async_stmt(/*dst_offset=*/dst_offset,
                              /*src_offset=*/src_offset);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // Insert commit+wait at statement boundaries to preserve synchronous
    // semantics for normal global->shared BufferStore copies.
    //
    // Important: avoid flushing inside inner loop bodies just because there
    // are trailing no-op statements (e.g., Evaluate(0)) after the injected
    // cp.async. Instead, treat "pure copy region" statements as part of the
    // copy run and only flush right before the next non-copy statement.
    Array<Stmt> out;
    out.reserve(op->seq.size() + 2);

    bool open_copy_region = pending_sync_copies_;
    bool uncommitted = uncommitted_sync_copies_;
    pending_sync_copies_ = false;
    uncommitted_sync_copies_ = false;

    for (const Stmt &stmt : op->seq) {
      pending_sync_copies_ = false;
      uncommitted_sync_copies_ = false;
      Stmt visited = this->VisitStmt(stmt);
      const CopyRegionAnalysis analysis = AnalyzeCopyRegion(visited);
      bool stmt_has_pending = pending_sync_copies_;
      bool stmt_has_uncommitted = uncommitted_sync_copies_;
      bool stmt_is_pure_copy_region = analysis.is_pure_copy_region;

      // Before we execute a non-copy statement, we must preserve synchronous
      // semantics for injected cp.async stores by making the data visible.
      if (open_copy_region && !stmt_is_pure_copy_region) {
        if (uncommitted) {
          out.push_back(MakeCommitGroupStmt());
        }
        out.push_back(MakeWaitGroupStmt(0));
        open_copy_region = false;
        uncommitted = false;
      }

      // If we are carrying uncommitted injected cp.async into an explicit wait,
      // ensure they are committed so the wait actually covers them.
      if (open_copy_region && uncommitted && analysis.wait > 0) {
        out.push_back(MakeCommitGroupStmt());
        uncommitted = false;
      }

      out.push_back(visited);

      if (stmt_has_pending) {
        open_copy_region = true;
        uncommitted = uncommitted || stmt_has_uncommitted;
      }

      if (analysis.commit > 0) {
        // A commit closes the currently open group, so there are no longer any
        // uncommitted injected cp.async transfers.
        uncommitted = false;
      }

      if (analysis.wait > 0) {
        // Any explicit wait serves as a synchronization boundary for injected
        // synchronous copies.
        open_copy_region = false;
        uncommitted = false;
      }
    }

    pending_sync_copies_ = open_copy_region;
    uncommitted_sync_copies_ = uncommitted;

    if (out.empty()) {
      return Evaluate(0);
    }
    if (out.size() == 1) {
      return out[0];
    }
    return SeqStmt(out);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // Treat branches as separate control flow paths. We propagate pending
    // synchronous copies into both branches (they occur before the branch),
    // but do not let mutations in one branch affect the other.
    bool pending_before = pending_sync_copies_;
    bool uncommitted_before = uncommitted_sync_copies_;

    pending_sync_copies_ = pending_before;
    uncommitted_sync_copies_ = uncommitted_before;
    Stmt then_case = this->VisitStmt(op->then_case);
    bool pending_then = pending_sync_copies_;
    bool uncommitted_then = uncommitted_sync_copies_;

    bool pending_else = pending_before;
    bool uncommitted_else = uncommitted_before;
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      pending_sync_copies_ = pending_before;
      uncommitted_sync_copies_ = uncommitted_before;
      else_case = this->VisitStmt(op->else_case.value());
      pending_else = pending_sync_copies_;
      uncommitted_else = uncommitted_sync_copies_;
    }

    pending_sync_copies_ = pending_then || pending_else;
    uncommitted_sync_copies_ = uncommitted_then || uncommitted_else;

    if (then_case.same_as(op->then_case) &&
        (!else_case.defined() || else_case.same_as(op->else_case))) {
      return tvm::ffi::GetRef<Stmt>(op);
    }
    return IfThenElse(op->condition, then_case, else_case);
  }

  Stmt VisitStmt_(const BufferStoreNode *store) final {
    if (!IsSharedBuffer(store->buffer)) {
      return StmtMutator::VisitStmt_(store);
    }
    // Only lower copies that are either explicitly
    // in the automatic lowering mode controlled by pass config.
    if (!enable_auto_async_copy_) {
      return StmtMutator::VisitStmt_(store);
    }

    if (auto *load = store->value.as<BufferLoadNode>()) {
      Optional<Stmt> injected = TryInjectPTX(load, store);
      if (injected.defined()) {
        pending_sync_copies_ = true;
        uncommitted_sync_copies_ = true;
        return injected.value();
      }
      return StmtMutator::VisitStmt_(store);
    }

    if (auto *call = store->value.as<CallNode>()) {
      // tir.if_then_else is a call to tir::builtin::if_then_else()
      if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3) {
        if (auto *load = call->args[1].as<BufferLoadNode>()) {
          // Only default value of 0 is supported since 0 is the default value
          // used by cp.async ptx.
          bool else_value_is_zero = IsZeroValue(call->args[2]);
          if (else_value_is_zero) {
            Optional<Stmt> injected =
                TryInjectPTX(load, store, /*predicated=*/true,
                             /*predicate_value=*/call->args[0]);
            if (injected.defined()) {
              pending_sync_copies_ = true;
              uncommitted_sync_copies_ = true;
              return injected.value();
            }
          }
        }
      }
    }

    return StmtMutator::VisitStmt_(store);
  }

private:
  static Optional<PrimExpr>
  FlattenToLinearOffset(const Buffer &buf,
                        const ffi::Array<PrimExpr> &indices) {
    // Convert N-D indices (potentially with axis_separators) into a single
    // row-major linear element offset.
    ffi::Array<PrimExpr> physical = buf.OffsetOf(indices);
    Buffer flattened_buf = buf.GetFlattenedBuffer();
    if (physical.size() != flattened_buf->shape.size() || physical.empty()) {
      return Optional<PrimExpr>();
    }

    PrimExpr linear = physical[0];
    for (size_t i = 1; i < physical.size(); ++i) {
      linear = linear * flattened_buf->shape[i] + physical[i];
    }
    return linear;
  }

  struct CopyRegionAnalysis {
    bool is_pure_copy_region = true;
    int commit = 0;
    int wait = 0;
  };

  static CopyRegionAnalysis
  MergeCopyRegionAnalysis(CopyRegionAnalysis a, const CopyRegionAnalysis &b) {
    a.is_pure_copy_region = a.is_pure_copy_region && b.is_pure_copy_region;
    a.commit += b.commit;
    a.wait += b.wait;
    return a;
  }

  static CopyRegionAnalysis AnalyzeCopyRegion(const Stmt &stmt) {
    CopyRegionAnalysis out;
    if (!stmt.defined()) {
      return out;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        out = MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(s));
      }
      return out;
    }
    if (const auto *ite = stmt.as<IfThenElseNode>()) {
      // Ignore the condition: treat it as pure control flow, and only care
      // whether the branches are pure copy regions so we can hoist sync out.
      out = MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(ite->then_case));
      if (ite->else_case.defined()) {
        out = MergeCopyRegionAnalysis(
            out, AnalyzeCopyRegion(ite->else_case.value()));
      }
      return out;
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (is_const_int(eval->value)) {
        return out;
      }
      const auto *call = eval->value.as<CallNode>();
      if (!call) {
        out.is_pure_copy_region = false;
        return out;
      }
      if (call->op.same_as(builtin::ptx_cp_async()) ||
          call->op.same_as(tl::ptx_cp_async())) {
        return out;
      }
      if (call->op.same_as(builtin::ptx_commit_group())) {
        out.commit += 1;
        return out;
      }
      if (call->op.same_as(builtin::ptx_wait_group())) {
        out.wait += 1;
        return out;
      }
      out.is_pure_copy_region = false;
      return out;
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return AnalyzeCopyRegion(let->body);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return AnalyzeCopyRegion(attr->body);
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      return AnalyzeCopyRegion(loop->body);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      if (block->init.defined()) {
        out = MergeCopyRegionAnalysis(out,
                                      AnalyzeCopyRegion(block->init.value()));
      }
      out = MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(block->body));
      return out;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      // Treat the predicate as pure control flow (no side effects). We only
      // care whether the realized body is a pure copy region so we can hoist
      // the final commit+wait out of sequential loop nests.
      const BlockNode *block = realize->block.get();
      if (block->init.defined()) {
        out = MergeCopyRegionAnalysis(out,
                                      AnalyzeCopyRegion(block->init.value()));
      }
      out = MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(block->body));
      return out;
    }
    out.is_pure_copy_region = false;
    return out;
  }

  static bool IsZeroValue(const PrimExpr &e) {
    if (auto *b = e.as<BroadcastNode>()) {
      return IsZeroValue(b->value);
    }
    if (auto *f = e.as<FloatImmNode>()) {
      return f->value == 0.0f;
    }
    if (auto *i = e.as<IntImmNode>()) {
      return i->value == 0;
    }
    return false;
  }

  static Stmt MakeCommitGroupStmt() {
    return Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
  }

  static Stmt MakeWaitGroupStmt(int n) {
    return Evaluate(Call(DataType::Handle(), builtin::ptx_wait_group(),
                         {IntImm(DataType::Int(32), n)}));
  }

  // Note: AnalyzeCopyRegion replaces both the old `IsPureCopyRegion` and
  // `SummarizeAsyncIntrinsics` helpers to avoid redundant traversals.

  bool enable_auto_async_copy_{true};
  int current_vectorized_lanes_{1};
  bool pending_sync_copies_{false};
  bool uncommitted_sync_copies_{false};
};

using namespace tir::transform;

tvm::transform::Pass LowerPTXAsyncCopy() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto target_opt = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target_opt.defined()) {
      return f;
    }
    Target target = target_opt.value();
    if (!TargetIsCuda(target)) {
      return f;
    }

    if (!TargetHasAsyncCopy(target)) {
      // Graceful fallback on older architectures.
      return f;
    }

    bool enable_auto_async_copy =
        ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();

    auto *n = f.CopyOnWrite();
    PTXAsyncCopyInjector injector(enable_auto_async_copy);
    n->body = injector.Finalize(injector(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerPTXAsyncCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerPTXAsyncCopy", LowerPTXAsyncCopy);
}

} // namespace tl
} // namespace tvm
