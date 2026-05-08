/*!
 * \file fuse_mbarrier_arrive_expect_tx.cc
 * \brief Fuse simple expect_tx -> TMA issue -> arrive sequences back into
 *        arrive_and_expect_tx before LowerOpaqueBlock.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "merge_if_stmt.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

class MBarrierArriveExpectTxFuser : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    Stmt merged = ApplyMergeIfStmt(f->body);
    f.CopyOnWrite()->body = MBarrierArriveExpectTxFuser().VisitStmt(merged);
    return f;
  }

private:
  static bool Is1DTmaLoad(const CallNode *op) {
    if (!op->op.same_as(tma_load())) {
      return false;
    }
    auto arg0 = op->args[0].as<Call>();
    return arg0 && !arg0.value()->op.same_as(create_tma_descriptor()) &&
           !arg0.value()->op.same_as(create_tma_im2col_descriptor());
  }

  static Optional<Call> GetEvaluateCall(const Stmt &stmt) {
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>()) {
        return tvm::ffi::GetRef<Call>(call);
      }
    }
    return std::nullopt;
  }

  static Optional<PrimExpr> GetTmaBarrier(const Call &call) {
    if (call->op.same_as(tma_load())) {
      return call->args[Is1DTmaLoad(call.get()) ? 2 : 1];
    }
    if (call->op.same_as(tma_load_im2col())) {
      return call->args[1];
    }
    return std::nullopt;
  }

  static bool MatchExpectTx(const Stmt &stmt, PrimExpr *barrier,
                            PrimExpr *bytes) {
    Optional<Call> call = GetEvaluateCall(stmt);
    if (!call.defined() || !call.value()->op.same_as(mbarrier_expect_tx())) {
      return false;
    }
    *barrier = call.value()->args[0];
    *bytes = call.value()->args[1];
    return true;
  }

  static bool MatchArrive(const Stmt &stmt, const PrimExpr &barrier) {
    Optional<Call> call = GetEvaluateCall(stmt);
    return call.defined() &&
           (call.value()->op.same_as(builtin::ptx_arrive_barrier()) ||
            call.value()->op.same_as(tl::ptx_arrive_cluster_barrier())) &&
           ExprDeepEqual()(call.value()->args[0], barrier);
  }

  static bool IsTransparentTmaIssueStmt(const Stmt &stmt,
                                        const PrimExpr &barrier,
                                        bool *saw_tma_load) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &sub_stmt : seq->seq) {
        if (!IsTransparentTmaIssueStmt(sub_stmt, barrier, saw_tma_load)) {
          return false;
        }
      }
      return true;
    }

    if (const auto *loop = stmt.as<ForNode>()) {
      return IsTransparentTmaIssueStmt(loop->body, barrier, saw_tma_load);
    }

    Optional<Call> call = GetEvaluateCall(stmt);
    if (!call.defined()) {
      return false;
    }

    if (call.value()->op.same_as(fence_proxy_async())) {
      return true;
    }

    if (call.value()->op.same_as(builtin::tvm_storage_sync())) {
      if (call.value()->args.size() == 1) {
        if (const auto *scope = call.value()->args[0].as<StringImmNode>()) {
          return scope->value == "shared" || scope->value == "shared.dyn";
        }
      }
      return false;
    }

    Optional<PrimExpr> load_barrier = GetTmaBarrier(call.value());
    if (!load_barrier.defined() ||
        !ExprDeepEqual()(load_barrier.value(), barrier)) {
      return false;
    }

    *saw_tma_load = true;
    return true;
  }

  void FlattenAppend(const Stmt &stmt, Array<Stmt> *out) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &sub_stmt : seq->seq) {
        FlattenAppend(sub_stmt, out);
      }
      return;
    }
    out->push_back(stmt);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> flat_seq;
    for (const Stmt &stmt : op->seq) {
      FlattenAppend(this->VisitStmt(stmt), &flat_seq);
    }

    Array<Stmt> new_seq;
    for (int i = 0, n = flat_seq.size(); i < n; ++i) {
      PrimExpr barrier;
      PrimExpr bytes;
      if (!MatchExpectTx(flat_seq[i], &barrier, &bytes)) {
        new_seq.push_back(flat_seq[i]);
        continue;
      }

      int j = i + 1;
      bool saw_tma_load = false;
      while (j < n &&
             IsTransparentTmaIssueStmt(flat_seq[j], barrier, &saw_tma_load)) {
        ++j;
      }

      if (!saw_tma_load || j >= n || !MatchArrive(flat_seq[j], barrier)) {
        new_seq.push_back(flat_seq[i]);
        continue;
      }

      new_seq.push_back(Evaluate(Call(DataType::Handle(),
                                      builtin::ptx_arrive_barrier_expect_tx(),
                                      {barrier, bytes})));
      for (int k = i + 1; k < j; ++k) {
        new_seq.push_back(flat_seq[k]);
      }
      i = j;
    }

    return new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq);
  }
};

} // namespace

tvm::transform::Pass FuseMBarrierArriveExpectTx() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &) {
    return MBarrierArriveExpectTxFuser::Rewrite(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.FuseMBarrierArriveExpectTx", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.FuseMBarrierArriveExpectTx",
                        FuseMBarrierArriveExpectTx);
}

} // namespace tl
} // namespace tvm
