/*!
 * \file inject_tcgen05_fence.cc
 * \brief Inject tcgen05.fence::before_thread_sync / after_thread_sync at
 *        conservative TCGEN05/TMEM synchronization boundaries on Blackwell
 *        (SM100+) targets.
 *
 * On Blackwell, the tcgen05 accumulator (TMEM) lives in its own address
 * space. Regular thread synchronization barriers (__syncthreads, mbarrier)
 * do NOT automatically make TMEM writes visible across threads. Two PTX
 * fence instructions bridge this gap:
 *
 *   tcgen05.fence::before_thread_sync  — flush TMEM state before barrier
 *   tcgen05.fence::after_thread_sync   — pull TMEM state after barrier
 *
 * This pass currently handles three patterns when the function targets SM100+
 * and contains tcgen05/TMEM operations:
 *
 *   1. Wrap every tvm_storage_sync("shared") / ("shared.dyn") with
 *      before+after fences.
 *   2. Insert after_thread_sync after mbarrier_wait_parity when a linear
 *      scan of following statements reaches tcgen05/TMEM use before another
 *      synchronization boundary.
 *   3. Insert before_thread_sync before ptx_arrive_barrier /
 *      ptx_arrive_cluster_barrier when a linear reverse scan reaches
 *      tcgen05/TMEM use before another synchronization boundary.
 *
 * It intentionally does not add an extra before_thread_sync around
 * tcgen05_mma_arrive(), because the underlying tcgen05.commit.*.mbarrier
 * already carries the producer-side ordering.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "../op/builtin.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::transform::PassContext;

namespace {

/*!
 * \brief Check if a call is tvm_storage_sync("shared") or
 *        tvm_storage_sync("shared.dyn").
 */
bool IsSharedStorageSync(const CallNode *call) {
  if (!call || !call->op.same_as(builtin::tvm_storage_sync())) {
    return false;
  }
  if (call->args.empty())
    return false;
  const auto *scope = call->args[0].as<StringImmNode>();
  if (!scope)
    return false;
  return scope->value == "shared" || scope->value == "shared.dyn";
}

bool IsMbarrierWaitParity(const CallNode *call) {
  return call && call->op.same_as(mbarrier_wait_parity());
}

bool IsPlainBarrierArrive(const CallNode *call) {
  return call && (call->op.same_as(builtin::ptx_arrive_barrier()) ||
                  call->op.same_as(ptx_arrive_cluster_barrier()));
}

bool IsBeforeFenceCall(const CallNode *call) {
  return call && call->op.same_as(tcgen05_before_thread_sync());
}

bool IsAfterFenceCall(const CallNode *call) {
  return call && call->op.same_as(tcgen05_after_thread_sync());
}

const CallNode *GetEvaluateCall(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    return eval->value.as<CallNode>();
  }
  return nullptr;
}

bool IsTcgen05OrTmemCall(const CallNode *call) {
  if (!call || IsBeforeFenceCall(call) || IsAfterFenceCall(call)) {
    return false;
  }

  return call->op.same_as(ptx_tcgen05_mma_ss()) ||
         call->op.same_as(ptx_tcgen05_mma_ts()) ||
         call->op.same_as(tcgen05_ld()) || call->op.same_as(tcgen05_st()) ||
         call->op.same_as(tcgen05_mma_arrive()) ||
         call->op.same_as(ptx_init_tensor_memory()) ||
         call->op.same_as(ptx_deallocate_tensor_memory());
}

bool StmtUsesTcgen05OrTmem(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *call = node.as<CallNode>()) {
      found = IsTcgen05OrTmemCall(call);
    }
  });
  return found;
}

bool IsBeforeFenceStmt(const Stmt &stmt) {
  return IsBeforeFenceCall(GetEvaluateCall(stmt));
}

bool IsAfterFenceStmt(const Stmt &stmt) {
  return IsAfterFenceCall(GetEvaluateCall(stmt));
}

bool IsFenceSyncBoundary(const CallNode *call) {
  return IsSharedStorageSync(call) || IsMbarrierWaitParity(call) ||
         IsPlainBarrierArrive(call) ||
         (call && call->op.same_as(tcgen05_mma_arrive()));
}

bool HasUpcomingTcgen05Use(const Array<Stmt> &seq, int start_index) {
  for (int i = start_index + 1; i < static_cast<int>(seq.size()); ++i) {
    const Stmt &stmt = seq[i];
    if (IsAfterFenceStmt(stmt)) {
      return false;
    }
    if (StmtUsesTcgen05OrTmem(stmt)) {
      return true;
    }
    if (IsBeforeFenceStmt(stmt) || IsFenceSyncBoundary(GetEvaluateCall(stmt))) {
      return false;
    }
  }
  return false;
}

bool HasPriorTcgen05Use(const Array<Stmt> &seq, int start_index) {
  for (int i = start_index - 1; i >= 0; --i) {
    const Stmt &stmt = seq[i];
    if (IsBeforeFenceStmt(stmt)) {
      return false;
    }
    if (StmtUsesTcgen05OrTmem(stmt)) {
      return true;
    }
    if (IsAfterFenceStmt(stmt) || IsFenceSyncBoundary(GetEvaluateCall(stmt))) {
      return false;
    }
  }
  return false;
}

/*!
 * \brief Check whether the function body contains any tcgen05 / TMEM
 *        operations that warrant fence insertion.
 */
bool HasTcgen05Operations(const Stmt &body) {
  return StmtUsesTcgen05OrTmem(body);
}

inline Stmt MakeBeforeFenceStmt() {
  return Evaluate(Call(DataType::Void(), tcgen05_before_thread_sync(), {}));
}

inline Stmt MakeAfterFenceStmt() {
  return Evaluate(Call(DataType::Void(), tcgen05_after_thread_sync(), {}));
}

inline void AppendFlattened(Array<Stmt> *out, const Stmt &stmt) {
  if (!stmt.defined()) {
    return;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &inner : seq->seq) {
      out->push_back(inner);
    }
    return;
  }
  out->push_back(stmt);
}

/*!
 * \brief Rewriter for conservative TCGEN05/TMEM handoff points.
 *
 * Supported rewrites:
 *
 *   tcgen05_before_thread_sync();
 *   __syncthreads();               // tvm_storage_sync("shared")
 *   tcgen05_after_thread_sync();
 *
 *   mbarrier_wait_parity(...);
 *   tcgen05_after_thread_sync();   // when the subsequent linear region uses
 *                                  // tcgen05/TMEM before another sync point
 *
 *   tcgen05_before_thread_sync();  // when the prior linear region used
 *   ptx_arrive_barrier(...);       // tcgen05/TMEM after the previous sync
 */
class Tcgen05FenceRewriter : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    bool saved_in_seq = in_seq_rewrite_;
    in_seq_rewrite_ = true;

    Array<Stmt> mutated_children;
    for (const Stmt &stmt : op->seq) {
      mutated_children.push_back(VisitStmt(stmt));
    }

    in_seq_rewrite_ = saved_in_seq;

    Array<Stmt> flat_seq;
    for (const Stmt &stmt : mutated_children) {
      AppendFlattened(&flat_seq, stmt);
    }

    Array<Stmt> rewritten;
    for (int i = 0; i < static_cast<int>(flat_seq.size()); ++i) {
      const Stmt &stmt = flat_seq[i];
      const CallNode *call = GetEvaluateCall(stmt);

      if (IsSharedStorageSync(call)) {
        if (i == 0 || !IsBeforeFenceStmt(flat_seq[i - 1])) {
          rewritten.push_back(MakeBeforeFenceStmt());
        }
        rewritten.push_back(stmt);
        if (i + 1 >= static_cast<int>(flat_seq.size()) ||
            !IsAfterFenceStmt(flat_seq[i + 1])) {
          rewritten.push_back(MakeAfterFenceStmt());
        }
        continue;
      }

      if (IsMbarrierWaitParity(call)) {
        rewritten.push_back(stmt);
        bool has_manual_after = i + 1 < static_cast<int>(flat_seq.size()) &&
                                IsAfterFenceStmt(flat_seq[i + 1]);
        if (!has_manual_after && HasUpcomingTcgen05Use(flat_seq, i)) {
          rewritten.push_back(MakeAfterFenceStmt());
        }
        continue;
      }

      if (IsPlainBarrierArrive(call)) {
        bool has_manual_before = i > 0 && IsBeforeFenceStmt(flat_seq[i - 1]);
        if (!has_manual_before && HasPriorTcgen05Use(flat_seq, i)) {
          rewritten.push_back(MakeBeforeFenceStmt());
        }
        rewritten.push_back(stmt);
        continue;
      }

      rewritten.push_back(stmt);
    }

    if (rewritten.size() == 1) {
      return rewritten[0];
    }
    return SeqStmt(std::move(rewritten));
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    if (in_seq_rewrite_) {
      return stmt;
    }
    const auto *call = GetEvaluateCall(stmt);
    if (IsSharedStorageSync(call)) {
      return SeqStmt(
          {MakeBeforeFenceStmt(), std::move(stmt), MakeAfterFenceStmt()});
    }
    return stmt;
  }

private:
  bool in_seq_rewrite_{false};
};

} // namespace

tvm::transform::Pass InjectTcgen05Fence() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    // Only apply on SM100+ (Blackwell) targets.
    Optional<Target> opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target.defined() ||
        !TargetHasSMVersionGE(opt_target.value(), 100)) {
      return f;
    }
    // Only apply if the function actually uses tcgen05 / TMEM operations.
    if (!HasTcgen05Operations(f->body)) {
      return f;
    }
    Tcgen05FenceRewriter rewriter;
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0,
                                            "tl.InjectTcgen05Fence", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectTcgen05Fence", InjectTcgen05Fence);
}

} // namespace tl
} // namespace tvm
