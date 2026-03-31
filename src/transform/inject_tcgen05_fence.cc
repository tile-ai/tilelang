/*!
 * \file inject_tcgen05_fence.cc
 * \brief Inject tcgen05.fence::before_thread_sync / after_thread_sync
 *        around __syncthreads() calls on Blackwell (SM100+) targets
 *        that use tensor memory (TMEM).
 *
 * On Blackwell, the tcgen05 accumulator (TMEM) lives in its own address
 * space. Regular thread synchronization barriers (__syncthreads, mbarrier)
 * do NOT automatically make TMEM writes visible across threads. Two PTX
 * fence instructions bridge this gap:
 *
 *   tcgen05.fence::before_thread_sync  — flush TMEM state before barrier
 *   tcgen05.fence::after_thread_sync   — pull TMEM state after barrier
 *
 * This pass wraps every tvm_storage_sync("shared") / ("shared.dyn") with
 * the fence pair when the function targets SM100+ and contains tcgen05/TMEM
 * operations.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

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

/*!
 * \brief Check whether the function body contains any tcgen05 / TMEM
 *        operations that warrant fence insertion.
 */
bool HasTcgen05Operations(const Stmt &body) {
  bool found = false;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *eval = node.as<EvaluateNode>()) {
      const auto *call = eval->value.as<CallNode>();
      if (call && (call->op.same_as(ptx_tcgen05_mma_ss()) ||
                   call->op.same_as(ptx_tcgen05_mma_ts()) ||
                   call->op.same_as(tcgen05_mma_arrive()) ||
                   call->op.same_as(ptx_init_tensor_memory()) ||
                   call->op.same_as(ptx_deallocate_tensor_memory()))) {
        found = true;
      }
    }
  });
  return found;
}

inline Stmt MakeBeforeFenceStmt() {
  return Evaluate(Call(DataType::Handle(), tcgen05_before_thread_sync(), {}));
}

inline Stmt MakeAfterFenceStmt() {
  return Evaluate(Call(DataType::Handle(), tcgen05_after_thread_sync(), {}));
}

/*!
 * \brief Rewriter that wraps every shared-memory storage sync with
 *        tcgen05 fence instructions.
 *
 *   tcgen05_before_thread_sync();
 *   __syncthreads();               // tvm_storage_sync("shared")
 *   tcgen05_after_thread_sync();
 */
class Tcgen05FenceRewriter : public StmtExprMutator {
public:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (IsSharedStorageSync(call)) {
      Stmt original = tvm::ffi::GetRef<Evaluate>(op);
      return SeqStmt({MakeBeforeFenceStmt(), original, MakeAfterFenceStmt()});
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

} // namespace

tvm::transform::Pass InjectTcgen05Fence() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    // Only apply on SM100+ (Blackwell) targets.
    Optional<Target> opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target.defined() || !TargetIsSm100(opt_target.value())) {
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
