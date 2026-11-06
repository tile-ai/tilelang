/*!
 * \file inject_tcgen05_fence.cc
 * \brief Keep TMEM visibility at native thread-synchronization boundaries.
 *
 * Ordinary shared-memory barriers do not alone order the TMEM proxy. Wrap
 * shared syncs with before/after fences and fence linear wait/use or use/arrive
 * handoffs. tcgen05.commit (tcgen05_mma_arrive) already carries producer-side
 * ordering and must not receive another before-thread-sync fence.
 */
#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "ir/annotations.h"
#include "op/builtin.h"
#include <tvm/ffi/extra/structural_visit.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

const CallObj *EvaluateCall(const Stmt &statement) {
  auto *evaluate = statement.as<EvaluateObj>();
  return evaluate ? evaluate->value.as<CallObj>() : nullptr;
}

bool SharedSync(const CallObj *call) {
  if (!call || !call->op.same_as(gpu_storage_sync_op()) || call->args.empty())
    return false;
  const auto *scope = call->args[0].as<StringImmObj>();
  return scope && (scope->value == "shared" || scope->value == "shared.dyn");
}

bool Wait(const CallObj *call) {
  return call && call->op.same_as(mbarrier_wait_parity());
}

bool Arrive(const CallObj *call) {
  return call && (call->op.same_as(ptx_arrive_barrier()) ||
                  call->op.same_as(ptx_arrive_cluster_barrier()));
}

bool BeforeFence(const CallObj *call) {
  return call && call->op.same_as(tcgen05_before_thread_sync());
}

bool AfterFence(const CallObj *call) {
  return call && call->op.same_as(tcgen05_after_thread_sync());
}

bool Tcgen05OrTmem(const CallObj *call) {
  if (!call || BeforeFence(call) || AfterFence(call))
    return false;
  return call->op.same_as(ptx_tcgen05_mma_ss()) ||
         call->op.same_as(ptx_tcgen05_mma_ts()) ||
         call->op.same_as(ptx_tcgen05_mma_blockscaled_ss()) ||
         call->op.same_as(ptx_tcgen05_cp_warpx4()) ||
         call->op.same_as(tcgen05_ld()) || call->op.same_as(tcgen05_st()) ||
         call->op.same_as(tcgen05_mma_arrive()) ||
         call->op.same_as(ptx_init_tensor_memory()) ||
         call->op.same_as(ptx_deallocate_tensor_memory());
}

bool UsesTmem(const Stmt &statement) {
  bool found = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      statement, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        found |= Tcgen05OrTmem(call);
        return ffi::WalkResult::Advance();
      });
  return found;
}

bool SynchronizationBoundary(const CallObj *call) {
  return SharedSync(call) || Wait(call) || Arrive(call) ||
         (call && call->op.same_as(tcgen05_mma_arrive()));
}

bool UpcomingUse(const Array<Stmt> &statements, size_t at) {
  for (size_t i = at + 1; i < statements.size(); ++i) {
    auto *call = EvaluateCall(statements[i]);
    if (AfterFence(call))
      return false;
    if (UsesTmem(statements[i]))
      return true;
    if (BeforeFence(call) || SynchronizationBoundary(call))
      return false;
  }
  return false;
}

bool PriorUse(const Array<Stmt> &statements, size_t at) {
  for (size_t i = at; i-- > 0;) {
    auto *call = EvaluateCall(statements[i]);
    if (BeforeFence(call))
      return false;
    if (UsesTmem(statements[i]))
      return true;
    if (AfterFence(call) || SynchronizationBoundary(call))
      return false;
  }
  return false;
}

Stmt Fence(bool before, Location loc) {
  auto op = before ? tcgen05_before_thread_sync() : tcgen05_after_thread_sync();
  return Evaluate(Call(PrimType::Void(), op, {}, {}, {}, loc), loc);
}

class Tcgen05FenceRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *sequence,
                            InplaceMode mode) final {
    Array<Stmt> children;
    for (const Stmt &child : sequence->seq)
      children.push_back(Mutate(child, mode).ValueOrUnchanged(child));
    auto flat = SeqStmt(children, sequence->loc)->seq;
    Array<Stmt> output;
    for (size_t i = 0; i < flat.size(); ++i) {
      const Stmt &statement = flat[i];
      const auto *call = EvaluateCall(statement);
      if (SharedSync(call)) {
        if (i == 0 || !BeforeFence(EvaluateCall(flat[i - 1])))
          output.push_back(Fence(true, statement->loc));
        output.push_back(statement);
        if (i + 1 == flat.size() || !AfterFence(EvaluateCall(flat[i + 1])))
          output.push_back(Fence(false, statement->loc));
      } else if (Wait(call)) {
        output.push_back(statement);
        bool manual =
            i + 1 < flat.size() && AfterFence(EvaluateCall(flat[i + 1]));
        if (!manual && UpcomingUse(flat, i))
          output.push_back(Fence(false, statement->loc));
      } else if (Arrive(call)) {
        bool manual = i > 0 && BeforeFence(EvaluateCall(flat[i - 1]));
        if (!manual && PriorUse(flat, i))
          output.push_back(Fence(true, statement->loc));
        output.push_back(statement);
      } else {
        output.push_back(statement);
      }
    }
    return SeqStmt(output, sequence->loc);
  }
};
} // namespace

tvm::transform::Pass InjectTcgen05Fence() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext) {
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    if (!target || !TargetHasSMVersionGE(*target, 100) ||
        function->GetAttr(tvm::attr::kDeclaration).value_or(false) ||
        !UsesTmem(function->body))
      return function;
    auto rewriter = make_object<Tcgen05FenceRewriter>();
    auto body = function->body;
    function.CopyOnWrite()->body =
        rewriter->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0,
                                             "tl.InjectTcgen05Fence");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.InjectTcgen05Fence",
                                   InjectTcgen05Fence);
}
} // namespace tvm::tl
