#include "backend/common/target_utils.h"
#include "compat/metadata.h"
#include "compat/visit.h"
#include "cuda/op/builtin.h"
#include "op/gpu.h"

#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;

const CallNode *EvaluateCall(const Stmt &statement) {
  const auto *evaluation = statement.as<EvaluateNode>();
  return evaluation ? evaluation->value.as<CallNode>() : nullptr;
}

bool IsBefore(const CallNode *call) {
  return call && call->op.same_as(tcgen05_before_thread_sync());
}

bool IsAfter(const CallNode *call) {
  return call && call->op.same_as(tcgen05_after_thread_sync());
}

bool IsSharedSync(const CallNode *call) {
  if (!call || !tl::IsStorageSyncOp(call->op) || call->args.empty())
    return false;
  const auto *scope = call->args[0].as<StringImmNode>();
  return scope && (scope->value == "shared" || scope->value == "shared.dyn" ||
                   scope->value == "cluster");
}

bool IsArrive(const CallNode *call) {
  if (call)
    if (const auto *operation = call->op.as<OpNode>();
        operation && operation->name.starts_with("tirx.ptx.mbarrier_arrive"))
      return true;
  return call && (call->op.same_as(mbarrier_arrive()) ||
                  call->op.same_as(ptx_arrive_cluster_barrier()) ||
                  call->op.same_as(mbarrier_arrive_expect_tx()));
}

bool IsWait(const CallNode *call) {
  return call && call->op.same_as(mbarrier_wait_parity());
}

bool IsSync(const CallNode *call) {
  return IsSharedSync(call) || IsArrive(call) || IsWait(call) ||
         (call && call->op.same_as(tcgen05_mma_arrive()));
}

bool UsesTmem(const Stmt &statement) {
  bool found = false;
  PostOrderVisit(statement, [&](const ObjectRef &object) {
    const auto *call = object.as<CallNode>();
    if (!call || IsBefore(call) || IsAfter(call))
      return;
    found |= call->op.same_as(ptx_tcgen05_mma_ss()) ||
             call->op.same_as(ptx_tcgen05_mma_ts()) ||
             call->op.same_as(ptx_tcgen05_mma_blockscaled_ss()) ||
             call->op.same_as(tcgen05_ld()) || call->op.same_as(tcgen05_st()) ||
             call->op.same_as(tcgen05_mma_arrive()) ||
             call->op.same_as(ptx_init_tensor_memory()) ||
             call->op.same_as(ptx_deallocate_tensor_memory());
  });
  return found;
}

bool HasNearbyUse(const Array<Stmt> &statements, int index, int direction) {
  for (int position = index + direction;
       position >= 0 && position < static_cast<int>(statements.size());
       position += direction) {
    const CallNode *call = EvaluateCall(statements[position]);
    if ((direction > 0 && IsAfter(call)) || (direction < 0 && IsBefore(call)))
      return false;
    if (UsesTmem(statements[position]))
      return true;
    if (IsBefore(call) || IsAfter(call) || IsSync(call))
      return false;
  }
  return false;
}

Stmt Fence(bool before, const Location &location) {
  return Evaluate(
      Call(PrimType::Void(),
           before ? tcgen05_before_thread_sync() : tcgen05_after_thread_sync(),
           {}, DictAttrs(), {}, location),
      location);
}

class FenceInjector : public tvm::tl::StmtExprMutator {
private:
  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node, InplaceMode mode) final {
    SeqStmt original = GetRef<SeqStmt>(node);
    SeqStmt visited = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                          .ValueOrUnchanged(original)
                          .as_or_throw<SeqStmt>();
    Array<Stmt> result;
    for (size_t index = 0; index < visited->seq.size(); ++index) {
      const Stmt &statement = visited->seq[index];
      const CallNode *call = EvaluateCall(statement);
      bool before = IsSharedSync(call) ||
                    (IsArrive(call) && HasNearbyUse(visited->seq, index, -1));
      bool after = IsSharedSync(call) ||
                   (IsWait(call) && HasNearbyUse(visited->seq, index, 1));
      if (before &&
          (index == 0 || !IsBefore(EvaluateCall(visited->seq[index - 1])))) {
        result.push_back(Fence(true, statement->loc));
      }
      result.push_back(statement);
      if (after && (index + 1 == visited->seq.size() ||
                    !IsAfter(EvaluateCall(visited->seq[index + 1])))) {
        result.push_back(Fence(false, statement->loc));
      }
    }
    return SeqStmt(result, node->loc);
  }
};
} // namespace

tvm::transform::Pass InjectTcgen05Fence() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    auto target = function->GetAttr<Target>("target");
    if (function->body && target && TargetHasSMVersionGE(*target, 100) &&
        UsesTmem(*function->body)) {
      SeqStmt body = *function->body;
      auto injector = ffi::make_object<FenceInjector>();
      function.CopyOnWrite()->body =
          injector->Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.cuda.InjectTcgen05Fence");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.InjectTcgen05Fence",
                                   InjectTcgen05Fence);
}
} // namespace tvm::tl
