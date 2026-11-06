/*!
 * \file fuse_mbarrier_arrive_expect_tx.cc
 * \brief Fuse a local expect_tx / TMA issue / arrive sequence in native IR.
 */
#include "cuda/op/builtin.h"
#include "ir/annotations.h"
#include "op/builtin.h"
#include "transform/merge_if_stmt.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

Optional<Call> EvaluateCall(const Stmt &statement) {
  const auto *evaluate = statement.as<EvaluateObj>();
  return evaluate ? evaluate->value.as<Call>() : std::nullopt;
}

bool HasAnnotations(const Call &call) {
  if (!call->attrs)
    return false;
  const auto *attrs = call->attrs.value().as<DictAttrsObj>();
  return !attrs || !attrs->dict.empty();
}

Optional<Expr> TmaBarrier(const Call &call) {
  auto args = DecodeCallOperands(call.get());
  if (call->op.same_as(tma_load())) {
    if (args.size() < 3)
      return std::nullopt;
    // The 1D form begins with the shared destination pointer; a tiled load
    // begins with a tensor-map descriptor, which is not in shared memory.
    const auto *pointer = args[0]->ty.as<PtrTypeObj>();
    bool one_dimensional = pointer && (pointer->scope.name() == "shared" ||
                                       pointer->scope.name() == "shared.dyn");
    return args[one_dimensional ? 2 : 1];
  }
  if (call->op.same_as(tma_load_im2col()) && args.size() >= 2)
    return args[1];
  return std::nullopt;
}

bool MatchExpectTx(const Stmt &statement, Expr *barrier, Expr *bytes) {
  auto call = EvaluateCall(statement);
  if (!call || !call.value()->op.same_as(mbarrier_expect_tx()) ||
      HasAnnotations(*call))
    return false;
  auto args = DecodeCallOperands(call.value().get());
  if (args.size() != 2)
    return false;
  *barrier = args[0];
  *bytes = args[1];
  return true;
}

bool MatchLocalArrive(const Stmt &statement, const Expr &barrier) {
  auto call = EvaluateCall(statement);
  // A cluster arrive updates the remote CTA's barrier, whereas expect_tx
  // updates the local one. Combining these into one local instruction drops
  // the remote-CTA operand. Retain the original two operations in that case.
  if (!call || !call.value()->op.same_as(ptx_arrive_barrier()) ||
      HasAnnotations(*call))
    return false;
  auto args = DecodeCallOperands(call.value().get());
  return args.size() == 1 && StructuralEqual()(args[0], barrier);
}

bool TransparentTmaIssue(const Stmt &statement, const Expr &barrier,
                         bool *saw_load) {
  if (const auto *sequence = statement.as<SeqStmtObj>()) {
    for (const Stmt &child : sequence->seq)
      if (!TransparentTmaIssue(child, barrier, saw_load))
        return false;
    return true;
  }
  if (const auto *loop = statement.as<ForObj>()) {
    if (SideEffect(loop->begin) > CallEffectKind::kPure ||
        SideEffect(loop->end) > CallEffectKind::kPure ||
        SideEffect(loop->step) > CallEffectKind::kPure)
      return false;
    return TransparentTmaIssue(loop->body, barrier, saw_load);
  }
  auto call = EvaluateCall(statement);
  if (!call)
    return false;
  if (call.value()->op.same_as(fence_proxy_async()))
    return true;
  if (call.value()->op.same_as(gpu_storage_sync_op())) {
    if (call.value()->args.size() != 1)
      return false;
    const auto *scope = call.value()->args[0].as<StringImmObj>();
    return scope && (scope->value == "shared" || scope->value == "shared.dyn");
  }
  auto load_barrier = TmaBarrier(*call);
  if (!load_barrier || !StructuralEqual()(*load_barrier, barrier))
    return false;
  *saw_load = true;
  return true;
}

class MBarrierArriveExpectTxFuser : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *sequence,
                            InplaceMode mode) final {
    Array<Stmt> flat;
    for (const Stmt &child : sequence->seq)
      FlattenAppend(Mutate(child, mode).ValueOrUnchanged(child), &flat);
    Array<Stmt> output;
    for (size_t i = 0; i < flat.size(); ++i) {
      Expr barrier(UnsafeInit{}), bytes(UnsafeInit{});
      if (!MatchExpectTx(flat[i], &barrier, &bytes)) {
        output.push_back(flat[i]);
        continue;
      }
      size_t j = i + 1;
      bool saw_load = false;
      while (j < flat.size() &&
             TransparentTmaIssue(flat[j], barrier, &saw_load))
        ++j;
      if (!saw_load || j == flat.size() ||
          !MatchLocalArrive(flat[j], barrier)) {
        output.push_back(flat[i]);
        continue;
      }
      output.push_back(
          Evaluate(Call(PrimType::Void(), ptx_arrive_barrier_expect_tx(),
                        {barrier, bytes}, {}, {}, flat[i]->loc),
                   flat[i]->loc));
      for (size_t k = i + 1; k < j; ++k)
        output.push_back(flat[k]);
      i = j;
    }
    return SeqStmt(output, sequence->loc);
  }

private:
  static void FlattenAppend(const Stmt &statement, Array<Stmt> *output) {
    if (const auto *sequence = statement.as<SeqStmtObj>()) {
      for (const Stmt &child : sequence->seq)
        FlattenAppend(child, output);
    } else {
      output->push_back(statement);
    }
  }
};
} // namespace

tvm::transform::Pass FuseMBarrierArriveExpectTx() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    Stmt merged = ApplyMergeIfStmt(function->body);
    auto fuser = make_object<MBarrierArriveExpectTxFuser>();
    function.CopyOnWrite()->body =
        fuser->Mutate(merged).ValueOrUnchanged(merged).as_or_throw<SeqStmt>();
    return function;
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0,
                                             "tl.FuseMBarrierArriveExpectTx");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.cuda.transform.FuseMBarrierArriveExpectTx",
      FuseMBarrierArriveExpectTx);
}
} // namespace tvm::tl
