#include "compat/expr.h"
#include "cuda/op/builtin.h"
#include "transform/merge_if_stmt.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

const CallNode *EvaluateCall(const Stmt &statement) {
  const auto *evaluation = statement.as<EvaluateNode>();
  return evaluation ? evaluation->value.as<CallNode>() : nullptr;
}

Expr BarrierReference(Expr expression) {
  while (const auto *call = expression.as<CallNode>()) {
    const auto *operation = call->op.as<OpNode>();
    if (!operation || call->args.size() != 1 ||
        (operation->name != "tirx.cuda.cvta_generic_to_shared" &&
         operation->name != "tirx.address_of" &&
         operation->name != "tirx.reinterpret"))
      break;
    expression = call->args[0];
  }
  return expression;
}

bool NativeUnitArrival(const CallNode *call) {
  const auto *operation = call ? call->op.as<OpNode>() : nullptr;
  if (!operation || operation->name != "tirx.ptx.mbarrier_arrive_nocount" ||
      call->args.size() != 7)
    return false;
  const auto *space = call->args[4].as<StringImmNode>();
  const auto *predicate = call->args[6].as<StringImmNode>();
  return space && space->value != "shared::cluster" && predicate &&
         predicate->value.empty();
}

bool StableBarrier(const Expr &expression) {
  if (const auto *load = expression.as<TensorLoadNode>()) {
    if (!load->source.as<TensorVar>())
      return false;
    for (const PrimExpr &index : load->indices) {
      if (SideEffect(index) != CallEffectKind::kPure)
        return false;
    }
    return true;
  }
  return SideEffect(expression) == CallEffectKind::kPure;
}

bool StableArgument(const Expr &expression) {
  if (SideEffect(expression) == CallEffectKind::kPure)
    return true;
  const auto *call = expression.as<CallNode>();
  if (!call)
    return false;
  static auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
  if (effects.get(call->op, CallEffectKind::kOpaque) != CallEffectKind::kPure)
    return false;
  bool address =
      call->op.same_as(access_ptr()) || call->op.same_as(address_of_op());
  for (size_t index = 0; index < call->args.size(); ++index) {
    if (address && index == 0 && call->args[index].as<TensorLoadNode>()) {
      if (!StableBarrier(call->args[index]))
        return false;
    } else if (!StableArgument(call->args[index])) {
      return false;
    }
  }
  return true;
}

Optional<size_t> TmaBarrierIndex(const CallNode *call) {
  if (call->op.same_as(tma_load_im2col()) ||
      call->op.same_as(tma_load_gather4())) {
    return call->args.size() > 1 ? Optional<size_t>(1) : std::nullopt;
  }
  if (!call->op.same_as(tma_load()) || call->args.size() < 3)
    return std::nullopt;
  if (const auto *descriptor = call->args[0].as<CallNode>()) {
    if (descriptor->op.same_as(create_tma_descriptor()) ||
        descriptor->op.same_as(create_tma_im2col_descriptor()))
      return 1;
  }
  Type type = call->args[0]->ty;
  if (const auto *pointer = type.as<PtrTypeNode>())
    type = pointer->element_type;
  return type.as<TensorMapTypeNode>() ? 1 : 2;
}

bool TransparentIssue(const Stmt &statement, const Expr &barrier,
                      bool *saw_load) {
  if (const auto *sequence = statement.as<SeqStmtNode>()) {
    for (const Stmt &child : sequence->seq) {
      if (!TransparentIssue(child, barrier, saw_load))
        return false;
    }
    return true;
  }
  if (const auto *loop = statement.as<ForNode>()) {
    if (SideEffect(loop->begin) != CallEffectKind::kPure ||
        SideEffect(loop->end) != CallEffectKind::kPure ||
        SideEffect(loop->step) != CallEffectKind::kPure)
      return false;
    return TransparentIssue(loop->body, barrier, saw_load);
  }
  const CallNode *call = EvaluateCall(statement);
  if (!call)
    return false;
  if (call->op.same_as(fence_proxy_async()))
    return call->args.empty();
  auto barrier_index = TmaBarrierIndex(call);
  if (!barrier_index ||
      !StructuralEqual::Equal(call->args[*barrier_index], barrier))
    return false;
  for (size_t index = 0; index < call->args.size(); ++index) {
    if (index != *barrier_index && !StableArgument(call->args[index]))
      return false;
  }
  *saw_load = true;
  return true;
}

class BarrierFuser : public StmtExprMutator {
private:
  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node, InplaceMode mode) final {
    SeqStmt original = GetRef<SeqStmt>(node);
    SeqStmt visited = StmtExprMutator::Mutate_(node, mode)
                          .ValueOrUnchanged(original)
                          .as_or_throw<SeqStmt>();
    Array<Stmt> result;
    for (size_t index = 0; index < visited->seq.size(); ++index) {
      const Stmt &statement = visited->seq[index];
      const CallNode *expect = EvaluateCall(statement);
      if (!expect || !expect->op.same_as(mbarrier_expect_tx()) ||
          expect->args.size() != 2 || !StableBarrier(expect->args[0]) ||
          SideEffect(expect->args[1]) != CallEffectKind::kPure) {
        result.push_back(statement);
        continue;
      }
      bool saw_load = false;
      size_t end = index + 1;
      while (end < visited->seq.size() &&
             TransparentIssue(visited->seq[end], expect->args[0], &saw_load))
        ++end;
      const CallNode *arrive =
          end < visited->seq.size() ? EvaluateCall(visited->seq[end]) : nullptr;
      bool native_arrival = NativeUnitArrival(arrive);
      if (!saw_load || !arrive ||
          (!native_arrival && (!arrive->op.same_as(mbarrier_arrive()) ||
                               arrive->args.size() != 1)) ||
          !StructuralEqual::Equal(BarrierReference(expect->args[0]),
                                  BarrierReference(arrive->args[0]))) {
        result.push_back(statement);
        continue;
      }
      Call fused = Call(std::nullopt, mbarrier_arrive_expect_tx(), expect->args,
                        expect->attrs, {}, expect->loc);
      if (native_arrival) {
        Array<Expr> arguments{
            arrive->args[0],
            prim::cast(expect->args[1].as_or_throw<PrimExpr>(),
                       PrimType::UInt(32)),
            arrive->args[1], StringImm("expect_tx")};
        for (size_t argument = 2; argument < arrive->args.size(); ++argument)
          arguments.push_back(arrive->args[argument]);
        fused =
            Call(std::nullopt, Op::Get("tirx.ptx.mbarrier_arrive_expect_tx"),
                 arguments, arrive->attrs, {}, expect->loc);
      }
      result.push_back(Evaluate(fused, statement->loc));
      for (size_t issue = index + 1; issue < end; ++issue)
        result.push_back(visited->seq[issue]);
      index = end;
    }
    return SeqStmt(result, node->loc);
  }
};
} // namespace

tvm::transform::Pass FuseMBarrierArriveExpectTx() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (function->body) {
      SeqStmt body =
          ApplyMergeIfStmt(function->body.value()).as_or_throw<SeqStmt>();
      auto fuser = ffi::make_object<BarrierFuser>();
      function.CopyOnWrite()->body =
          fuser->Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(
      pass, 0, "tl.cuda.FuseMBarrierArriveExpectTx");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.cuda.transform.FuseMBarrierArriveExpectTx",
      FuseMBarrierArriveExpectTx);
}
} // namespace tvm::tl
