#include "backend/common/target_utils.h"
#include "compat/visit.h"
#include "cuda/op/builtin.h"
#include "op/operator.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;

enum class ProxyEvent { None, Generic, Async, Fence };
constexpr unsigned kClean = 1;
constexpr unsigned kDirty = 2;

bool Shared(const Type &type) {
  if (const auto *pointer = type.as<PtrTypeNode>()) {
    return pointer->scope.name().find("shared") == 0;
  }
  if (const auto *tensor = type.as<tirx::TensorTypeNode>()) {
    return tensor->scope.name().find("shared") == 0;
  }
  return false;
}

bool MayWriteThrough(const Expr &expression) {
  const auto *call = expression.as<CallNode>();
  if (call && call->op.same_as(access_ptr()) && call->args.size() == 3) {
    if (!(GetConservativeAccessMask(call->args[2].as_or_throw<PrimExpr>()) &
          kAccessWrite))
      return false;
    return Shared(expression->ty);
  }
  if (call && call->attrs) {
    auto mask = call->attrs.value().as_or_throw<DictAttrs>().GetAttr<PrimExpr>(
        "tl.access_mask");
    if (mask && !(GetConservativeAccessMask(*mask) & kAccessWrite))
      return false;
  }
  if (expression.as<TensorLoadNode>())
    return false;
  if (Shared(expression->ty))
    return true;
  if (call) {
    for (const Expr &argument : call->args) {
      if (MayWriteThrough(argument))
        return true;
    }
  }
  return false;
}

ProxyEvent Classify(const CallNode *call) {
  if (call->op.same_as(fence_proxy_async()))
    return ProxyEvent::Fence;
  if (const auto *operation = call->op.as<OpNode>();
      operation && (operation->name.starts_with("tirx.ptx.mbarrier_") ||
                    operation->name == "tirx.ptx.cp_async_mbarrier_arrive" ||
                    operation->name == "tirx.cuda.cvta_generic_to_shared"))
    return ProxyEvent::None;
  for (const Op &operation :
       {tma_load(), tma_load_im2col(), tma_load_multicast(), tma_load_gather4(),
        tma_store(), tma_store_scatter4(), ptx_wgmma_ss(), ptx_wgmma_rs(),
        ptx_wgmma_sp_ss(), ptx_wgmma_sp_rs(), ptx_tcgen05_mma_ss(),
        ptx_tcgen05_mma_blockscaled_ss(), ptx_tcgen05_mma_ts()}) {
    if (call->op.same_as(operation))
      return ProxyEvent::Async;
  }
  for (const Op &operation :
       {mbarrier_arrive(), mbarrier_init(), mbarrier_expect_tx(),
        mbarrier_wait_parity(), mbarrier_arrive_expect_tx(),
        ptx_arrive_cluster_barrier(), tcgen05_mma_arrive()}) {
    if (call->op.same_as(operation))
      return ProxyEvent::None;
  }
  if (SideEffect(GetRef<Call>(call)) <= CallEffectKind::kReadState)
    return ProxyEvent::None;
  for (const Expr &argument : call->args) {
    if (MayWriteThrough(argument))
      return ProxyEvent::Generic;
  }
  return ProxyEvent::None;
}

struct EventSummary {
  bool generic = false;
  bool async = false;
  bool fence = false;
};

EventSummary Summarize(const ObjectRef &value) {
  EventSummary summary;
  PostOrderVisit(value, [&](const ObjectRef &object) {
    if (const auto *store = object.as<TensorStoreNode>())
      summary.generic |= Shared(store->dest->ty);
    if (const auto *call = object.as<CallNode>()) {
      ProxyEvent event = Classify(call);
      summary.generic |= event == ProxyEvent::Generic;
      summary.async |= event == ProxyEvent::Async;
      summary.fence |= event == ProxyEvent::Fence;
    }
  });
  return summary;
}

Stmt Fence(const Location &location) {
  return Evaluate(
      Call(std::nullopt, fence_proxy_async(), {}, DictAttrs(), {}, location),
      location);
}

struct RewriteResult {
  Stmt statement;
  unsigned fallthrough;
  unsigned breaks = 0;
  unsigned continues = 0;
};

class ProxyFenceRewriter {
public:
  RewriteResult Rewrite(const Stmt &statement, unsigned input) {
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      Array<Stmt> result;
      unsigned state = input;
      unsigned breaks = 0;
      unsigned continues = 0;
      for (const Stmt &child : sequence->seq) {
        auto rewritten = Rewrite(child, state);
        result.push_back(rewritten.statement);
        state = rewritten.fallthrough;
        breaks |= rewritten.breaks;
        continues |= rewritten.continues;
      }
      return {SeqStmt(result, statement->loc), state, breaks, continues};
    }
    if (statement.as<BreakNode>())
      return {statement, 0, input};
    if (statement.as<ContinueNode>())
      return {statement, 0, 0, input};
    if (const auto *branch = statement.as<IfNode>()) {
      auto condition = Summarize(branch->condition);
      TVM_FFI_CHECK(!condition.async && !condition.generic && !condition.fence,
                    ValueError)
          << "Proxy operations in an if condition must be bound before the "
             "condition";
      auto summary = Summarize(statement);
      bool hoist = (input & kDirty) && summary.async && !summary.generic &&
                   !summary.fence &&
                   SideEffect(branch->condition) == CallEffectKind::kPure;
      unsigned entry = hoist ? kClean : input;
      auto then_result = Rewrite(branch->then_case, entry);
      auto else_result = branch->else_case ? Rewrite(*branch->else_case, entry)
                                           : RewriteResult{SeqStmt({}), entry};
      Stmt conditional = If(
          branch->condition, SeqStmt(then_result.statement),
          branch->else_case ? Optional<SeqStmt>(SeqStmt(else_result.statement))
                            : std::nullopt,
          branch->loc);
      if (hoist)
        conditional = SeqStmt({Fence(branch->loc), conditional});
      return {conditional, then_result.fallthrough | else_result.fallthrough,
              then_result.breaks | else_result.breaks,
              then_result.continues | else_result.continues};
    }
    if (const auto *loop = statement.as<ForNode>()) {
      for (const PrimExpr &header : {loop->begin, loop->end, loop->step}) {
        auto summary = Summarize(header);
        TVM_FFI_CHECK(!summary.generic && !summary.async && !summary.fence,
                      ValueError)
            << "Proxy operations in loop bounds must be bound before the loop";
      }
      return RewriteLoop(statement, loop->body, input);
    }
    if (const auto *loop = statement.as<WhileNode>()) {
      auto condition = Summarize(loop->condition);
      TVM_FFI_CHECK(!condition.generic && !condition.async && !condition.fence,
                    ValueError)
          << "Proxy operations in a while condition must be expressed in the "
             "loop body";
      return RewriteLoop(statement, loop->body, input);
    }
    if (const auto *region = statement.as<RegionStmtNode>()) {
      bool device = region->op->name == "tirx.device_scope";
      auto body = Rewrite(region->body, device ? kClean : input);
      return {RegionStmt(region->op, region->args, region->body_params,
                         region->attrs, SeqStmt(body.statement),
                         region->result_vars, region->loc),
              device ? input : body.fallthrough, body.breaks, body.continues};
    }
    auto summary = Summarize(statement);
    const auto *store = statement.as<TensorStoreNode>();
    bool store_only_generic = store && !Summarize(store->value).generic;
    TVM_FFI_CHECK(!summary.async || (!summary.fence &&
                                     (!summary.generic || store_only_generic)),
                  ValueError)
        << "Mixed generic/async proxy calls must be split into separate "
           "statements";
    unsigned output = input;
    bool inject = input && (input & kDirty) && summary.async;
    if (summary.async || summary.fence)
      output = input ? kClean : 0;
    if (summary.generic)
      output = input ? kDirty : 0;
    if (statement.as<ReturnNode>())
      output = 0;
    return {inject ? Stmt(SeqStmt({Fence(statement->loc), statement},
                                  statement->loc))
                   : statement,
            output};
  }

private:
  RewriteResult RewriteLoop(const Stmt &statement, const SeqStmt &body,
                            unsigned input) {
    EventSummary summary = Summarize(body);
    bool hoist = (input & kDirty) && summary.async && !summary.generic;
    unsigned entry = hoist ? kClean : input;
    unsigned header = entry;
    RewriteResult result{body, entry};
    for (int iteration = 0; iteration < 4; ++iteration) {
      result = Rewrite(body, header);
      unsigned next = entry | result.fallthrough | result.continues;
      if (next == header)
        break;
      header = next;
      TVM_FFI_ICHECK_LT(iteration, 3);
    }
    unsigned output =
        entry | result.fallthrough | result.continues | result.breaks;
    Stmt rewritten = statement;
    if (const auto *loop = statement.as<ForNode>()) {
      rewritten = For(loop->var, loop->begin, loop->end, loop->step, loop->kind,
                      SeqStmt(result.statement), loop->annotations, loop->loc);
    } else {
      const auto *while_loop = statement.as<WhileNode>();
      rewritten = While(while_loop->condition, SeqStmt(result.statement),
                        while_loop->loc);
    }
    if (hoist)
      rewritten = SeqStmt({Fence(statement->loc), rewritten}, statement->loc);
    return {rewritten, output};
  }
};
} // namespace

tvm::transform::Pass InjectFenceProxy() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    auto target = function->GetAttr<Target>("target");
    if (function->body && target && TargetHasBulkCopy(*target)) {
      function.CopyOnWrite()->body = SeqStmt(
          ProxyFenceRewriter().Rewrite(*function->body, kClean).statement);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.cuda.InjectFenceProxy");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.InjectFenceProxy",
                                   InjectFenceProxy);
}
} // namespace tvm::tl
