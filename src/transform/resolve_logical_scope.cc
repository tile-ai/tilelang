#include <optional>

#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/data_type.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "../op/utils.h"
#include "runtime/thread_storage_scope.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {
PrimExpr MakeLinearThreadId(const Array<IterVar> &thread_vars) {
  DataType dtype = DataType::Int(64);
  PrimExpr linear_id = make_const(dtype, 0);
  PrimExpr stride = make_const(dtype, 1);

  static const char *thread_tags[] = {
      "threadIdx.x",
      "threadIdx.y",
      "threadIdx.z",
  };
  for (const char *tag : thread_tags) {
    for (const IterVar &thread : thread_vars) {
      if (thread->thread_tag != tag) {
        continue;
      }

      PrimExpr index = Cast(dtype, thread->var - thread->dom->min);
      linear_id = linear_id + index * stride;
      stride = stride * Cast(dtype, thread->dom->extent);
      break;
    }
  }
  return linear_id;
}

class ThreadPrivateLoadDetector : public ExprVisitor {
public:
  bool Detect(const PrimExpr &expr) {
    found_ = false;
    VisitExpr(expr);
    return found_;
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    runtime::StorageScope scope =
        runtime::StorageScope::Create(op->buffer.scope());

    switch (scope.rank) {
    case runtime::StorageRank::kWarp:
    case runtime::StorageRank::kLocal:
    case runtime::StorageRank::kWMMAMatrixA:
    case runtime::StorageRank::kWMMAMatrixB:
    case runtime::StorageRank::kWMMAAccumulator:
    case runtime::StorageRank::kAMXTMM:
    case runtime::StorageRank::kMMAMatrixA:
    case runtime::StorageRank::kMMAMatrixB:
    case runtime::StorageRank::kMMAMatrixC:
    case runtime::StorageRank::kMetalSimdGroup:
      found_ = true;
      return;

    case runtime::StorageRank::kGlobal:
    case runtime::StorageRank::kShared:
    case runtime::StorageRank::kTexture:
      ExprVisitor::VisitExpr_(op);
      return;
    }
  }
  bool found_{false};
};

class LogicalScopeResolver : public StmtExprMutator {
public:
  explicit LogicalScopeResolver(int warp_size) : warp_size_(warp_size) {}

private:
  bool HasFullWarps() const {
    if (env_threads_.empty()) {
      return false;
    }

    DataType dtype = DataType::Int(64);
    PrimExpr thread_count = make_const(dtype, 1);
    for (const IterVar &thread : env_threads_) {
      thread_count = thread_count * Cast(dtype, thread->dom->extent);
    }

    arith::Analyzer analyzer;
    PrimExpr warp_size = make_const(dtype, warp_size_);
    return analyzer.CanProve(FloorMod(thread_count, warp_size) == 0);
  }

  bool IsWarpUniformExpr(const PrimExpr &expr) const {
    PrimExpr expanded_expr = Substitute(expr, bind_values_);
    ThreadPrivateLoadDetector detector;
    if (detector.Detect(expanded_expr)) {
      return false;
    }
    Map<Var, PrimExpr> thread_one;
    Map<Var, PrimExpr> thread_two;
    arith::Analyzer analyzer;

    for (const IterVar &thread : env_threads_) {
      Var one(thread->var->name_hint + "<T1>", thread->var->dtype);
      Var two(thread->var->name_hint + "<T2>", thread->var->dtype);

      thread_one.Set(thread->var, one);
      thread_two.Set(thread->var, two);

      analyzer.Bind(one, thread->dom);
      analyzer.Bind(two, thread->dom);
    }
    if (thread_one.empty()) {
      return true;
    }
    PrimExpr linear_id = MakeLinearThreadId(env_threads_);

    PrimExpr lhs = Substitute(expanded_expr, thread_one);
    PrimExpr rhs = Substitute(expanded_expr, thread_two);
    PrimExpr linear_one = Substitute(linear_id, thread_one);
    PrimExpr linear_two = Substitute(linear_id, thread_two);

    PrimExpr warp_size = make_const(DataType::Int(64), warp_size_);

    PrimExpr same_warp =
        FloorDiv(linear_one, warp_size) == FloorDiv(linear_two, warp_size);
    PrimExpr agree;
    if (expanded_expr.dtype().is_bool()) {
      agree = Or(And(lhs, rhs), And(Not(lhs), Not(rhs)));
    } else {
      agree = lhs == rhs;
    }
    return analyzer.CanProve(Or(Not(same_warp), agree));
  }

  bool IsWarpUniformAccess(const CallNode *logical_call) const {
    ICHECK(logical_call->args.size() == 2U || logical_call->args.size() == 3U);
    const auto *access_ptr = logical_call->args[0].as<CallNode>();
    ICHECK(access_ptr != nullptr);
    ICHECK(access_ptr->op.same_as(tl::access_ptr()));
    ICHECK_EQ(access_ptr->args.size(), 3U);
    const auto *base_load = access_ptr->args[0].as<BufferLoadNode>();
    ICHECK(base_load != nullptr)
        << "tl.access_ptr base must be BufferLoad, but got "
        << access_ptr->args[0];
    if (!IsSharedBuffer(base_load->buffer) &&
        !IsGlobalBuffer(base_load->buffer)) {
      return false;
    }
    for (const PrimExpr &index : base_load->indices) {
      if (!IsWarpUniformExpr(index)) {
        return false;
      }
    }
    const PrimExpr &access_extent = access_ptr->args[1];
    if (!IsWarpUniformExpr(access_extent)) {
      return false;
    }

    const PrimExpr &reduction_size = logical_call->args[1];
    if (!IsWarpUniformExpr(reduction_size)) {
      return false;
    }

    return true;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    static const Op &if_then_else_op = Op::Get("tirx.if_then_else");

    if (op->op.same_as(if_then_else_op)) {
      ICHECK_EQ(op->args.size(), 3U);
      PrimExpr condition = VisitExpr(op->args[0]);
      const bool condition_is_uniform = IsWarpUniformExpr(condition);
      const bool parent_is_uniform = is_warp_uniform_;
      is_warp_uniform_ = condition_is_uniform && parent_is_uniform;
      PrimExpr then_case = VisitExpr(op->args[1]);
      is_warp_uniform_ = condition_is_uniform && parent_is_uniform;
      PrimExpr else_case = VisitExpr(op->args[2]);
      is_warp_uniform_ = parent_is_uniform;
      return Call(op->dtype, op->op, {condition, then_case, else_case},
                  op->span);
    }
    PrimExpr visited = StmtExprMutator::VisitExpr_(op);
    const auto *call = visited.as<CallNode>();
    ICHECK(call != nullptr);

    if (!call->op.same_as(Op::Get("tl.any_of")) &&
        !call->op.same_as(Op::Get("tl.all_of"))) {
      return visited;
    }

    ICHECK(call->args.size() == 2U || call->args.size() == 3U);
    if (call->args.size() == 3U) {
      const auto *scope = call->args[2].as<StringImmNode>();
      ICHECK(scope != nullptr);
      if (scope->value != "auto") {
        return visited;
      }
    }

    Array<PrimExpr> args = call->args;
    const bool use_warp =
        is_warp_uniform_ && HasFullWarps() && IsWarpUniformAccess(call);
    if (args.size() == 2U) {
      args.push_back(StringImm(use_warp ? "warp" : "thread"));
    } else {
      args.Set(2, StringImm(use_warp ? "warp" : "thread"));
    }
    return Call(call->dtype, call->op, args, call->span);
  }

  PrimExpr VisitExpr_(const LetNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    PrimExpr expanded_value = Substitute(value, bind_values_);

    Map<Var, PrimExpr> parent_bind_values = bind_values_;
    bind_values_.Set(op->var, expanded_value);
    PrimExpr body = VisitExpr(op->body);
    bind_values_ = parent_bind_values;
    return Let(op->var, value, body, op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tirx::attr::thread_extent) {
      return StmtExprMutator::VisitStmt_(op);
    }
    IterVar thread = Downcast<IterVar>(op->node);
    runtime::ThreadScope scope =
        runtime::ThreadScope::Create(thread->thread_tag);
    if (scope.rank != 1) {
      return StmtExprMutator::VisitStmt_(op);
    }
    env_threads_.push_back(thread);
    Stmt visited = StmtExprMutator::VisitStmt_(op);
    env_threads_.pop_back();
    return visited;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    auto condition = VisitExpr(op->condition);

    bool condition_is_uniform = IsWarpUniformExpr(condition);

    bool parent_is_uniform = is_warp_uniform_;
    is_warp_uniform_ = parent_is_uniform && condition_is_uniform;
    Stmt then_case = VisitStmt(op->then_case);

    Optional<Stmt> else_case = std::nullopt;
    if (op->else_case.defined()) {
      is_warp_uniform_ = parent_is_uniform && condition_is_uniform;
      else_case = VisitStmt(op->else_case.value());
    }
    is_warp_uniform_ = parent_is_uniform;
    return IfThenElse(condition, then_case, else_case, op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto min = VisitExpr(op->min);
    auto extent = VisitExpr(op->extent);

    Optional<PrimExpr> new_step = std::nullopt;
    auto effective_step = make_const(op->loop_var.dtype(), 1);
    if (op->step.defined()) {
      effective_step = VisitExpr(op->step.value());
      new_step = effective_step;
    }
    bool condition_is_uniform = IsWarpUniformExpr(min) &&
                                IsWarpUniformExpr(extent) &&
                                IsWarpUniformExpr(effective_step);

    bool parent_is_uniform = is_warp_uniform_;
    is_warp_uniform_ = parent_is_uniform && condition_is_uniform;
    Stmt body = VisitStmt(op->body);

    is_warp_uniform_ = parent_is_uniform;
    return For(op->loop_var, min, extent, op->kind, body, op->thread_binding,
               op->annotations, new_step, op->span);
  }
  Stmt VisitStmt_(const WhileNode *op) final {
    // In a while loop we always turn "auto" -> "thread"
    bool parent_is_uniform = is_warp_uniform_;
    is_warp_uniform_ = false;
    const auto condition = VisitExpr(op->condition);
    const auto body = VisitStmt(op->body);
    is_warp_uniform_ = parent_is_uniform;
    return While(condition, body, op->span);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    PrimExpr expanded_value = Substitute(value, bind_values_);

    bind_values_.Set(op->var, expanded_value);
    return Bind(op->var, value, op->span);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    auto const parent_bind_values = bind_values_;
    Array<Stmt> statements;
    statements.reserve(op->size());
    for (const Stmt &stmt : op->seq) {
      statements.push_back(VisitStmt(stmt));
    }
    bind_values_ = parent_bind_values;
    return SeqStmt(statements, op->span);
  }

  Array<IterVar> env_threads_;
  Map<Var, PrimExpr> bind_values_;
  bool is_warp_uniform_{true};
  int warp_size_;
};

PrimFunc ResolveLogicalScopePrimFunc(PrimFunc func) {
  if (!func.defined() || !func->body.defined()) {
    return func;
  }

  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined()) << "ResolveLogicalScope requires a bound target";
  auto warp_size_attr = target.value()->GetAttr<Integer>("thread_warp_size");
  ICHECK(warp_size_attr.defined())
      << "ResolveLogicalScope requires target attribute thread_warp_size";
  int warp_size = warp_size_attr.value().IntValue();
  ICHECK_GT(warp_size, 0);

  LogicalScopeResolver resolver(warp_size);
  PrimFuncNode *node = func.CopyOnWrite();
  node->body = resolver(std::move(node->body));
  return func;
}

} // namespace

namespace transform {

tvm::transform::Pass ResolveLogicalScope() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return ResolveLogicalScopePrimFunc(std::move(func));
  };

  return tvm::tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                                  "tl.ResolveLogicalScope", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.ResolveLogicalScope",
                        ResolveLogicalScope);
}

} // namespace transform
} // namespace tl
} // namespace tvm
