  #include "support/check.h"
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/target/target.h>

  #include "runtime/thread_storage_scope.h"
  #include "tvm/runtime/data_type.h"
  #include "tvm/tirx/stmt.h"
  #include <tvm/arith/analyzer.h>
  #include <tvm/ir/cast.h>
  #include <tvm/tirx/analysis.h>

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
    for(const char *tag : thread_tags) {
      for(const IterVar &thread : thread_vars) {
        if (thread->thread_tag != tag) {
          continue;
        }

        PrimExpr index =
            Cast(dtype, thread->var - thread->dom->min);
        linear_id = linear_id + index * stride;
        stride = stride * Cast(dtype, thread->dom->extent);
        break;
      }
    }
    return linear_id;
  }

  class LogicalScopeResolver : public StmtExprMutator {
  public:
    explicit LogicalScopeResolver(int warp_size) : warp_size_(warp_size) {}

    PrimExpr Visit(PrimExpr expr) {
      return VisitExpr(std::move(expr));
    }

  private:
    bool IsWarpUniformCondition(const PrimExpr &condition) {
      Map<Var, PrimExpr> thread_one;
      Map<Var, PrimExpr> thread_two;
      arith::Analyzer analyzer;

      for(const IterVar &thread : env_threads_) {
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

      PrimExpr condition_one = Substitute(condition, thread_one);
      PrimExpr condition_two = Substitute(condition, thread_two);
      PrimExpr linear_one = Substitute(linear_id, thread_one);
      PrimExpr linear_two = Substitute(linear_id, thread_two);

      PrimExpr warp_size = make_const(DataType::Int(64), warp_size_);

      PrimExpr same_warp = FloorDiv(linear_one, warp_size) == FloorDiv(linear_two, warp_size);
      PrimExpr agree = Or(And(condition_one, condition_two), And(Not(condition_one), Not(condition_two)));
      return analyzer.CanProve(Or(Not(same_warp), agree));
    }
    PrimExpr VisitExpr_(const CallNode *op) final {
      PrimExpr visited = StmtExprMutator::VisitExpr_(op);
      const auto *call = visited.as<CallNode>();
      ICHECK(call != nullptr);

      if (!call->op.same_as(Op::Get("tl.any_of")) &&
          !call->op.same_as(Op::Get("tl.all_of"))) {
        return visited;
      }

      ICHECK_EQ(call->args.size(), 3U);
      const auto *scope = call->args[2].as<StringImmNode>();
      ICHECK(scope != nullptr);

      if (scope->value != "auto") {
        return visited;
      }

      Array<PrimExpr> args = call->args;
      args.Set(2, StringImm( is_warp_uniform_? "warp" : "thread"));
      return Call(call->dtype, call->op, args, call->span);
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

      bool condition_is_uniform = IsWarpUniformCondition(condition);

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

    Array<IterVar> env_threads_;
    bool is_warp_uniform_{true};
    int warp_size_;
  };

  PrimFunc ResolveLogicalScopePrimFunc(PrimFunc func) {
    if (!func.defined() || !func->body.defined()) {
      return func;
    }

    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined()) << "ResolveLogicalScope requires a bound target";
    auto warp_size_attr =
        target.value()->GetAttr<Integer>("thread_warp_size");
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
