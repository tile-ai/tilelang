#include "compat/analyzer.h"
#include "op/dtype.h"
#include "transform/common/loop_aware_mutator.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include "tirx_compat/transform.h"

#include "transform/tirx_passes.h"

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {

using namespace tvm::prim;

struct RemoveNoOpConfigObj : public ffi::Object {
  int64_t max_simplification_steps;
  bool ignore_profiler_call;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RemoveNoOpConfigObj>()
        .def_ro("max_simplification_steps",
                &RemoveNoOpConfigObj::max_simplification_steps,
                "If non-zero, RewriteSimplifier will throw an error "
                "after the number of steps specified.  "
                "For use in debug and testing purposes.",
                refl::DefaultValue(0))
        .def_ro("ignore_profiler_call",
                &RemoveNoOpConfigObj::ignore_profiler_call,
                "If true, profiler calls are rendered as no-ops.",
                refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.transform.RemoveNoOpConfig",
                                    RemoveNoOpConfigObj, ffi::Object);
};

class RemoveNoOpConfig : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RemoveNoOpConfig,
                                                ffi::ObjectRef,
                                                RemoveNoOpConfigObj);
};

TVM_FFI_STATIC_INIT_BLOCK() { RemoveNoOpConfigObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  ::tvm::transform::PassContext::RegisterConfigOption<RemoveNoOpConfig>(
      "tl.RemoveNoOp");
}

class NoOpRemover : public LoopAwareMutator {
public:
  using LoopAwareMutator::Mutate;
  using LoopAwareMutator::Mutate_;
  static Stmt Apply(Stmt stmt, const sym::Analyzer &analyzer,
                    bool ignore_profiler_call = false) {
    auto visitor =
        ffi::make_object<NoOpRemover>(analyzer, ignore_profiler_call);
    return visitor->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
  }

private:
  using Parent = LoopAwareMutator;

public:
  NoOpRemover(const sym::Analyzer &analyzer, bool ignore_profiler_call = false)
      : Parent(analyzer), ignore_profiler_call_(ignore_profiler_call) {}

private:
  UnchangedOr<Stmt> Mutate_(const IfNode *op, InplaceMode inplace_mode) final {
    Stmt stmt = Parent::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<IfNode>();

    if (!op) {
      return stmt;
    }
    if (op->else_case) {
      bool no_op_else = tirx::IsNoOp(op->else_case.value());
      bool no_op_then = tirx::IsNoOp(op->then_case);
      if (no_op_else && no_op_then) {
        return MakeEvaluate(op->condition);
      } else if (no_op_else) {
        return If(op->condition, op->then_case);
      } else if (no_op_then) {
        return If(!op->condition, op->else_case.value());
      } else {
        return stmt;
      }
    } else {
      if (tirx::IsNoOp(op->then_case)) {
        return MakeEvaluate(op->condition);
      } else {
        return stmt;
      }
    }
  }
  UnchangedOr<Stmt> Mutate_(const ForNode *op, InplaceMode inplace_mode) final {
    PrimExpr enters = (op->step > 0 && op->begin < op->end) ||
                      (op->step < 0 && op->begin > op->end);
    if (tl::arith::CanProve(analyzer_, !enters)) {
      return MakeEvaluate({op->begin, op->end, op->step});
    }
    Stmt stmt = Parent::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<ForNode>();
    if (tirx::IsNoOp(op->body) &&
        SideEffect(op->begin) == CallEffectKind::kPure &&
        SideEffect(op->end) == CallEffectKind::kPure &&
        SideEffect(op->step) == CallEffectKind::kPure) {
      return Evaluate(0);
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode *op,
                            InplaceMode inplace_mode) final {
    if (HasSideEffect(op->value)) {
      return ffi::Unchanged();
    } else {
      return Evaluate(0);
    }
  }

  bool HasSideEffect(const Expr &value) {
    if (auto prim = value.as<PrimExpr>())
      return HasSideEffect(prim.value());

    return value.as<CallNode>() != nullptr;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *op,
                            InplaceMode inplace_mode) final {
    TensorStore store = ffi::GetRef<TensorStore>(op);

    auto only_side_effects = [&]() {
      ffi::Array<Stmt> statements;
      statements.push_back(MakeEvaluate(store->value));
      for (const auto &index : store->indices) {
        statements.push_back(MakeEvaluate(index));
      }
      Stmt input = SeqStmt(statements);
      return this->Mutate(input, inplace_mode).ValueOrUnchanged(input);
    };

    if (!::tvm::tl::IsCustomDType(store->value.ty())) {
      PrimExpr stores_existing_value =
          store->value - TensorLoad(store->dest.as_or_throw<TensorVar>(),
                                    store->indices) ==
          0;
      stores_existing_value =
          tl::arith::Simplify(analyzer_, stores_existing_value);
      if (IsOne(stores_existing_value)) {
        return only_side_effects();
      }
    }

    if (const TensorLoadNode *load = store->value.as<TensorLoadNode>()) {
      TensorVar tensor = load->source.as_or_throw<tvm::tirx::TensorVar>();
      if (tensor.same_as(store->dest.as_or_throw<TensorVar>()) &&
          ArrayValueEqual(load->indices, store->indices)) {
        return only_side_effects();
      }
    }

    return store;
  }

  bool ArrayValueEqual(const ffi::Array<PrimExpr> &a,
                       const ffi::Array<PrimExpr> &b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
      if (!tl::arith::CanProveEqual(analyzer_, a[i], b[i])) {
        return false;
      }
    }
    return true;
  }

  bool HasSideEffect(const PrimExpr &value) {
    if (ignore_profiler_call_) {
      if (const CallNode *call = value.as<CallNode>()) {
        static const Op timer_init_op = Op::Get("tirx.cuda.timer_init");
        static const Op timer_start_op = Op::Get("tirx.cuda.timer_start");
        static const Op timer_end_op = Op::Get("tirx.cuda.timer_end");
        static const Op timer_finalize_op = Op::Get("tirx.cuda.timer_finalize");
        if (call->op.same_as(timer_init_op) ||
            call->op.same_as(timer_start_op) ||
            call->op.same_as(timer_end_op) ||
            call->op.same_as(timer_finalize_op)) {
          return false;
        }
      }
    }
    return SideEffect(value) > CallEffectKind::kReadState;
  }

  Stmt MakeEvaluate(Expr value) {
    if (SideEffect(value) > CallEffectKind::kReadState) {
      return Evaluate(value);
    } else {
      return Evaluate(0);
    }
  }
  Stmt MakeEvaluate(const ffi::Array<PrimExpr> &values) {
    ffi::Array<Stmt> stmts;
    for (PrimExpr e : values) {
      if (SideEffect(e) > CallEffectKind::kReadState) {
        stmts.push_back(Evaluate(e));
      }
    }

    return SeqStmt(stmts);
  }

  ffi::Map<Var, sym::IntSet> var_range_map_;
  bool ignore_profiler_call_{false};
};

Stmt RemoveNoOp(Stmt stmt, const sym::Analyzer &analyzer,
                bool ignore_profiler_call) {
  return NoOpRemover::Apply(std::move(stmt), analyzer, ignore_profiler_call);
}

} // namespace

Pass RemoveNoOp() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value())
      return f;
    RemoveNoOpConfig config =
        ctx->GetConfigOrDefault<RemoveNoOpConfig>("tl.RemoveNoOp");

    sym::Analyzer analyzer;
    analyzer->rewrite_simplify.SetMaximumRewriteSteps(
        config->max_simplification_steps);

    bool ignore_profiler_call = config->ignore_profiler_call;

    {
      auto *write_ptr = f.CopyOnWrite();
      write_ptr->body = NoOpRemover::Apply(std::move(write_ptr->body).value(),
                                           analyzer, ignore_profiler_call);
    }
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tl.RemoveNoOp");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl._tirx_passes.RemoveNoOp",
                                   [] { return RemoveNoOp(); });
}

} // namespace tvm::tl::tirx_passes
