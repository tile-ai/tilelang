#include "compat/analyzer.h"
#include "transform/common/loop_aware_mutator.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include "transform/tirx_passes.h"

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {

using namespace tvm::prim;

namespace {

class StmtSimplifier : public LoopAwareMutator {
public:
  using LoopAwareMutator::Mutate;
  using LoopAwareMutator::Mutate_;
  explicit StmtSimplifier(const sym::Analyzer &analyzer)
      : LoopAwareMutator(analyzer) {}

  Function Run(Function func);

protected:
  using Parent = LoopAwareMutator;

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                               InplaceMode inplace_mode) final;

  UnchangedOr<Stmt> Mutate_(const BindNode *op,
                            InplaceMode inplace_mode) override;

  UnchangedOr<Stmt> Mutate_(const IfNode *op,
                            InplaceMode inplace_mode) override;

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *op,
                            InplaceMode inplace_mode) override;

private:
  bool ArrayDeepEqual(const ffi::Array<PrimExpr> &lhs,
                      const ffi::Array<PrimExpr> &rhs);

  ffi::Optional<bool> ProveCondition(PrimExpr condition) const;

  ffi::Map<Var, PrimExpr> non_inlined_bindings_;
};

Function StmtSimplifier::Run(Function func) {
  MarkTensorParamShapes(func);
  auto *n = func.CopyOnWrite();

  ffi::Map<Var, StringImm> string_bindings;
  n->body = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                std::move(n->body),
                [&](const Bind &bind) -> ffi::UnchangedOr<ffi::Any> {
                  if (!bind.defined())
                    return ffi::Unchanged();
                  if (auto value = bind->value.as<StringImm>()) {
                    string_bindings.Set(bind->var, *value);
                    return ffi::Any(Evaluate(0));
                  }
                  return ffi::Unchanged();
                },
                [&](const Var &var) -> ffi::UnchangedOr<ffi::Any> {
                  if (auto value = string_bindings.Get(var))
                    return ffi::Any(*value);
                  return ffi::Unchanged();
                })
                .as_or_throw<ffi::Optional<SeqStmt>>();
  n->body = Mutate(n->body, InplaceMode::kAllow).ValueOrUnchanged(n->body);
  return func;
}

UnchangedOr<ffi::Any> StmtSimplifier::Mutate(ffi::AnyView input,
                                             InplaceMode inplace_mode) {
  if (input.as<TensorType>()) {
    return ffi::Unchanged();
  }
  if (def_region_kind() != kTVMFFIDefRegionKindNone && input.as<Var>()) {
    return Parent::Mutate(input, inplace_mode);
  }
  if (auto expr = input.as<PrimExpr>()) {
    PrimExpr simplified = tl::arith::Simplify(analyzer_, *expr);
    if (simplified.same_as(*expr))
      return ffi::Unchanged();
    return simplified;
  }
  return Parent::Mutate(input, inplace_mode);
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const BindNode *op,
                                          InplaceMode inplace_mode) {
  if (const auto *call = op->value.as<CallNode>()) {

    if (call->op.same_as(tirx::alloc_tensor_op()))
      return ffi::Unchanged();
    if (call->op.same_as(tirx::decl_tensor_op())) {

      auto data = this->Mutate(call->args[0], InplaceMode::kDisallow);
      if (data.UnchangedOrSameAs(call->args[0]))
        return ffi::Unchanged();
      auto value = ffi::make_object<CallNode>(*call);
      value->args.Set(0, std::move(data).ValueUnchecked());
      if (inplace_mode == InplaceMode::kAllow) {
        const_cast<BindNode *>(op)->value = Expr(std::move(value));
        return ffi::Unchanged();
      }
      auto copy = ffi::make_object<BindNode>(*op);
      copy->value = Expr(std::move(value));
      return Stmt(std::move(copy));
    }
  }
  auto prim_value = op->value.as<PrimExpr>();
  if (!prim_value) {
    return Parent::Mutate_(op, inplace_mode);
  }
  PrimExpr value = this->Mutate(prim_value.value(), inplace_mode)
                       .ValueOrUnchanged(prim_value.value());

  if (SideEffect(value) <= CallEffectKind::kPure) {
    tl::arith::Bind(analyzer_, op->var, value);

    non_inlined_bindings_.Set(op->var, value);
  }

  if (value.same_as(op->value)) {
    return ffi::Unchanged();
  } else {
    if (inplace_mode == InplaceMode::kAllow) {
      auto *n = const_cast<BindNode *>(op);
      n->value = std::move(value);
      return ffi::Unchanged();
    }
    auto n = ffi::make_object<BindNode>(*op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const IfNode *op,
                                          InplaceMode inplace_mode) {
  if (ffi::Optional<bool> cond = ProveCondition(op->condition)) {
    if (cond.value()) {
      return this->Mutate(op->then_case, inplace_mode)
          .ValueOrUnchanged(op->then_case);
    } else if (op->else_case) {
      return this->Mutate(op->else_case.value(), inplace_mode)
          .ValueOrUnchanged(op->else_case.value());
    } else {
      return Evaluate(0);
    }
  } else {
    return Parent::Mutate_(op, inplace_mode);
  }
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const TensorStoreNode *op,
                                          InplaceMode inplace_mode) {
  TensorStore store = Parent::Mutate_(op, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                          .as_or_throw<TensorStore>();
  if (const TensorLoadNode *load = store->value.as<TensorLoadNode>()) {
    TensorVar tensor = load->source.as_or_throw<tvm::tirx::TensorVar>();
    if (tensor.same_as(store->dest.as_or_throw<TensorVar>()) &&
        ArrayDeepEqual(load->indices, store->indices)) {
      return Evaluate(0);
    }
  }
  return store;
}

bool StmtSimplifier::ArrayDeepEqual(const ffi::Array<PrimExpr> &lhs,
                                    const ffi::Array<PrimExpr> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); i++) {
    if (!prim::ExprDeepEqual()(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

ffi::Optional<bool> StmtSimplifier::ProveCondition(PrimExpr condition) const {
  auto f_substitute =
      [this](const Var &var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = non_inlined_bindings_.Get(var))
      return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  condition =
      ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(condition, f_substitute)
          .as_or_throw<PrimExpr>();
  condition = tl::arith::Simplify(analyzer_, condition);
  if (const auto *as_int = condition.as<IntImmNode>()) {
    return as_int->value != 0;
  } else {
    return std::nullopt;
  }
}

} // namespace

Function StmtSimplify(Function func, const sym::Analyzer &analyzer) {
  return ffi::make_object<StmtSimplifier>(analyzer)->Run(std::move(func));
}

} // namespace

Pass StmtSimplify() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    sym::Analyzer analyzer;
    return StmtSimplify(std::move(f), analyzer);
  };
  return CreateFunctionPass(pass_func, 0, "tl.StmtSimplify");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl._tirx_passes.StmtSimplify",
                                   [] { return StmtSimplify(); });
}

} // namespace tvm::tl::tirx_passes
