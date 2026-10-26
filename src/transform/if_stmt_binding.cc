#include "compat/metadata.h"

#include "common/memory_effects.h"
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

using analysis::AliasOrigins;
using analysis::EffectCollector;
using analysis::MemoryEffects;

class IfGuardRewriter : public tvm::tl::StmtExprMutator {
public:
  IfGuardRewriter(const tirx::Function &function, bool inline_bindings)
      : aliases_(function), inline_bindings_(inline_bindings) {}
  using tvm::tl::StmtExprMutator::Mutate;

private:
  bool Replayable(const PrimExpr &expression, const MemoryEffects &body) const {
    if (!expression.ty().IsScalar() ||
        SideEffect(expression) > CallEffectKind::kReadState) {
      return false;
    }
    auto effects =
        ffi::make_object<EffectCollector>(aliases_)->Collect(expression);
    return !effects.unknown_read && !effects.unknown_write &&
           (effects.reads.empty() ||
            (!body.unknown_write &&
             !aliases_.MayAlias(effects.reads, body.writes)));
  }

  ffi::UnchangedOr<Stmt> Mutate_(const IfNode *node,
                                 ffi::InplaceMode mode) final {
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(node))
                      .as_or_throw<If>();
    if (result->else_case)
      return result;
    const auto *sequence = result->then_case.as<SeqStmtNode>();
    if (!sequence || sequence->seq.size() <= 1)
      return result;
    MemoryEffects effects =
        ffi::make_object<EffectCollector>(aliases_)->Collect(result->then_case);
    ffi::Map<PrimVar, PrimExpr> substitutions;
    ffi::Array<Stmt> bodies;
    ffi::Array<Stmt> binding_scope;
    bool keep_scope = false;
    for (const Stmt &statement : sequence->seq) {
      Stmt rewritten = substitutions.empty()
                           ? statement
                           : tl::Substitute(statement, substitutions);
      if (const auto *binding = rewritten.as<BindNode>();
          binding && !keep_scope) {
        auto variable = binding->var.as<PrimVar>();
        auto value = binding->value.as<PrimExpr>();
        if (inline_bindings_ && variable && value &&
            Replayable(value.value(), effects)) {
          substitutions.Set(variable.value(), value.value());
          continue;
        }
        keep_scope = true;
      }
      (keep_scope ? binding_scope : bodies).push_back(rewritten);
    }
    if (!binding_scope.empty())
      bodies.push_back(SeqStmt(binding_scope, result->loc));
    if (bodies.empty())
      return Evaluate(result->condition, result->loc);
    PrimExpr condition = result->condition;
    ffi::Array<Stmt> guarded;
    if (bodies.size() > 1 && !Replayable(condition, effects)) {
      PrimVar snapshot("if_condition", condition.ty());
      guarded.push_back(Bind(snapshot, condition, result->loc));
      condition = snapshot;
    }
    for (const Stmt &body : bodies)
      guarded.push_back(If(condition, body, std::nullopt, result->loc));
    return SeqStmt(guarded, result->loc);
  }

  AliasOrigins aliases_;
  bool inline_bindings_;
};

} // namespace

tvm::transform::Pass IfStmtBinding() {
  auto pass = [](tirx::Function function, IRModule module,
                 tvm::transform::PassContext context) {
    if (!function->body)
      return function;
    bool inline_bindings =
        context->GetConfig<bool>(kIfStmtBindingInlineReplayableBinds)
            .value_or(true);
    Stmt body =
        ffi::make_object<IfGuardRewriter>(function, inline_bindings)
            ->Mutate(function->body.value(), ffi::InplaceMode::kDisallow)
            .ValueOrUnchanged(function->body.value());
    function.CopyOnWrite()->body = body;
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.IfStmtBinding");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.IfStmtBinding", IfStmtBinding);
}

} // namespace tvm::tl
