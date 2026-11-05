/*!
 * \file if_stmt_binding.cc
 * \brief Distribute guards while preserving native definitions and effects.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "common/bind_utils.h"
#include "common/pipeline_access.h"

namespace tvm::tl {
namespace {

using namespace tirx;

class IfStmtBindingRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  IfStmtBindingRewriter(const tirx::Function &function,
                        bool inline_replayable_binds)
      : memory_(function),
        target_(function->GetAttr<Target>("target").value_or(Target("llvm"))),
        inline_replayable_binds_(inline_replayable_binds) {}

  ffi::UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode mode) final {
    auto rewritten = StmtExprMutator::Mutate_(op, mode)
                         .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                         .as_or_throw<If>();
    // An else branch is one control-flow choice, not two independent guards.
    if (rewritten->else_case)
      return rewritten;
    return Distribute(rewritten->condition, rewritten->then_case,
                      rewritten->loc);
  }

private:
  struct AccessSummary {
    ffi::Array<TensorRegion> reads;
    TensorSet writes;
    bool unknown{false};
  };

  AccessSummary Accesses(const Stmt &statement) const {
    auto collector =
        ffi::make_object<PipelineAccessCollector>(&memory_, target_);
    collector->Visit(statement);
    AccessSummary result{
        collector->GetReads(), {}, collector->HasUnknownAccess()};
    for (const auto &write : collector->GetWrites())
      result.writes.insert(RegionTensor(write));
    // Calls may change state not represented by a tensor operand (e.g. an
    // external callback). Never infer disjointness solely from absent writes.
    // Pure scalar values remain replayable across these calls.
    VisitNativePostOrder(statement, [&](const ffi::ObjectRef &node) {
      if (const auto *call = node.as<CallObj>()) {
        if (SideEffect(ffi::GetRef<Call>(call)) >= CallEffectKind::kReadState &&
            !ParseOperator(ffi::GetRef<Call>(call)).defined())
          result.unknown = true;
      }
    });
    return result;
  }

  bool Replayable(const Stmt &statement, const AccessSummary &body) const {
    const auto *bind = statement.as<BindObj>();
    if (!bind)
      return false;
    auto access = Accesses(statement);
    if ((access.unknown || body.unknown) &&
        SideEffect(bind->value) > CallEffectKind::kPure)
      return false;
    return IsReplayableScalarBind(
        statement, access.reads, body.writes,
        [this](const TensorVar &a, const TensorVar &b) {
          return memory_.MayAlias(a, b);
        });
  }

  static ffi::Array<Stmt> Flatten(const SeqStmt &body) {
    ffi::Array<Stmt> result;
    for (const Stmt &statement : body->seq) {
      if (auto nested = statement.as<SeqStmt>()) {
        for (const Stmt &child : Flatten(*nested))
          result.push_back(child);
      } else {
        result.push_back(statement);
      }
    }
    return result;
  }

  static bool DefinesInSequence(const Stmt &statement) {
    if (statement.as<BindObj>())
      return true;
    if (const auto *region = statement.as<RegionStmtObj>())
      return !region->result_vars.empty();
    return false;
  }

  static Stmt Guard(const PrimExpr &condition, const Stmt &body,
                    const Location &loc) {
    auto seq = body.as<SeqStmt>();
    return If(condition, seq.value_or(SeqStmt({body}, body->loc)), std::nullopt,
              loc);
  }

  Stmt Distribute(const PrimExpr &condition, const SeqStmt &body,
                  const Location &loc) const {
    const auto statements = Flatten(body);
    const auto accesses = Accesses(body);
    // One mutator owns all remappings for this sequence. Rewriting each
    // statement independently loses fresh tensor identities when a substituted
    // scalar occurs only in a definition's type, shape, stride, or layout.
    auto inliner = ffi::make_object<StmtExprMutator>();
    ffi::Array<Stmt> groups, tail;
    bool in_definition_scope = false;
    for (const Stmt &statement : statements) {
      if (!in_definition_scope && inline_replayable_binds_ &&
          Replayable(statement, accesses)) {
        const auto *bind = statement.as<BindObj>();
        Expr value = inliner->Mutate(bind->value).ValueOrUnchanged(bind->value);
        inliner->VarRemapSet(bind->var, value);
        continue;
      }
      Stmt rewritten = inliner->Mutate(statement).ValueOrUnchanged(statement);
      if (DefinesInSequence(statement))
        in_definition_scope = true;
      if (in_definition_scope)
        tail.push_back(rewritten);
      else
        groups.push_back(rewritten);
    }
    // Bind and RegionStmt results cannot escape a separately guarded lexical
    // block. Keep the first materialized definition and every consumer
    // together.
    if (!tail.empty())
      groups.push_back(SeqStmt(tail, body->loc));
    if (groups.empty())
      return If(condition, SeqStmt({}, body->loc), std::nullopt, loc);
    if (groups.size() == 1)
      return Guard(condition, groups[0], loc);

    ffi::Array<Stmt> result;
    PrimExpr guard = condition;
    PrimVar snapshot("if_condition", condition.ty(), condition->loc);
    Bind condition_bind(snapshot, condition, condition->loc);
    // The source if evaluates its condition once. A body write through another
    // view or parameter may alias that condition even when Var identities
    // differ.
    if (!Replayable(condition_bind, accesses)) {
      result.push_back(condition_bind);
      guard = snapshot;
    }
    for (const Stmt &group : groups)
      result.push_back(Guard(guard, group, loc));
    return SeqStmt(result, loc);
  }

  PipelineMemoryAnalysis memory_;
  Target target_;
  bool inline_replayable_binds_;
};

} // namespace

tvm::transform::Pass IfStmtBinding() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext context) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    bool inline_binds =
        context->GetConfig<bool>(kIfStmtBindingInlineReplayableBinds)
            .value_or(true);
    auto rewriter =
        ffi::make_object<IfStmtBindingRewriter>(function, inline_binds);
    auto body = function->body;
    function.CopyOnWrite()->body =
        rewriter->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.IfStmtBinding");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.IfStmtBinding", IfStmtBinding);
}

} // namespace tvm::tl
