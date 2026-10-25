#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include "common/memory_effects.h"
#include "op/builtin.h"

namespace tvm::tl::transform {

namespace {

bool SafePredicate(const PrimExpr &expression) {
  bool safe = expression.ty().IsScalar();
  PostOrderVisit(expression, [&](const ffi::ObjectRef &object) {
    ffi::Optional<PrimExpr> divisor;
    if (const auto *node = object.as<DivNode>())
      divisor = node->b;
    if (const auto *node = object.as<ModNode>())
      divisor = node->b;
    if (const auto *node = object.as<FloorDivNode>())
      divisor = node->b;
    if (const auto *node = object.as<FloorModNode>())
      divisor = node->b;
    if (divisor) {
      auto value = GetConstInt(*divisor);
      safe &= value && *value > 0;
    }
    if (object.as<LShiftNode>() || object.as<RShiftNode>())
      safe = false;
    if (const auto *cast = object.as<CastNode>())
      safe &= !cast->ty.as_or_throw<PrimType>().MatchesCode(kDLInt, kDLUInt) ||
              cast->value.ty().MatchesCode(kDLInt, kDLUInt);
    if (const auto *call = object.as<CallNode>())
      safe &= call->op.same_as(prim::likely_op());
  });
  return safe;
}

bool EmptyBranch(const Stmt &statement) {
  if (const auto *sequence = statement.as<SeqStmtNode>()) {
    for (const Stmt &child : sequence->seq)
      if (!EmptyBranch(child))
        return false;
    return true;
  }
  if (const auto *evaluate = statement.as<EvaluateNode>()) {
    auto value = evaluate->value.as<PrimExpr>();
    return value && SideEffect(*value) == CallEffectKind::kPure &&
           SafePredicate(*value);
  }
  if (const auto *bind = statement.as<BindNode>()) {
    auto value = bind->value.as<PrimExpr>();
    return value && SideEffect(*value) == CallEffectKind::kPure &&
           SafePredicate(*value);
  }
  if (const auto *loop = statement.as<ForNode>())
    return SideEffect(loop->begin) == CallEffectKind::kPure &&
           SideEffect(loop->end) == CallEffectKind::kPure &&
           SideEffect(loop->step) == CallEffectKind::kPure &&
           SafePredicate(loop->begin) && SafePredicate(loop->end) &&
           SafePredicate(loop->step) && EmptyBranch(loop->body);
  return false;
}

class GuardSpecializer : public tirx::StmtExprMutator {
public:
  GuardSpecializer(PrimExpr condition, bool enabled,
                   ffi::Map<PrimVar, PrimExpr> definitions,
                   std::unordered_set<const VarNode *> lifted)
      : condition_(std::move(condition)), enabled_(enabled),
        definitions_(std::move(definitions)), lifted_(std::move(lifted)) {}

private:
  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    if (lifted_.count(node->var.get()))
      return SeqStmt({}, node->loc);
    return StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const IfNode *node,
                                 ffi::InplaceMode mode) final {
    if (!ffi::StructuralEqual()(Substitute(node->condition, definitions_),
                                condition_))
      return StmtExprMutator::Mutate_(node, mode);
    SeqStmt branch =
        enabled_ ? node->then_case : node->else_case.value_or(SeqStmt({}));
    return Mutate(branch, mode).ValueOrUnchanged(branch);
  }

  PrimExpr condition_;
  bool enabled_;
  ffi::Map<PrimVar, PrimExpr> definitions_;
  std::unordered_set<const VarNode *> lifted_;
};

class LoopGuardHoister : public tirx::StmtExprMutator {
public:
  LoopGuardHoister(const tirx::Function &function, bool allow_else)
      : aliases_(function), allow_else_(allow_else) {
    PostOrderVisit(function->body.value(), [&](const ffi::ObjectRef &object) {
      const auto *region = object.as<RegionStmtNode>();
      if (region && region->op.same_as(tirx::launch_thread_op()) &&
          region->args[0].as_or_throw<StringImm>()->value.find("threadIdx.") ==
              0)
        for (const Var &variable : region->body_params)
          thread_variables_.insert(variable.get());
    });
  }

private:
  static ffi::Array<Stmt> Flatten(const SeqStmt &body) {
    ffi::Array<Stmt> statements;
    for (const Stmt &statement : body->seq) {
      if (auto nested = statement.as<SeqStmt>()) {
        for (const Stmt &child : Flatten(*nested))
          statements.push_back(child);
      } else {
        statements.push_back(statement);
      }
    }
    return statements;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    For loop = StmtExprMutator::Mutate_(node, mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(node))
                   .as_or_throw<For>();
    if (loop->kind == ForKind::kVectorized)
      return loop;
    for (const PrimExpr &header : {loop->begin, loop->end, loop->step})
      if (SideEffect(header) != CallEffectKind::kPure || !SafePredicate(header))
        return loop;
    ffi::Array<Stmt> statements = Flatten(loop->body);
    ffi::Map<PrimVar, PrimExpr> definitions;
    std::unordered_map<const VarNode *, Bind> bindings;
    ffi::Optional<PrimExpr> condition;
    ffi::Optional<PrimExpr> original_condition;
    for (const Stmt &statement : statements) {
      if (auto bind = statement.as<Bind>()) {
        auto variable = bind.value()->var.as<PrimVar>();
        auto value = bind.value()->value.as<PrimExpr>();
        if (variable && value) {
          definitions.Set(*variable, Substitute(*value, definitions));
          bindings.emplace(variable.value().get(), *bind);
        }
      }
      if (const auto *branch = statement.as<IfNode>()) {
        PrimExpr expanded = Substitute(branch->condition, definitions);
        if (condition && !ffi::StructuralEqual()(*condition, expanded))
          return loop;
        if (!condition) {
          condition = expanded;
          original_condition = branch->condition;
        }
      }
    }
    if (!condition || !SafePredicate(*condition))
      return loop;
    std::unordered_set<const VarNode *> local_variables{loop->var.get()};
    bool control_boundary = false;
    PostOrderVisit(loop->body, [&](const ffi::ObjectRef &object) {
      if (const auto *bind = object.as<BindNode>())
        local_variables.insert(bind->var.get());
      if (const auto *nested = object.as<ForNode>())
        local_variables.insert(nested->var.get());
      control_boundary |= object.as<ReturnNode>() ||
                          object.as<AssertStmtNode>() ||
                          object.as<BreakNode>() || object.as<ContinueNode>() ||
                          object.as<RegionStmtNode>();
    });
    if (control_boundary)
      return loop;
    for (const Var &variable : tvm::UndefinedVars(*condition))
      if (local_variables.count(variable.get()) ||
          thread_variables_.count(variable.get()))
        return loop;
    auto effects =
        ffi::make_object<analysis::EffectCollector>(aliases_)->Collect(
            loop->body);
    auto reads = ffi::make_object<analysis::EffectCollector>(aliases_)->Collect(
        *condition);
    if (reads.unknown_read || reads.unknown_write ||
        (!reads.reads.empty() &&
         (effects.unknown_write ||
          aliases_.MayAlias(reads.reads, effects.writes))))
      return loop;
    std::unordered_set<const VarNode *> lifted;
    std::function<void(const Expr &)> collect_bindings =
        [&](const Expr &value) {
          for (const Var &variable : tvm::UndefinedVars(value)) {
            auto binding = bindings.find(variable.get());
            if (binding != bindings.end() &&
                lifted.insert(variable.get()).second)
              collect_bindings(binding->second->value);
          }
        };
    collect_bindings(*original_condition);
    ffi::Array<Stmt> prefix;
    for (const Stmt &statement : statements)
      if (const auto *bind = statement.as<BindNode>();
          bind && lifted.count(bind->var.get()))
        prefix.push_back(statement);
    auto specialize = [&](bool enabled) {
      return ffi::make_object<GuardSpecializer>(*condition, enabled,
                                                definitions, lifted)
          ->Mutate(loop->body, ffi::InplaceMode::kDisallow)
          .ValueOrUnchanged(loop->body)
          .as_or_throw<SeqStmt>();
    };
    SeqStmt then_body = specialize(true);
    SeqStmt else_body = specialize(false);
    bool has_else = !EmptyBranch(else_body);
    if (has_else && !allow_else_)
      return loop;
    For then_loop = loop;
    then_loop.CopyOnWrite()->body = then_body;
    ffi::Optional<SeqStmt> otherwise;
    if (has_else) {
      For else_loop = loop;
      else_loop.CopyOnWrite()->body = else_body;
      otherwise = SeqStmt({else_loop});
    }
    prefix.push_back(
        If(*original_condition, SeqStmt({then_loop}), otherwise, loop->loc));
    SeqStmt result(prefix, loop->loc);
    if (!reads.reads.empty()) {
      sym::Analyzer analyzer;
      PrimExpr nonempty = IntImm::Bool(false);
      if (analyzer->CanProve(loop->step > 0))
        nonempty = loop->begin < loop->end;
      else if (analyzer->CanProve(loop->step < 0))
        nonempty = loop->begin > loop->end;
      else
        return loop;
      if (!analyzer->CanProve(nonempty))
        return If(nonempty, result, std::nullopt, loop->loc);
    }
    return result;
  }

  analysis::AliasOrigins aliases_;
  std::unordered_set<const VarNode *> thread_variables_;
  bool allow_else_;
};

} // namespace

tvm::transform::Pass LoopUnswitching() {
  auto pass = [](IRModule module, tvm::transform::PassContext context) {
    if (context->GetConfig<bool>(kDisableLoopUnswitching).value_or(false)) {
      return module;
    }
    bool allow_else =
        context->GetConfig<bool>(kLoopUnswitchingAllowNonTrivialElse)
            .value_or(false);
    auto hoist = [allow_else](tirx::Function function, IRModule,
                              tvm::transform::PassContext) {
      if (function->body) {
        Stmt body = ffi::make_object<LoopGuardHoister>(function, allow_else)
                        ->Mutate(*function->body, ffi::InplaceMode::kDisallow)
                        .ValueOrUnchanged(*function->body);
        function.CopyOnWrite()->body = body;
      }
      return function;
    };
    module = tirx::transform::CreateFunctionPass(
        hoist, 0, "tl.HoistReplayableLoopGuards")(module);
    module = tirx::transform::ConvertSSA()(module);
    return tirx::transform::HoistIf(allow_else)(std::move(module));
  };
  return tvm::transform::CreateModulePass(pass, 0, "tl.LoopUnswitching");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LoopUnswitching",
                                   LoopUnswitching);
}

} // namespace tvm::tl::transform
