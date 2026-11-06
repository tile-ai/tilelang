/*!
 * \file loop_unswitching.cc
 * \brief Hoist invariant guards while retaining native scopes and storage
 * effects.
 */
#include "../ir/analyzer.h"
#include "../op/builtin.h"
#include "common/pipeline_access.h"
#include "ir/stmt_functor.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/op/region.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace tirx;
using VarSet = std::unordered_set<const VarObj *>;

// Unlike structural equality with free-variable mapping, ExprDeepEqual keeps
// two equal-shaped tensors and two same-named scalar definitions distinct.
bool SameGuard(const PrimExpr &lhs, const PrimExpr &rhs) {
  return prim::ExprDeepEqual()(lhs, rhs);
}

bool IsNoWork(const Stmt &statement) {
  if (const auto *op = statement.as<EvaluateObj>())
    return SideEffect(op->value) <= CallEffectKind::kReadState;
  if (const auto *op = statement.as<BindObj>())
    return SideEffect(op->value) <= CallEffectKind::kReadState;
  if (const auto *op = statement.as<SeqStmtObj>()) {
    for (const Stmt &child : op->seq)
      if (!IsNoWork(child))
        return false;
    return true;
  }
  if (const auto *op = statement.as<IfObj>())
    return SideEffect(op->condition) <= CallEffectKind::kReadState &&
           IsNoWork(op->then_case) &&
           (!op->else_case || IsNoWork(*op->else_case));
  if (const auto *op = statement.as<ForObj>())
    return SideEffect(op->begin) <= CallEffectKind::kReadState &&
           SideEffect(op->end) <= CallEffectKind::kReadState &&
           SideEffect(op->step) <= CallEffectKind::kReadState &&
           IsNoWork(op->body);
  return false;
}

class OutsideCallChecker : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  explicit OutsideCallChecker(PrimExpr guard) : guard_(std::move(guard)) {}
  bool found{false};

  ffi::Optional<VisitInterrupt> Visit_(const IfObj *op) final {
    if (SameGuard(op->condition, guard_))
      return std::nullopt;
    return StmtExprVisitor::Visit_(op);
  }
  ffi::Optional<VisitInterrupt> Visit_(const CallObj *) final {
    found = true;
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj *) final {
    // A dialect region can own synchronization or a collective scope. Even
    // without expression calls, duplicating it under a branch is not neutral.
    found = true;
    return std::nullopt;
  }

private:
  PrimExpr guard_;
};

class BranchReplacer : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate_;
  BranchReplacer(PrimExpr guard, bool take_then, VarSet hoisted)
      : guard_(std::move(guard)), take_then_(take_then),
        hoisted_(std::move(hoisted)) {}

  ffi::UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode mode) final {
    if (!SameGuard(op->condition, guard_))
      return LayoutMapMutator::Mutate_(op, mode);
    if (take_then_)
      return Mutate(op->then_case).ValueOrUnchanged(op->then_case);
    if (op->else_case)
      return Mutate(*op->else_case).ValueOrUnchanged(*op->else_case);
    return Evaluate(IntImm(PrimType::Int(32), 0), op->loc);
  }
  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (hoisted_.count(op->var.get()))
      return SeqStmt({}, op->loc);
    return LayoutMapMutator::Mutate_(op, mode);
  }

private:
  PrimExpr guard_;
  bool take_then_;
  VarSet hoisted_;
};

class LoopUnswitcher : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate_;
  LoopUnswitcher(const tirx::Function &function, bool allow_non_trivial_else)
      : memory_(function),
        target_(function->GetAttr<Target>("target").value_or(Target("llvm"))),
        allow_non_trivial_else_(allow_non_trivial_else),
        analyzer_(CreateTileLangAnalyzer()) {}

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    VarSet saved = thread_vars_;
    if (op->op.same_as(launch_thread_op()) && !op->args.empty()) {
      if (auto tag = op->args[0].as<StringImm>();
          tag && std::string((*tag)->value).starts_with("threadIdx."))
        for (const Var &var : op->body_params)
          thread_vars_.insert(var.get());
    } else if (op->op->name == "tl.launch_kernel" && op->args.size() >= 2) {
      // Also cover public pass use before MaterializeKernelLaunch.
      if (auto ends = op->args[1].as<tvm::Tuple>())
        for (size_t i = (*ends)->fields.size(); i < op->body_params.size(); ++i)
          thread_vars_.insert(op->body_params[i].get());
    }
    auto result = LayoutMapMutator::Mutate_(op, mode);
    thread_vars_ = std::move(saved);
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    auto loop = LayoutMapMutator::Mutate_(op, mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                    .as_or_throw<For>();
    // Evaluate the original range unchanged. An effectful/dynamic step needs a
    // separate range snapshot; this optimization must not invent one.
    const auto *step = loop->step.as<IntImmObj>();
    auto step_value = step ? step->value.as<int64_t>() : std::nullopt;
    if (!step_value || *step_value == 0 ||
        SideEffect(loop->begin) > CallEffectKind::kPure ||
        SideEffect(loop->end) > CallEffectKind::kPure)
      return loop;
    memory_.ObserveDefinitions(loop);
    auto accesses =
        ffi::make_object<PipelineAccessCollector>(&memory_, target_);
    accesses->Visit(loop->body);
    Candidate candidate;
    candidate.accesses = accesses;
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        loop, [&](const VarObj *var, TVMFFIDefRegionKind kind) {
          if (kind != kTVMFFIDefRegionKindNone)
            candidate.loop_defs.insert(var);
          return ffi::WalkResult::Advance();
        });
    Find(loop->body, &candidate);
    if (!candidate.guard)
      return loop;
    auto guard = candidate.guard.value();
    auto outside = ffi::make_object<OutsideCallChecker>(guard);
    outside->Visit(loop->body);
    if (outside->found)
      return loop;

    auto then_rewriter =
        ffi::make_object<BranchReplacer>(guard, true, candidate.hoisted_vars);
    auto else_rewriter =
        ffi::make_object<BranchReplacer>(guard, false, candidate.hoisted_vars);
    SeqStmt then_body =
        then_rewriter->Mutate(loop->body).ValueOrUnchanged(loop->body);
    SeqStmt else_body =
        else_rewriter->Mutate(loop->body).ValueOrUnchanged(loop->body);
    if (!allow_non_trivial_else_ && !IsNoWork(else_body))
      return loop;

    auto version = [&](SeqStmt body) -> SeqStmt {
      For copy(loop->var, loop->begin, loop->end, loop->step, loop->kind,
               std::move(body), loop->annotations, loop->loc);
      // Each branch gets a separate definition traversal. Renewing the whole
      // result after duplication would map both copies of a Bind to one Var.
      tirx::Function wrapper({}, SeqStmt({copy}), VoidType());
      return tl::RenewDef(std::move(wrapper))->body;
    };
    ffi::Array<Stmt> statements = candidate.hoisted;
    statements.push_back(If(guard, version(std::move(then_body)),
                            version(std::move(else_body)), candidate.loc));
    SeqStmt result(statements, loop->loc);
    // Raw native For always tests var < end, including for a negative step.
    // Public T.serial handles descending Python ranges before constructing it.
    PrimExpr executes = loop->begin < loop->end;
    if (!analyzer_->CanProve(executes)) {
      // The original zero-trip loop evaluates no predicate or predicate Bind.
      // Retain that boundary, including for loads/division in the condition.
      return If(executes, result, std::nullopt, loop->loc);
    }
    return result;
  }

private:
  struct Candidate {
    ffi::ObjectPtr<PipelineAccessCollector> accesses;
    VarSet loop_defs;
    std::unordered_map<const VarObj *, Bind> bindings;
    ffi::Optional<PrimExpr> guard;
    Location loc{UnknownLoc()};
    VarSet hoisted_vars;
    ffi::Array<Stmt> hoisted;
  };

  bool DependsOnThread(const Expr &expression, VarSet *seen) const {
    for (const Var &var : UndefinedVars(expression)) {
      if (thread_vars_.count(var.get()))
        return true;
      if (!seen->insert(var.get()).second)
        continue;
      auto value = memory_.definitions().Get(var);
      if (value && DependsOnThread(*value, seen))
        return true;
    }
    return false;
  }

  bool Invariant(const Expr &expression, Candidate *candidate, VarSet *visiting,
                 VarSet *hoisted, ffi::Array<Stmt> *bindings) const {
    if (SideEffect(expression) > CallEffectKind::kReadState)
      return false;
    bool call = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        expression,
        [&](const ffi::ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (node.as<CallObj>())
            call = true;
          return ffi::WalkResult::Advance();
        });
    if (call)
      return false;
    VarSet thread_seen;
    if (DependsOnThread(expression, &thread_seen))
      return false;
    auto read = ffi::make_object<PipelineAccessCollector>(&memory_, target_);
    read->Visit(expression);
    if (read->HasUnknownAccess() ||
        (candidate->accesses->HasUnknownAccess() && !read->GetReads().empty()))
      return false;
    for (const TensorRegion &load : read->GetReads())
      for (const TensorRegion &store : candidate->accesses->GetWrites())
        if (memory_.MayAlias(RegionTensor(load), RegionTensor(store)))
          return false;
    for (const Var &var : UndefinedVars(expression)) {
      if (!candidate->loop_defs.count(var.get()) || hoisted->count(var.get()))
        continue;
      auto found = candidate->bindings.find(var.get());
      if (found == candidate->bindings.end() || !var->ty.as<PrimTypeObj>() ||
          !visiting->insert(var.get()).second)
        return false;
      const Bind &bind = found->second;
      if (!Invariant(bind->value, candidate, visiting, hoisted, bindings))
        return false;
      visiting->erase(var.get());
      hoisted->insert(var.get());
      bindings->push_back(bind);
    }
    return true;
  }

  void Find(const SeqStmt &body, Candidate *candidate) const {
    for (const Stmt &statement : body->seq) {
      if (candidate->guard)
        return;
      if (auto sequence = statement.as<SeqStmt>()) {
        Find(*sequence, candidate);
      } else if (auto binding = statement.as<Bind>()) {
        candidate->bindings.insert_or_assign((*binding)->var.get(), *binding);
      } else if (const auto *branch = statement.as<IfObj>()) {
        VarSet visiting, hoisted;
        ffi::Array<Stmt> bindings;
        if (Invariant(branch->condition, candidate, &visiting, &hoisted,
                      &bindings)) {
          candidate->guard = branch->condition;
          candidate->loc = branch->loc;
          candidate->hoisted_vars = std::move(hoisted);
          candidate->hoisted = std::move(bindings);
        }
        // Do not speculate through a nested conditional or dialect region.
        // Its loads may only be defined when the outer predicate holds.
      }
      // A preceding exit, assertion, or nonterminating nested loop can keep
      // the original predicate from being reached. Do not move its loads
      // ahead of that boundary, even with the aggressive option enabled.
      bool may_exit = false;
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          statement,
          [&](const ffi::ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
            if (node.as<BreakObj>() || node.as<ContinueObj>() ||
                node.as<ReturnObj>() || node.as<AssertStmtObj>() ||
                node.as<ForObj>() || node.as<WhileObj>())
              may_exit = true;
            return ffi::WalkResult::Advance();
          });
      if (may_exit)
        return;
    }
  }

  PipelineMemoryAnalysis memory_;
  Target target_;
  bool allow_non_trivial_else_;
  sym::Analyzer analyzer_;
  VarSet thread_vars_;
};
} // namespace

tvm::transform::Pass LoopUnswitching() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext context) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false) ||
        context->GetConfig<bool>(kDisableLoopUnswitching).value_or(false))
      return function;
    bool allow_else =
        context->GetConfig<bool>(kLoopUnswitchingAllowNonTrivialElse)
            .value_or(false);
    auto rewriter = ffi::make_object<LoopUnswitcher>(function, allow_else);
    SeqStmt body = function->body;
    function.CopyOnWrite()->body =
        rewriter->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0, "tl.LoopUnswitching");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LoopUnswitching",
                                   LoopUnswitching);
}
} // namespace tvm::tl
