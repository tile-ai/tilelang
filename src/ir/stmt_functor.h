/*!
 * \file tl/ir/stmt_functor.h
 * \brief TileLang proofs over the native constraint-aware traversal.
 */
#ifndef TILELANG_IR_STMT_FUNCTOR_H_
#define TILELANG_IR_STMT_FUNCTOR_H_

#include "analyzer.h"
#include "annotations.h"
#include "loop_math.h"
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <tvm/tirx/stmt_functor_with_constraint_context.h>

namespace tvm::tl {

// TileLang layout declarations carry tensor identities and coordinate captures
// in attrs. Native Call traversal deliberately leaves attrs to their dialect.
TVM_TIRX_DLL ffi::Expected<ffi::Optional<ffi::VisitInterrupt>>
VisitLayoutMapCall(const CallObj *call, ffi::StructuralVisitorObj *visitor);
TVM_TIRX_DLL ffi::Expected<ffi::UnchangedOr<ffi::Any>>
MutateLayoutMapCall(const CallObj *call, ffi::StructuralMutatorObj *mutator,
                    ffi::InplaceMode mode);

class LayoutMapMutator : public tirx::StmtExprMutator {
public:
  using tirx::StmtExprMutator::Mutate;
  using tirx::StmtExprMutator::Mutate_;
  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) override {
    if (!op->op.same_as(layout_map_op()))
      return tirx::StmtExprMutator::Mutate_(op, mode);
    auto result = MutateLayoutMapCall(op, this, mode).value();
    if (result.IsUnchanged())
      return ffi::Unchanged();
    return std::move(result).ValueUnchecked().cast<Expr>();
  }
};

class LayoutMapVisitor : public tirx::StmtExprVisitor {
public:
  using tirx::StmtExprVisitor::Visit;
  using tirx::StmtExprVisitor::Visit_;
  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallObj *op) override {
    if (op->op.same_as(layout_map_op()))
      return VisitLayoutMapCall(op, this).value();
    return tirx::StmtExprVisitor::Visit_(op);
  }
};

// Validate attrs-only uses at the declaration's lexical position. This builds
// a validation-only view for the native scope checker, never production IR.
TVM_TIRX_DLL void CheckLayoutMapScopes(const tirx::Function &function);

// Preserve native definition/type semantics while visiting TileLang layouts.
// Each call owns a fresh remap environment; free captures retain their
// identity.
TVM_TIRX_DLL tirx::Function RenewDef(tirx::Function function);
TVM_TIRX_DLL Stmt SubstituteWithDataTypeLegalization(
    Stmt statement,
    ffi::TypedFunction<ffi::Optional<PrimExpr>(const Var &)> mapping);

// Inspection may look through a singleton sequence. Reconstruction must still
// use SeqStmt for native scope bodies; empty and multi-statement bodies stay
// intact, and this does not cross a RegionStmt definition scope.
inline Stmt SingleStatement(const Stmt &stmt) {
  const auto *seq = stmt.as<SeqStmtObj>();
  return seq && seq->seq.size() == 1 ? seq->seq[0] : stmt;
}

// The native collector treats every positive constant step as a range. Keep
// TileLang's machine-integer no-wrap contract at the For consumer boundary.
// Other definitions and branch facts still use the native structural engine.
template <typename Traversal>
void AppendForConstraints(Traversal *traversal, const PrimVar &var,
                          const PrimExpr &begin, const PrimExpr &end,
                          const PrimExpr &step, sym::AnalyzerObj *analyzer) {
  // Mixed-width headers may truncate on assignment or use different compare
  // promotions. Do not interpret them as same-type mathematical induction.
  if (var.ty() != begin.ty() || var.ty() != end.ty() || var.ty() != step.ty())
    return;
  if (CanUseLoopBounds(begin, end, step, analyzer)) {
    Range range(begin, end);
    if (prim::IsZero(range->extent))
      return;
    traversal->AppendConstraint(sym::RangeConstraint(var, range));
  }
  // Even a wrapping/nonpositive loop tests var < end before each iteration.
  // Pure headers are stable, so an executed body also proves initial entry.
  if (SideEffect(end) <= CallEffectKind::kPure)
    traversal->AppendConstraint(sym::PredicateConstraint(var < end));
  if (SideEffect(begin) <= CallEffectKind::kPure &&
      SideEffect(end) <= CallEffectKind::kPure)
    traversal->AppendConstraint(sym::PredicateConstraint(end > begin));
}

// TileLang's prim.assume spelling supplies an optimizer premise to later
// statements in the same native scope. The target collector handles its own
// tirx.assume_always spelling, but does not collect this one.
// Mutable observations are not durable facts; consumers that model snapshots
// own those observations separately.
template <typename Traversal>
void AppendPureAssumeConstraint(Traversal *traversal, const EvaluateObj *op) {
  const auto *call = op->value.as<CallObj>();
  if (!call || !call->op.same_as(prim::assume_op()) || call->args.size() != 1)
    return;
  if (auto condition = call->args[0].as<PrimExpr>();
      condition && condition->ty() == PrimType::Bool() &&
      SideEffect(*condition) <= CallEffectKind::kPure)
    traversal->AppendConstraint(sym::PredicateConstraint(*condition, true));
}

// The upstream structural engine owns scope entry, definition remapping and
// premise collection. Keep the caller's query settings, and explicitly sync
// them at each proof site (including proofs after a child traversal).
class StmtExprMutatorWithAnalyzer
    : public tirx::StmtExprMutatorWithConstraintContext {
public:
  using tirx::StmtExprMutatorWithConstraintContext::Mutate_;

  explicit StmtExprMutatorWithAnalyzer(
      const sym::Analyzer &analyzer = CreateTileLangAnalyzer())
      : tirx::StmtExprMutatorWithConstraintContext(
            analyzer->GetConstraintContext()),
        proof_analyzer_(analyzer->Clone()) {}

  explicit StmtExprMutatorWithAnalyzer(sym::AnalyzerObj *analyzer)
      : StmtExprMutatorWithAnalyzer(ffi::GetRef<sym::Analyzer>(analyzer)) {}

protected:
  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) override {
    if (!op->op.same_as(layout_map_op()))
      return tirx::StmtExprMutatorWithConstraintContext::Mutate_(op, mode);
    auto result = MutateLayoutMapCall(op, this, mode).value();
    if (result.IsUnchanged())
      return ffi::Unchanged();
    return std::move(result).ValueUnchecked().cast<Expr>();
  }

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj *op,
                                 InplaceMode mode) override {
    auto result = tirx::StmtExprMutatorWithConstraintContext::Mutate_(op, mode);
    const auto *evaluate =
        result.IsUnchanged() ? op : ffi::AnyView(result).as<EvaluateObj>();
    if (evaluate)
      AppendPureAssumeConstraint(this, evaluate);
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) override {
    auto mapped_var = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                        return MutateExpected(op->var, mode);
                      }).value();
    PrimVar var = std::move(mapped_var)
                      .ValueOrUnchanged(ffi::Any(op->var))
                      .cast<PrimVar>();
    if (!var.same_as(op->var))
      VarRemapSetExpected(op->var, var).value();
    PrimExpr begin = Mutate(op->begin, mode).ValueOrUnchanged(op->begin);
    PrimExpr end = Mutate(op->end, mode).ValueOrUnchanged(op->end);
    PrimExpr step = Mutate(op->step, mode).ValueOrUnchanged(op->step);
    SeqStmt body = WithConstraintScope([&]() -> ffi::Expected<SeqStmt> {
                     AppendForConstraints(this, var, begin, end, step,
                                          GetSyncedAnalyzer());
                     return Mutate(op->body, mode).ValueOrUnchanged(op->body);
                   }).value();
    if (var.same_as(op->var) && begin.same_as(op->begin) &&
        end.same_as(op->end) && step.same_as(op->step) &&
        body.same_as(op->body))
      return ffi::Unchanged();
    auto result = ffi::make_object<ForObj>(*op);
    result->var = std::move(var);
    result->begin = std::move(begin);
    result->end = std::move(end);
    result->step = std::move(step);
    result->body = std::move(body);
    return Stmt(std::move(result));
  }

  sym::AnalyzerObj *GetSyncedAnalyzer() const {
    proof_analyzer_->SyncConstraintContext(constraints);
    return proof_analyzer_.get();
  }

  sym::Analyzer GetAnalyzer() const {
    GetSyncedAnalyzer();
    return proof_analyzer_;
  }

  void MarkTensorParamShapes(const tirx::Function &function) {
    AddParamConstraints(function);
  }

  // The upstream convenience method uses a separate private analyzer. Require
  // TileLang callers to select their iterator premises and proof owner
  // explicitly.
  ffi::Array<PrimExpr> IterMapSimplifyWithContext(const ffi::Array<PrimExpr> &,
                                                  bool) = delete;

private:
  sym::Analyzer proof_analyzer_;
};

class StmtExprVisitorWithAnalyzer
    : public tirx::StmtExprVisitorWithConstraintContext {
public:
  using tirx::StmtExprVisitorWithConstraintContext::Visit_;

  explicit StmtExprVisitorWithAnalyzer(
      const sym::Analyzer &analyzer = CreateTileLangAnalyzer())
      : proof_analyzer_(analyzer->Clone()) {
    constraints = analyzer->GetConstraintContext();
  }

protected:
  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallObj *op) override {
    if (op->op.same_as(layout_map_op()))
      return VisitLayoutMapCall(op, this).value();
    return tirx::StmtExprVisitorWithConstraintContext::Visit_(op);
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const EvaluateObj *op) override {
    auto result = tirx::StmtExprVisitorWithConstraintContext::Visit_(op);
    if (!result)
      AppendPureAssumeConstraint(this, op);
    return result;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const ForObj *op) override {
    auto result = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                    return VisitExpected(op->var);
                  }).value();
    if (result)
      return result;
    for (const PrimExpr &header : {op->begin, op->end, op->step}) {
      result = VisitExpected(header).value();
      if (result)
        return result;
    }
    return WithConstraintScope(
               [&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
                 AppendForConstraints(this, op->var, op->begin, op->end,
                                      op->step, GetSyncedAnalyzer());
                 return VisitExpected(op->body);
               })
        .value();
  }

  sym::AnalyzerObj *GetSyncedAnalyzer() const {
    proof_analyzer_->SyncConstraintContext(constraints);
    return proof_analyzer_.get();
  }

  sym::Analyzer GetAnalyzer() const {
    GetSyncedAnalyzer();
    return proof_analyzer_;
  }

private:
  sym::Analyzer proof_analyzer_;
};

} // namespace tvm::tl

#endif // TILELANG_IR_STMT_FUNCTOR_H_
