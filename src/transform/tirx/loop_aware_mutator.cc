#include "transform/common/loop_aware_mutator.h"
#include "compat/analyzer.h"

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/analysis.h>

namespace tvm::tl {
using namespace tvm::tirx;
using namespace tvm::prim;

UnchangedOr<Stmt> LoopAwareMutator::Mutate_(const SeqStmtNode *op,
                                            InplaceMode inplace_mode) {
  return constraint_scope_.WithNewScope([&]() -> UnchangedOr<Stmt> {
    ffi::Array<Stmt> statements;
    bool changed = false;
    for (const Stmt &statement : op->seq) {
      Stmt rewritten =
          Mutate(statement, inplace_mode).ValueOrUnchanged(statement);
      changed |= !rewritten.same_as(statement);
      statements.push_back(rewritten);
      const auto *evaluate = rewritten.as<EvaluateNode>();
      const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
      if (call && call->op.same_as(prim::assume_op())) {
        PrimExpr condition = call->args[0].as_or_throw<PrimExpr>();
        if (SideEffect(condition) == CallEffectKind::kPure) {
          constraint_scope_.Current().Emplace(analyzer_, condition, true);
        }
      }
    }
    if (!changed)
      return ffi::Unchanged();
    return SeqStmt(statements, op->loc);
  });
}

UnchangedOr<Stmt> LoopAwareMutator::Mutate_(const ForNode *op,
                                            InplaceMode inplace_mode) {
  auto enter_constraint = [&](PrimExpr condition) {
    constraint_scope_.Current().Emplace(
        analyzer_, tl::arith::Simplify(analyzer_, condition));
  };
  return constraint_scope_.WithNewScope([&]() -> UnchangedOr<Stmt> {
    auto begin_result = this->Mutate(op->begin, inplace_mode);
    bool begin_unchanged = begin_result.UnchangedOrSameAs(op->begin);
    PrimExpr begin = std::move(begin_result).ValueOrUnchanged(op->begin);
    auto end_result = this->Mutate(op->end, inplace_mode);
    bool end_unchanged = end_result.UnchangedOrSameAs(op->end);
    PrimExpr end = std::move(end_result).ValueOrUnchanged(op->end);
    PrimExpr step =
        this->Mutate(op->step, inplace_mode).ValueOrUnchanged(op->step);
    ffi::Optional<Range> domain;
    if (SideEffect(begin) == CallEffectKind::kPure &&
        SideEffect(end) == CallEffectKind::kPure &&
        SideEffect(step) == CallEffectKind::kPure) {
      if (tl::arith::CanProve(analyzer_, step > 0))
        domain = Range(begin, end);
    }
    if (domain) {
      tl::arith::Bind(analyzer_, op->var, domain.value());
      iter_vars_.Set(op->var, domain.value());
    }
    auto annotations =
        this->Mutate(op->annotations, inplace_mode)
            .as_or_throw<UnchangedOr<decltype(op->annotations)>>();
    Stmt body = constraint_scope_.WithNewScope([&]() -> Stmt {
      if (tl::arith::CanProve(analyzer_, step > 0)) {
        enter_constraint(end > begin);
        enter_constraint(end - begin > 0);
      } else if (tl::arith::CanProve(analyzer_, step < 0)) {
        enter_constraint(begin > end);
        enter_constraint(begin - end > 0);
      }
      PrimExpr inside = (step > 0 && op->var >= begin && op->var < end) ||
                        (step < 0 && op->var <= begin && op->var > end);
      enter_constraint(inside);
      return this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    });
    if (begin_unchanged && end_unchanged && body.same_as(op->body) &&
        step.same_as(op->step) &&
        annotations.UnchangedOrSameAs(op->annotations)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto *updated = const_cast<ForNode *>(op);
        updated->begin = std::move(begin);
        updated->end = std::move(end);
        updated->step = std::move(step);
        updated->body = std::move(body);
        updated->annotations =
            std::move(annotations).ValueOrUnchanged(op->annotations);
        return ffi::Unchanged();
      }
      auto updated = ffi::make_object<ForNode>(*op);
      updated->begin = std::move(begin);
      updated->end = std::move(end);
      updated->step = std::move(step);
      updated->body = std::move(body);
      updated->annotations =
          std::move(annotations).ValueOrUnchanged(op->annotations);
      return Stmt(updated);
    }
  });
}

} // namespace tvm::tl
