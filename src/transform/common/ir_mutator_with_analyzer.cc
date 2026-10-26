#include "compat/analyzer.h"
#include "compat/metadata.h"
#include <tvm/backend/cuda/attr.h>
#include <tvm/ffi/cast.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/region.h>

#include "transform/common/constraint_helpers.h"

namespace tvm {
namespace tl {
using namespace tvm::tirx;
void StmtExprMutatorWithAnalyzer::InitVTable(VTable *vtable) {
  StmtExprMutator::InitVTable(vtable);
}

const StmtExprMutatorWithAnalyzer::VTable *
StmtExprMutatorWithAnalyzer::GlobalVTable() {
  static const VTable table = [] {
    VTable table;
    InitVTable(&table);
    table.Finalize();
    return table;
  }();
  return &table;
}

using namespace tvm::prim;

using tl::arith::detail::EnterConstraintFacts;

void StmtExprMutatorWithAnalyzer::MarkTensorParamShapes(
    const tirx::Function &func) {

  for (const tvm::Var &param : func->params) {
    if (!param->ty.as<tirx::TensorTypeNode>()) {
      continue;
    }
    tirx::TensorVar tensor = param.as_or_throw<tirx::TensorVar>();
    for (PrimExpr shape : tensor.shape()) {
      analyzer_->MarkGlobalNonNegValue(shape);
    }
  }
}

ffi::Array<PrimExpr> StmtExprMutatorWithAnalyzer::IterMapSimplifyWithContext(
    const ffi::Array<PrimExpr> &indices, bool non_trivial_only) {
  PrimExpr pred = IntImm::Bool(true);
  for (PrimExpr val : iter_predicates_) {
    pred = pred && val;
  }
  int updated = indices.size();
  sym::Analyzer analyzer_ref = ffi::GetRef<sym::Analyzer>(this->analyzer_);
  ffi::Array<PrimExpr> simplified =
      sym::IterMapSimplify(indices, this->iter_vars_, pred,
                           sym::IterMapLevel::Surjective, analyzer_ref);
  if (non_trivial_only) {
    for (int index = 0; index < updated; ++index) {
      if (simplified[index]->IsInstance<IntImmNode>() &&
          indices[index].as<PrimVar>()) {
        simplified.Set(index, indices[index]);
      }
    }
  }
  return simplified;
}

UnchangedOr<Stmt>
StmtExprMutatorWithAnalyzer::Mutate_(const ForNode *op,
                                     InplaceMode inplace_mode) {
  return constraint_scope_.WithNewScope([&]() -> UnchangedOr<Stmt> {
    Range dom = Range(op->begin, op->end);
    tl::arith::Bind(analyzer_, op->var, dom);
    iter_vars_.Set(op->var, dom);
    auto begin_result = this->Mutate(op->begin, inplace_mode);
    bool begin_unchanged = begin_result.UnchangedOrSameAs(op->begin);
    PrimExpr begin = std::move(begin_result).ValueOrUnchanged(op->begin);
    auto end_result = this->Mutate(op->end, inplace_mode);
    bool end_unchanged = end_result.UnchangedOrSameAs(op->end);
    PrimExpr end = std::move(end_result).ValueOrUnchanged(op->end);
    PrimExpr step =
        this->Mutate(op->step, inplace_mode).ValueOrUnchanged(op->step);
    Stmt body = constraint_scope_.WithNewScope([&]() -> Stmt {
      EnterConstraintFacts(&constraint_scope_.Current(), analyzer_,
                           end > begin);
      PrimExpr extent = Range::GetExtentFromBeginEnd(begin, end);
      EnterConstraintFacts(&constraint_scope_.Current(), analyzer_,
                           extent > IntImm(extent.ty(), 0));
      return this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    });
    if (begin_unchanged && end_unchanged && body.same_as(op->body) &&
        step.same_as(op->step)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto *updated = const_cast<ForNode *>(op);
        updated->begin = std::move(begin);
        updated->end = std::move(end);
        updated->step = std::move(step);
        updated->body = std::move(body);
        return ffi::Unchanged();
      }
      auto updated = ffi::make_object<ForNode>(*op);
      updated->begin = std::move(begin);
      updated->end = std::move(end);
      updated->step = std::move(step);
      updated->body = std::move(body);
      return Stmt(updated);
    }
  });
}

UnchangedOr<Stmt>
StmtExprMutatorWithAnalyzer::Mutate_(const BindNode *op,
                                     InplaceMode inplace_mode) {
  if (!op->value.as<PrimExpr>()) {
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }
  auto value_result = this->Mutate(op->value, inplace_mode);
  bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
  Expr value = std::move(value_result).ValueOrUnchanged(op->value);
  if (auto prim_value = value.as<PrimExpr>()) {
    if (SideEffect(prim_value.value()) <= CallEffectKind::kPure) {
      tl::arith::Bind(analyzer_, op->var, prim_value.value());
    }
  }
  if (value_unchanged)
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto *updated = const_cast<BindNode *>(op);
    updated->value = std::move(value);
    return ffi::Unchanged();
  }
  auto updated = ffi::make_object<BindNode>(*op);
  updated->value = std::move(value);
  return Stmt(updated);
}

UnchangedOr<Stmt>
StmtExprMutatorWithAnalyzer::Mutate_(const IfNode *op,
                                     InplaceMode inplace_mode) {
  return constraint_scope_.WithNewScope([&]() -> UnchangedOr<Stmt> {
    auto condition_result = this->Mutate(op->condition, inplace_mode);
    bool condition_unchanged =
        condition_result.UnchangedOrSameAs(op->condition);
    PrimExpr condition =
        std::move(condition_result).ValueOrUnchanged(op->condition);
    PrimExpr real_condition = condition;

    if (auto call = condition.as<CallNode>()) {
      static const Op likely_op = Op::Get("prim.likely");
      if (call->op.same_as(likely_op)) {
        real_condition = call->args[0].as_or_throw<PrimExpr>();
      }
    }

    bool condition_is_pure =
        SideEffect(real_condition) <= CallEffectKind::kPure;
    SeqStmt then_case = op->then_case;
    ffi::Optional<SeqStmt> else_case;
    constraint_scope_.WithNewScope([&]() {
      if (condition_is_pure) {
        EnterConstraintFacts(&constraint_scope_.Current(), analyzer_,
                             real_condition);
      }
      WithRecordIterPredicate(
          condition_is_pure ? real_condition : IntImm::Bool(true), [&] {
            then_case = this->Mutate(op->then_case, inplace_mode)
                            .ValueOrUnchanged(op->then_case);
          });
    });
    if (op->else_case) {
      PrimExpr neg_condition =
          tl::arith::RewriteSimplify(analyzer_, prim::Not(real_condition));
      constraint_scope_.WithNewScope([&]() {
        if (condition_is_pure) {
          constraint_scope_.Current().Emplace(analyzer_, neg_condition);
        }
        else_case = this->Mutate(op->else_case.value(), inplace_mode)
                        .ValueOrUnchanged(op->else_case.value());
      });
    }
    if (IsOne(real_condition))
      return then_case;
    if (IsZero(real_condition)) {
      return else_case.value_or(Evaluate(0));
    }

    if (condition_unchanged && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto *updated = const_cast<IfNode *>(op);
        updated->condition = std::move(condition);
        updated->then_case = std::move(then_case);
        updated->else_case = std::move(else_case);
        return ffi::Unchanged();
      }
      auto updated = ffi::make_object<IfNode>(*op);
      updated->condition = std::move(condition);
      updated->then_case = std::move(then_case);
      updated->else_case = std::move(else_case);
      return Stmt(updated);
    }
  });
}

UnchangedOr<Stmt>
StmtExprMutatorWithAnalyzer::Mutate_(const RegionStmtNode *op,
                                     InplaceMode inplace_mode) {
  return constraint_scope_.WithNewScope([&]() -> UnchangedOr<Stmt> {
    if (op->op.same_as(tirx::launch_thread_op())) {
      PrimVar var = op->body_params[0].as_or_throw<PrimVar>();
      PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
      Range dom = Range::FromMinExtent(IntImm(extent.ty(), 0), extent);
      tl::arith::Bind(analyzer_, var, dom);
      iter_vars_.Set(var, dom);
    }

    if (auto description =
            op->attrs->dict.Get(tvm::backend::cuda::attr::kLaunchFields.name)) {
      auto fields = description->as_or_throw<ffi::Array<ffi::String>>();
      for (size_t index = 0; index < fields.size(); ++index) {
        std::string field(fields[index]);
        std::string intrinsic;
        if (field.rfind("grid.", 0) == 0)
          intrinsic = "tirx.cuda.block_idx";
        if (field.rfind("block.", 0) == 0)
          intrinsic = "tirx.cuda.thread_idx";
        if (intrinsic.empty())
          continue;
        PrimExpr coordinate = Call(PrimType::Int(32), Op::Get(intrinsic),
                                   {StringImm(field.substr(field.size() - 1))})
                                  .as_or_throw<PrimExpr>();
        constraint_scope_.Current().Emplace(analyzer_, coordinate >= 0);
        constraint_scope_.Current().Emplace(
            analyzer_, coordinate < op->args[index].as_or_throw<PrimExpr>());
      }
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  });
}

UnchangedOr<Stmt>
StmtExprMutatorWithAnalyzer::Mutate_(const AssertStmtNode *op,
                                     InplaceMode inplace_mode) {
  auto condition_result = this->Mutate(op->condition, inplace_mode);
  bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
  PrimExpr condition =
      std::move(condition_result).ValueOrUnchanged(op->condition);

  if (SideEffect(condition) <= CallEffectKind::kPure) {
    constraint_scope_.Current().Emplace(analyzer_, condition);
  }

  if (condition_unchanged) {
    return ffi::Unchanged();
  } else {
    if (inplace_mode == InplaceMode::kAllow) {
      auto *updated = const_cast<AssertStmtNode *>(op);
      updated->condition = std::move(condition);
      return ffi::Unchanged();
    }
    auto updated = ffi::make_object<AssertStmtNode>(*op);
    updated->condition = std::move(condition);
    return Stmt(updated);
  }
}

UnchangedOr<Expr>
StmtExprMutatorWithAnalyzer::Mutate_(const CallNode *op,
                                     InplaceMode inplace_mode) {

  static const Op if_then_else_op = Op::Get("prim.if_then_else");
  if (op->op.same_as(if_then_else_op)) {
    PrimExpr cond = this->Mutate(op->args[0])
                        .ValueOrUnchanged(op->args[0])
                        .as_or_throw<PrimExpr>();
    Expr true_value = op->args[1], false_value = op->args[2];
    constraint_scope_.WithNewScope([&]() {
      EnterConstraintFacts(&constraint_scope_.Current(), analyzer_, cond);
      WithRecordIterPredicate(cond, [&] {
        true_value = this->Mutate(op->args[1])
                         .ValueOrUnchanged(op->args[1])
                         .as_or_throw<Expr>();
      });
    });
    {
      PrimExpr not_cond = prim::Not(cond);
      constraint_scope_.WithNewScope([&]() {
        constraint_scope_.Current().Emplace(analyzer_, not_cond);
        WithRecordIterPredicate(not_cond, [&] {
          false_value = this->Mutate(op->args[2])
                            .ValueOrUnchanged(op->args[2])
                            .as_or_throw<Expr>();
        });
      });
    }
    if (IsZero(cond)) {
      return false_value;
    }
    if (IsOne(cond)) {
      return true_value;
    }
    if (cond.same_as(op->args[0]) && true_value.same_as(op->args[1]) &&
        false_value.same_as(op->args[2])) {
      return ffi::Unchanged();
    } else {
      return Call(op->ty, op->op, {cond, true_value, false_value}, op->attrs,
                  {}, op->loc);
    }
  }
  return StmtExprMutator::Mutate_(op, inplace_mode);
}

UnchangedOr<PrimExpr>
StmtExprMutatorWithAnalyzer::Mutate_(const prim::LetNode *op,
                                     InplaceMode inplace_mode) {
  auto value_result = this->Mutate(op->value, inplace_mode);
  bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
  PrimExpr value = std::move(value_result).ValueOrUnchanged(op->value);
  if (SideEffect(value) <= CallEffectKind::kPure) {
    tl::arith::Bind(analyzer_, op->var, value);
  }

  auto body_result = this->Mutate(op->body, inplace_mode);
  bool body_unchanged = body_result.UnchangedOrSameAs(op->body);
  PrimExpr body = std::move(body_result).ValueOrUnchanged(op->body);
  if (value_unchanged && body_unchanged) {
    return ffi::Unchanged();
  } else {
    return prim::Let(op->var, value, body);
  }
}

UnchangedOr<PrimExpr>
StmtExprMutatorWithAnalyzer::Mutate_(const prim::SelectNode *op,
                                     InplaceMode inplace_mode) {
  auto cond_result = this->Mutate(op->condition, inplace_mode);
  bool cond_unchanged = cond_result.UnchangedOrSameAs(op->condition);
  PrimExpr cond = std::move(cond_result).ValueOrUnchanged(op->condition);
  PrimExpr true_value = op->true_value, false_value = op->false_value;
  constraint_scope_.WithNewScope([&]() {
    EnterConstraintFacts(&constraint_scope_.Current(), analyzer_, cond);
    true_value =
        Mutate(op->true_value, inplace_mode).ValueOrUnchanged(op->true_value);
  });
  {
    PrimExpr neg_cond = tl::arith::RewriteSimplify(analyzer_, prim::Not(cond));
    constraint_scope_.WithNewScope([&]() {
      constraint_scope_.Current().Emplace(analyzer_, neg_cond);
      false_value = Mutate(op->false_value, inplace_mode)
                        .ValueOrUnchanged(op->false_value);
    });
  }
  if (IsZero(cond)) {
    return false_value;
  }
  if (IsOne(cond)) {
    return true_value;
  }

  if (cond_unchanged && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return ffi::Unchanged();
  } else {
    return prim::Select(cond, true_value, false_value);
  }
}

} // namespace tl
} // namespace tvm
