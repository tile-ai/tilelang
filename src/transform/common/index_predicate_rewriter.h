/*!
 * \file index_predicate_rewriter.h
 * \brief Keep predicates consistent with promoted buffer-index arithmetic.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_INDEX_PREDICATE_REWRITER_H_
#define TVM_TL_TRANSFORM_COMMON_INDEX_PREDICATE_REWRITER_H_

#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include "int64_promoter.h"

namespace tvm {
namespace tl {

/*!
 * \brief Associate promoted address arithmetic with its controlling predicates.
 *
 * Each access owns its promotions. Only predicates controlling that access use
 * them; arithmetic in data selections and fixed-width value boundaries is kept.
 */
class IndexPredicateRewriter : public tirx::StmtExprMutator {
private:
  using PromotionMap =
      std::unordered_map<PrimExpr, PrimExpr, ffi::StructuralHash,
                         tirx::ExprDeepEqual>;

public:
  class Promoter : public Int64Promoter {
  public:
    PrimExpr VisitExpr(const PrimExpr &expr) final {
      PrimExpr promoted = Int64Promoter::VisitExpr(expr);
      if (expr.dtype().is_int() && expr.dtype().bits() < 64 &&
          promoted.dtype().is_int() && promoted.dtype().bits() == 64 &&
          !expr.as<tirx::VarNode>() && !expr.as<tirx::IntImmNode>() &&
          !expr.as<tirx::CastNode>() && !expr.as<tirx::BufferLoadNode>()) {
        promotions_.insert_or_assign(expr, promoted);
      }
      return promoted;
    }

  private:
    friend class IndexPredicateRewriter;
    PromotionMap promotions_;
  };

  void Record(const ffi::ObjectRef &access, Promoter &promoter) {
    if (!promoter.promotions_.empty()) {
      access_promotions_.insert_or_assign(access,
                                          std::move(promoter.promotions_));
    }
  }

  tirx::Stmt Rewrite(const tirx::Stmt &stmt) {
    return access_promotions_.empty() ? stmt : VisitStmt(stmt);
  }

private:
  class PredicateLegalizer : public tirx::DataTypeLegalizer {
  public:
    explicit PredicateLegalizer(const PromotionMap &promotions)
        : promotions_(promotions) {}

    PrimExpr VisitExpr(const PrimExpr &expr) final {
      auto it = promotions_.find(expr);
      return it != promotions_.end() ? it->second
                                     : DataTypeLegalizer::VisitExpr(expr);
    }

  private:
    PrimExpr VisitExpr_(const tirx::CastNode *op) final {
      // A widening conversion may have been inserted to match the bound's
      // dtype. Widen the matching arithmetic before this conversion as well.
      // Narrowing conversions remain explicit fixed-width value boundaries.
      if (op->dtype.is_int() && op->value.dtype().is_int() &&
          op->dtype.bits() > op->value.dtype().bits()) {
        return DataTypeLegalizer::VisitExpr_(op);
      }
      return ffi::GetRef<PrimExpr>(op);
    }

    PrimExpr VisitExpr_(const tirx::BufferLoadNode *op) final {
      return ffi::GetRef<PrimExpr>(op);
    }

    PrimExpr VisitExpr_(const tirx::LetNode *op) final {
      return tirx::Let(op->var, op->value, VisitExpr(op->body), op->span);
    }

    PrimExpr VisitExpr_(const tirx::SelectNode *op) final {
      // A data-valued selection is a fixed-width producer. Only a selection
      // promoted as part of this access's address may be replaced as a whole.
      return op->dtype.is_bool() ? DataTypeLegalizer::VisitExpr_(op)
                                 : ffi::GetRef<PrimExpr>(op);
    }

    PrimExpr VisitExpr_(const tirx::CallNode *op) final {
      if (op->op.same_as(tirx::builtin::if_then_else()) &&
          !op->dtype.is_bool()) {
        return ffi::GetRef<PrimExpr>(op);
      }
      if (op->op.same_as(tirx::builtin::if_then_else()) ||
          op->op.same_as(tirx::builtin::likely()) ||
          op->op.same_as(tirx::builtin::shift_left()) ||
          op->op.same_as(tirx::builtin::shift_right()) ||
          op->op.same_as(tirx::builtin::bitwise_and()) ||
          op->op.same_as(tirx::builtin::bitwise_or()) ||
          op->op.same_as(tirx::builtin::bitwise_xor())) {
        PrimExpr value = DataTypeLegalizer::VisitExpr_(op);
        if (const auto *call = value.as<tirx::CallNode>()) {
          return tirx::Call(call->dtype, call->op, call->args, op->annotations,
                            op->span);
        }
        return value;
      }
      // Do not alter arguments of calls with an independent ABI.
      return ffi::GetRef<PrimExpr>(op);
    }

    const PromotionMap &promotions_;
  };

  static void Merge_(PromotionMap &target, const PromotionMap &source) {
    target.insert(source.begin(), source.end());
  }

  PrimExpr RewritePredicate_(const PrimExpr &predicate,
                             const PromotionMap &promotions) {
    if (promotions.empty()) {
      return predicate;
    }
    PredicateLegalizer legalizer(promotions);
    return legalizer(predicate);
  }

  tirx::Stmt RewriteDominatingAssertions_(const tirx::Stmt &stmt,
                                          const PromotionMap &suffix) {
    if (suffix.empty()) {
      return stmt;
    }
    if (auto assertion = stmt.as<tirx::AssertStmt>()) {
      PrimExpr condition =
          RewritePredicate_(assertion.value()->condition, suffix);
      if (!condition.same_as(assertion.value()->condition)) {
        tirx::AssertStmt updated = assertion.value();
        updated.CopyOnWrite()->condition = condition;
        return updated;
      }
    } else if (auto attr = stmt.as<tirx::AttrStmt>()) {
      // Thread launches may execute the body zero times. Other attributes are
      // annotations whose body executes before the enclosing sequence resumes.
      if (attr.value()->attr_key == tirx::attr::thread_extent ||
          attr.value()->attr_key == s_tir::attr::virtual_thread) {
        return stmt;
      }
      tirx::Stmt body =
          RewriteDominatingAssertions_(attr.value()->body, suffix);
      if (!body.same_as(attr.value()->body)) {
        tirx::AttrStmt updated = attr.value();
        updated.CopyOnWrite()->body = body;
        return updated;
      }
    } else if (auto sequence = stmt.as<tirx::SeqStmt>()) {
      // Local suffixes were already handled by VisitStmt_. Every straight-line
      // assertion here also controls the suffix outside this nested sequence.
      ffi::Array<tirx::Stmt> seq =
          sequence.value()->seq.Map([&](const tirx::Stmt &child) {
            return RewriteDominatingAssertions_(child, suffix);
          });
      if (!seq.same_as(sequence.value()->seq)) {
        tirx::SeqStmt updated = sequence.value();
        updated.CopyOnWrite()->seq = seq;
        return updated;
      }
    }
    // Branches, loops, and block predicates can skip their bodies.
    return stmt;
  }

  const PromotionMap &AccessPromotions_(const ffi::ObjectRef &access) const {
    auto it = access_promotions_.find(access);
    return it != access_promotions_.end() ? it->second : empty_promotions_;
  }

  void FinishScope_(const ffi::ObjectRef &node) {
    PromotionMap promotions = std::move(scopes_.back());
    scopes_.pop_back();
    Merge_(promotions, AccessPromotions_(node));
    last_promotions_ = std::move(promotions);
    if (!scopes_.empty()) {
      Merge_(scopes_.back(), last_promotions_);
    }
  }

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    scopes_.emplace_back();
    PrimExpr result = StmtExprMutator::VisitExpr(expr);
    FinishScope_(expr);
    return result;
  }

  tirx::Stmt VisitStmt(const tirx::Stmt &stmt) final {
    scopes_.emplace_back();
    tirx::Stmt result = StmtExprMutator::VisitStmt(stmt);
    FinishScope_(stmt);
    return result;
  }

  tirx::Stmt VisitStmt_(const tirx::IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    tirx::Stmt then_case = VisitStmt(op->then_case);
    PromotionMap promotions = last_promotions_;
    ffi::Optional<tirx::Stmt> else_case = std::nullopt;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
      Merge_(promotions, last_promotions_);
    }
    condition = RewritePredicate_(condition, promotions);
    return tirx::IfThenElse(condition, then_case, else_case, op->span);
  }

  tirx::Stmt VisitStmt_(const tirx::WhileNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    tirx::Stmt body = VisitStmt(op->body);
    return tirx::While(RewritePredicate_(condition, last_promotions_), body,
                       op->span);
  }

  tirx::Stmt VisitStmt_(const tirx::SeqStmtNode *op) final {
    // An assertion controls only the subsequent statements in this sequence.
    // Reverse traversal collects those accesses without including earlier ones.
    PromotionMap suffix;
    ffi::Array<tirx::Stmt> seq = op->seq;
    for (size_t i = seq.size(); i > 0; --i) {
      tirx::Stmt stmt = VisitStmt(op->seq[i - 1]);
      PromotionMap promotions = last_promotions_;
      seq.Set(i - 1, RewriteDominatingAssertions_(stmt, suffix));
      Merge_(suffix, promotions);
    }
    return tirx::SeqStmt(seq, op->span);
  }

  tirx::Stmt VisitStmt_(const tirx::SBlockRealizeNode *op) final {
    ffi::Array<PrimExpr> values = op->iter_values.Map(
        [this](const PrimExpr &value) { return VisitExpr(value); });
    PrimExpr predicate = VisitExpr(op->predicate);
    tirx::SBlock block = tvm::Downcast<tirx::SBlock>(VisitStmt(op->block));
    return tirx::SBlockRealize(values,
                               RewritePredicate_(predicate, last_promotions_),
                               block, op->span);
  }

  tirx::Stmt VisitStmt_(const tirx::BufferStoreNode *op) final {
    tirx::BufferStore store =
        tvm::Downcast<tirx::BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (op->predicate.defined()) {
      store.CopyOnWrite()->predicate = RewritePredicate_(
          VisitExpr(op->predicate.value()),
          AccessPromotions_(ffi::GetRef<tirx::BufferStore>(op)));
    }
    return store;
  }

  PrimExpr VisitExpr_(const tirx::BufferLoadNode *op) final {
    tirx::BufferLoad load =
        tvm::Downcast<tirx::BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (op->predicate.defined()) {
      load.CopyOnWrite()->predicate = RewritePredicate_(
          VisitExpr(op->predicate.value()),
          AccessPromotions_(ffi::GetRef<tirx::BufferLoad>(op)));
    }
    return load;
  }

  PrimExpr VisitExpr_(const tirx::CallNode *op) final {
    if (!op->op.same_as(tirx::builtin::if_then_else())) {
      return StmtExprMutator::VisitExpr_(op);
    }
    PrimExpr condition = VisitExpr(op->args[0]);
    PrimExpr true_value = VisitExpr(op->args[1]);
    PromotionMap promotions = last_promotions_;
    PrimExpr false_value = VisitExpr(op->args[2]);
    Merge_(promotions, last_promotions_);
    return tirx::Call(
        op->dtype, op->op,
        {RewritePredicate_(condition, promotions), true_value, false_value},
        op->annotations, op->span);
  }

  std::unordered_map<ffi::ObjectRef, PromotionMap, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      access_promotions_;
  const PromotionMap empty_promotions_;
  std::vector<PromotionMap> scopes_;
  PromotionMap last_promotions_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_INDEX_PREDICATE_REWRITER_H_
