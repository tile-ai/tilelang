/*!
 * \file index_predicate_rewriter.h
 * \brief Keep predicates consistent with promoted buffer-index arithmetic.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_INDEX_PREDICATE_REWRITER_H_
#define TVM_TL_TRANSFORM_COMMON_INDEX_PREDICATE_REWRITER_H_

#include <unordered_map>

#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include "int64_promoter.h"

namespace tvm {
namespace tl {

/*!
 * \brief Record index promotions and apply them to matching predicate
 * arithmetic.
 *
 * Rewrite predicates after visiting all accesses, so enclosing guards and
 * guards introduced by safe-memory legalization use the same arithmetic as the
 * address. Only matching arithmetic is changed; data expressions and
 * fixed-width value boundaries retain their original semantics.
 */
class IndexPredicateRewriter : public tirx::StmtExprMutator {
public:
  PrimExpr Promote(const PrimExpr &index) {
    RecordingPromoter promoter(&promotions_);
    return promoter(index);
  }

  tirx::Stmt Rewrite(const tirx::Stmt &stmt) {
    return promotions_.empty() ? stmt : VisitStmt(stmt);
  }

private:
  using PromotionMap =
      std::unordered_map<PrimExpr, PrimExpr, ffi::StructuralHash,
                         tirx::ExprDeepEqual>;

  class RecordingPromoter : public Int64Promoter {
  public:
    explicit RecordingPromoter(PromotionMap *promotions)
        : promotions_(promotions) {}

    PrimExpr VisitExpr(const PrimExpr &expr) final {
      PrimExpr promoted = Int64Promoter::VisitExpr(expr);
      if (expr.dtype().is_int() && expr.dtype().bits() < 64 &&
          promoted.dtype().is_int() && promoted.dtype().bits() == 64 &&
          !expr.as<tirx::VarNode>() && !expr.as<tirx::IntImmNode>() &&
          !expr.as<tirx::CastNode>() && !expr.as<tirx::BufferLoadNode>()) {
        promotions_->insert_or_assign(expr, promoted);
      }
      return promoted;
    }

  private:
    PromotionMap *promotions_;
  };

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

    PrimExpr VisitExpr_(const tirx::CallNode *op) final {
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

  PrimExpr RewritePredicate_(const PrimExpr &predicate) {
    PredicateLegalizer legalizer(promotions_);
    return legalizer(predicate);
  }

  tirx::Stmt VisitStmt_(const tirx::IfThenElseNode *op) final {
    tirx::IfThenElse stmt =
        tvm::Downcast<tirx::IfThenElse>(StmtExprMutator::VisitStmt_(op));
    stmt.CopyOnWrite()->condition = RewritePredicate_(stmt->condition);
    return stmt;
  }

  tirx::Stmt VisitStmt_(const tirx::WhileNode *op) final {
    tirx::While stmt =
        tvm::Downcast<tirx::While>(StmtExprMutator::VisitStmt_(op));
    stmt.CopyOnWrite()->condition = RewritePredicate_(stmt->condition);
    return stmt;
  }

  tirx::Stmt VisitStmt_(const tirx::AssertStmtNode *op) final {
    tirx::AssertStmt stmt =
        tvm::Downcast<tirx::AssertStmt>(StmtExprMutator::VisitStmt_(op));
    stmt.CopyOnWrite()->condition = RewritePredicate_(stmt->condition);
    return stmt;
  }

  tirx::Stmt VisitStmt_(const tirx::SBlockRealizeNode *op) final {
    tirx::SBlockRealize stmt =
        tvm::Downcast<tirx::SBlockRealize>(StmtExprMutator::VisitStmt_(op));
    stmt.CopyOnWrite()->predicate = RewritePredicate_(stmt->predicate);
    return stmt;
  }

  tirx::Stmt VisitStmt_(const tirx::BufferStoreNode *op) final {
    tirx::BufferStore store =
        tvm::Downcast<tirx::BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (store->predicate.defined()) {
      store.CopyOnWrite()->predicate =
          RewritePredicate_(store->predicate.value());
    }
    return store;
  }

  PrimExpr VisitExpr_(const tirx::BufferLoadNode *op) final {
    tirx::BufferLoad load =
        tvm::Downcast<tirx::BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (load->predicate.defined()) {
      load.CopyOnWrite()->predicate =
          RewritePredicate_(load->predicate.value());
    }
    return load;
  }

  PrimExpr VisitExpr_(const tirx::SelectNode *op) final {
    tirx::Select expr =
        tvm::Downcast<tirx::Select>(StmtExprMutator::VisitExpr_(op));
    expr.CopyOnWrite()->condition = RewritePredicate_(expr->condition);
    return expr;
  }

  PrimExpr VisitExpr_(const tirx::CallNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    if (op->op.same_as(tirx::builtin::if_then_else())) {
      tirx::Call call = tvm::Downcast<tirx::Call>(expr);
      call.CopyOnWrite()->args.Set(0, RewritePredicate_(call->args[0]));
      return call;
    }
    return expr;
  }

  PromotionMap promotions_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_INDEX_PREDICATE_REWRITER_H_
