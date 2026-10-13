/*!
 * \file index_predicate_rewriter.h
 * \brief Keep predicates consistent with promoted buffer-index arithmetic.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_INDEX_PREDICATE_REWRITER_H_
#define TVM_TL_TRANSFORM_COMMON_INDEX_PREDICATE_REWRITER_H_

#include <unordered_map>
#include <unordered_set>
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
  // Only compare pure, same-width integer addition/multiplication modulo
  // associativity and commutativity. Casts, calls, loads, and Let values remain
  // opaque leaves; no analyzer assumptions or arithmetic distribution are used.
  static bool IsCommutativeArithmetic_(const PrimExpr &expr) {
    return expr.dtype().is_int() &&
           (expr.as<tirx::AddNode>() || expr.as<tirx::MulNode>()) &&
           tirx::SideEffect(expr) <= tirx::CallEffectKind::kPure;
  }

  static void CollectTerms_(const PrimExpr &expr, bool addition, DataType dtype,
                            std::vector<PrimExpr> &terms) {
    if (expr.dtype() == dtype) {
      if (const auto *add = expr.as<tirx::AddNode>(); addition && add) {
        CollectTerms_(add->a, addition, dtype, terms);
        CollectTerms_(add->b, addition, dtype, terms);
        return;
      }
      if (const auto *mul = expr.as<tirx::MulNode>(); !addition && mul) {
        CollectTerms_(mul->a, addition, dtype, terms);
        CollectTerms_(mul->b, addition, dtype, terms);
        return;
      }
    }
    terms.push_back(expr);
  }

  static std::vector<PrimExpr> ArithmeticTerms_(const PrimExpr &expr) {
    std::vector<PrimExpr> terms;
    CollectTerms_(expr, expr.as<tirx::AddNode>() != nullptr, expr.dtype(),
                  terms);
    return terms;
  }

  struct IndexArithmeticHash {
    size_t operator()(const PrimExpr &expr) const {
      if (!IsCommutativeArithmetic_(expr)) {
        return ffi::StructuralHash()(expr);
      }
      bool addition = expr.as<tirx::AddNode>() != nullptr;
      size_t hash = ffi::StructuralHash()(
          tirx::make_const(expr.dtype(), addition ? 0 : 1));
      const auto terms = ArithmeticTerms_(expr);
      for (const auto &term : terms) {
        hash += (*this)(term);
      }
      return hash ^ terms.size();
    }
  };

  struct IndexArithmeticEqual {
    bool operator()(const PrimExpr &lhs, const PrimExpr &rhs) const {
      if (tirx::ExprDeepEqual()(lhs, rhs)) {
        return true;
      }
      if (lhs.dtype() != rhs.dtype() || !IsCommutativeArithmetic_(lhs) ||
          !IsCommutativeArithmetic_(rhs) ||
          (lhs.as<tirx::AddNode>() != nullptr) !=
              (rhs.as<tirx::AddNode>() != nullptr)) {
        return false;
      }
      const auto left = ArithmeticTerms_(lhs);
      const auto right = ArithmeticTerms_(rhs);
      if (left.size() != right.size()) {
        return false;
      }
      std::vector<bool> matched(right.size(), false);
      for (const auto &term : left) {
        size_t i = 0;
        for (; i < right.size(); ++i) {
          if (!matched[i] && (*this)(term, right[i])) {
            matched[i] = true;
            break;
          }
        }
        if (i == right.size()) {
          return false;
        }
      }
      return true;
    }
  };

  using PromotionMap =
      std::unordered_map<PrimExpr, PrimExpr, IndexArithmeticHash,
                         IndexArithmeticEqual>;
  using AccessSet = std::unordered_set<ffi::ObjectRef, ffi::ObjectPtrHash,
                                       ffi::ObjectPtrEqual>;

  struct AccessPromotions {
    PromotionMap own;
    AccessSet nested;
  };

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

    /*! \brief Associate bounds derived from this access's actual extent. */
    void RecordExtent(const PrimExpr &offset, const PrimExpr &promoted_offset,
                      const PrimExpr &extent) {
      if (!promoted_offset.dtype().is_int() ||
          promoted_offset.dtype().bits() != 64) {
        return;
      }
      PrimExpr end = offset + extent;
      PrimExpr promoted_end =
          promoted_offset + tvm::cast(promoted_offset.dtype(), extent);
      PrimExpr one = tirx::make_const(end.dtype(), 1);
      PrimExpr promoted_one = tirx::make_const(promoted_end.dtype(), 1);
      auto record = [&](const PrimExpr &value, const PrimExpr &promoted) {
        if (!value.as<tirx::IntImmNode>() &&
            !IndexArithmeticEqual()(value, offset)) {
          promotions_.insert_or_assign(value, promoted);
        }
      };
      record(end, promoted_end);
      record(extent + offset, promoted_end);
      record(end - one, promoted_end - promoted_one);
      record(offset + (extent - one), promoted_end - promoted_one);
      record((extent + offset) - one, promoted_end - promoted_one);
    }

    /*! \brief Include scalar endpoints of vector index arithmetic. */
    void RecordIndex(const PrimExpr &index, const PrimExpr &promoted) {
      RecordExtent(index, promoted, tirx::make_const(index.dtype(), 1));
      const auto *ramp = index.as<tirx::RampNode>();
      const auto *wide_ramp = promoted.as<tirx::RampNode>();
      if (!ramp || !wide_ramp) {
        return;
      }
      PrimExpr one = tirx::make_const(ramp->base.dtype(), 1);
      PrimExpr wide_one = tirx::make_const(wide_ramp->base.dtype(), 1);
      PrimExpr displacement =
          ramp->stride * (tvm::cast(ramp->base.dtype(), ramp->lanes) - one);
      PrimExpr wide_last =
          wide_ramp->base +
          wide_ramp->stride *
              (tvm::cast(wide_ramp->base.dtype(), wide_ramp->lanes) - wide_one);
      // Keep the stride/lanes arithmetic at the actual address width; unlike
      // an access_ptr extent, these are index producers, not ABI values.
      for (const PrimExpr &last :
           {ramp->base + displacement, ramp->base - (-displacement)}) {
        if (!last.as<tirx::IntImmNode>() && !last.as<tirx::VarNode>()) {
          promotions_.insert_or_assign(last, wide_last);
        }
        RecordExtent(last, wide_last, one);
      }
    }

  private:
    friend class IndexPredicateRewriter;
    PromotionMap promotions_;
    AccessSet nested_accesses_;
  };

  void Record(const ffi::ObjectRef &access, Promoter &promoter) {
    if (!promoter.promotions_.empty() || !promoter.nested_accesses_.empty()) {
      access_promotions_.insert_or_assign(
          access, AccessPromotions{std::move(promoter.promotions_),
                                   std::move(promoter.nested_accesses_)});
    }
  }

  /*! \brief Rewrite nested guards before an index transformation clones them.
   */
  PrimExpr PrepareIndex(const PrimExpr &index, Promoter &promoter) {
    if (!HasPromotedAccess(index)) {
      return index;
    }
    // Keep provenance on the enclosing access as well: later simplification
    // may clone its children before an outer statement guard is rewritten.
    tirx::PostOrderVisit(index, [&](const ffi::ObjectRef &node) {
      MergeAccesses_(promoter.nested_accesses_, node);
    });
    return VisitExpr(index);
  }

  /*! \brief Whether an expression contains an access awaiting guard rewriting.
   */
  bool HasPromotedAccess(const PrimExpr &expr) const {
    if (access_promotions_.empty()) {
      return false;
    }
    bool found = false;
    tirx::PostOrderVisit(expr, [this, &found](const ffi::ObjectRef &node) {
      found |= access_promotions_.count(node) != 0;
    });
    return found;
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
      if (it != promotions_.end()) {
        return it->second;
      }
      // Match complete integer producers used by promoted addresses. Recursing
      // into other integer arithmetic could change an unpromoted address or a
      // business condition which merely contains the same index subexpression.
      if (!expr.dtype().is_bool() && !expr.as<tirx::CastNode>()) {
        return expr;
      }
      return DataTypeLegalizer::VisitExpr(expr);
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

  static void Merge_(AccessSet &target, const AccessSet &source) {
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

  PrimExpr RewritePredicate_(const PrimExpr &predicate,
                             const AccessSet &accesses) {
    PromotionMap common;
    std::unordered_set<PrimExpr, IndexArithmeticHash, IndexArithmeticEqual>
        conflicts;
    for (const auto &access : accesses) {
      for (const auto &[value, promoted] : AccessPromotions_(access)) {
        auto [it, inserted] = common.emplace(value, promoted);
        if (!inserted && !IndexArithmeticEqual()(it->second, promoted)) {
          conflicts.insert(value);
        }
      }
    }
    if (conflicts.empty()) {
      return RewritePredicate_(predicate, common);
    }
    // An extent is an ABI value, while another access may use the same source
    // expression as index arithmetic. Accept only a single interpretation of
    // the controlling predicate; do not combine, copy, or restructure control
    // flow to accommodate incompatible integer semantics.
    for (const auto &value : conflicts) {
      common.erase(value);
    }
    ffi::Optional<PrimExpr> result = std::nullopt;
    for (const auto &access : accesses) {
      PromotionMap promotions = common;
      const auto &own = AccessPromotions_(access);
      promotions.insert(own.begin(), own.end());
      PrimExpr condition = RewritePredicate_(predicate, promotions);
      if (!result.defined()) {
        result = condition;
      } else if (!tirx::ExprDeepEqual()(result.value(), condition)) {
        TVM_FFI_THROW(NotImplementedError)
            << "Cannot preserve predicate semantics with conflicting index "
               "widths: "
            << predicate
            << ". Split the accesses into separately guarded statements with "
               "an explicit fallback where required.";
      }
    }
    return result.value_or(predicate);
  }

  tirx::Stmt RewriteDominatingAssertions_(const tirx::Stmt &stmt,
                                          const AccessSet &suffix) {
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
    return it != access_promotions_.end() ? it->second.own : empty_promotions_;
  }

  void MergeAccesses_(AccessSet &target, const ffi::ObjectRef &access) const {
    auto it = access_promotions_.find(access);
    if (it != access_promotions_.end()) {
      if (!it->second.own.empty()) {
        target.insert(access);
      }
      Merge_(target, it->second.nested);
    }
  }

  void FinishScope_(const ffi::ObjectRef &node) {
    AccessSet accesses = std::move(scopes_.back());
    scopes_.pop_back();
    MergeAccesses_(accesses, node);
    last_accesses_ = std::move(accesses);
    if (!scopes_.empty()) {
      Merge_(scopes_.back(), last_accesses_);
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

  class BufferUseRemapper : public tirx::StmtExprMutator {
  public:
    explicit BufferUseRemapper(
        const ffi::Map<tirx::Buffer, tirx::Buffer> &remap)
        : remap_(remap) {}

  private:
    tirx::Buffer VisitBufferUse(const tirx::Buffer &buffer) final {
      auto it = remap_.find(buffer);
      return it != remap_.end() ? (*it).second : buffer;
    }

    const ffi::Map<tirx::Buffer, tirx::Buffer> &remap_;
  };

  tirx::Buffer VisitBufferDef(const tirx::Buffer &buffer,
                              bool alloc_data) final {
    tirx::Buffer updated = StmtExprMutator::VisitBufferDef(buffer, alloc_data);
    if (!updated.same_as(buffer)) {
      // Predicates and addresses now use the rebuilt buffer. Keep both sides
      // of each recorded substitution in the same buffer namespace.
      BufferUseRemapper remapper(buffer_remap_);
      for (auto &[access, promotions] : access_promotions_) {
        PromotionMap remapped;
        for (const auto &[value, promoted] : promotions.own) {
          remapped.insert_or_assign(remapper(value), remapper(promoted));
        }
        promotions.own = std::move(remapped);
      }
    }
    return updated;
  }

  tirx::Stmt VisitStmt_(const tirx::IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    tirx::Stmt then_case = VisitStmt(op->then_case);
    AccessSet accesses = last_accesses_;
    ffi::Optional<tirx::Stmt> else_case = std::nullopt;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
      Merge_(accesses, last_accesses_);
    }
    condition = RewritePredicate_(condition, accesses);
    return tirx::IfThenElse(condition, then_case, else_case, op->span);
  }

  tirx::Stmt VisitStmt_(const tirx::WhileNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    tirx::Stmt body = VisitStmt(op->body);
    return tirx::While(RewritePredicate_(condition, last_accesses_), body,
                       op->span);
  }

  tirx::Stmt VisitStmt_(const tirx::SeqStmtNode *op) final {
    // Visit definitions before uses so the mutator's buffer remapping remains
    // valid when a buffer's shape, strides, or elem_offset changes.
    ffi::Array<tirx::Stmt> seq = op->seq;
    std::vector<AccessSet> accesses;
    accesses.reserve(seq.size());
    for (size_t i = 0; i < seq.size(); ++i) {
      seq.Set(i, VisitStmt(op->seq[i]));
      accesses.push_back(last_accesses_);
    }
    // Only suffix analysis runs backwards; this does not revisit buffer fields.
    AccessSet suffix;
    for (size_t i = seq.size(); i > 0; --i) {
      seq.Set(i - 1, RewriteDominatingAssertions_(seq[i - 1], suffix));
      Merge_(suffix, accesses[i - 1]);
    }
    return tirx::SeqStmt(seq, op->span);
  }

  tirx::Stmt VisitStmt_(const tirx::SBlockRealizeNode *op) final {
    ffi::Array<PrimExpr> values = op->iter_values.Map(
        [this](const PrimExpr &value) { return VisitExpr(value); });
    PrimExpr predicate = VisitExpr(op->predicate);
    tirx::SBlock block = tvm::Downcast<tirx::SBlock>(VisitStmt(op->block));
    return tirx::SBlockRealize(
        values, RewritePredicate_(predicate, last_accesses_), block, op->span);
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
    AccessSet accesses = last_accesses_;
    PrimExpr false_value = VisitExpr(op->args[2]);
    Merge_(accesses, last_accesses_);
    condition = RewritePredicate_(condition, accesses);
    if (condition.same_as(op->args[0]) && true_value.same_as(op->args[1]) &&
        false_value.same_as(op->args[2])) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return tirx::Call(op->dtype, op->op, {condition, true_value, false_value},
                      op->annotations, op->span);
  }

  std::unordered_map<ffi::ObjectRef, AccessPromotions, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      access_promotions_;
  const PromotionMap empty_promotions_;
  std::vector<AccessSet> scopes_;
  AccessSet last_accesses_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_INDEX_PREDICATE_REWRITER_H_
