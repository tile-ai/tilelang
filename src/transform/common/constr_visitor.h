#ifndef TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
#define TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_

#include "support/check.h"
#include "tvm/arith/analyzer.h"
#include "tvm/ir/expr.h"
#include <ostream>
#include <string>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/var.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {

/*!
 * \brief Replace every mutable read in an expression with a fresh, independent
 *        variable.
 *
 * `Analyzer::Bind` installs a rewrite `var -> value`, so a value reading
 * mutable state lets the analyzer equate two reads across a store it does not
 * track, and makes two instances of one constraint set agree where two threads
 * would read two different registers. A fresh unknown per read only ever
 * weakens equalities, and keeps the pure structure around the read.
 */
class FreshenMutableReads : public tirx::ExprMutator {
public:
  using tirx::ExprMutator::operator();

private:
  PrimExpr Fresh(const PrimExpr &e) {
    auto it = memo_.find(e);
    if (it != memo_.end())
      return it->second;
    tirx::Var var("free" + std::to_string(memo_.size()), e.dtype());
    memo_.emplace(e, var);
    return var;
  }

  PrimExpr VisitExpr_(const tirx::BufferLoadNode *op) override {
    return Fresh(ffi::GetRef<PrimExpr>(op));
  }
  PrimExpr VisitExpr_(const tirx::ProducerLoadNode *op) override {
    return Fresh(ffi::GetRef<PrimExpr>(op));
  }
  PrimExpr VisitExpr_(const tirx::ReduceNode *op) override {
    return Fresh(ffi::GetRef<PrimExpr>(op));
  }
  PrimExpr VisitExpr_(const tirx::CallNode *op) override {
    // Opaque or stateful calls are unknowns; pure calls recurse so that their
    // pure sub-structure survives.
    if (tirx::SideEffect(ffi::GetRef<PrimExpr>(op)) >
        tirx::CallEffectKind::kPure)
      return Fresh(ffi::GetRef<PrimExpr>(op));
    return tirx::ExprMutator::VisitExpr_(op);
  }

  std::unordered_map<PrimExpr, PrimExpr, ffi::StructuralHash,
                     tirx::ExprDeepEqual>
      memo_;
};

struct Constr {

  enum Kind {
    kConstr,
    kBindValue,
    kBindRange,
  } kind;
  bool is_assume = false;
  tirx::Var var;
  PrimExpr value;
  Range range;

  Constr(PrimExpr constr, bool is_assume = false)
      : kind(kConstr), value(constr), is_assume(is_assume) {};
  Constr(tirx::Var var, PrimExpr val)
      : kind(kBindValue), var(var), value(val) {};
  Constr(tirx::Var var, Range range)
      : kind(kBindRange), var(var), range(range) {};

  Constr() = default;
  Constr(const Constr &other) = default;
  Constr(Constr &&other) = default;
  Constr &operator=(const Constr &other) = default;

  void Format(std::ostream &os) const {
    os << "Constr(kind=";
    switch (kind) {
    case kConstr:
      os << "kConstr";
      os << ", is_assume=" << (is_assume ? "true" : "false");
      os << ", value=" << value;
      break;
    case kBindValue:
      os << "kBindValue";
      os << ", var=" << var->name_hint;
      os << ", value=" << value;
      break;
    case kBindRange:
      os << "kBindRange";
      os << ", var=" << var->name_hint;
      os << ", range=Range(min=" << range->min;
      os << ", extent=" << range->extent << ")";
      break;
    default:
      os << "Unknown";
    }
    os << ")";
  }

  PrimExpr ToGenericConstr() const {
    switch (kind) {
    case kConstr:
      return value;
    case kBindValue:
      // A vector-typed bind cannot be stated as an equality.
      if (var.dtype().is_vector())
        return Bool(true);
      return var == value;
    case kBindRange:
      return tirx::And(var >= range->min, var < (range->min + range->extent));
    }
    LOG(FATAL) << "Unreachable";
    return PrimExpr();
  }
  /*!
   * \brief Rewrite through \p subs, keeping the kind whenever possible, so that
   *        renaming one side of a two-instance comparison keeps its binds.
   */
  Constr Substitute(ffi::Map<tirx::Var, PrimExpr> subs) const {
    switch (kind) {
    case kConstr:
      return Constr(tirx::Substitute(value, subs), is_assume);
    case kBindValue:
    case kBindRange: {
      auto it = subs.find(var);
      const auto *new_var =
          it == subs.end() ? var.get() : (*it).second.as<tirx::VarNode>();
      // A bind remapped onto a non-variable is only expressible as a predicate.
      if (new_var == nullptr)
        return Constr(tirx::Substitute(ToGenericConstr(), subs), is_assume);
      if (kind == kBindValue)
        return Constr(ffi::GetRef<tirx::Var>(new_var),
                      tirx::Substitute(value, subs));
      return Constr(
          ffi::GetRef<tirx::Var>(new_var),
          Range::FromMinExtent(tirx::Substitute(range->min, subs),
                               tirx::Substitute(range->extent, subs)));
    }
    }
    LOG(FATAL) << "Unreachable";
    return Constr();
  }
  void Populate(arith::Analyzer &analyzer) const {
    switch (kind) {
    case kConstr:
      analyzer.EnterConstraint(value, is_assume);
      break;
    case kBindValue:
      analyzer.Bind(var, FreshenMutableReads()(value));
      break;
    case kBindRange: {
      FreshenMutableReads freshen;
      analyzer.Bind(var, Range::FromMinExtent(freshen(range->min),
                                              freshen(range->extent)));
      break;
    }
    default:
      LOG(FATAL) << "Unreachable";
    }
  }
};

struct ConstrSet {
  ConstrSet Substitute(ffi::Map<tirx::Var, PrimExpr> subs) const {
    ConstrSet new_set;
    for (const auto &c : constrs_) {
      new_set.constrs_.push_back(c.Substitute(subs));
    }
    return new_set;
  }
  /*!
   * \brief Rename \p from and every bind defined at or after it by appending
   *        \p suffix; binds defined before \p from stay shared.
   *
   * Used when one set is instantiated twice in a single analyzer to model two
   * concurrent executions: two threads in ThreadSync, two logical iterations in
   * VerifyParallelLoop. Sharing a variable private to an execution turns
   * `v == f(a)` and `v == f(b)` into `f(a) == f(b)`, contradicting `a != b` for
   * an injective `f` and leaving a set under which every query holds vacuously.
   *
   * \param subs Map to extend; the new renames accumulate so the caller can
   *        apply the same map to expressions held outside the set.
   * \param from Pivot; renames every bind when not given. Has to be an
   *        `Optional` as `tirx::Var()` builds a real variable named "v".
   * \param rename_ranges Pass false to keep iteration variables shared, which
   *        ThreadSync's loop-carry model needs to compare `i` with `i + 1`.
   */
  ConstrSet RenameFrom(const std::string &suffix,
                       ffi::Map<tirx::Var, PrimExpr> &subs,
                       const ffi::Optional<tirx::Var> &from = std::nullopt,
                       bool rename_ranges = true) const {
    bool active = !from.has_value();
    for (const Constr &c : constrs_) {
      if (c.kind != Constr::kBindValue && c.kind != Constr::kBindRange)
        continue;
      if (from.has_value() && c.var.same_as(from.value()))
        active = true;
      if (!rename_ranges && c.kind == Constr::kBindRange)
        continue;
      if (active && !subs.count(c.var))
        subs.Set(c.var, tirx::Var(c.var->name_hint + suffix, c.var.dtype()));
    }
    return Substitute(subs);
  }

  /*!
   * \brief Union with \p other, dropping duplicates.
   *
   * A variable bound to conflicting values on the two sides is a caller error
   * -- it should have been renamed per side -- and is reported rather than
   * merged. `is_assume` is cleared: an assume is trusted only at the program
   * point that stated it, and being trusted it may reference mutable reads, so
   * it must not become a global fact once two points are merged.
   */
  ConstrSet Merge(const ConstrSet &other) const {
    ConstrSet out = *this;
    std::unordered_map<const tirx::VarNode *, Constr> bound;
    std::unordered_set<PrimExpr, ffi::StructuralHash, tirx::ExprDeepEqual>
        preds;
    for (const Constr &c : out.constrs_) {
      if (c.kind == Constr::kConstr)
        preds.insert(c.value);
      else
        bound.emplace(c.var.get(), c);
    }
    for (const Constr &c : other.constrs_) {
      if (c.kind == Constr::kConstr) {
        if (preds.insert(c.value).second)
          out.constrs_.push_back(c);
        continue;
      }
      auto it = bound.find(c.var.get());
      if (it == bound.end()) {
        bound.emplace(c.var.get(), c);
        out.constrs_.push_back(c);
      } else if (it->second.kind != c.kind ||
                 !tirx::ExprDeepEqual()(it->second.ToGenericConstr(),
                                        c.ToGenericConstr())) {
        LOG(WARNING) << "ConstrSet::Merge: var '" << c.var->name_hint
                     << "' bound to conflicting values across merged sets; "
                        "caller should rename per-side-varying vars. Dropping "
                        "the incoming bind. existing="
                     << it->second.ToGenericConstr()
                     << " incoming=" << c.ToGenericConstr();
      }
    }
    for (Constr &c : out.constrs_) {
      c.is_assume = false;
    }
    return out;
  }

  /*!
   * \brief Lower every bind to a predicate, leaving predicates as they are.
   *
   * Use this when merging into an analyzer that binds one of the shared
   * variables itself: a second `Bind` of it trips the analyzer's re-bind check,
   * while predicates coexist with any external bind.
   */
  ConstrSet ToConstraints() const {
    ConstrSet out;
    out.constrs_.reserve(constrs_.size());
    for (const auto &c : constrs_)
      out.constrs_.push_back(Constr(c.ToGenericConstr(), c.is_assume));
    return out;
  }

  void Populate(arith::Analyzer &analyzer) const {
    // Keep program order: `Analyzer::Bind` evaluates the bounds and modular set
    // of the value at bind time, so entering the binds first would widen them
    // -- a `v = tx` inside `if tx < 64` would lose its upper bound.
    for (const auto &c : constrs_) {
      c.Populate(analyzer);
    }
  }
  bool CanProve(const PrimExpr &expr) const {
    arith::Analyzer analyzer;
    Populate(analyzer);
    return analyzer.CanProve(expr);
  }
  template <typename... Args> void AddConstr(Args... args) {
    constrs_.push_back(Constr(args...));
  }

  /*! \brief Convert the constraint set to a conjunction (AND) of all
   * constraints */
  PrimExpr ToConjunction() const {
    if (constrs_.empty())
      return Bool(true);
    PrimExpr result = constrs_[0].ToGenericConstr();
    for (size_t i = 1; i < constrs_.size(); ++i) {
      result = tirx::And(result, constrs_[i].ToGenericConstr());
    }
    return result;
  }

  void Format(std::ostream &os) const {
    os << "ConstrSet(size=" << constrs_.size() << ") {\n";
    for (size_t i = 0; i < constrs_.size(); ++i) {
      os << "  [" << i << "] ";
      constrs_[i].Format(os);
      os << "\n";
    }
    os << "}";
  }

  std::vector<Constr> constrs_;
};

struct ConstrVisitor : public tirx::StmtExprVisitor {
private:
  using Base = tirx::StmtExprVisitor;

  struct Guard {
    std::vector<Constr> &constrs;
    ~Guard() { constrs.pop_back(); }
  };

protected:
  template <typename... Args> Guard MakeGuard(const Args... args) {
    constr_stack_.push_back(Constr(args...));
    return Guard{constr_stack_};
  }

public:
  using StmtExprVisitor::VisitExpr_;
  using StmtExprVisitor::VisitStmt_;
  void VisitIfThenElseExpr(const PrimExpr cond, const PrimExpr true_value,
                           const PrimExpr false_value) {
    // Visit the condition first without any guard, as it is always evaluated
    // This ensures any buffer accesses in the condition are recorded
    Base::VisitExpr(cond);
    {
      auto guard = MakeGuard(cond);
      Base::VisitExpr(true_value);
    }
    {
      auto guard = MakeGuard(tirx::Not(cond));
      Base::VisitExpr(false_value);
    }
  }
  void VisitStmt_(const tirx::BindNode *op) override { Base::VisitStmt_(op); }
  void VisitStmt_(const tirx::SeqStmtNode *op) override {
    size_t old_size = constr_stack_.size();
    for (const tirx::Stmt &stmt : op->seq) {
      Base::VisitStmt(stmt);
      if (const auto *bind = stmt.as<tirx::BindNode>()) {
        // A flat Bind defines its variable for following statements in this
        // sequence, but not while its own value is being evaluated.
        constr_stack_.emplace_back(bind->var, bind->value);
      } else if (const auto *assert_stmt = stmt.as<tirx::AssertStmtNode>()) {
        // A tirx AssertStmt carries no body, so it sits next to the statements
        // it guards rather than around them; control reaches them only once it
        // has passed. Pure conditions only: an assert states what held when
        // control passed it, not a standing promise like an assume, so a
        // condition reading mutable state may be falsified by a later store,
        // which is not tracked here.
        if (tirx::SideEffect(assert_stmt->condition) <=
            tirx::CallEffectKind::kPure) {
          constr_stack_.emplace_back(assert_stmt->condition);
        }
      }
    }
    constr_stack_.resize(old_size);
  }
  void VisitStmt_(const tirx::AttrStmtNode *op) override {
    if (op->attr_key == tirx::attr::tilelang_assume) {
      auto expr = Downcast<PrimExpr>(op->node);
      auto guard = MakeGuard(expr, true);
      Base::VisitStmt_(op);
    } else if (op->attr_key == tirx::attr::thread_extent ||
               op->attr_key == s_tir::attr::virtual_thread) {
      tirx::IterVar iv = Downcast<tirx::IterVar>(op->node);
      Range dom =
          Range::FromMinExtent(tirx::make_zero(op->value.dtype()), op->value);
      auto guard = MakeGuard(iv->var, dom);
      Base::VisitStmt_(op);
    } else {
      Base::VisitStmt_(op);
    }
  }
  void VisitStmt_(const tirx::IfThenElseNode *op) override {
    {
      auto guard = MakeGuard(op->condition);
      Base::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      auto guard = MakeGuard(tirx::Not(op->condition));
      Base::VisitStmt(op->else_case.value());
    }
  }
  void VisitExpr_(const tirx::SelectNode *op) override {
    VisitIfThenElseExpr(op->condition, op->true_value, op->false_value);
  }
  void VisitExpr_(const tirx::CallNode *op) override {
    static auto op_if_then_else = Op::Get("tirx.if_then_else");
    if (op->op.same_as(op_if_then_else)) {
      VisitIfThenElseExpr(op->args[0], op->args[1], op->args[2]);
    } else {
      Base::VisitExpr_(op);
    }
  }
  void VisitStmt_(const tirx::ForNode *op) override {
    if (op->kind == tirx::ForKind::kParallel ||
        op->kind == tirx::ForKind::kVectorized) {
      auto guard_1 =
          MakeGuard(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      auto guard_2 = MakeGuard(op->extent > 0);
      Base::VisitStmt_(op);
    } else {
      auto guard_1 =
          MakeGuard(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      auto guard_2 = MakeGuard(op->extent > 0);
      Base::VisitStmt_(op);
    }
  }
  void VisitStmt_(const tirx::WhileNode *op) override {
    {
      auto guard = MakeGuard(op->condition);
      Base::VisitStmt(op->body);
    }
  }
  ConstrSet GetConstrSet() const {
    return ConstrSet{.constrs_ = constr_stack_};
  }
  std::vector<Constr> constr_stack_;
};
} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
