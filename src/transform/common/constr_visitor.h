#ifndef TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
#define TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_

#include "compat/expr.h"
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/with_context.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {

class FreshenMutableReads {
public:
  enum class Mode { kPerOccurrence, kSnapshot };

private:
  class Rewriter : public tirx::StmtExprMutator {
  public:
    explicit Rewriter(Mode mode) : mode_(mode) {}

  private:
    PrimExpr Fresh(const PrimExpr &expression) {
      if (mode_ == Mode::kSnapshot) {
        if (auto found = memo_.find(expression); found != memo_.end()) {
          return found->second;
        }
      }
      PrimVar fresh("free" + std::to_string(count_++), expression.ty());
      if (mode_ == Mode::kSnapshot) {
        memo_.emplace(expression, fresh);
      }
      return fresh;
    }

    ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                       ffi::InplaceMode) final {
      return Fresh(ffi::GetRef<PrimExpr>(node));
    }

    ffi::UnchangedOr<PrimExpr> Mutate_(const LetNode *node,
                                       ffi::InplaceMode mode) final {
      PrimExpr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
      PrimExpr body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
      PrimVar previous = node->var.as_or_throw<PrimVar>();
      PrimVar variable(previous->name, previous.ty(), previous->loc);
      return Let(variable, value, tl::Substitute(body, {{previous, variable}}),
                 node->loc);
    }

    ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                   ffi::InplaceMode mode) final {
      Expr expression = ffi::GetRef<Expr>(node);
      if (auto scalar = expression.as<PrimExpr>();
          scalar && SideEffect(expression) > CallEffectKind::kPure) {
        return Fresh(scalar.value());
      }
      return StmtExprMutator::Mutate_(node, mode);
    }

    Mode mode_;
    int count_{0};
    std::unordered_map<PrimExpr, PrimExpr, ffi::StructuralHash,
                       ffi::StructuralEqual>
        memo_;
  };

public:
  explicit FreshenMutableReads(Mode mode = Mode::kPerOccurrence)
      : rewriter_(ffi::make_object<Rewriter>(mode)) {}

  PrimExpr operator()(const PrimExpr &expression) {
    return rewriter_->Mutate(expression, ffi::InplaceMode::kDisallow)
        .ValueOrUnchanged(expression);
  }

private:
  ffi::ObjectPtr<Rewriter> rewriter_;
};

struct Constr {

  enum Kind {
    kConstr,
    kBindValue,
    kBindRange,
  } kind;
  bool is_assume = false;
  PrimVar var{ffi::UnsafeInit{}};
  PrimExpr value = IntImm::Bool(true);
  Range range{ffi::UnsafeInit{}};

  Constr(PrimExpr constr, bool is_assume = false)
      : kind(kConstr), value(constr), is_assume(is_assume) {};
  Constr(PrimVar var, PrimExpr val) : kind(kBindValue), var(var), value(val) {};
  Constr(PrimVar var, Range range)
      : kind(kBindRange), var(var), range(range) {};

  Constr() : kind(kConstr) {}
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
      os << ", var=" << var->name;
      os << ", value=" << value;
      break;
    case kBindRange:
      os << "kBindRange";
      os << ", var=" << var->name;
      os << ", range=Range(min=" << range->begin;
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
      if (!var.ty().IsScalar())
        return IntImm::Bool(true);
      return var == value;
    case kBindRange:
      return prim::And(var >= range->begin,
                       var < (range->begin + range->extent));
    }
    TVM_FFI_THROW(InternalError) << "Unreachable";
    return IntImm::Bool(true);
  }
  /*!
   * \brief Rewrite through \p subs, keeping the kind whenever possible, so that
   *        renaming one side of a two-instance comparison keeps its binds.
   */
  Constr Substitute(ffi::Map<PrimVar, PrimExpr> subs) const {
    switch (kind) {
    case kConstr:
      return Constr(tl::Substitute(value, subs), is_assume);
    case kBindValue:
    case kBindRange: {
      auto it = subs.find(var);
      const auto *new_var =
          it == subs.end() ? var.get() : (*it).second.as<VarNode>();
      // A bind remapped onto a non-variable is only expressible as a predicate.
      if (new_var == nullptr)
        return Constr(tl::Substitute(ToGenericConstr(), subs), is_assume);
      if (kind == kBindValue)
        return Constr(ffi::GetRef<Var>(new_var).as_or_throw<PrimVar>(),
                      tl::Substitute(value, subs));
      return Constr(ffi::GetRef<Var>(new_var).as_or_throw<PrimVar>(),
                    Range::FromMinExtent(tl::Substitute(range->begin, subs),
                                         tl::Substitute(range->extent, subs)));
    }
    }
    TVM_FFI_THROW(InternalError) << "Unreachable";
    return Constr();
  }
  /*!
   * \brief A copy whose definition no longer reads mutable state.
   *
   * Only a bind needs it: `Bind` installs a rewrite, so `v == A[i]` would keep
   * being substituted after a store invalidated it. A predicate states a fact
   * that held on entry and stays usable as a premise inside the scope it
   * guards.
   *
   * Every consumer goes through here, so no path can miss the substitution.
   */
  Constr FreshenReads() const {
    FreshenMutableReads freshen;
    switch (kind) {
    case kConstr:
      return *this;
    case kBindValue:
      return Constr(var, freshen(value));
    case kBindRange:
      return Constr(var, Range::FromMinExtent(freshen(range->begin),
                                              freshen(range->extent)));
    }
    TVM_FFI_THROW(InternalError) << "Unreachable";
    return Constr();
  }
  void Populate(sym::Analyzer &analyzer,
                WithGroup<sym::ConstraintContext> &contexts) const {
    Constr c = FreshenReads();
    switch (c.kind) {
    case kConstr:
      // Simplify against the binds already installed, so a predicate held
      // in a bound boolean var (`cond` where `cond = (i < n)` is a bind)
      // expands into the condition it represents before the analyzer
      // digests it. EnterConstraint evaluates eagerly, exactly like Bind:
      // an opaque boolean entered as-is contributes nothing.
      contexts.Emplace(analyzer, analyzer->Simplify(c.value), c.is_assume);
      break;
    case kBindValue:
      analyzer->Bind(c.var, c.value);
      break;
    case kBindRange:
      analyzer->Bind(c.var, c.range);
      break;
    default:
      TVM_FFI_THROW(InternalError) << "Unreachable";
    }
  }
};

struct ConstrSet {
  ConstrSet Substitute(ffi::Map<PrimVar, PrimExpr> subs) const {
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
   * \p subs accumulates the renames, so the caller can apply the same map to
   * expressions held outside the set. \p from has to be an `Optional`, as
   * `PrimVar()` builds a real variable named "v". Pass \p rename_ranges false
   * to keep iteration variables shared, as ThreadSync's loop-carry model needs.
   */
  ConstrSet RenameFrom(const std::string &suffix,
                       ffi::Map<PrimVar, PrimExpr> &subs,
                       const ffi::Optional<PrimVar> &from = std::nullopt,
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
        subs.Set(c.var, PrimVar(c.var->name + suffix, c.var.ty()));
    }
    return Substitute(subs);
  }

  /*!
   * \brief Union with \p other, dropping duplicates.
   *
   * A variable bound to conflicting values on the two sides is a caller error
   * -- it should have been renamed per side -- and is reported rather than
   * merged. `is_assume` is cleared: an assume is trusted only where it was
   * stated, and being trusted it may reference mutable reads.
   */
  ConstrSet Merge(const ConstrSet &other) const {
    ConstrSet out = *this;
    std::unordered_map<const VarNode *, Constr> bound;
    std::unordered_set<PrimExpr, ffi::StructuralHash, ffi::StructuralEqual>
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
                 !ffi::StructuralEqual()(it->second.ToGenericConstr(),
                                         c.ToGenericConstr())) {
        LOG(WARNING) << "ConstrSet::Merge: var '" << c.var->name
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
    return out.Normalized();
  }

  /*!
   * \brief Restore the bind-before-use invariant after concatenation.
   *
   * Lexical collection emits a bind before every entry that reads its
   * var, so a single set never needs this. `Merge` concatenates two such
   * sets, and an entry of the first set may read a bind var the second
   * set defines: replaying that order digests the predicate before the
   * definition exists, and the fact is lost -- `EnterConstraint`
   * evaluates eagerly, exactly like `Bind` (see the ordering discussion
   * in issue #3220).
   *
   * Entries stay in their current order, except that a bind is hoisted
   * (with the binds its own definition reads, recursively) directly
   * before the first entry that reads its var. Within one
   * lexically-collected set no entry reads a later bind, so this is a
   * no-op there and the program-order guarantee of #2805 stays intact.
   * A hoisted bind can still miss bounds that a first-set predicate
   * would have tightened; that residue is inherent to eager `Bind` and
   * only a lazily-evaluated bind (issue #3220, option 3) removes it.
   */
  ConstrSet Normalized() const {
    std::unordered_map<const VarNode *, size_t> bind_at;
    for (size_t i = 0; i < constrs_.size(); ++i) {
      const Constr &c = constrs_[i];
      if ((c.kind == Constr::kBindValue || c.kind == Constr::kBindRange) &&
          !bind_at.count(c.var.get())) {
        bind_at.emplace(c.var.get(), i);
      }
    }
    enum : uint8_t { kNew = 0, kVisiting = 1, kEmitted = 2 };
    std::vector<uint8_t> state(constrs_.size(), kNew);
    ConstrSet out;
    out.constrs_.reserve(constrs_.size());
    std::function<void(size_t)> emit = [&](size_t index) {
      if (state[index] == kEmitted) {
        return;
      }
      if (state[index] == kVisiting) {
        // SSA-form collection cannot produce a definition cycle; if one
        // arrives anyway, keep the current relative order instead of
        // crashing the compile.
        LOG(WARNING) << "ConstrSet::Normalized: cyclic bind definitions; "
                        "keeping collection order";
        return;
      }
      state[index] = kVisiting;
      for (const Var &var : ReadVars(constrs_[index])) {
        auto it = bind_at.find(var.get());
        if (it != bind_at.end() && it->second != index) {
          emit(it->second);
        }
      }
      state[index] = kEmitted;
      out.constrs_.push_back(constrs_[index]);
    };
    for (size_t i = 0; i < constrs_.size(); ++i) {
      emit(i);
    }
    return out;
  }

  /*!
   * \brief Lower every bind to a predicate, leaving predicates as they are.
   *
   * Use this when the analyzer binds one of the shared variables itself: a
   * second `Bind` of it trips the re-bind check, while predicates coexist with
   * any bind.
   */
  ConstrSet ToConstraints() const {
    ConstrSet out;
    out.constrs_.reserve(constrs_.size());
    for (const Constr &c : constrs_)
      out.constrs_.push_back(
          Constr(c.FreshenReads().ToGenericConstr(), c.is_assume));
    return out;
  }

  [[nodiscard]] WithGroup<sym::ConstraintContext>
  Populate(sym::Analyzer &analyzer) const {
    WithGroup<sym::ConstraintContext> contexts;
    // Keep program order: `Analyzer::Bind` evaluates the bounds and modular set
    // of the value at bind time, so entering the binds first would widen them
    // -- a `v = tx` inside `if tx < 64` would lose its upper bound. Merged
    // sets are additionally normalized (see `Normalized`) so a bind always
    // precedes the entries that read its var.
    for (const auto &c : constrs_) {
      c.Populate(analyzer, contexts);
    }
    return contexts;
  }

  /*! \brief Free vars an entry's definition or predicate reads (the bind's
   *  own var excluded by construction: it does not occur in its value). */
  static ffi::Array<Var> ReadVars(const Constr &c) {
    switch (c.kind) {
    case Constr::kConstr:
    case Constr::kBindValue:
      return tvm::UndefinedVars(c.value);
    case Constr::kBindRange: {
      ffi::Array<Var> vars = tvm::UndefinedVars(c.range->begin);
      for (const Var &var : tvm::UndefinedVars(c.range->extent)) {
        vars.push_back(var);
      }
      return vars;
    }
    }
    TVM_FFI_THROW(InternalError) << "Unreachable";
    return {};
  }
  bool CanProve(const PrimExpr &expr) const {
    sym::Analyzer analyzer;
    auto constraints = Populate(analyzer);
    return analyzer->CanProve(expr);
  }
  template <typename... Args> void AddConstr(Args... args) {
    constrs_.push_back(Constr(args...));
  }

  /*! \brief Convert the constraint set to a conjunction (AND) of all
   * constraints */
  PrimExpr ToConjunction() const {
    if (constrs_.empty())
      return IntImm::Bool(true);
    PrimExpr result = constrs_[0].ToGenericConstr();
    for (size_t i = 1; i < constrs_.size(); ++i) {
      result = prim::And(result, constrs_[i].ToGenericConstr());
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
  using Result = ffi::Optional<ffi::VisitInterrupt>;

  struct Guard {
    std::vector<Constr> &constraints;
    size_t size;
    ~Guard() { constraints.resize(size); }
  };

protected:
  template <typename... Args> Guard MakeGuard(const Args &...args) {
    Guard guard{constr_stack_, constr_stack_.size()};
    constr_stack_.emplace_back(args...);
    return guard;
  }

public:
  using Base::Visit_;

  Result VisitConditional(const PrimExpr &condition, const Expr &true_value,
                          const Expr &false_value) {
    if (auto interrupt = Base::Visit(condition)) {
      return interrupt;
    }
    {
      auto guard = MakeGuard(condition);
      if (auto interrupt = Base::Visit(true_value)) {
        return interrupt;
      }
    }
    auto guard = MakeGuard(!condition);
    return Base::Visit(false_value);
  }

  Result Visit_(const SeqStmtNode *node) override {
    Guard guard{constr_stack_, constr_stack_.size()};
    for (const Stmt &statement : node->seq) {
      if (auto interrupt = Base::Visit(statement)) {
        return interrupt;
      }
      if (const auto *bind = statement.as<BindNode>()) {
        auto variable = bind->var.as<PrimVar>();
        auto value = bind->value.as<PrimExpr>();
        if (variable && value) {
          constr_stack_.emplace_back(variable.value(), value.value());
        }
      } else if (const auto *assertion = statement.as<AssertStmtNode>()) {
        if (SideEffect(assertion->condition) == CallEffectKind::kPure) {
          constr_stack_.emplace_back(assertion->condition);
        }
      } else if (const auto *evaluate = statement.as<EvaluateNode>()) {
        if (const auto *call = evaluate->value.as<CallNode>();
            call && call->op.same_as(prim::assume_op())) {
          constr_stack_.emplace_back(call->args[0].as_or_throw<PrimExpr>(),
                                     true);
        }
      }
    }
    return std::nullopt;
  }

  Result Visit_(const RegionStmtNode *node) override {
    if (node->op.same_as(tirx::launch_thread_op())) {
      for (const Expr &argument : node->args) {
        if (auto interrupt = Base::Visit(argument)) {
          return interrupt;
        }
      }
      auto variable = node->body_params[0].as_or_throw<PrimVar>();
      auto extent = node->args[1].as_or_throw<PrimExpr>();
      auto guard = MakeGuard(
          variable, Range::FromMinExtent(MakeConst(extent.ty(), 0), extent));
      return Base::Visit(node->body);
    }
    for (const char *key :
         {"tilelang_assume", "tl.assume_requires_runtime_check"}) {
      if (auto condition = node->attrs.GetAttr<PrimExpr>(key)) {
        auto guard = MakeGuard(condition.value(), true);
        return Base::Visit_(node);
      }
    }
    return Base::Visit_(node);
  }

  Result Visit_(const IfNode *node) override {
    if (auto interrupt = Base::Visit(node->condition)) {
      return interrupt;
    }
    {
      auto guard = MakeGuard(node->condition);
      if (auto interrupt = Base::Visit(node->then_case)) {
        return interrupt;
      }
    }
    if (node->else_case) {
      auto guard = MakeGuard(!node->condition);
      return Base::Visit(node->else_case.value());
    }
    return std::nullopt;
  }

  Result Visit_(const CallNode *node) override {
    if (node->op.same_as(prim::if_then_else_op())) {
      return VisitConditional(node->args[0].as_or_throw<PrimExpr>(),
                              node->args[1], node->args[2]);
    }
    return Base::Visit_(node);
  }

  Result Visit_(const ForNode *node) override {
    if (auto interrupt = Base::Visit(node->begin)) {
      return interrupt;
    }
    if (auto interrupt = Base::Visit(node->end)) {
      return interrupt;
    }
    if (auto interrupt = Base::Visit(node->step)) {
      return interrupt;
    }
    sym::Analyzer analyzer;
    auto constraints = GetConstrSet().Populate(analyzer);
    if (analyzer->CanProve(node->step > 0)) {
      auto range_guard = MakeGuard(node->var, Range(node->begin, node->end));
      auto extent_guard = MakeGuard(node->begin < node->end);
      return Base::Visit(node->body);
    }
    if (analyzer->CanProve(node->step < 0)) {
      auto range_guard =
          MakeGuard(node->var > node->end && node->var <= node->begin);
      auto extent_guard = MakeGuard(node->begin > node->end);
      return Base::Visit(node->body);
    }
    auto condition_guard = MakeGuard(
        (node->step > 0 && node->var >= node->begin && node->var < node->end) ||
        (node->step < 0 && node->var <= node->begin && node->var > node->end));
    return Base::Visit(node->body);
  }

  Result Visit_(const WhileNode *node) override {
    if (auto interrupt = Base::Visit(node->condition)) {
      return interrupt;
    }
    auto guard = MakeGuard(node->condition);
    return Base::Visit(node->body);
  }

  ConstrSet GetConstrSet() const {
    return ConstrSet{.constrs_ = constr_stack_};
  }
  std::vector<Constr> constr_stack_;
};
} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
