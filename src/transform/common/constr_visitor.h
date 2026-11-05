#ifndef TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
#define TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_

#include "ir/analyzer.h"
#include "ir/annotations.h"
#include "ir/loop_math.h"
#include "support/check.h"
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
using namespace prim;

/*! \brief Rewrite primitive uses without changing their native type contract.
 */
inline PrimExpr SubstituteConstraintExpr(const PrimExpr &value,
                                         const ffi::Map<Var, PrimExpr> &subs) {
  auto mutator = ffi::make_object<tvm::ExprMutator>();
  for (const auto &[var, replacement] : subs)
    mutator->VarRemapSet(var, replacement);
  return mutator->Mutate(value).ValueOrUnchanged(value);
}

/*! \brief Own the native lexical contexts used by one proof.
 * Keep the returned object alive through all queries. Contexts are popped in
 * reverse order. Bindings still belong to the receiving analyzer, so callers
 * populate a fresh analyzer rather than sharing one with unrelated proofs.
 */
class ConstrAnalyzerContext {
public:
  void Enter(const sym::Analyzer &analyzer, const PrimExpr &condition,
             bool is_assume) {
    contexts_.push_back(std::make_unique<ScopedAnalyzerConstraints>(
        analyzer, condition, is_assume));
  }
  ~ConstrAnalyzerContext() {
    while (!contexts_.empty())
      contexts_.pop_back();
  }

private:
  std::vector<std::unique_ptr<ScopedAnalyzerConstraints>> contexts_;
};

/*! \brief Replace each evaluation of mutable state with a fresh primitive.
 * Analyzer bindings outlive stores, so binding a variable to a TensorLoad
 * must capture an unknown value, never install a rewrite to a live memory
 * read. Snapshot mode shares structurally equal reads only within this
 * explicit mutator instance; different access points use different instances.
 */
class FreshenMutableReads : public tvm::ExprMutator {
public:
  using ExprMutator::Mutate;
  using ExprMutator::Mutate_;
  enum class Mode { kPerOccurrence, kSnapshot };
  explicit FreshenMutableReads(Mode mode = Mode::kPerOccurrence)
      : mode_(mode) {}

  PrimExpr operator()(const PrimExpr &value) {
    return Mutate(value).ValueOrUnchanged(value);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                     InplaceMode) override {
    return Fresh(ffi::GetRef<PrimExpr>(op));
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) override {
    auto expression = ffi::GetRef<Expr>(op);
    if (auto primitive = expression.as<PrimExpr>();
        primitive.has_value() && SideEffect(expression) > CallEffectKind::kPure)
      return Fresh(primitive.value());
    return ExprMutator::Mutate_(op, mode);
  }

private:
  PrimExpr Fresh(const PrimExpr &expr) {
    if (mode_ == Mode::kSnapshot) {
      auto it = memo_.find(expr);
      if (it != memo_.end())
        return it->second;
    }
    PrimExpr fresh = PrimVar("free" + std::to_string(count_++), expr.ty());
    if (mode_ == Mode::kSnapshot)
      memo_.emplace(expr, fresh);
    return fresh;
  }
  Mode mode_;
  int count_{0};
  std::unordered_map<PrimExpr, PrimExpr, ffi::StructuralHash,
                     prim::ExprDeepEqual>
      memo_;
};

struct Constr {

  enum Kind {
    kConstr,
    kBindValue,
    kBindRange,
  } kind;
  bool is_assume = false;
  Var var{ffi::UnsafeInit{}};
  PrimExpr value{ffi::UnsafeInit{}};
  Range range{ffi::UnsafeInit{}};

  Constr(PrimExpr constr, bool is_assume = false)
      : kind(kConstr), value(constr), is_assume(is_assume) {};
  Constr(Var var, PrimExpr val) : kind(kBindValue), var(var), value(val) {};
  Constr(Var var, Range range) : kind(kBindRange), var(var), range(range) {};

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
      if (!var->ty.as_or_throw<PrimType>().IsScalar())
        return IntImm(PrimType::Bool(), 1);
      return var.as_or_throw<PrimExpr>() == value;
    case kBindRange:
      return prim::And(var.as_or_throw<PrimExpr>() >= range->begin,
                       var.as_or_throw<PrimExpr>() <
                           (range->begin + range->extent));
    }
    LOG(FATAL) << "Unreachable";
    return IntImm(PrimType::Bool(), 1);
  }
  /*!
   * \brief Rewrite through \p subs, keeping the kind whenever possible, so that
   *        renaming one side of a two-instance comparison keeps its binds.
   */
  Constr Substitute(ffi::Map<Var, PrimExpr> subs) const {
    switch (kind) {
    case kConstr:
      return Constr(SubstituteConstraintExpr(value, subs), is_assume);
    case kBindValue:
    case kBindRange: {
      auto it = subs.find(var);
      const auto *new_var =
          it == subs.end() ? var.get() : (*it).second.as<VarObj>();
      // A bind remapped onto a non-variable is only expressible as a predicate.
      if (new_var == nullptr)
        return Constr(SubstituteConstraintExpr(ToGenericConstr(), subs),
                      is_assume);
      if (kind == kBindValue)
        return Constr(ffi::GetRef<Var>(new_var),
                      SubstituteConstraintExpr(value, subs));
      return Constr(
          ffi::GetRef<Var>(new_var),
          Range::FromMinExtent(SubstituteConstraintExpr(range->begin, subs),
                               SubstituteConstraintExpr(range->extent, subs)));
    }
    }
    LOG(FATAL) << "Unreachable";
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
    auto freshen = ffi::make_object<FreshenMutableReads>();
    switch (kind) {
    case kConstr:
      return *this;
    case kBindValue:
      return Constr(var, (*freshen)(value));
    case kBindRange:
      return Constr(var, Range::FromMinExtent((*freshen)(range->begin),
                                              (*freshen)(range->extent)));
    }
    LOG(FATAL) << "Unreachable";
    return Constr();
  }
  void Populate(const sym::Analyzer &analyzer,
                ConstrAnalyzerContext *context) const {
    Constr c = FreshenReads();
    switch (c.kind) {
    case kConstr:
      // Simplify against the binds already installed, so a predicate held
      // in a bound boolean var (`cond` where `cond = (i < n)` is a bind)
      // expands into the condition it represents before the analyzer
      // digests it. EnterConstraint evaluates eagerly, exactly like Bind:
      // an opaque boolean entered as-is contributes nothing.
      context->Enter(analyzer, analyzer->Simplify(c.value), c.is_assume);
      break;
    case kBindValue:
      // A self bind records a definition for RenameFrom without installing a
      // cyclic rewrite or sharing an auxiliary unknown across loop instances.
      if (!c.var.same_as(c.value))
        BindAnalyzer(analyzer, c.var, c.value);
      break;
    case kBindRange:
      BindAnalyzer(analyzer, c.var, c.range);
      break;
    default:
      LOG(FATAL) << "Unreachable";
    }
  }
};

struct ConstrSet {
  ConstrSet Substitute(ffi::Map<Var, PrimExpr> subs) const {
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
   * a missing marker is distinct from a real variable. Pass \p rename_ranges
   * false to keep iteration variables shared, as ThreadSync's loop-carry model
   * needs.
   */
  ConstrSet RenameFrom(const std::string &suffix, ffi::Map<Var, PrimExpr> &subs,
                       const ffi::Optional<Var> &from = std::nullopt,
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
        subs.Set(c.var, PrimVar(c.var->name + suffix, c.var->ty));
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
    std::unordered_map<const VarObj *, Constr> bound;
    std::unordered_set<PrimExpr, ffi::StructuralHash, prim::ExprDeepEqual>
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
                 !prim::ExprDeepEqual()(it->second.ToGenericConstr(),
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
    std::unordered_map<const VarObj *, size_t> bind_at;
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

  [[nodiscard]] std::unique_ptr<ConstrAnalyzerContext>
  Populate(const sym::Analyzer &analyzer) const {
    auto context = std::make_unique<ConstrAnalyzerContext>();
    // Preserve source order and bind-before-use after Merge normalization.
    for (const auto &entry : constrs_)
      entry.Populate(analyzer, context.get());
    return context;
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
    LOG(FATAL) << "Unreachable";
    return {};
  }
  bool CanProve(const PrimExpr &expr) const {
    auto analyzer = CreateTileLangAnalyzer();
    auto context = Populate(analyzer);
    return analyzer->CanProve(expr);
  }
  template <typename... Args> void AddConstr(Args... args) {
    constrs_.push_back(Constr(args...));
  }

  /*! \brief Convert the constraint set to a conjunction (AND) of all
   * constraints */
  PrimExpr ToConjunction() const {
    if (constrs_.empty())
      return IntImm(PrimType::Bool(), 1);
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

/*! \brief Collect premises in source order using native IR definition sites.
 * Bind results are available to subsequent statements of the same SeqStmt.
 * Arguments of regions/loops are evaluated outside their new definitions;
 * guards and body definitions never leak into siblings after the scope ends.
 */
struct ConstrVisitor : public tirx::StmtExprVisitor {
private:
  using Base = tirx::StmtExprVisitor;
  struct Guard {
    std::vector<Constr> &constrs;
    size_t old_size;
    ~Guard() { constrs.resize(old_size); }
  };

protected:
  template <typename... Args> Guard MakeGuard(const Args &...args) {
    size_t old_size = constr_stack_.size();
    constr_stack_.emplace_back(args...);
    return Guard{constr_stack_, old_size};
  }
  Guard Scope() { return Guard{constr_stack_, constr_stack_.size()}; }

public:
  using Base::Visit;
  using Base::Visit_;
  using Result = ffi::Optional<ffi::VisitInterrupt>;

  Result VisitIfThenElseExpr(const PrimExpr &cond, const PrimExpr &true_value,
                             const PrimExpr &false_value) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(cond));
    {
      auto guard = MakeGuard(cond);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(true_value));
    }
    {
      auto guard = MakeGuard(prim::Not(cond));
      return Visit(false_value);
    }
  }

  Result Visit_(const SeqStmtObj *op) override {
    auto scope = Scope();
    for (const Stmt &stmt : op->seq) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(stmt));
      if (const auto *bind = stmt.as<BindObj>()) {
        if (auto value = bind->value.as<PrimExpr>())
          constr_stack_.emplace_back(bind->var, value.value());
      } else if (const auto *assertion = stmt.as<AssertStmtObj>()) {
        if (SideEffect(assertion->condition) <= CallEffectKind::kPure)
          constr_stack_.emplace_back(assertion->condition);
      } else if (const auto *evaluate = stmt.as<EvaluateObj>()) {
        const auto *call = evaluate->value.as<CallObj>();
        if (call && call->op.same_as(prim::assume_op()))
          constr_stack_.emplace_back(call->args[0].as_or_throw<PrimExpr>(),
                                     true);
      }
    }
    return std::nullopt;
  }

  Result Visit_(const RegionStmtObj *op) override {
    // Dynamic metadata and launch bounds are evaluated before body_params.
    for (const Expr &arg : op->args)
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(arg));
    auto scope = Scope();
    if (op->op.same_as(tirx::launch_thread_op())) {
      PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
      constr_stack_.emplace_back(
          op->body_params[0],
          Range::FromMinExtent(IntImm(extent.ty(), 0), extent));
    } else if (op->op->name == "tl.launch_kernel") {
      const auto begins = op->args[0].as_or_throw<tvm::Tuple>()->fields;
      const auto ends = op->args[1].as_or_throw<tvm::Tuple>()->fields;
      const auto threads = op->args[3].as_or_throw<tvm::Tuple>()->fields;
      for (size_t i = 0; i < ends.size(); ++i)
        constr_stack_.emplace_back(op->body_params[i],
                                   Range(begins[i].as_or_throw<PrimExpr>(),
                                         ends[i].as_or_throw<PrimExpr>()));
      for (size_t i = 0; i < threads.size(); ++i) {
        PrimExpr extent = threads[i].as_or_throw<PrimExpr>();
        constr_stack_.emplace_back(
            op->body_params[ends.size() + i],
            Range::FromMinExtent(IntImm(extent.ty(), 0), extent));
      }
    } else if (op->op->name == "tl.attr") {
      auto values =
          DecodeAnnotations(
              op->args[0].as_or_throw<tvm::Tuple>(),
              op->attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema")
                  .value())
              .cast<ffi::Map<ffi::Any, ffi::Any>>();
      if (values.at("key").cast<ffi::String>() == "tl.assume")
        constr_stack_.emplace_back(values.at("node").cast<PrimExpr>(), true);
    }
    return Visit(op->body);
  }

  Result Visit_(const IfObj *op) override {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->condition));
    {
      auto guard = MakeGuard(op->condition);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->then_case));
    }
    if (op->else_case) {
      auto guard = MakeGuard(prim::Not(op->condition));
      return Visit(op->else_case.value());
    }
    return std::nullopt;
  }

  Result Visit_(const prim::SelectObj *op) override {
    return VisitIfThenElseExpr(op->condition, op->true_value, op->false_value);
  }

  Result Visit_(const CallObj *op) override {
    if (op->op.same_as(prim::if_then_else_op()))
      return VisitIfThenElseExpr(op->args[0].as_or_throw<PrimExpr>(),
                                 op->args[1].as_or_throw<PrimExpr>(),
                                 op->args[2].as_or_throw<PrimExpr>());
    if (auto callee = op->op.as<Op>();
        callee.has_value() && callee.value()->name == "tl.annotation")
      return std::nullopt;
    return Base::Visit_(op);
  }

  Result Visit_(const ForObj *op) override {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->begin));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->end));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->step));
    auto scope = Scope();
    auto analyzer = CreateTileLangAnalyzer();
    auto context = GetConstrSet().Populate(analyzer);
    if (CanUseLoopBounds(op->begin, op->end, op->step, analyzer.get())) {
      constr_stack_.emplace_back(op->var, Range(op->begin, op->end));
      constr_stack_.emplace_back(op->end > op->begin);
    } else {
      // Retain a definition for per-iteration renaming without asserting an
      // incorrect lower bound for negative, zero or dynamically signed steps.
      // Native For always tests var < end; it is not Python's range.
      constr_stack_.emplace_back(op->var, op->var.as_or_throw<PrimExpr>());
      if (SideEffect(op->end) <= CallEffectKind::kPure)
        constr_stack_.emplace_back(op->var < op->end);
      if (SideEffect(op->begin) <= CallEffectKind::kPure &&
          SideEffect(op->end) <= CallEffectKind::kPure)
        constr_stack_.emplace_back(op->begin < op->end);
    }
    return Visit(op->body);
  }

  Result Visit_(const WhileObj *op) override {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->condition));
    auto guard = MakeGuard(op->condition);
    return Visit(op->body);
  }

  ConstrSet GetConstrSet() const {
    return ConstrSet{.constrs_ = constr_stack_};
  }
  std::vector<Constr> constr_stack_;
};
} // namespace tvm::tl
#endif // TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
