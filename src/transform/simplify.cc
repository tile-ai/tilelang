/*!
 * \file simplify.cc
 * \brief TileLang simplification policy over native statements and types.
 */
#include "../ir/analyzer.h"
#include "ir/annotations.h"
#include "tensor_descriptor.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>

#include "ir/stmt_functor.h"
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_set>

namespace tvm::tl {

struct SimplifyConfigNode : public ffi::Object {
  bool transitively_prove_inequalities{false};
  bool propagate_knowns_to_prove_conditional{false};
  bool propagate_knowns_to_simplify_expressions{false};
  bool convert_boolean_to_and_of_ors{false};
  bool apply_constraints_to_boolean_branches{false};
  bool enable_simplify_let_inline{true};

  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<SimplifyConfigNode>()
        .def_ro("transitively_prove_inequalities",
                &SimplifyConfigNode::transitively_prove_inequalities,
                "Use scoped native proofs for transitive boolean conditions",
                refl::DefaultValue(false))
        .def_ro("propagate_knowns_to_prove_conditional",
                &SimplifyConfigNode::propagate_knowns_to_prove_conditional,
                "Use known tensor values when simplifying conditions",
                refl::DefaultValue(false))
        .def_ro("propagate_knowns_to_simplify_expressions",
                &SimplifyConfigNode::propagate_knowns_to_simplify_expressions,
                "Replace loads with exact known tensor values",
                refl::DefaultValue(false))
        .def_ro("convert_boolean_to_and_of_ors",
                &SimplifyConfigNode::convert_boolean_to_and_of_ors,
                "Distribute pure scalar boolean expressions into AND of ORs",
                refl::DefaultValue(false))
        .def_ro("apply_constraints_to_boolean_branches",
                &SimplifyConfigNode::apply_constraints_to_boolean_branches,
                "Simplify pure AND/OR operands under the other operand's "
                "constraint",
                refl::DefaultValue(false))
        .def_ro("enable_simplify_let_inline",
                &SimplifyConfigNode::enable_simplify_let_inline,
                "Inline pure native Bind values", refl::DefaultValue(true));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.transform.SimplifyConfig",
                                    SimplifyConfigNode, ffi::Object);
};

class SimplifyConfig : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SimplifyConfig, ffi::ObjectRef,
                                                SimplifyConfigNode);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  SimplifyConfigNode::RegisterReflection();
  tvm::transform::PassContext::RegisterConfigOption<SimplifyConfig>(
      "tl.Simplify");
}

namespace {
using namespace tvm::prim;

bool PureScalarBoolean(const PrimExpr &value) {
  return value.ty().IsScalar() && value.ty().MatchesCode(kDLBool) &&
         SideEffect(value) <= CallEffectKind::kPure;
}

// Track uses through the structural protocol, including types and attribute
// payloads that ordinary expression visitors intentionally skip. A definition
// is not a use, but variables inside a simple definition's type are uses.
struct BindingUsage {
  std::unordered_set<const VarObj *> used;
  bool has_volatile_memory{false};
};

BindingUsage CollectBindingUsage(const tirx::Function &function) {
  BindingUsage result;
  ffi::StructuralVisit(
      function,
      [&](const Var &var, ffi::StructuralVisitorObj *visitor)
          -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        if (visitor->def_region_kind() == kTVMFFIDefRegionKindNone)
          result.used.insert(var.get());
        return visitor->DefaultVisitExpected(var);
      },
      [&](const tirx::Function &current, ffi::StructuralVisitorObj *visitor)
          -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
            visitor->VisitExpected(current->attrs));
        return visitor->DefaultVisitExpected(current);
      },
      [&](const Call &call, ffi::StructuralVisitorObj *visitor)
          -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        if (call->op.same_as(layout_map_op()))
          return VisitLayoutMapCall(call.get(), visitor);
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(call->attrs));
        return visitor->DefaultVisitExpected(call);
      },
      [&](const For &loop, ffi::StructuralVisitorObj *visitor)
          -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        // The generic protocol leaves loop metadata to its owning passes.
        // DCE must nevertheless retain definitions referenced by that metadata,
        // including variables nested inside arrays and attribute dictionaries.
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
            visitor->VisitExpected(loop->annotations));
        return visitor->DefaultVisitExpected(loop);
      },
      [&](const RegionStmt &region, ffi::StructuralVisitorObj *visitor)
          -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        if (region->op->name == "tl.attr") {
          auto fields = ReadAnnotations(region.get());
          result.has_volatile_memory |=
              fields.at("key").cast<ffi::String>() == "volatile_scope";
        }
        return visitor->DefaultVisitExpected(region);
      },
      [&](const DictAttrs &attrs, ffi::StructuralVisitorObj *visitor)
          -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        result.has_volatile_memory |=
            attrs.defined() && (attrs->dict.count("tirx.volatile") ||
                                attrs->dict.count("volatile_scope"));
        return visitor->DefaultVisitExpected(attrs);
      });
  return result;
}

class UnusedBindRemover : public tl::LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate;
  using LayoutMapMutator::Mutate_;

  explicit UnusedBindRemover(BindingUsage usage) : usage_(std::move(usage)) {}

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    auto primitive = op->value.as<PrimExpr>();
    if (primitive && !usage_.used.count(op->var.get())) {
      auto effect = SideEffect(*primitive);
      // Keep every potentially volatile read, including through aliases. No
      // alias-disjointness assumption is needed for this conservative rule.
      if (effect <= CallEffectKind::kPure ||
          (!usage_.has_volatile_memory &&
           effect <= CallEffectKind::kReadState)) {
        removed_ = true;
        return SeqStmt({}, op->loc);
      }
    }
    return LayoutMapMutator::Mutate_(op, mode);
  }

  bool removed() const { return removed_; }

private:
  BindingUsage usage_;
  bool removed_{false};
};

tirx::Function RemoveUnusedBindings(tirx::Function function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  while (true) {
    auto remover =
        ffi::make_object<UnusedBindRemover>(CollectBindingUsage(function));
    auto body = function->body;
    auto updated = remover->Mutate(body).ValueOrUnchanged(body);
    if (!remover->removed())
      return function;
    function.CopyOnWrite()->body = std::move(updated);
  }
}

class ContextualIndexFolder : public tvm::ExprMutator {
public:
  using ExprMutator::Mutate;
  using ExprMutator::Mutate_;

  explicit ContextualIndexFolder(sym::Analyzer analyzer)
      : analyzer_(std::move(analyzer)) {}

  ffi::UnchangedOr<Expr> Mutate_(const VarObj *, InplaceMode) final {
    // Type-bearing variables are identities, not numeric subexpressions.
    return ffi::Unchanged();
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const FloorDivObj *op,
                                     InplaceMode mode) final {
    PrimExpr original = ffi::GetRef<PrimExpr>(op);
    PrimExpr value = ExprMutator::Mutate_(op, mode).ValueOrUnchanged(original);
    PrimType type = value.ty();
    if (type.IsScalar() && type.MatchesCode(kDLInt) &&
        (type.bits() == 32 || type.bits() == 64) &&
        SideEffect(value) <= CallEffectKind::kPure) {
      auto bound = analyzer_->const_int_bound(value);
      if (bound->min_value == bound->max_value &&
          bound->min_value != sym::ConstIntBound::kNegInf &&
          bound->max_value != sym::ConstIntBound::kPosInf)
        return IntImm(type, bound->min_value);
    }
    if (value.same_as(original))
      return ffi::Unchanged();
    return value;
  }

private:
  sym::Analyzer analyzer_;
};

// Numeric rewrites and integer classification stay with the native analyzer.
// An optional proof never upgrades a query to SYMBOLIC_BOUND.
class NativeSimplifier : public tl::StmtExprMutatorWithAnalyzer {
public:
  using Parent = tl::StmtExprMutatorWithAnalyzer;
  using Parent::Mutate;
  using Parent::Mutate_;

  NativeSimplifier(const sym::Analyzer &analyzer, SimplifyConfig config,
                   bool remove_no_op = false, bool ignore_profiler_call = false)
      : Parent(analyzer), config_(std::move(config)),
        non_inline_analyzer_(analyzer->Clone()),
        index_folder_(ffi::make_object<ContextualIndexFolder>(
            config_->enable_simplify_let_inline ? GetAnalyzer()
                                                : non_inline_analyzer_)),
        remove_no_op_(remove_no_op),
        ignore_profiler_call_(ignore_profiler_call) {}

  tirx::Function Run(tirx::Function function) {
    MarkTensorParamShapes(function);
    has_volatile_memory_ = CollectBindingUsage(function).has_volatile_memory;
    auto body = function->body;
    function.CopyOnWrite()->body = Mutate(body).ValueOrUnchanged(body);
    return function;
  }

  ffi::UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                                    InplaceMode mode) final {
    // Tensor contracts are identity-bearing metadata. Their defining
    // expressions require a separate typed replacement transaction.
    if (input.as<tirx::TensorType>())
      return ffi::Unchanged();
    if (def_region_kind() != kTVMFFIDefRegionKindNone && input.as<Var>())
      return Parent::Mutate(input, mode);
    // Cleanup reuses the typed traversal and safe store identity policy, but
    // must not enable Simplify's arithmetic, value propagation or Bind
    // inlining.
    if (remove_no_op_)
      return Parent::Mutate(input, mode);
    ffi::Any value = Parent::Mutate(input, mode).ValueOrUnchanged(input);
    if (auto primitive = value.as<PrimExpr>()) {
      PrimExpr simplified = GetSyncedAnalyzer()->Simplify(*primitive);
      // Canonical rewriting can introduce a quotient below the result root,
      // e.g. (tx - 128) // 128 -> tx // 128 - 1. Fold the resulting expression
      // tree with the current constraints, then combine the exposed constants.
      PrimExpr folded =
          index_folder_->Mutate(simplified).ValueOrUnchanged(simplified);
      if (!folded.same_as(simplified))
        simplified = GetSyncedAnalyzer()->Simplify(folded);
      if (PureScalarBoolean(simplified))
        simplified = SimplifyBoolean(std::move(simplified));
      else if (config_->convert_boolean_to_and_of_ors &&
               CanDuplicateBoolean(simplified)) {
        int budget = 128;
        simplified = ToConjunction(std::move(simplified), &budget);
      }
      value = std::move(simplified);
    }
    if (value.same_as(input))
      return ffi::Unchanged();
    return value;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    // Knowledge never crosses a branch, loop, or nested lexical boundary.
    // Stores within this sequence remain available to its later statements.
    known_.reset();
    auto result = Parent::Mutate_(op, mode);
    known_.reset();
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (remove_no_op_)
      return Parent::Mutate_(op, mode);
    Expr value = Mutate(op->value, mode).ValueOrUnchanged(op->value);
    if (auto literal = value.as<StringImm>()) {
      VarRemapSetExpected(op->var, *literal).value();
      return SeqStmt({}, op->loc);
    }
    if (auto primitive = value.as<PrimExpr>();
        primitive && SideEffect(*primitive) <= CallEffectKind::kPure) {
      condition_bindings_.Set(op->var, *primitive);
      // Native SeqStmt traversal records the resulting definition once, after
      // this callback. Appending it here would define the same identity twice.
    }
    if (value.same_as(op->value))
      return ffi::Unchanged();
    auto copy = ffi::make_object<BindObj>(*op);
    copy->value = std::move(value);
    return Stmt(copy);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode mode) final {
    PrimExpr condition =
        remove_no_op_ ? op->condition : RewriteCondition(op->condition);
    if (remove_no_op_ && SideEffect(condition) <= CallEffectKind::kPure)
      condition = GetSyncedAnalyzer()->Simplify(condition);
    known_.reset();
    // The native constraint traversal visits both branches but deliberately
    // does not simplify control flow. Remove only a proven literal condition,
    // retaining scalar integer nonzero truth values as well as booleans.
    if (const auto *literal = condition.as<IntImmObj>()) {
      if (literal->value == 0 && !op->else_case)
        return SeqStmt({}, op->loc);
      SeqStmt selected = literal->value != 0 ? op->then_case : *op->else_case;
      // Keep the selected branch's definitions in a temporary collector scope.
      // The enclosing sequence records the returned statements once when it
      // flattens this replacement; recording them here would double-bind Vars.
      auto result = WithTraversalConstraints(this, {}, [&] {
        return Mutate(selected, InplaceMode::kDisallow)
            .ValueOrUnchanged(selected);
      });
      known_.reset();
      return result;
    }
    auto copy = ffi::make_object<IfObj>(*op);
    if (!remove_no_op_)
      copy->condition = std::move(condition);
    Stmt result = Parent::Mutate_(copy.get(), InplaceMode::kDisallow)
                      .ValueOrUnchanged(Stmt(copy));
    const auto *branch = result.as<IfObj>();
    if (remove_no_op_ && branch) {
      bool empty_else = !branch->else_case || tirx::IsNoOp(*branch->else_case);
      if (tirx::IsNoOp(branch->then_case) && empty_else)
        return PreserveEvaluation(branch->condition, branch->loc);
      if (branch->else_case && empty_else) {
        auto cleaned = ffi::make_object<IfObj>(*branch);
        cleaned->else_case = std::nullopt;
        return Stmt(cleaned);
      }
    }
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    known_.reset();
    const bool pure_bounds = SideEffect(op->begin) <= CallEffectKind::kPure &&
                             SideEffect(op->end) <= CallEffectKind::kPure;
    if (remove_no_op_ && pure_bounds &&
        GetSyncedAnalyzer()->CanProve(op->begin >= op->end)) {
      // Prune before traversing a zero-trip body: shape-dependent operations
      // there may be undefined (e.g. layout division by a zero extent).
      if (SideEffect(op->step) <= CallEffectKind::kPure)
        return SeqStmt({}, op->loc);
      // Preserve the loop's bound/step evaluation policy when its step has
      // effects. Its unreachable body still must not be analyzed.
      auto empty = ffi::make_object<ForObj>(*op);
      empty->body = SeqStmt({}, op->body->loc);
      return Stmt(empty);
    }
    auto result = Parent::Mutate_(op, mode);
    known_.reset();
    if (remove_no_op_) {
      Stmt statement =
          std::move(result).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
      const auto *loop = statement.as<ForObj>();
      if (loop && tirx::IsNoOp(loop->body) && pure_bounds &&
          SideEffect(loop->step) <= CallEffectKind::kPure &&
          GetSyncedAnalyzer()->CanProve(loop->step > 0))
        return SeqStmt({}, loop->loc);
      return statement;
    }
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj *op,
                                 InplaceMode mode) final {
    if (!remove_no_op_)
      return Parent::Mutate_(op, mode);
    if (ignore_profiler_call_) {
      if (const auto *call = op->value.as<CallObj>()) {
        const auto *callee = call->op.as<OpObj>();
        if (callee && (callee->name == "tirx.cuda.timer_init" ||
                       callee->name == "tirx.cuda.timer_start" ||
                       callee->name == "tirx.cuda.timer_end" ||
                       callee->name == "tirx.cuda.timer_finalize")) {
          ffi::Array<Stmt> effects;
          for (const Expr &argument : call->args)
            effects.push_back(PreserveEvaluation(argument, op->loc));
          return SeqStmt(effects, op->loc);
        }
      }
    }
    return PreserveEvaluation(op->value, op->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const WhileObj *op, InplaceMode mode) final {
    // The header executes repeatedly; a preceding store is only the value
    // at its first evaluation, not an invariant of the loop.
    known_.reset();
    auto result = Parent::Mutate_(op, mode);
    known_.reset();
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const AssertStmtObj *op,
                                 InplaceMode mode) final {
    if (remove_no_op_)
      return Parent::Mutate_(op, mode);
    auto copy = ffi::make_object<AssertStmtObj>(*op);
    copy->condition = RewriteCondition(op->condition);
    return Parent::Mutate_(copy.get(), InplaceMode::kDisallow)
        .ValueOrUnchanged(Stmt(copy));
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    known_.reset();
    bool volatile_scope = false;
    if (op->op->name == "tl.attr") {
      auto fields = ReadAnnotations(op);
      volatile_scope = fields.at("key").cast<ffi::String>() == "volatile_scope";
    }
    volatile_depth_ += volatile_scope;
    auto result = Parent::Mutate_(op, mode);
    volatile_depth_ -= volatile_scope;
    known_.reset();
    return result;
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    // Even a read-only foreign callback can observe mutable/volatile state.
    if (SideEffect(ffi::GetRef<Expr>(op)) > CallEffectKind::kPure)
      known_.reset();
    return Parent::Mutate_(op, mode);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                     InplaceMode mode) final {
    PrimExpr value =
        Parent::Mutate_(op, mode).ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
    auto load = value.as<TensorLoadObj>();
    bool propagate =
        !remove_no_op_ &&
        (config_->propagate_knowns_to_simplify_expressions ||
         (in_condition_ && config_->propagate_knowns_to_prove_conditional));
    if (load && propagate && !has_volatile_memory_ && !volatile_depth_ &&
        known_ && load->source.same_as(known_->tensor) &&
        SameIndices(load->indices, known_->indices)) {
      return known_->value;
    }
    if (value.same_as(ffi::GetRef<PrimExpr>(op)))
      return ffi::Unchanged();
    return value;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op,
                                 InplaceMode mode) final {
    Stmt statement =
        Parent::Mutate_(op, mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    const auto *store = statement.as<TensorStoreObj>();
    if (store && !has_volatile_memory_ && !volatile_depth_) {
      if (const auto *load = store->value.as<TensorLoadObj>();
          load && load->source.same_as(store->dest) &&
          SameIndices(load->indices, store->indices) &&
          SideEffect(store->dest) <= CallEffectKind::kPure) {
        bool pure_indices = true;
        for (const PrimExpr &index : store->indices)
          pure_indices &= SideEffect(index) <= CallEffectKind::kPure;
        if (pure_indices)
          return SeqStmt({}, op->loc);
      }
    }
    // Any store can alias the previous tensor. Retain only the newest exact
    // store; no distinct-tensor or private-allocation assumption is needed.
    known_.reset();
    if (store && !has_volatile_memory_ && !volatile_depth_ &&
        SideEffect(store->value) <= CallEffectKind::kPure) {
      bool stable_indices = true;
      for (const PrimExpr &index : store->indices)
        stable_indices &= SideEffect(index) <= CallEffectKind::kPure;
      if (stable_indices)
        known_ = KnownStore{store->dest, store->indices, store->value};
    }
    if (statement.same_as(ffi::GetRef<Stmt>(op)))
      return ffi::Unchanged();
    return statement;
  }

private:
  Stmt PreserveEvaluation(const Expr &value, const Location &loc) const {
    auto effect = SideEffect(value);
    if (effect <= CallEffectKind::kPure ||
        (!has_volatile_memory_ && effect <= CallEffectKind::kReadState))
      return SeqStmt({}, loc);
    return Evaluate(value, loc);
  }

  sym::AnalyzerObj *GetSyncedAnalyzer() const {
    if (config_->enable_simplify_let_inline)
      return Parent::GetSyncedAnalyzer();
    // Keep the collector's full definition stream for structural traversal,
    // but do not substitute named Bind values when this option is disabled.
    // Branch and iterator facts remain available; RewriteCondition explicitly
    // expands pure condition bindings before asking this analyzer for a proof.
    sym::ConstraintContext context;
    constraints->ForEach([&](const sym::Constraint &entry) {
      if (const auto *binding = entry.as<sym::BindingConstraintObj>();
          binding && condition_bindings_.count(binding->var))
        return;
      context->Append(entry);
    });
    non_inline_analyzer_->SyncConstraintContext(context);
    return non_inline_analyzer_.get();
  }

  struct KnownStore {
    Expr tensor;
    ffi::Array<PrimExpr> indices;
    PrimExpr value;
  };

  bool CanDuplicateBoolean(const PrimExpr &value) const {
    if (has_volatile_memory_ || !value.ty().IsScalar() ||
        !value.ty().MatchesCode(kDLBool) ||
        SideEffect(value) > CallEffectKind::kReadState)
      return false;
    bool stable = true;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        value,
        [&](const ffi::ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (auto call = object.as<Call>())
            stable &= SideEffect(*call) <= CallEffectKind::kPure;
          return ffi::WalkResult::Advance();
        });
    return stable;
  }

  bool SameIndices(const ffi::Array<PrimExpr> &a,
                   const ffi::Array<PrimExpr> &b) const {
    if (a.size() != b.size())
      return false;
    for (size_t i = 0; i < a.size(); ++i)
      if (!ExprDeepEqual()(a[i], b[i]))
        return false;
    return true;
  }

  PrimExpr RewriteCondition(PrimExpr condition) {
    condition = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                    condition,
                    [&](const Var &var) -> ffi::UnchangedOr<ffi::Any> {
                      if (auto value = condition_bindings_.Get(var))
                        return ffi::Any(*value);
                      return ffi::Unchanged();
                    })
                    .as_or_throw<PrimExpr>();
    bool previous = in_condition_;
    in_condition_ = true;
    auto result =
        Mutate(condition, InplaceMode::kDisallow).ValueOrUnchanged(condition);
    in_condition_ = previous;
    return result;
  }

  PrimExpr ProveBoolean(PrimExpr value) {
    value = GetSyncedAnalyzer()->Simplify(value);
    if (config_->transitively_prove_inequalities && PureScalarBoolean(value)) {
      if (GetSyncedAnalyzer()->CanProve(value))
        return IntImm::Bool(true);
      if (GetSyncedAnalyzer()->CanProve(Not(value)))
        return IntImm::Bool(false);
    }
    return value;
  }

  PrimExpr SimplifyBoolean(PrimExpr value) {
    if (config_->apply_constraints_to_boolean_branches) {
      const auto *and_node = value.as<AndObj>();
      const auto *or_node = value.as<OrObj>();
      if (and_node || or_node) {
        PrimExpr left = and_node ? and_node->a : or_node->a;
        PrimExpr right = and_node ? and_node->b : or_node->b;
        {
          // ConstraintContext consumes canonical comparisons, while an OR
          // branch contributes a negated predicate. Normalize that predicate
          // with the existing rules before asking for contextual bounds.
          PrimExpr constraint =
              GetSyncedAnalyzer()->Simplify(and_node ? left : Not(left));
          right = WithTraversalConstraints(
              this, {sym::PredicateConstraint(constraint)},
              [&] { return ProveBoolean(right); });
        }
        {
          PrimExpr constraint =
              GetSyncedAnalyzer()->Simplify(and_node ? right : Not(right));
          left = WithTraversalConstraints(
              this, {sym::PredicateConstraint(constraint)},
              [&] { return ProveBoolean(left); });
        }
        value =
            and_node ? PrimExpr(And(left, right)) : PrimExpr(Or(left, right));
      }
    }
    value = ProveBoolean(std::move(value));
    if (config_->convert_boolean_to_and_of_ors) {
      int budget = 128;
      value = ToConjunction(value, &budget);
    }
    return value;
  }

  PrimExpr ToConjunction(PrimExpr value, int *budget) {
    if (--*budget < 0)
      return value;
    if (const auto *conjunction = value.as<AndObj>())
      return And(ToConjunction(conjunction->a, budget),
                 ToConjunction(conjunction->b, budget));
    if (const auto *disjunction = value.as<OrObj>()) {
      PrimExpr left = ToConjunction(disjunction->a, budget);
      PrimExpr right = ToConjunction(disjunction->b, budget);
      if (*budget < 0)
        return value;
      if (const auto *nested = left.as<AndObj>())
        return And(ToConjunction(Or(nested->a, right), budget),
                   ToConjunction(Or(nested->b, right), budget));
      if (const auto *nested = right.as<AndObj>())
        return And(ToConjunction(Or(left, nested->a), budget),
                   ToConjunction(Or(left, nested->b), budget));
      return Or(left, right);
    }
    return value;
  }

  SimplifyConfig config_;
  sym::Analyzer non_inline_analyzer_;
  ffi::ObjectPtr<ContextualIndexFolder> index_folder_;
  ffi::Map<Var, PrimExpr> condition_bindings_;
  std::optional<KnownStore> known_;
  bool in_condition_{false};
  bool has_volatile_memory_{false};
  int volatile_depth_{0};
  bool remove_no_op_{false};
  bool ignore_profiler_call_{false};
};

tirx::Function PruneUnusedParameters(tirx::Function function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  auto uses = UndefinedVars(function->body);
  ffi::Array<Expr> expressions;
  for (const Var &use : uses)
    expressions.push_back(use);
  auto free = UndefinedVars(Tuple(expressions));
  std::unordered_set<const VarObj *> live;
  for (const Var &use : free)
    live.insert(use.get());
  auto usage = CollectBindingUsage(function);
  live.insert(usage.used.begin(), usage.used.end());
  CollectInputTensorDescriptorUses(function->body, function->params, &live);
  ffi::Array<Var> kept;
  for (const Var &parameter : function->params) {
    // The legacy option only removed unused tensor parameters. Scalar and
    // pointer parameters retain their public positional ABI.
    bool used =
        !parameter->ty.as<tirx::TensorTypeObj>() || live.count(parameter.get());
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        parameter->ty,
        [&](const VarObj *variable) -> ffi::Expected<ffi::WalkResult> {
          used |= live.count(variable) != 0;
          return ffi::WalkResult::Advance();
        });
    if (used)
      kept.push_back(parameter);
  }
  if (kept.size() != function->params.size()) {
    auto body = RemapInputTensorDescriptors(function, kept);
    auto *writable = function.CopyOnWrite();
    writable->body = std::move(body);
    writable->params = kept;
    writable->ty = writable->func_type_annotation();
  }
  return function;
}
} // namespace

tvm::transform::Pass Simplify(bool simplify_arguments) {
  auto pass = [simplify_arguments](IRModule module,
                                   tvm::transform::PassContext context) {
    auto config = context->GetConfigOrDefault<SimplifyConfig>("tl.Simplify");
    auto core = tirx::transform::CreateFunctionPass(
        [config](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
          if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
            return function;
          auto analyzer = CreateTileLangAnalyzer();
          return RemoveUnusedBindings(
              ffi::make_object<NativeSimplifier>(analyzer, config)
                  ->Run(function));
        },
        0, "tl.NativeSimplify");
    module = core(std::move(module));
    if (simplify_arguments) {
      auto cleanup = tirx::transform::CreateFunctionPass(
          [](tirx::Function function, IRModule, tvm::transform::PassContext) {
            return PruneUnusedParameters(function);
          },
          0, "tl.RemoveUnusedParameters");
      module = cleanup(std::move(module));
    }
    return module;
  };
  return tvm::transform::CreateModulePass(pass, 0, "tl.Simplify");
}

tvm::transform::Pass RemoveNoOp() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext context) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    bool ignore_profiler_call = false;
    int64_t maximum_steps = 0;
    if (auto options = context->GetConfig<ffi::ObjectRef>("tirx.RemoveNoOp")) {
      // Consume the upstream reflected configuration without duplicating its
      // private C++ config class or adding another user-facing option.
      const auto type_key = options->GetTypeKey();
      ignore_profiler_call = ffi::reflection::FieldGetter(
                                 type_key, "ignore_profiler_call")(*options)
                                 .cast<bool>();
      maximum_steps = ffi::reflection::FieldGetter(
                          type_key, "max_simplification_steps")(*options)
                          .cast<int64_t>();
    }
    auto analyzer = CreateTileLangAnalyzer();
    analyzer->rewrite_simplify.SetMaximumRewriteSteps(maximum_steps);
    auto config_node = ffi::make_object<SimplifyConfigNode>();
    auto config = ffi::GetRef<SimplifyConfig>(config_node.get());
    return ffi::make_object<NativeSimplifier>(analyzer, std::move(config), true,
                                              ignore_profiler_call)
        ->Run(std::move(function));
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.RemoveNoOp");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.transform.Simplify", Simplify)
      .def("tl.transform.RemoveNoOp", RemoveNoOp);
}
} // namespace tvm::tl
