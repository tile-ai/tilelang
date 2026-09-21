/*!
 * \file lower_invariant_arithmetic.cc
 * \brief Host-prepared integer arithmetic for launch-invariant divisors.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "common/attr.h"
#include "common/launch_plan.h"

namespace tvm {
namespace tl {
using namespace tirx;

TVM_REGISTER_PASS_CONFIG_OPTION("tl.enable_invariant_arithmetic", Bool);

namespace {
using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using ExprSet =
    std::unordered_set<PrimExpr, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

bool IsSupportedInteger(DataType dtype) {
  return dtype.is_scalar() && (dtype.is_int() || dtype.is_uint()) &&
         (dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32 ||
          dtype.bits() == 64);
}

// Drop only casts that preserve every value of the source type. In particular,
// sign extension into an unsigned type is not a value-preserving widening.
bool IsValuePreservingWiden(DataType from, DataType to) {
  return IsSupportedInteger(from) && IsSupportedInteger(to) &&
         from.bits() < to.bits() && (from.is_uint() || to.is_int());
}

struct ArithmeticFacts {
  VarSet inputs;
  VarSet range_opaque;
  ffi::Map<Var, PrimExpr> aliases;
  ExprSet seen;
  ExprSet exact;
  ExprSet nonnegative;
  ExprSet positive_divisor;
  ExprSet fits_signed32;
  ExprSet fits_unsigned32;
  std::vector<PrimExpr> fast_divisors;

  PrimExpr Resolve(const PrimExpr &expr) const {
    return Substitute(expr, aliases);
  }

  PrimExpr ResolveDivisor(const PrimExpr &expr) const {
    PrimExpr d = Resolve(expr);
    while (const auto *cast = d.as<CastNode>()) {
      if (!IsValuePreservingWiden(cast->value.dtype(), cast->dtype)) {
        break;
      }
      d = cast->value;
    }
    if (IsSupportedInteger(d.dtype()) && d.dtype().bits() < 32) {
      d = cast(d.dtype().is_int() ? DataType::Int(32) : DataType::UInt(32), d);
    }
    return d;
  }

  bool CanPrepare(const PrimExpr &expr) const {
    if (!IsSupportedInteger(expr.dtype())) {
      return false;
    }
    if (const auto *var = expr.as<VarNode>()) {
      return inputs.count(ffi::GetRef<Var>(var));
    }
    if (expr.as<IntImmNode>()) {
      return true;
    }
    if (const auto *cast = expr.as<CastNode>()) {
      return CanPrepare(cast->value);
    }
#define TL_HOST_BINARY(Node)                                                   \
  if (const auto *node = expr.as<Node>()) {                                    \
    return CanPrepare(node->a) && CanPrepare(node->b);                         \
  }
    TL_HOST_BINARY(AddNode)
    TL_HOST_BINARY(SubNode)
    TL_HOST_BINARY(MulNode)
    TL_HOST_BINARY(MinNode)
    TL_HOST_BINARY(MaxNode)
#undef TL_HOST_BINARY
    return false;
  }

  bool HasOpaqueRange(const PrimExpr &expr) const {
    bool opaque = false;
    PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
      if (const auto *cast = node.as<CastNode>()) {
        // Analyzer bounds can model a signed-to-unsigned cast as a widening
        // of the signed interval, missing the wrapped negative values.
        if (IsSupportedInteger(cast->value.dtype()) &&
            IsSupportedInteger(cast->dtype) &&
            cast->dtype != cast->value.dtype() &&
            !IsValuePreservingWiden(cast->value.dtype(), cast->dtype)) {
          opaque = true;
        }
      } else if (const auto *var = node.as<VarNode>()) {
        opaque = opaque || range_opaque.count(ffi::GetRef<Var>(var));
      }
    });
    return opaque;
  }

  bool HasFastDivisor(const PrimExpr &d) const {
    for (const PrimExpr &other : fast_divisors) {
      if (ffi::StructuralEqual()(d, other)) {
        return true;
      }
    }
    return false;
  }
};

// Collect proofs before rewriting predicates: replacing x % d in a condition
// must not hide the exact-divisibility fact from its dominated division sites.
class ArithmeticAnalyzer : public arith::IRMutatorWithAnalyzer {
public:
  ArithmeticAnalyzer(arith::Analyzer *analyzer, ArithmeticFacts *facts)
      : IRMutatorWithAnalyzer(analyzer), facts_(facts) {}

  Stmt VisitStmt_(const BindNode *op) final {
    PrimExpr value = facts_->Resolve(op->value);
    if (facts_->HasOpaqueRange(op->value)) {
      facts_->range_opaque.insert(op->var);
    }
    if (facts_->CanPrepare(value)) {
      facts_->aliases.Set(op->var, value);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

#define TL_ANALYZE_DIVMOD(Node, is_remainder)                                  \
  PrimExpr VisitExpr_(const Node *op) final {                                  \
    Record(ffi::GetRef<PrimExpr>(op), op->a, op->b, is_remainder);             \
    return IRMutatorWithAnalyzer::VisitExpr_(op);                              \
  }
  TL_ANALYZE_DIVMOD(FloorDivNode, false)
  TL_ANALYZE_DIVMOD(FloorModNode, true)
  TL_ANALYZE_DIVMOD(DivNode, false)
  TL_ANALYZE_DIVMOD(ModNode, true)
#undef TL_ANALYZE_DIVMOD

private:
  void Record(const PrimExpr &expr, const PrimExpr &x, const PrimExpr &divisor,
              bool remainder) {
    PrimExpr d = facts_->ResolveDivisor(divisor);
    if (!facts_->CanPrepare(d) || d.as<IntImmNode>() ||
        !IsSupportedInteger(x.dtype())) {
      return;
    }
    // Only stable scalar expressions can inherit divisibility from a predicate.
    // Re-reading a mutable buffer is not the same value as its earlier load.
    bool stable = SideEffect(x) == CallEffectKind::kPure;
    bool range_safe = !facts_->HasOpaqueRange(x);
    bool divisor_safe = !facts_->HasOpaqueRange(divisor);
    bool exact = stable && range_safe && divisor_safe &&
                 analyzer_->CanProve(floormod(x, divisor) == 0);
    bool first = facts_->seen.insert(expr).second;
    auto record_proof = [&](ExprSet *set, bool proven) {
      // A shared expression node can occur under different predicates. A
      // proof must hold at every occurrence, not merely at the first visit.
      if (first && proven) {
        set->insert(expr);
      } else if (!proven) {
        set->erase(expr);
      }
    };
    record_proof(&facts_->exact, exact);
    // Type-derived bounds remain valid for mutable loads. Predicate-derived
    // bounds are used only for stable values, just like exactness proofs.
    arith::Analyzer type_analyzer;
    arith::Analyzer *range_analyzer = stable ? analyzer_ : &type_analyzer;
    record_proof(&facts_->nonnegative,
                 x.dtype().is_uint() ||
                     (range_safe && range_analyzer->CanProve(x >= 0)));
    record_proof(&facts_->fits_signed32,
                 x.dtype().bits() < 32 ||
                     (x.dtype().bits() == 32 && x.dtype().is_int()) ||
                     (range_safe &&
                      range_analyzer->CanProve(
                          x <= make_const(x.dtype(), int64_t{2147483647}))));
    record_proof(&facts_->fits_unsigned32,
                 x.dtype().bits() <= 32 ||
                     (range_safe &&
                      range_analyzer->CanProve(
                          x <= make_const(x.dtype(), uint64_t{4294967295}))));
    record_proof(&facts_->positive_divisor,
                 divisor_safe && analyzer_->CanProve(divisor > 0));
    if ((!exact || x.dtype().bits() == 64) && !remainder &&
        d.dtype() == DataType::Int(32) && !facts_->HasFastDivisor(d)) {
      facts_->fast_divisors.push_back(d);
    }
  }

  ArithmeticFacts *facts_;
};

class InvariantArithmeticRewriter : public StmtExprMutator {
public:
  InvariantArithmeticRewriter(const ArithmeticFacts &facts, LaunchPlan *plan)
      : facts_(facts), plan_(plan) {}

#define TL_REWRITE_DIVMOD(Node, remainder, truncating)                         \
  PrimExpr VisitExpr_(const Node *op) final {                                  \
    return Rewrite(ffi::GetRef<PrimExpr>(op), VisitExpr(op->a),                \
                   VisitExpr(op->b), remainder, truncating);                   \
  }
  TL_REWRITE_DIVMOD(FloorDivNode, false, false)
  TL_REWRITE_DIVMOD(FloorModNode, true, false)
  TL_REWRITE_DIVMOD(DivNode, false, true)
  TL_REWRITE_DIVMOD(ModNode, true, true)
#undef TL_REWRITE_DIVMOD

private:
  PrimExpr FastDiv(const PrimExpr &x, const PrimExpr &d, bool remainder,
                   const PrimExpr &valid, bool truncating) {
    DataType i32 = DataType::Int(32);
    DataType u32 = DataType::UInt(32);
    DataType u64 = DataType::UInt(64);
    Var safe = plan_->Prepare(max(d, make_const(i32, 1)), "fastdiv_d");
    Var k = plan_->Prepare(
        32 - cast(i32, clz(cast(u32, max(safe, make_const(i32, 2)) - 1))),
        "fastdiv_k");
    Var shift = plan_->Prepare(max(k - 1, make_const(i32, 0)), "fastdiv_shift");
    PrimExpr power = make_const(u64, 1) << cast(u64, shift + 32);
    Var multiplier = plan_->Prepare(
        cast(u32, floordiv(power + cast(u64, safe) - 1, cast(u64, safe))),
        "fastdiv_multiplier");
    return Call(x.dtype(), remainder ? tl::fast_rem() : tl::fast_div(),
                {x, d, multiplier, shift, valid, Bool(truncating)});
  }

  // mu = floor(2^word_bits / d). For a dividend in that word's range,
  // the quotient is at most one too small, even if d is wider than the word.
  PrimExpr BarrettReduction(const PrimExpr &x, const PrimExpr &d,
                            bool remainder, const PrimExpr &valid,
                            bool truncating, int word_bits) {
    DataType word = DataType::UInt(word_bits);
    DataType u64 = DataType::UInt(64);
    PrimExpr safe = cast(u64, max(d, make_const(d.dtype(), 1)));
    PrimExpr reciprocal_expr;
    if (word.bits() == 64) {
      // floor(2^64 / d), represented modulo 2^64 for d == 1. Compute
      // via UINT64_MAX so neither host int128 nor an overflowing literal is
      // required. The device handles d == 1 separately.
      PrimExpr top = make_const(u64, uint64_t{0xffffffffffffffff});
      reciprocal_expr =
          floordiv(top, safe) + cast(u64, floormod(top, safe) == safe - 1);
    } else {
      reciprocal_expr =
          cast(word, floordiv(make_const(u64, uint64_t{1} << 32), safe));
    }
    PrimExpr reciprocal =
        plan_->PrepareArgument(reciprocal_expr, "barrett_reciprocal");
    if (remainder) {
      return Call(x.dtype(), tl::barrett_reduce(),
                  {x, d, reciprocal, valid, Bool(truncating)});
    }
    return Call(x.dtype(), tl::fast_div(),
                {x, d, reciprocal, make_const(DataType::Int(32), 0), valid,
                 Bool(truncating)});
  }

  // For an exactly divisible x, remove d's power-of-two factor and multiply
  // by the inverse of its odd part modulo 2^32. Five Newton steps from 1
  // double the number of correct bits each time (1 -> 2 -> ... -> 32).
  PrimExpr ExactDiv(const PrimExpr &x, const PrimExpr &d,
                    const PrimExpr &valid) {
    DataType u32 = DataType::UInt(32);
    Var safe =
        plan_->Prepare(cast(u32, max(d, make_const(d.dtype(), 1))), "exact_d");
    PrimExpr lowbit = bitwise_and(safe, make_const(u32, 0) - safe);
    Var shift =
        plan_->Prepare(make_const(u32, 31) - clz(lowbit), "exact_shift");
    Var odd = plan_->Prepare(safe >> shift, "exact_odd");
    PrimExpr inverse = make_const(u32, 1);
    for (int i = 0; i < 5; ++i) {
      inverse = plan_->Prepare(inverse * (make_const(u32, 2) - odd * inverse),
                               "exact_inverse");
    }
    return Call(x.dtype(), tl::exact_div(), {x, d, inverse, shift, valid});
  }

  PrimExpr Rewrite(const PrimExpr &site, const PrimExpr &x,
                   const PrimExpr &divisor, bool remainder, bool truncating) {
    PrimExpr d = facts_.ResolveDivisor(divisor);
    PrimExpr original =
        truncating ? (remainder ? truncmod(x, divisor) : truncdiv(x, divisor))
                   : (remainder ? floormod(x, divisor) : floordiv(x, divisor));
    if (!facts_.CanPrepare(d) || d.as<IntImmNode>() ||
        !IsSupportedInteger(x.dtype())) {
      return original;
    }
    // The operation type is independent of the reciprocal's word size.
    // Narrow operations use CUDA's scalar 32-bit arithmetic and cast the
    // result back; wide operations retain their original fallback width.
    DataType compute_type =
        x.dtype().bits() < 32
            ? (x.dtype().is_int() ? DataType::Int(32) : DataType::UInt(32))
            : x.dtype();
    PrimExpr value = cast(compute_type, x);
    bool exact = facts_.exact.count(site) && compute_type.bits() == 32;
    bool fast32 = d.dtype() == DataType::Int(32) && facts_.HasFastDivisor(d);
    PrimExpr valid = facts_.positive_divisor.count(site) ? const_true() : d > 0;
    if (!exact && !facts_.nonnegative.count(site)) {
      valid = valid && (value >= 0);
    }
    if (d.dtype().bits() == 32 && compute_type.bits() == 64) {
      const ExprSet &bounded =
          fast32 ? facts_.fits_signed32 : facts_.fits_unsigned32;
      if (!bounded.count(site)) {
        uint64_t limit = fast32 ? uint64_t{2147483647} : uint64_t{4294967295};
        valid = valid && (value <= make_const(compute_type, limit));
      }
    }
    PrimExpr result;
    if (exact) {
      result = remainder ? make_zero(compute_type) : ExactDiv(value, d, valid);
    } else if (fast32) {
      result = FastDiv(value, d, remainder, valid, truncating);
    } else {
      int word_bits =
          d.dtype().bits() == 32 || facts_.fits_unsigned32.count(site) ? 32
                                                                       : 64;
      result =
          BarrettReduction(value, d, remainder, valid, truncating, word_bits);
    }
    return cast(x.dtype(), result);
  }

  const ArithmeticFacts &facts_;
  LaunchPlan *plan_;
};

class InvariantArithmeticPlanner : public StmtMutator {
public:
  explicit InvariantArithmeticPlanner(const PrimFunc &func) {
    for (const Var &param : func->params) {
      inputs_.insert(param);
    }
    for (const auto &entry : func->buffer_map) {
      for (const PrimExpr &extent : entry.second->shape) {
        if (const auto *var = extent.as<VarNode>()) {
          inputs_.insert(ffi::GetRef<Var>(var));
        }
      }
      for (const PrimExpr &stride : entry.second->strides) {
        if (const auto *var = stride.as<VarNode>()) {
          inputs_.insert(ffi::GetRef<Var>(var));
        }
      }
    }
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tvm::attr::kTarget) {
      return StmtMutator::VisitStmt_(op);
    }
    LaunchPlan plan;
    ArithmeticFacts facts;
    facts.inputs = inputs_;
    arith::Analyzer analyzer;
    ArithmeticAnalyzer(&analyzer, &facts)(op->body);
    Stmt body = InvariantArithmeticRewriter(facts, &plan)(op->body);
    // Canonicalize original narrow/unsigned scalar captures as well as
    // preparation results, without changing their arithmetic types.
    ffi::Map<Var, PrimExpr> abi_values;
    PostOrderVisit(body, [&](const ffi::ObjectRef &node) {
      if (const auto *vn = node.as<VarNode>()) {
        Var var = ffi::GetRef<Var>(vn);
        if (inputs_.count(var) && IsSupportedInteger(var.dtype()) &&
            (var.dtype().bits() < 32 || var.dtype() == DataType::UInt(64)) &&
            !abi_values.count(var)) {
          abi_values.Set(var, plan.PrepareArgument(var, "scalar_bits"));
        }
      }
    });
    body = Substitute(body, abi_values);
    return plan.Materialize(AttrStmt(op->node, op->attr_key, op->value, body));
  }

private:
  VarSet inputs_;
};

// Keep arithmetic opaque through symbolic simplification, then share calls only
// inside the statement/branch where they execute. Never extract fallback math.
bool IsInvariantArithmetic(const PrimExpr &expr) {
  const auto *call = expr.as<CallNode>();
  return call && (call->op.same_as(tl::fast_div()) ||
                  call->op.same_as(tl::fast_rem()) ||
                  call->op.same_as(tl::barrett_reduce()) ||
                  call->op.same_as(tl::exact_div()));
}

class ArithmeticCallBinder : public ExprMutator {
public:
  PrimExpr VisitExpr_(const CallNode *op) final {
    // Do not move expressions out of a lazy arm or cache mutable loads.
    if (op->op.same_as(builtin::if_then_else()) ||
        SideEffect(ffi::GetRef<PrimExpr>(op)) != CallEffectKind::kPure) {
      return ffi::GetRef<PrimExpr>(op);
    }
    PrimExpr value = ExprMutator::VisitExpr_(op);
    if (!IsInvariantArithmetic(value)) {
      return value;
    }
    for (const auto &entry : values_) {
      if (ffi::StructuralEqual()(entry.first, value)) {
        return entry.second;
      }
    }
    Var var("invariant_value", value.dtype());
    values_.emplace_back(value, var);
    bindings_.push_back(Bind(var, value, op->span));
    return var;
  }

  // The caller can lower statement conditions into separate branches. Within
  // other expressions these nodes remain barriers to speculative evaluation.
  PrimExpr VisitExpr_(const AndNode *op) final {
    return ffi::GetRef<PrimExpr>(op);
  }
  PrimExpr VisitExpr_(const OrNode *op) final {
    return ffi::GetRef<PrimExpr>(op);
  }
  PrimExpr VisitExpr_(const LetNode *op) final {
    return ffi::GetRef<PrimExpr>(op);
  }
  PrimExpr VisitExpr_(const SelectNode *op) final {
    return ffi::GetRef<PrimExpr>(op);
  }

  Stmt Materialize(const Stmt &stmt) {
    bindings_.push_back(stmt);
    return SeqStmt::Flatten(bindings_);
  }

private:
  std::vector<std::pair<PrimExpr, Var>> values_;
  ffi::Array<Stmt> bindings_;
};

class InvariantArithmeticMaterializer : public StmtMutator {
public:
  Stmt VisitStmt_(const BindNode *op) final {
    ArithmeticCallBinder binder;
    PrimExpr value = binder(op->value);
    return binder.Materialize(Bind(op->var, value, op->span));
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    // Predicated stores may suppress evaluation of their operands.
    if (op->predicate.defined()) {
      return ffi::GetRef<Stmt>(op);
    }
    if (const auto *call = op->value.as<CallNode>();
        call && call->op.same_as(builtin::if_then_else()) &&
        ContainsArithmetic(op->value)) {
      Stmt yes = VisitStmt(BufferStore(op->buffer, call->args[1], op->indices,
                                       op->predicate, op->span));
      Stmt no = VisitStmt(BufferStore(op->buffer, call->args[2], op->indices,
                                      op->predicate, op->span));
      return MaterializeCondition(call->args[0], yes, no, op->span);
    }
    ArithmeticCallBinder binder;
    PrimExpr value = binder(op->value);
    return binder.Materialize(
        BufferStore(op->buffer, value, op->indices, op->predicate, op->span));
  }

private:
  static bool ContainsArithmetic(const PrimExpr &expr) {
    bool found = false;
    PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        found |= IsInvariantArithmetic(ffi::GetRef<PrimExpr>(call));
      }
    });
    return found;
  }

  Stmt MaterializeCondition(const PrimExpr &condition, const Stmt &yes,
                            const Stmt &no, const Span &span) {
    if (!ContainsArithmetic(condition)) {
      return IfThenElse(condition, yes, no, span);
    }
    // Only split a flat two-term guard around a single store. Do not expand
    // arbitrary Boolean trees: duplicating their branches can grow
    // exponentially.
    if (const auto *op = condition.as<AndNode>();
        op && !op->a.as<AndNode>() && !op->a.as<OrNode>() &&
        !op->b.as<AndNode>() && !op->b.as<OrNode>()) {
      return MaterializeCondition(
          op->a, MaterializeCondition(op->b, yes, no, span), no, span);
    }
    ArithmeticCallBinder binder;
    PrimExpr value = binder(condition);
    return binder.Materialize(IfThenElse(value, yes, no, span));
  }
};
} // namespace

namespace transform {
tvm::transform::Pass LowerInvariantArithmetic() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    func.CopyOnWrite()->body = InvariantArithmeticPlanner(func)(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.LowerInvariantArithmetic", {});
}

tvm::transform::Pass MaterializeInvariantArithmetic() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    func.CopyOnWrite()->body = InvariantArithmeticMaterializer()(func->body);
    return func;
  };
  // Splitting short-circuit conditions can duplicate branch-local bindings.
  return tvm::transform::Sequential(
      {tirx::transform::CreatePrimFuncPass(
           pass_func, 0, "tl.MaterializeInvariantArithmetic", {}),
       tirx::transform::ConvertSSA()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.transform.LowerInvariantArithmetic", LowerInvariantArithmetic)
      .def("tl.transform.MaterializeInvariantArithmetic",
           MaterializeInvariantArithmetic);
}
} // namespace transform
} // namespace tl
} // namespace tvm
