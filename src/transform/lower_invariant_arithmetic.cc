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

#include <algorithm>
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

bool CanProvePositiveDivisor(const PrimExpr &expr, arith::Analyzer *analyzer) {
  if (const auto *maximum = expr.as<MaxNode>()) {
    return CanProvePositiveDivisor(maximum->a, analyzer) ||
           CanProvePositiveDivisor(maximum->b, analyzer);
  }
  if (const auto *minimum = expr.as<MinNode>()) {
    return CanProvePositiveDivisor(minimum->a, analyzer) &&
           CanProvePositiveDivisor(minimum->b, analyzer);
  }
  if (expr.as<VarNode>() || expr.as<IntImmNode>()) {
    return analyzer->CanProve(expr > 0);
  }
  if (const auto *cast = expr.as<CastNode>()) {
    return IsValuePreservingWiden(cast->value.dtype(), cast->dtype) &&
           CanProvePositiveDivisor(cast->value, analyzer);
  }
  if (const auto *mul = expr.as<MulNode>()) {
    if (!CanProvePositiveDivisor(mul->a, analyzer) ||
        !CanProvePositiveDivisor(mul->b, analyzer)) {
      return false;
    }
    auto a = analyzer->const_int_bound(mul->a);
    auto b = analyzer->const_int_bound(mul->b);
    uint64_t limit = expr.dtype().is_int()
                         ? (uint64_t{1} << (expr.dtype().bits() - 1)) - 1
                         : (~uint64_t{0} >> (64 - expr.dtype().bits()));
    // Positivity of the factors is insufficient if their product wraps.
    return a->max_value > 0 && b->max_value > 0 &&
           a->max_value != arith::ConstIntBoundNode::kPosInf &&
           b->max_value != arith::ConstIntBoundNode::kPosInf &&
           uint64_t(a->max_value) <= limit / uint64_t(b->max_value);
  }
  return false;
}

// Remainder canonicalization and factor matching substantially follow the
// approach in penguin-wwy's PR #3267. Apply these identities after index
// widening, before introducing opaque arithmetic intrinsics.
class InvariantRemainderNormalizer : public arith::IRMutatorWithAnalyzer {
public:
  explicit InvariantRemainderNormalizer(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  static PrimExpr ReuseRemainder(const PrimExpr &product,
                                 const std::vector<PrimExpr> &remainders) {
    if (!IsSupportedInteger(product.dtype()) ||
        SideEffect(product) != CallEffectKind::kPure) {
      return product;
    }
    for (const PrimExpr &expr : remainders) {
      const auto *mod = expr.as<FloorModNode>();
      if (product.dtype() != expr.dtype()) {
        continue;
      }
      if (auto divisor = MatchProduct_(product, mod->a);
          divisor && ffi::StructuralEqual()(divisor.value(), mod->b)) {
        // q*d = x-r also holds under word wrapping; retain the input width.
        return mod->a - expr;
      }
    }
    return product;
  }

  PrimExpr VisitExpr_(const FloorModNode *op) final {
    PrimExpr value = StmtExprMutator::VisitExpr_(op);
    const auto *mod = value.as<FloorModNode>();
    const auto *radix = mod ? mod->b.as<IntImmNode>() : nullptr;
    if (radix && radix->value > 0 && (radix->value & (radix->value - 1)) == 0 &&
        IsSupportedInteger(value.dtype()) &&
        SideEffect(value) == CallEffectKind::kPure) {
      if (auto reduced = RecoverRemainder_(mod->a, mod->b)) {
        return floormod(reduced.value(), mod->b);
      }
    }
    return value;
  }

  PrimExpr VisitExpr_(const SubNode *op) final {
    PrimExpr value = StmtExprMutator::VisitExpr_(op);
    const auto *sub = value.as<SubNode>();
    if (!sub || !IsSupportedInteger(value.dtype()) ||
        SideEffect(value) != CallEffectKind::kPure) {
      return value;
    }
    if (auto divisor = MatchProduct_(sub->b, sub->a)) {
      return floormod(sub->a, divisor.value());
    }
    if (auto reduced = CancelTerm_(sub->a, sub->b)) {
      return reduced.value();
    }
    if (auto reduced = RecoverRemainder_(value)) {
      return reduced.value();
    }
    return value;
  }

  PrimExpr VisitExpr_(const AddNode *op) override {
    PrimExpr value = StmtExprMutator::VisitExpr_(op);
    const auto *add = value.as<AddNode>();
    if (!add || !IsSupportedInteger(value.dtype()) ||
        SideEffect(value) != CallEffectKind::kPure) {
      return value;
    }
    for (int i = 0; i < 2; ++i) {
      PrimExpr product = i == 0 ? add->a : add->b;
      PrimExpr remainder = i == 0 ? add->b : add->a;
      if (const auto *mod = remainder.as<FloorModNode>()) {
        auto divisor = MatchProduct_(product, mod->a);
        if (divisor && ffi::StructuralEqual()(divisor.value(), mod->b)) {
          // q*d + r = x, including signed floor semantics and wrapping
          // multiplication. No positive-factor or no-overflow assumption.
          return mod->a;
        }
      }
    }
    return value;
  }

private:
  // Simplification can interleave a swizzle's offsets with x - (x/d)*d.
  // Recover the remainder without distributing products or crossing casts.
  static void CollectTerms_(const PrimExpr &value, bool negative,
                            std::vector<std::pair<PrimExpr, bool>> *terms) {
    if (const auto *add = value.as<AddNode>()) {
      CollectTerms_(add->a, negative, terms);
      CollectTerms_(add->b, negative, terms);
    } else if (const auto *sub = value.as<SubNode>()) {
      CollectTerms_(sub->a, negative, terms);
      CollectTerms_(sub->b, !negative, terms);
    } else {
      terms->emplace_back(value, negative);
    }
  }

  ffi::Optional<PrimExpr>
  RecoverRemainder_(const PrimExpr &value,
                    ffi::Optional<PrimExpr> modulus = std::nullopt) {
    std::vector<std::pair<PrimExpr, bool>> terms;
    CollectTerms_(value, false, &terms);
    if (terms.size() > 32) {
      return std::nullopt;
    }
    for (size_t i = 0; i < terms.size(); ++i) {
      if (!terms[i].second) {
        continue;
      }
      std::vector<PrimExpr> factors;
      CollectFactors_(terms[i].first, &factors);
      for (const PrimExpr &factor : factors) {
        const auto *div = factor.as<FloorDivNode>();
        if (!div || !MatchProduct_(terms[i].first, div->a)) {
          continue;
        }
        std::vector<std::pair<PrimExpr, bool>> dividend;
        CollectTerms_(div->a, false, &dividend);
        std::vector<bool> used(terms.size(), false);
        used[i] = true;
        bool matched = true;
        for (const auto &term : dividend) {
          size_t j = 0;
          for (; j < terms.size(); ++j) {
            if (!used[j] && term.second == terms[j].second &&
                ffi::StructuralEqual()(term.first, terms[j].first)) {
              used[j] = true;
              break;
            }
          }
          if (j == terms.size()) {
            // A power-of-two outer modulus also discards word overflow.
            // Missing multiples of it can be restored before matching x-q*d.
            if (modulus.defined() &&
                analyzer_->CanProve(floormod(term.first, modulus.value()) ==
                                    0)) {
              continue;
            }
            matched = false;
            break;
          }
        }
        if (matched) {
          PrimExpr result = floormod(div->a, div->b);
          for (size_t j = 0; j < terms.size(); ++j) {
            if (!used[j]) {
              result = terms[j].second ? result - terms[j].first
                                       : result + terms[j].first;
            }
          }
          return result;
        }
      }
    }
    return std::nullopt;
  }

  static ffi::Optional<PrimExpr> CancelTerm_(const PrimExpr &sum,
                                             const PrimExpr &term) {
    std::vector<PrimExpr> left, right;
    CollectFactors_(sum, &left);
    CollectFactors_(term, &right);
    if (SameFactors_(left, std::move(right))) {
      return make_zero(sum.dtype());
    }
    if (const auto *add = sum.as<AddNode>()) {
      if (auto reduced = CancelTerm_(add->a, term)) {
        return is_zero(reduced.value()) ? add->b : reduced.value() + add->b;
      }
      if (auto reduced = CancelTerm_(add->b, term)) {
        return is_zero(reduced.value()) ? add->a : add->a + reduced.value();
      }
    }
    return std::nullopt;
  }

  static ffi::Optional<PrimExpr> MatchProduct_(const PrimExpr &product,
                                               const PrimExpr &dividend) {
    std::vector<PrimExpr> factors;
    CollectFactors_(product, &factors);
    for (size_t i = 0; i < factors.size(); ++i) {
      const auto *quotient = factors[i].as<FloorDivNode>();
      if (!quotient || !ffi::StructuralEqual()(quotient->a, dividend)) {
        continue;
      }
      std::vector<PrimExpr> divisor_factors;
      CollectFactors_(quotient->b, &divisor_factors);
      auto remaining = factors;
      remaining.erase(remaining.begin() + i);
      if (SameFactors_(remaining, std::move(divisor_factors))) {
        return quotient->b;
      }
    }
    return std::nullopt;
  }

  static bool SameFactors_(const std::vector<PrimExpr> &left,
                           std::vector<PrimExpr> right) {
    if (left.size() != right.size()) {
      return false;
    }
    for (const PrimExpr &factor : left) {
      auto it =
          std::find_if(right.begin(), right.end(), [&](const PrimExpr &other) {
            return ffi::StructuralEqual()(factor, other);
          });
      if (it == right.end()) {
        return false;
      }
      right.erase(it);
    }
    return true;
  }

  static void CollectFactors_(const PrimExpr &value,
                              std::vector<PrimExpr> *factors) {
    if (const auto *mul = value.as<MulNode>()) {
      CollectFactors_(mul->a, factors);
      CollectFactors_(mul->b, factors);
    } else {
      factors->push_back(value);
    }
  }
};

struct ArithmeticFacts {
  VarSet inputs;
  VarSet range_opaque;
  ffi::Map<Var, PrimExpr> aliases;
  ExprSet seen;
  ExprSet exact;
  ExprSet nonzero_divisor;
  ExprSet positive_divisor;
  ExprSet nonnegative;
  ExprSet bounded_remainder;
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

// Undo redundant coordinate decoding before div/rem become opaque intrinsics.
class LayoutDecodeSimplifier : public InvariantRemainderNormalizer {
public:
  explicit LayoutDecodeSimplifier(arith::Analyzer *analyzer)
      : InvariantRemainderNormalizer(analyzer) {}

  PrimExpr VisitExpr_(const AddNode *op) final {
    PrimExpr value = InvariantRemainderNormalizer::VisitExpr_(op);
    const auto *sum = value.as<AddNode>();
    if (!sum || SideEffect(value) != CallEffectKind::kPure) {
      return value;
    }
    PrimExpr left = analyzer_->Simplify(sum->a);
    PrimExpr right = analyzer_->Simplify(sum->b);
    if (auto result = Combine(left, right)) {
      return result.value();
    }
    if (auto result = Combine(right, left)) {
      return result.value();
    }
    return value;
  }

private:
  ffi::Optional<PrimExpr> Combine(const PrimExpr &product,
                                  const PrimExpr &remainder) {
    const auto *mul = product.as<MulNode>();
    const auto *mod = remainder.as<FloorModNode>();
    if (!mul || !mod) {
      return std::nullopt;
    }
    const auto *inner = mod->a.as<FloorDivNode>();
    if (!inner || !IsSupportedInteger(inner->dtype) ||
        !CanProvePositiveDivisor(inner->b, analyzer_) ||
        !CanProvePositiveDivisor(mod->b, analyzer_)) {
      return std::nullopt;
    }
    for (int i = 0; i < 2; ++i) {
      PrimExpr quotient = i == 0 ? mul->a : mul->b;
      PrimExpr radix = i == 0 ? mul->b : mul->a;
      const auto *outer = quotient.as<FloorDivNode>();
      if (!outer || !analyzer_->CanProveEqual(radix, mod->b) ||
          !analyzer_->CanProveEqual(outer->a, inner->a)) {
        continue;
      }
      PrimExpr combined = inner->b * mod->b;
      if (CanProvePositiveDivisor(combined, analyzer_) &&
          CanProvePositiveDivisor(outer->b, analyzer_) &&
          analyzer_->CanProveEqual(outer->b, combined)) {
        // q = floor(x/a); floor(x/(a*b))*b + q%b = q. Positive
        // denominators and non-wrapping products justify the factorization.
        return mod->a;
      }
    }
    return std::nullopt;
  }
};

// Reuse an existing coordinate quotient instead of preparing another product
// divisor. This changes the factorization, not the physical tensor layout.
class SharedLayoutQuotient : public arith::IRMutatorWithAnalyzer {
public:
  SharedLayoutQuotient(arith::Analyzer *analyzer, const Stmt &body)
      : IRMutatorWithAnalyzer(analyzer) {
    PostOrderVisit(body, [&](const ffi::ObjectRef &node) {
      if (const auto *div = node.as<FloorDivNode>()) {
        quotients_.emplace_back(div->a, div->b);
      }
    });
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto outer = std::move(remainders_);
    remainders_.clear();
    bool conditional = op->predicate.defined();
    // Only reuse within this load's address. A remainder in another statement
    // or a lazy arm does not justify changing the address's arithmetic recipe.
    for (const PrimExpr &index : op->indices) {
      PostOrderVisit(index, [&](const ffi::ObjectRef &node) {
        const auto *call = node.as<CallNode>();
        conditional |= node.as<SelectNode>() || node.as<LetNode>() ||
                       node.as<AndNode>() || node.as<OrNode>() ||
                       (call && call->op.same_as(builtin::if_then_else()));
        if (const auto *mod = node.as<FloorModNode>()) {
          PrimExpr expr = ffi::GetRef<PrimExpr>(mod);
          if (SideEffect(expr) == CallEffectKind::kPure &&
              std::none_of(remainders_.begin(), remainders_.end(),
                           [&](const PrimExpr &other) {
                             return ffi::StructuralEqual()(expr, other);
                           })) {
            remainders_.push_back(expr);
          }
        }
      });
    }
    if (conditional) {
      remainders_.clear();
    }
    PrimExpr value = IRMutatorWithAnalyzer::VisitExpr_(op);
    remainders_ = std::move(outer);
    return value;
  }

  PrimExpr VisitExpr_(const MulNode *op) final {
    PrimExpr value = IRMutatorWithAnalyzer::VisitExpr_(op);
    return InvariantRemainderNormalizer::ReuseRemainder(value, remainders_);
  }

  PrimExpr VisitExpr_(const FloorDivNode *op) final {
    PrimExpr value = IRMutatorWithAnalyzer::VisitExpr_(op);
    const auto *div = value.as<FloorDivNode>();
    if (!div || SideEffect(value) != CallEffectKind::kPure) {
      return value;
    }
    PrimExpr denominator = analyzer_->Simplify(div->b);
    const auto *product = denominator.as<MulNode>();
    if (!product || !IsSupportedInteger(denominator.dtype()) ||
        !CanProvePositiveDivisor(denominator, analyzer_)) {
      return value;
    }
    PrimExpr numerator = analyzer_->Simplify(div->a);
    for (int i = 0; i < 2; ++i) {
      PrimExpr factor = i == 0 ? product->a : product->b;
      PrimExpr remaining = i == 0 ? product->b : product->a;
      for (const auto &entry : quotients_) {
        if (ffi::StructuralEqual()(numerator,
                                   analyzer_->Simplify(entry.first)) &&
            ffi::StructuralEqual()(factor, analyzer_->Simplify(entry.second))) {
          return floordiv(floordiv(div->a, factor), remaining);
        }
      }
    }
    return value;
  }

private:
  std::vector<std::pair<PrimExpr, PrimExpr>> quotients_;
  std::vector<PrimExpr> remainders_;
};

// Collect proofs before rewriting predicates: replacing x % d in a condition
// must not hide the exact-divisibility fact from its dominated division sites.
class ArithmeticAnalyzer : public arith::IRMutatorWithAnalyzer {
public:
  ArithmeticAnalyzer(arith::Analyzer *analyzer, ArithmeticFacts *facts)
      : IRMutatorWithAnalyzer(analyzer), facts_(facts) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      const auto *div = op->value.as<FloorDivNode>();
      const auto *factor = div ? div->b.as<IntImmNode>() : nullptr;
      if (factor && factor->value > 0 && IsSupportedInteger(div->a.dtype()) &&
          div->a.dtype().bits() <= 32) {
        DataType type = div->a.dtype();
        // A launch extent is evaluated at its own width, unlike device index
        // intermediates that FlattenBuffer may widen. Even a wrapped numerator
        // cannot exceed its type's maximum. Do not use an unbounded signed
        // product's mathematical range to claim that it cannot wrap.
        int64_t maximum = (int64_t{1} << (type.bits() - type.is_int())) - 1;
        int64_t extent_upper = maximum / factor->value;
        if (extent_upper > 0) {
          IterVar thread = Downcast<IterVar>(op->node);
          DataType thread_type = thread->var.dtype();
          if (thread_type.bits() <= 32 &&
              extent_upper >
                  ((int64_t{1} << (thread_type.bits() - thread_type.is_int())) -
                   1)) {
            return IRMutatorWithAnalyzer::VisitStmt_(op);
          }
          With<arith::ConstraintContext> scope(
              analyzer_,
              thread->var < make_const(thread->var.dtype(), extent_upper));
          return IRMutatorWithAnalyzer::VisitStmt_(op);
        }
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    PrimExpr value = facts_->Resolve(op->value);
    if (facts_->HasOpaqueRange(op->value) ||
        HasUnprovenWrap_(value, analyzer_)) {
      facts_->range_opaque.insert(op->var);
    }
    if (facts_->CanPrepare(value)) {
      facts_->aliases.Set(op->var, value);
    }
    Stmt result = IRMutatorWithAnalyzer::VisitStmt_(op);
    arith::Analyzer type_analyzer;
    arith::Analyzer *range_analyzer =
        SideEffect(value) == CallEffectKind::kPure ? analyzer_ : &type_analyzer;
    if (!facts_->HasOpaqueRange(value) && FitsSigned32(value, range_analyzer)) {
      // Bind captures a value even when its RHS reads mutable memory. Retain
      // type-derived bounds on that captured value, not on subsequent reloads.
      auto bound = analyzer_->const_int_bound(op->var);
      analyzer_->const_int_bound.Update(
          op->var,
          arith::ConstIntBound(std::max(bound->min_value, int64_t{-2147483648}),
                               std::min(bound->max_value, int64_t{2147483647})),
          true);
    }
    return result;
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
  static bool HasUnprovenWrap_(const PrimExpr &expr,
                               arith::Analyzer *analyzer) {
    bool may_wrap = false;
    PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
      if (!node.as<AddNode>() && !node.as<SubNode>() && !node.as<MulNode>()) {
        return;
      }
      PrimExpr value = Downcast<PrimExpr>(node);
      if (!IsSupportedInteger(value.dtype())) {
        return;
      }
      auto bound = analyzer->const_int_bound(value);
      int bits = value.dtype().bits();
      int64_t minimum =
          value.dtype().is_uint()
              ? 0
              : (bits == 64 ? INT64_MIN : -(int64_t{1} << (bits - 1)));
      uint64_t maximum = value.dtype().is_int()
                             ? (uint64_t{1} << (bits - 1)) - 1
                             : (~uint64_t{0} >> (64 - bits));
      // Analyzer intervals can describe mathematical arithmetic past the
      // node's word width. Such bounds cannot prove the wrapped result's sign
      // or the range of a later widening cast.
      may_wrap |=
          bound->min_value == arith::ConstIntBoundNode::kNegInf ||
          bound->max_value == arith::ConstIntBoundNode::kPosInf ||
          bound->min_value < minimum ||
          (bound->max_value >= 0 && uint64_t(bound->max_value) > maximum);
    });
    return may_wrap;
  }

  bool CanReduceOnce(const PrimExpr &x, const PrimExpr &d) const {
    if (!CanProvePositiveDivisor(d, analyzer_)) {
      return false;
    }
    auto divisor_bound = analyzer_->const_int_bound(d);
    auto bound = analyzer_->const_int_bound(x);
    if (divisor_bound->min_value <= 0) {
      return false;
    }
    if (bound->min_value >= 0 &&
        bound->max_value != arith::ConstIntBoundNode::kPosInf &&
        uint64_t(bound->max_value) / 2 < uint64_t(divisor_bound->min_value)) {
      return true;
    }
    PrimExpr value = analyzer_->Simplify(x);
    const auto *call = value.as<CallNode>();
    if (!call || !call->op.same_as(builtin::bitwise_xor())) {
      return false;
    }
    for (int i = 0; i < 2; ++i) {
      PrimExpr base = call->args[i], mask = call->args[1 - i];
      auto mask_bound = analyzer_->const_int_bound(mask);
      // For nonnegative operands, base ^ mask <= base + mask. If each
      // operand is below d, the XOR is below 2*d without forming that product.
      if (mask_bound->min_value >= 0 &&
          mask_bound->max_value < divisor_bound->min_value &&
          analyzer_->CanProve(base >= 0) && analyzer_->CanProve(base < d)) {
        return true;
      }
    }
    return false;
  }

  bool FitsSigned32(const PrimExpr &expr, arith::Analyzer *analyzer) const {
    if (!IsSupportedInteger(expr.dtype())) {
      return false;
    }
    if (expr.dtype().bits() < 32 ||
        (expr.dtype().is_int() && expr.dtype().bits() == 32)) {
      return true;
    }
    if (!facts_->HasOpaqueRange(expr) && !HasUnprovenWrap_(expr, analyzer) &&
        (expr.dtype().is_uint() ||
         analyzer->CanProve(expr >=
                            make_const(expr.dtype(), int64_t{-2147483648}))) &&
        analyzer->CanProve(expr <=
                           make_const(expr.dtype(), int64_t{2147483647}))) {
      return true;
    }
    // For either floor or truncating remainder, a signed-int32 divisor bounds
    // every defined result to int32, even when the dividend is wide. Generic
    // interval analysis loses this bound when the divisor interval spans zero.
    if (const auto *mod = expr.as<FloorModNode>()) {
      return facts_->ResolveDivisor(mod->b).dtype() == DataType::Int(32);
    }
    if (const auto *mod = expr.as<ModNode>()) {
      return facts_->ResolveDivisor(mod->b).dtype() == DataType::Int(32);
    }
    if (const auto *call = expr.as<CallNode>();
        call && (call->op.same_as(builtin::bitwise_xor()) ||
                 call->op.same_as(builtin::bitwise_and()) ||
                 call->op.same_as(builtin::bitwise_or()))) {
      // Bitwise operations preserve sign extension when both operands fit.
      return FitsSigned32(call->args[0], analyzer) &&
             FitsSigned32(call->args[1], analyzer);
    }
    return false;
  }

  bool CanProveNonnegative_(const PrimExpr &expr,
                            arith::Analyzer *analyzer) const {
    if (expr.dtype().is_uint() ||
        (!facts_->HasOpaqueRange(expr) && !HasUnprovenWrap_(expr, analyzer) &&
         analyzer->CanProve(expr >= 0))) {
      return true;
    }
    // These operations bound their result independently of overflow in the
    // dividend. An opaque input range need not poison the entire decode chain.
    if (const auto *mod = expr.as<FloorModNode>()) {
      return !facts_->HasOpaqueRange(mod->b) &&
             CanProvePositiveDivisor(mod->b, analyzer);
    }
    if (const auto *call = expr.as<CallNode>()) {
      if (call->op.same_as(builtin::bitwise_and())) {
        return CanProveNonnegative_(call->args[0], analyzer) ||
               CanProveNonnegative_(call->args[1], analyzer);
      }
      if (call->op.same_as(builtin::bitwise_xor()) ||
          call->op.same_as(builtin::bitwise_or())) {
        return CanProveNonnegative_(call->args[0], analyzer) &&
               CanProveNonnegative_(call->args[1], analyzer);
      }
    }
    return false;
  }

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
    arith::Analyzer type_analyzer;
    arith::Analyzer *range_analyzer = stable ? analyzer_ : &type_analyzer;
    bool range_safe =
        !facts_->HasOpaqueRange(x) && !HasUnprovenWrap_(x, range_analyzer);
    bool divisor_safe = !facts_->HasOpaqueRange(divisor);
    bool exact = stable && range_safe && divisor_safe &&
                 !HasUnprovenWrap_(divisor, analyzer_) &&
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
    record_proof(&facts_->bounded_remainder, remainder && stable &&
                                                 range_safe && divisor_safe &&
                                                 CanReduceOnce(x, d));
    // Type-derived bounds remain valid for mutable loads. Predicate-derived
    // bounds are used only for stable values, just like exactness proofs.
    record_proof(&facts_->nonnegative, CanProveNonnegative_(x, range_analyzer));
    record_proof(&facts_->fits_signed32, FitsSigned32(x, range_analyzer));
    record_proof(&facts_->fits_unsigned32,
                 x.dtype().bits() <= 32 || facts_->fits_signed32.count(expr) ||
                     (range_safe &&
                      (x.dtype().is_uint() ||
                       range_analyzer->CanProve(
                           x >= make_const(x.dtype(), int64_t{-4294967295}))) &&
                      range_analyzer->CanProve(
                          x <= make_const(x.dtype(), uint64_t{4294967295}))));
    record_proof(&facts_->positive_divisor,
                 divisor_safe && CanProvePositiveDivisor(d, analyzer_));
    record_proof(&facts_->nonzero_divisor,
                 divisor_safe && analyzer_->CanProve(divisor != 0));
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
  PrimExpr PreserveHostWrap(const PrimExpr &value) const {
    DataType word = DataType::UInt(value.dtype().bits());
#define TL_UNSIGNED_HOST_ARITH(Node, Operator)                                 \
  if (const auto *op = value.as<Node>()) {                                     \
    return cast(value.dtype(),                                                 \
                cast(word, PreserveHostWrap(op->a))                            \
                    Operator cast(word, PreserveHostWrap(op->b)));             \
  }
    TL_UNSIGNED_HOST_ARITH(AddNode, +)
    TL_UNSIGNED_HOST_ARITH(SubNode, -)
    TL_UNSIGNED_HOST_ARITH(MulNode, *)
#undef TL_UNSIGNED_HOST_ARITH
    if (const auto *op = value.as<CastNode>()) {
      return cast(op->dtype, PreserveHostWrap(op->value));
    }
    if (const auto *op = value.as<MinNode>()) {
      return min(PreserveHostWrap(op->a), PreserveHostWrap(op->b));
    }
    if (const auto *op = value.as<MaxNode>()) {
      return max(PreserveHostWrap(op->a), PreserveHostWrap(op->b));
    }
    return value;
  }

  PrimExpr Magnitude(const PrimExpr &value) const {
    DataType word = DataType::UInt(value.dtype().bits());
    PrimExpr bits = cast(word, PreserveHostWrap(value));
    if (value.dtype().is_uint()) {
      return bits;
    }
    // Read the sign bit after wrapping. A signed comparison can be folded
    // incorrectly from positivity of the factors even when their product wraps.
    PrimExpr sign =
        make_zero(word) - (bits >> make_const(word, word.bits() - 1));
    return bitwise_xor(bits, sign) - sign;
  }

  PrimExpr FastDiv(const PrimExpr &x, const PrimExpr &d, bool remainder,
                   const PrimExpr &valid, bool truncating, bool nonnegative,
                   bool positive_divisor) {
    DataType i32 = DataType::Int(32);
    DataType u32 = DataType::UInt(32);
    DataType u64 = DataType::UInt(64);
    Var safe =
        plan_->Prepare(max(Magnitude(d), make_const(u32, 1)), "fastdiv_d");
    Var k = plan_->Prepare(
        32 - cast(i32, clz(cast(u32, max(safe, make_const(u32, 2)) - 1))),
        "fastdiv_k");
    Var shift = plan_->Prepare(max(k - 1, make_const(i32, 0)), "fastdiv_shift");
    PrimExpr power = make_const(u64, 1) << cast(u64, shift + 32);
    Var multiplier = plan_->Prepare(
        cast(u32, floordiv(power + cast(u64, safe) - 1, cast(u64, safe))),
        "fastdiv_multiplier");
    return Call(x.dtype(), remainder ? tl::fast_rem() : tl::fast_div(),
                {x, d, multiplier, shift, valid, Bool(truncating),
                 Bool(nonnegative), Bool(positive_divisor)});
  }

  // mu = floor(2^word_bits / abs(d)). For a magnitude in that word's range,
  // the quotient is at most one too small, even if abs(d) is wider than the
  // word.
  PrimExpr BarrettReduction(const PrimExpr &x, const PrimExpr &d,
                            bool remainder, const PrimExpr &valid,
                            bool truncating, int word_bits, bool nonnegative,
                            bool positive_divisor) {
    DataType word = DataType::UInt(word_bits);
    DataType u64 = DataType::UInt(64);
    PrimExpr magnitude = Magnitude(d);
    PrimExpr safe = cast(u64, max(magnitude, make_const(magnitude.dtype(), 1)));
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
    // Widen only the device operand, after canonical host preparation. This
    // preserves sharing with equivalent narrow uses and selects the Barrett
    // helper rather than the signed-int32 magic-divisor overload.
    PrimExpr device_d =
        word_bits == 64 || (!remainder && d.dtype() == DataType::Int(32) &&
                            x.dtype().bits() == 64)
            ? cast(x.dtype(), d)
            : d;
    if (remainder) {
      return Call(x.dtype(), tl::barrett_reduce(),
                  {x, device_d, reciprocal, valid, Bool(truncating),
                   Bool(nonnegative), Bool(positive_divisor)});
    }
    return Call(x.dtype(), tl::fast_div(),
                {x, device_d, reciprocal, make_const(DataType::Int(32), 0),
                 valid, Bool(truncating), Bool(nonnegative),
                 Bool(positive_divisor)});
  }

  // For an exactly divisible x, remove d's power-of-two factor and multiply
  // by the inverse of its odd part modulo 2^32. Five Newton steps from 1
  // double the number of correct bits each time (1 -> 2 -> ... -> 32).
  PrimExpr ExactDiv(const PrimExpr &x, const PrimExpr &d,
                    const PrimExpr &valid) {
    DataType u32 = DataType::UInt(32);
    Var safe = plan_->Prepare(max(cast(u32, Magnitude(d)), make_const(u32, 1)),
                              "exact_d");
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
    if (remainder && facts_.bounded_remainder.count(site)) {
      return cast(x.dtype(), Call(compute_type, tl::bounded_rem(),
                                  {value, cast(compute_type, d)}));
    }
    bool exact = facts_.exact.count(site) && compute_type.bits() == 32;
    // Select the algorithm from the dividend range before preparing parameters.
    // A narrow divisor does not imply a narrow quotient or dividend.
    bool fast32 = d.dtype() == DataType::Int(32) && facts_.HasFastDivisor(d) &&
                  facts_.fits_signed32.count(site);
    // Every nonzero divisor is handled by the selected magnitude algorithm.
    // Keep only the original zero-divisor behavior, not a per-lane sign guard.
    PrimExpr valid = facts_.nonzero_divisor.count(site) ? const_true() : d != 0;
    PrimExpr result;
    if (exact) {
      result = remainder ? make_zero(compute_type) : ExactDiv(value, d, valid);
    } else if (fast32) {
      result = FastDiv(value, d, remainder, valid, truncating,
                       facts_.nonnegative.count(site),
                       facts_.positive_divisor.count(site));
    } else {
      int word_bits = facts_.fits_unsigned32.count(site) ? 32 : 64;
      result = BarrettReduction(value, d, remainder, valid, truncating,
                                word_bits, facts_.nonnegative.count(site),
                                facts_.positive_divisor.count(site));
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
    if (op->attr_key == tirx::attr::tilelang_assume) {
      assumptions_.push_back(ffi::GetRef<AttrStmt>(op));
      Stmt result = StmtMutator::VisitStmt_(op);
      assumptions_.pop_back();
      return result;
    }
    if (op->attr_key != tvm::attr::kTarget) {
      return StmtMutator::VisitStmt_(op);
    }
    LaunchPlan plan;
    ArithmeticFacts facts;
    facts.inputs = inputs_;
    // Shape assumptions can enclose the device target rather than occur in
    // its body. Preserve that context for proofs, without moving evaluation.
    Stmt context_body = op->body;
    for (auto it = assumptions_.rbegin(); it != assumptions_.rend(); ++it) {
      context_body =
          AttrStmt((*it)->node, (*it)->attr_key, (*it)->value, context_body);
    }
    arith::Analyzer layout_analyzer;
    Stmt simplified = LayoutDecodeSimplifier(&layout_analyzer)(context_body);
    arith::Analyzer quotient_analyzer;
    simplified =
        SharedLayoutQuotient(&quotient_analyzer, simplified)(simplified);
    arith::Analyzer analyzer;
    ArithmeticAnalyzer(&analyzer, &facts)(simplified);
    for (size_t i = 0; i < assumptions_.size(); ++i) {
      simplified = Downcast<AttrStmt>(simplified)->body;
    }
    Stmt body = InvariantArithmeticRewriter(facts, &plan)(simplified);
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
  std::vector<AttrStmt> assumptions_;
};

// Keep arithmetic opaque through symbolic simplification, then share calls only
// inside the statement/branch where they execute. Never extract fallback math.
bool IsInvariantArithmetic(const PrimExpr &expr) {
  const auto *call = expr.as<CallNode>();
  return call && (call->op.same_as(tl::fast_div()) ||
                  call->op.same_as(tl::fast_rem()) ||
                  call->op.same_as(tl::barrett_reduce()) ||
                  call->op.same_as(tl::exact_div()) ||
                  call->op.same_as(tl::bounded_rem()));
}

class ArithmeticCallBinder : public ExprMutator {
public:
  using Values = std::vector<std::pair<PrimExpr, Var>>;

  explicit ArithmeticCallBinder(Values values) : values_(std::move(values)) {}

  const Values &GetValues() const { return values_; }

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
  Values values_;
  ffi::Array<Stmt> bindings_;
};

class InvariantArithmeticMaterializer : public StmtMutator {
public:
  explicit InvariantArithmeticMaterializer(
      ArithmeticCallBinder::Values values = {})
      : values_(std::move(values)) {}

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    return MaterializeCondition(op->condition, op->then_case, op->else_case,
                                op->span);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    ArithmeticCallBinder binder(values_);
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
      Stmt yes = BufferStore(op->buffer, call->args[1], op->indices,
                             op->predicate, op->span);
      Stmt no = BufferStore(op->buffer, call->args[2], op->indices,
                            op->predicate, op->span);
      return MaterializeCondition(call->args[0], yes, no, op->span, true);
    }
    ArithmeticCallBinder binder(values_);
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
                            const ffi::Optional<Stmt> &no, const Span &span,
                            bool split_store_guard = false) {
    if (!ContainsArithmetic(condition)) {
      return IfThenElse(condition, VisitStmt(yes),
                        no.defined()
                            ? ffi::Optional<Stmt>(VisitStmt(no.value()))
                            : std::nullopt,
                        span);
    }
    // Only split a flat two-term guard around a single store. Do not expand
    // arbitrary Boolean trees: duplicating their branches can grow
    // exponentially.
    if (const auto *op = condition.as<AndNode>();
        split_store_guard && op && !op->a.as<AndNode>() &&
        !op->a.as<OrNode>() && !op->b.as<AndNode>() && !op->b.as<OrNode>()) {
      return MaterializeCondition(op->a, IfThenElse(op->b, yes, no, span), no,
                                  span);
    }
    ArithmeticCallBinder binder(values_);
    PrimExpr value = binder(condition);
    // Reuse only pure values already evaluated by a dominating condition.
    // Branch-local bindings never escape to siblings or later statements.
    InvariantArithmeticMaterializer nested(binder.GetValues());
    return binder.Materialize(IfThenElse(
        value, nested(yes),
        no.defined() ? ffi::Optional<Stmt>(nested(no.value())) : std::nullopt,
        span));
  }

  ArithmeticCallBinder::Values values_;
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
