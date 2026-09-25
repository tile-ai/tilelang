/*!
 * \file lower_invariant_arithmetic.cc
 * \brief Host-prepared integer arithmetic for launch-invariant divisors.
 */
#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/product_normal_form.h"
#include "common/attr.h"
#include "common/launch_plan.h"

namespace tvm {
namespace tl {
using namespace tirx;

TVM_REGISTER_PASS_CONFIG_OPTION("tl.enable_invariant_arithmetic", Bool);

namespace {
using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

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

bool FitsInType(const arith::ConstIntBound &bound, DataType dtype) {
  int bits = dtype.bits();
  int64_t minimum = dtype.is_uint()
                        ? 0
                        : (bits == 64 ? std::numeric_limits<int64_t>::min()
                                      : -(int64_t{1} << (bits - 1)));
  uint64_t maximum =
      dtype.is_int() ? (uint64_t{1} << (bits - 1)) - 1
                     : (std::numeric_limits<uint64_t>::max() >> (64 - bits));
  return bound->min_value != arith::ConstIntBoundNode::kNegInf &&
         bound->max_value != arith::ConstIntBoundNode::kPosInf &&
         bound->min_value >= minimum &&
         (bound->max_value < 0 || uint64_t(bound->max_value) <= maximum);
}

// Analyzer and its Z3 prover use mathematical integers. Only ask them for word
// properties when each intermediate and cast agrees with that interpretation.
// Bound-preserving casts are legal even when the source type itself is wider.
bool HasUnprovenWrap(const PrimExpr &expr, arith::Analyzer *analyzer,
                     const VarSet &opaque_vars = {}) {
  bool may_wrap = false;
  PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    if (may_wrap) {
      return;
    }
    if (const auto *var = node.as<VarNode>()) {
      may_wrap = opaque_vars.count(ffi::GetRef<Var>(var));
    } else if (const auto *cast = node.as<CastNode>()) {
      may_wrap =
          !IsSupportedInteger(cast->value.dtype()) ||
          !IsSupportedInteger(cast->dtype) ||
          (cast->dtype != cast->value.dtype() &&
           !IsValuePreservingWiden(cast->value.dtype(), cast->dtype) &&
           !FitsInType(analyzer->const_int_bound(cast->value), cast->dtype));
    } else if (node.as<AddNode>() || node.as<SubNode>() || node.as<MulNode>()) {
      PrimExpr value = Downcast<PrimExpr>(node);
      may_wrap = !IsSupportedInteger(value.dtype()) ||
                 !FitsInType(analyzer->const_int_bound(value), value.dtype());
    }
  });
  return may_wrap;
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
  if (expr.as<AddNode>() || expr.as<SubNode>() || expr.as<MulNode>()) {
    return !HasUnprovenWrap(expr, analyzer) &&
           analyzer->const_int_bound(expr)->min_value > 0;
  }
  return false;
}

// Remainder canonicalization and factor matching substantially follow the
// approach in penguin-wwy's PR #3267. Apply these identities after index
// widening, before introducing opaque arithmetic intrinsics.
class LayoutArithmeticSimplifier : public arith::IRMutatorWithAnalyzer {
public:
  explicit LayoutArithmeticSimplifier(arith::Analyzer *analyzer)
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

  PrimExpr VisitExpr_(const AddNode *op) final {
    PrimExpr value = StmtExprMutator::VisitExpr_(op);
    const auto *add = value.as<AddNode>();
    if (!add || SideEffect(value) != CallEffectKind::kPure) {
      return value;
    }
    if (IsSupportedInteger(value.dtype())) {
      for (int i = 0; i < 2; ++i) {
        PrimExpr product = i == 0 ? add->a : add->b;
        PrimExpr remainder = i == 0 ? add->b : add->a;
        if (const auto *mod = remainder.as<FloorModNode>()) {
          auto divisor = MatchProduct_(product, mod->a);
          if (divisor && ffi::StructuralEqual()(divisor.value(), mod->b)) {
            // q*d + r = x also holds under signed word wrapping.
            value = mod->a;
            break;
          }
        }
      }
    }
    const auto *sum = value.as<AddNode>();
    if (!sum) {
      return value;
    }
    PrimExpr left = analyzer_->Simplify(sum->a);
    PrimExpr right = analyzer_->Simplify(sum->b);
    if (auto result = Combine_(left, right)) {
      return result.value();
    }
    if (auto result = Combine_(right, left)) {
      return result.value();
    }
    return value;
  }

private:
  ffi::Optional<PrimExpr> Combine_(const PrimExpr &product,
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

  // Simplification can interleave a swizzle's offsets with x - (x/d)*d.
  // Recover the remainder without distributing products or crossing casts.
  ffi::Optional<PrimExpr>
  RecoverRemainder_(const PrimExpr &value,
                    ffi::Optional<PrimExpr> modulus = std::nullopt) {
    std::vector<std::pair<PrimExpr, bool>> terms;
    arith::UnpackSum(value, [&](const PrimExpr &term, int sign) {
      terms.emplace_back(term, sign < 0);
    });
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
        arith::UnpackSum(div->a, [&](const PrimExpr &term, int sign) {
          dividend.emplace_back(term, sign < 0);
        });
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
    if (SameFactors_(left, right)) {
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
      if (SameFactors_(remaining, divisor_factors)) {
        return quotient->b;
      }
    }
    return std::nullopt;
  }

  static bool SameFactors_(const std::vector<PrimExpr> &left,
                           const std::vector<PrimExpr> &right) {
    return std::is_permutation(left.begin(), left.end(), right.begin(),
                               right.end(), ffi::StructuralEqual());
  }

  static void CollectFactors_(const PrimExpr &value,
                              std::vector<PrimExpr> *factors) {
    arith::UnpackReduction<MulNode>(
        value, [&](const PrimExpr &factor) { factors->push_back(factor); });
  }
};

enum DivisionProperty : unsigned {
  kNonzeroDivisor = 1 << 0,
  kPositiveDivisor = 1 << 1,
  kNonnegativeDividend = 1 << 2,
  kBoundedRemainder = 1 << 3,
  kSigned32Dividend = 1 << 4,
  kUnsigned32Magnitude = 1 << 5,
  kAllDivisionProperties = (1 << 6) - 1,
};

struct ArithmeticFacts {
  VarSet inputs;
  ffi::Map<Var, PrimExpr> aliases;
  std::unordered_map<PrimExpr, unsigned, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      properties;
  std::vector<PrimExpr> fast_divisors;
  std::vector<PrimExpr> product_divisors;

  PrimExpr Resolve(const PrimExpr &expr) const {
    return Substitute(expr, aliases);
  }

  PrimExpr ResolveDivisor(const PrimExpr &expr) const {
    PrimExpr d = Resolve(expr);
    while (const auto *cast = d.as<CastNode>()) {
      if (cast->value.dtype() != cast->dtype &&
          !IsValuePreservingWiden(cast->value.dtype(), cast->dtype)) {
        break;
      }
      d = cast->value;
    }
    if (IsSupportedInteger(d.dtype()) && d.dtype().bits() < 32) {
      d = cast(d.dtype().is_int() ? DataType::Int(32) : DataType::UInt(32), d);
    }
    // Reassociate only multiplication in one integer word. Casts remain
    // indivisible factors: widening a wrapped product is not the same as
    // multiplying widened operands. Reuse an existing expression rather than
    // asking the mathematical-integer analyzer to prove word equivalence.
    if (d.as<MulNode>()) {
      std::vector<PrimExpr> factors;
      arith::UnpackReduction<MulNode>(
          d, [&](const PrimExpr &factor) { factors.push_back(factor); });
      for (const PrimExpr &other : product_divisors) {
        if (other.dtype() != d.dtype()) {
          continue;
        }
        std::vector<PrimExpr> other_factors;
        arith::UnpackReduction<MulNode>(other, [&](const PrimExpr &factor) {
          other_factors.push_back(factor);
        });
        if (std::is_permutation(factors.begin(), factors.end(),
                                other_factors.begin(), other_factors.end(),
                                ffi::StructuralEqual())) {
          return other;
        }
      }
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

  bool HasFastDivisor(const PrimExpr &d) const {
    for (const PrimExpr &other : fast_divisors) {
      if (ffi::StructuralEqual()(d, other)) {
        return true;
      }
    }
    return false;
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
    return LayoutArithmeticSimplifier::ReuseRemainder(value, remainders_);
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

// Collect sign and range proofs before rewriting their supporting predicates.
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
    if (HasUnprovenWrap(value, analyzer_, range_opaque_)) {
      range_opaque_.insert(op->var);
    }
    if (facts_->CanPrepare(value)) {
      facts_->aliases.Set(op->var, value);
    }
    Stmt result = IRMutatorWithAnalyzer::VisitStmt_(op);
    arith::Analyzer type_analyzer;
    arith::Analyzer *range_analyzer =
        SideEffect(value) == CallEffectKind::kPure ? analyzer_ : &type_analyzer;
    if (!HasUnprovenWrap(value, range_analyzer, range_opaque_) &&
        FitsSigned32_(value, range_analyzer)) {
      // Bind captures a value even when its RHS reads mutable memory. Retain
      // type-derived bounds on that captured value, not on subsequent reloads.
      auto bound = analyzer_->const_int_bound(op->var);
      analyzer_->const_int_bound.Update(
          op->var,
          arith::ConstIntBound(
              std::max<int64_t>(bound->min_value,
                                std::numeric_limits<int32_t>::min()),
              std::min<int64_t>(bound->max_value,
                                std::numeric_limits<int32_t>::max())),
          true);
    }
    return result;
  }

#define TL_ANALYZE_DIVMOD(Node, is_remainder)                                  \
  PrimExpr VisitExpr_(const Node *op) final {                                  \
    Record_(ffi::GetRef<PrimExpr>(op), op->a, op->b, is_remainder);            \
    return IRMutatorWithAnalyzer::VisitExpr_(op);                              \
  }
  TL_ANALYZE_DIVMOD(FloorDivNode, false)
  TL_ANALYZE_DIVMOD(FloorModNode, true)
  TL_ANALYZE_DIVMOD(DivNode, false)
  TL_ANALYZE_DIVMOD(ModNode, true)
#undef TL_ANALYZE_DIVMOD

private:
  VarSet range_opaque_;

  bool CanReduceOnce_(const PrimExpr &x, const PrimExpr &d) const {
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
    if (!call || !(call->op.same_as(builtin::bitwise_xor()) ||
                   call->op.same_as(builtin::bitwise_or()))) {
      return false;
    }
    for (int i = 0; i < 2; ++i) {
      PrimExpr base = call->args[i], mask = call->args[1 - i];
      auto mask_bound = analyzer_->const_int_bound(mask);
      // For nonnegative operands, XOR and OR are bounded by their sum.
      // If each operand is below d, the result is below 2*d without
      // forming that product.
      if (mask_bound->min_value >= 0 &&
          mask_bound->max_value < divisor_bound->min_value &&
          analyzer_->CanProve(base >= 0) && analyzer_->CanProve(base < d)) {
        return true;
      }
    }
    return false;
  }

  bool FitsSigned32_(const PrimExpr &expr, arith::Analyzer *analyzer) const {
    if (!IsSupportedInteger(expr.dtype())) {
      return false;
    }
    if (expr.dtype().bits() < 32 ||
        (expr.dtype().is_int() && expr.dtype().bits() == 32)) {
      return true;
    }
    if (!HasUnprovenWrap(expr, analyzer, range_opaque_) &&
        (expr.dtype().is_uint() ||
         analyzer->CanProve(
             expr >=
             make_const(expr.dtype(), std::numeric_limits<int32_t>::min()))) &&
        analyzer->CanProve(
            expr <=
            make_const(expr.dtype(), std::numeric_limits<int32_t>::max()))) {
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
      return FitsSigned32_(call->args[0], analyzer) &&
             FitsSigned32_(call->args[1], analyzer);
    }
    return false;
  }

  bool CanProveNonnegative_(const PrimExpr &expr,
                            arith::Analyzer *analyzer) const {
    if (expr.dtype().is_uint() ||
        (!HasUnprovenWrap(expr, analyzer, range_opaque_) &&
         analyzer->CanProve(expr >= 0))) {
      return true;
    }
    // These operations bound their result independently of overflow in the
    // dividend. An opaque input range need not poison the entire decode chain.
    if (const auto *mod = expr.as<FloorModNode>()) {
      return !HasUnprovenWrap(mod->b, analyzer, range_opaque_) &&
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

  void Record_(const PrimExpr &expr, const PrimExpr &x, const PrimExpr &divisor,
               bool remainder) {
    PrimExpr d = facts_->ResolveDivisor(divisor);
    if (!facts_->CanPrepare(d) || d.as<IntImmNode>() ||
        !IsSupportedInteger(x.dtype())) {
      return;
    }
    if (d.as<MulNode>() && std::none_of(facts_->product_divisors.begin(),
                                        facts_->product_divisors.end(),
                                        [&](const PrimExpr &other) {
                                          return ffi::StructuralEqual()(d,
                                                                        other);
                                        })) {
      facts_->product_divisors.push_back(d);
    }
    // Only stable scalar expressions can inherit bounds from a predicate.
    // Re-reading a mutable buffer is not the same value as its earlier load.
    bool stable = SideEffect(x) == CallEffectKind::kPure;
    arith::Analyzer type_analyzer;
    arith::Analyzer *range_analyzer = stable ? analyzer_ : &type_analyzer;
    bool range_safe = !HasUnprovenWrap(x, range_analyzer, range_opaque_);
    bool divisor_safe = !HasUnprovenWrap(divisor, analyzer_, range_opaque_);
    auto entry =
        facts_->properties.try_emplace(expr, kAllDivisionProperties).first;
    unsigned &properties = entry->second;
    auto retain = [&](DivisionProperty property, bool proven) {
      // A shared IR node must satisfy the property at every occurrence.
      if (!proven) {
        properties &= ~property;
      }
    };
    retain(kBoundedRemainder, remainder && stable && range_safe &&
                                  divisor_safe && CanReduceOnce_(x, d));
    // Mutable loads may use type bounds, but not predicates on an earlier read.
    retain(kNonnegativeDividend, CanProveNonnegative_(x, range_analyzer));
    retain(kSigned32Dividend, FitsSigned32_(x, range_analyzer));
    retain(
        kUnsigned32Magnitude,
        x.dtype().bits() <= 32 || (properties & kSigned32Dividend) ||
            (range_safe &&
             (x.dtype().is_uint() ||
              range_analyzer->CanProve(
                  x >= make_const(
                           x.dtype(),
                           -int64_t{std::numeric_limits<uint32_t>::max()}))) &&
             range_analyzer->CanProve(
                 x <=
                 make_const(x.dtype(), std::numeric_limits<uint32_t>::max()))));
    retain(kPositiveDivisor,
           divisor_safe && CanProvePositiveDivisor(d, analyzer_));
    retain(kNonzeroDivisor, divisor_safe && analyzer_->CanProve(divisor != 0));
    if (!remainder && d.dtype() == DataType::Int(32) &&
        !facts_->HasFastDivisor(d)) {
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
    return Rewrite_(ffi::GetRef<PrimExpr>(op), VisitExpr(op->a),               \
                    VisitExpr(op->b), remainder, truncating);                  \
  }
  TL_REWRITE_DIVMOD(FloorDivNode, false, false)
  TL_REWRITE_DIVMOD(FloorModNode, true, false)
  TL_REWRITE_DIVMOD(DivNode, false, true)
  TL_REWRITE_DIVMOD(ModNode, true, true)
#undef TL_REWRITE_DIVMOD

private:
  PrimExpr PreserveWordWrap_(const PrimExpr &value) const {
    DataType word = DataType::UInt(value.dtype().bits());
#define TL_UNSIGNED_WORD_ARITH(Node, Operator)                                 \
  if (const auto *op = value.as<Node>()) {                                     \
    return cast(value.dtype(),                                                 \
                cast(word, PreserveWordWrap_(op->a))                           \
                    Operator cast(word, PreserveWordWrap_(op->b)));            \
  }
    TL_UNSIGNED_WORD_ARITH(AddNode, +)
    TL_UNSIGNED_WORD_ARITH(SubNode, -)
    TL_UNSIGNED_WORD_ARITH(MulNode, *)
#undef TL_UNSIGNED_WORD_ARITH
    if (const auto *op = value.as<CastNode>()) {
      return cast(op->dtype, PreserveWordWrap_(op->value));
    }
    if (const auto *op = value.as<MinNode>()) {
      return min(PreserveWordWrap_(op->a), PreserveWordWrap_(op->b));
    }
    if (const auto *op = value.as<MaxNode>()) {
      return max(PreserveWordWrap_(op->a), PreserveWordWrap_(op->b));
    }
    return value;
  }

  PrimExpr Magnitude_(const PrimExpr &value) const {
    DataType word = DataType::UInt(value.dtype().bits());
    PrimExpr bits = cast(word, PreserveWordWrap_(value));
    if (value.dtype().is_uint()) {
      return bits;
    }
    // Read the sign bit after wrapping. A signed comparison can be folded
    // incorrectly from positivity of the factors even when their product wraps.
    PrimExpr sign =
        make_zero(word) - (bits >> make_const(word, word.bits() - 1));
    return bitwise_xor(bits, sign) - sign;
  }

  PrimExpr FastDiv_(const PrimExpr &x, const PrimExpr &d, bool remainder,
                    const PrimExpr &valid, bool truncating, bool nonnegative,
                    bool positive_divisor) {
    DataType i32 = DataType::Int(32);
    DataType u32 = DataType::UInt(32);
    DataType u64 = DataType::UInt(64);
    Var safe =
        plan_->Prepare(max(Magnitude_(d), make_const(u32, 1)), "fastdiv_d");
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
  PrimExpr BarrettReduction_(const PrimExpr &x, const PrimExpr &d,
                             bool remainder, const PrimExpr &valid,
                             bool truncating, int word_bits, bool nonnegative,
                             bool positive_divisor) {
    DataType word = DataType::UInt(word_bits);
    DataType u64 = DataType::UInt(64);
    PrimExpr magnitude = Magnitude_(d);
    PrimExpr safe = cast(u64, max(magnitude, make_const(magnitude.dtype(), 1)));
    PrimExpr reciprocal_expr;
    if (word.bits() == 64) {
      // floor(2^64 / d), represented modulo 2^64 for d == 1. Compute
      // via UINT64_MAX so neither host int128 nor an overflowing literal is
      // required. The device handles d == 1 separately.
      PrimExpr top = make_const(u64, std::numeric_limits<uint64_t>::max());
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

  PrimExpr Rewrite_(const PrimExpr &site, const PrimExpr &x,
                    const PrimExpr &divisor, bool remainder, bool truncating) {
    auto found = facts_.properties.find(site);
    if (found == facts_.properties.end()) {
      return truncating
                 ? (remainder ? truncmod(x, divisor) : truncdiv(x, divisor))
                 : (remainder ? floormod(x, divisor) : floordiv(x, divisor));
    }
    unsigned properties = found->second;
    PrimExpr d = facts_.ResolveDivisor(divisor);
    // The operation type is independent of the reciprocal's word size.
    // Narrow operations use CUDA's scalar 32-bit arithmetic and cast the
    // result back; wide operations retain their original fallback width.
    DataType compute_type =
        x.dtype().bits() < 32
            ? (x.dtype().is_int() ? DataType::Int(32) : DataType::UInt(32))
            : x.dtype();
    // Preserve evaluated-word semantics when the helper is inlined by C++
    // codegen. Signed overflow in a swizzle recipe must not erase its sign.
    PrimExpr value = cast(compute_type, PreserveWordWrap_(x));
    if (remainder && (properties & kBoundedRemainder)) {
      return cast(x.dtype(), Call(compute_type, tl::bounded_rem(),
                                  {value, cast(compute_type, d)}));
    }
    // Select the algorithm from the dividend range before preparing parameters.
    // A narrow divisor does not imply a narrow quotient or dividend.
    bool fast32 = d.dtype() == DataType::Int(32) && facts_.HasFastDivisor(d) &&
                  (properties & kSigned32Dividend);
    // Every nonzero divisor is handled by the selected magnitude algorithm.
    // Keep only the original zero-divisor behavior, not a per-lane sign guard.
    PrimExpr valid = (properties & kNonzeroDivisor) ? const_true() : d != 0;
    PrimExpr result;
    if (fast32) {
      result = FastDiv_(value, d, remainder, valid, truncating,
                        (properties & kNonnegativeDividend),
                        (properties & kPositiveDivisor));
    } else {
      int word_bits = (properties & kUnsigned32Magnitude) ? 32 : 64;
      result = BarrettReduction_(value, d, remainder, valid, truncating,
                                 word_bits, (properties & kNonnegativeDividend),
                                 (properties & kPositiveDivisor));
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
    Stmt simplified =
        LayoutArithmeticSimplifier(&layout_analyzer)(context_body);
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
                  call->op.same_as(tl::bounded_rem()));
}

class ArithmeticCallBinder : public ExprMutator {
public:
  explicit ArithmeticCallBinder(ffi::Array<Bind> bindings)
      : bindings_(std::move(bindings)), inherited_count_(bindings_.size()) {}

  const ffi::Array<Bind> &GetBindings() const { return bindings_; }

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
    for (const Bind &binding : bindings_) {
      if (ffi::StructuralEqual()(binding->value, value)) {
        return binding->var;
      }
    }
    Var var("invariant_value", value.dtype());
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

  Stmt Materialize(const Stmt &stmt) const {
    ffi::Array<Stmt> statements;
    for (size_t i = inherited_count_; i < bindings_.size(); ++i) {
      statements.push_back(bindings_[i]);
    }
    statements.push_back(stmt);
    return SeqStmt::Flatten(statements);
  }

private:
  ffi::Array<Bind> bindings_;
  size_t inherited_count_;
};

class InvariantArithmeticMaterializer : public StmtMutator {
public:
  explicit InvariantArithmeticMaterializer(ffi::Array<Bind> bindings = {})
      : bindings_(std::move(bindings)) {}

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    return MaterializeCondition_(op->condition, op->then_case, op->else_case,
                                 op->span);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    ArithmeticCallBinder binder(bindings_);
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
        ContainsArithmetic_(op->value)) {
      Stmt yes = BufferStore(op->buffer, call->args[1], op->indices,
                             op->predicate, op->span);
      Stmt no = BufferStore(op->buffer, call->args[2], op->indices,
                            op->predicate, op->span);
      return MaterializeCondition_(call->args[0], yes, no, op->span, true);
    }
    ArithmeticCallBinder binder(bindings_);
    PrimExpr value = binder(op->value);
    return binder.Materialize(
        BufferStore(op->buffer, value, op->indices, op->predicate, op->span));
  }

private:
  static bool ContainsArithmetic_(const PrimExpr &expr) {
    bool found = false;
    PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        found |= IsInvariantArithmetic(ffi::GetRef<PrimExpr>(call));
      }
    });
    return found;
  }

  Stmt MaterializeCondition_(const PrimExpr &condition, const Stmt &yes,
                             const ffi::Optional<Stmt> &no, const Span &span,
                             bool split_store_guard = false) {
    if (!ContainsArithmetic_(condition)) {
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
      return MaterializeCondition_(op->a, IfThenElse(op->b, yes, no, span), no,
                                   span);
    }
    ArithmeticCallBinder binder(bindings_);
    PrimExpr value = binder(condition);
    // Reuse only pure values already evaluated by a dominating condition.
    // Branch-local bindings never escape to siblings or later statements.
    InvariantArithmeticMaterializer nested(binder.GetBindings());
    return binder.Materialize(IfThenElse(
        value, nested(yes),
        no.defined() ? ffi::Optional<Stmt>(nested(no.value())) : std::nullopt,
        span));
  }

  ffi::Array<Bind> bindings_;
};
} // namespace

namespace transform {
tvm::transform::Pass LowerInvariantArithmetic(const ffi::String &stage) {
  if (stage != "prepare" && stage != "materialize") {
    TVM_FFI_THROW(ValueError)
        << "LowerInvariantArithmetic stage must be 'prepare' or 'materialize', "
           "got '"
        << stage << "'";
  }
  bool materialize = stage == "materialize";
  auto pass_func = [materialize](PrimFunc func, const IRModule &,
                                 const tvm::transform::PassContext &) {
    Stmt body = materialize ? InvariantArithmeticMaterializer()(func->body)
                            : InvariantArithmeticPlanner(func)(func->body);
    func.CopyOnWrite()->body = std::move(body);
    return func;
  };
  tvm::transform::Pass pass = tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.LowerInvariantArithmetic." + stage, {});
  if (!materialize) {
    return pass;
  }
  // Splitting short-circuit conditions can duplicate branch-local bindings.
  return tvm::transform::Sequential({pass, tirx::transform::ConvertSSA()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerInvariantArithmetic",
                                   LowerInvariantArithmetic);
}
} // namespace transform
} // namespace tl
} // namespace tvm
