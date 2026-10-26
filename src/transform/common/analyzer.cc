#include "compat/analyzer.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/expr_functor.h>

#include "op/prim.h"
#include "transform/common/data_type_rewriter.h"

#include <bit>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl::arith {
namespace {

using namespace tvm::prim;
constexpr int64_t kPositiveInfinity = sym::ConstIntBound::kPosInf;
constexpr int64_t kNegativeInfinity = sym::ConstIntBound::kNegInf;

bool IsBoolean(PrimType type) {
  return type.MatchesCode(kDLBool) || type.MatchesElementType(kDLInt, 1);
}

bool IsInteger(PrimType type) {
  return type.MatchesCode(kDLInt, kDLUInt, kDLBool);
}

std::pair<ffi::BigInt, ffi::BigInt> Limits(PrimType type) {
  if (IsBoolean(type))
    return {0, 1};
  bool is_signed = type.MatchesCode(kDLInt);
  ffi::BigInt limit = ffi::BigInt(1) << (type.bits() - is_signed);
  return {is_signed ? -limit : ffi::BigInt(0), limit - 1};
}

sym::ConstIntBound Everything(PrimType type) {
  if (!IsInteger(type))
    return sym::ConstIntBound(kNegativeInfinity, kPositiveInfinity);
  auto [minimum, maximum] = Limits(type);
  return sym::ConstIntBound(
      minimum <= kNegativeInfinity ? kNegativeInfinity
                                   : static_cast<int64_t>(minimum),
      maximum >= kPositiveInfinity ? kPositiveInfinity
                                   : static_cast<int64_t>(maximum));
}

bool Finite(const sym::ConstIntBound &bound) {
  return bound->min_value > kNegativeInfinity &&
         bound->max_value < kPositiveInfinity;
}

bool Fits(const sym::ConstIntBound &bound, PrimType type) {
  if (!Finite(bound))
    return false;
  auto [minimum, maximum] = Limits(type);
  return minimum <= bound->min_value && maximum >= bound->max_value;
}

bool Lossless(PrimType source, PrimType target) {
  auto [source_min, source_max] = Limits(source);
  auto [target_min, target_max] = Limits(target);
  return source_min >= target_min && source_max <= target_max;
}

ffi::BigInt FloorQuotient(ffi::BigInt value, const ffi::BigInt &divisor) {
  ffi::BigInt quotient = value / divisor;
  return value % divisor < 0 ? quotient - 1 : quotient;
}

class CustomTypeFinder : public tvm::ExprVisitor {
public:
  using ExprVisitor::Visit;
  ffi::Optional<ffi::VisitInterrupt> Visit(ffi::AnyView input) final {
    if (auto expression = input.as<PrimExpr>();
        expression && tl::IsCustomDType(expression->ty()))
      return ffi::VisitInterrupt();
    return ExprVisitor::Visit(input);
  }
};

bool HasCustomType(const PrimExpr &expression) {
  return ffi::make_object<CustomTypeFinder>()->Visit(expression).has_value();
}

class OpaqueRestorer : public tvm::ExprMutator {
public:
  explicit OpaqueRestorer(
      const std::vector<std::pair<PrimExpr, PrimVar>> &replacements)
      : replacements_(replacements) {}
  using ExprMutator::Mutate_;
  ffi::UnchangedOr<Expr> Mutate_(const VarNode *node, ffi::InplaceMode) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone)
      return ffi::Unchanged();
    for (const auto &[original, replacement] : replacements_)
      if (node == replacement.get())
        return Mutate(original, ffi::InplaceMode::kDisallow)
            .ValueOrUnchanged(original);
    return ffi::Unchanged();
  }

private:
  const std::vector<std::pair<PrimExpr, PrimVar>> &replacements_;
};

class QueryNormalizer : public tvm::ExprMutator {
public:
  explicit QueryNormalizer(sym::AnalyzerObj *analyzer,
                           bool arithmetic_casts = true)
      : analyzer_(analyzer), arithmetic_casts_(arithmetic_casts) {}
  using ExprMutator::Mutate_;
  std::vector<std::pair<PrimExpr, PrimVar>> opaque;
  bool has_effectful_let = false;

  ffi::UnchangedOr<Expr> Mutate_(const VarNode *node, ffi::InplaceMode) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone ||
        !node->ty.as<PrimType>() || active_.count(node))
      return ffi::Unchanged();
    if (auto binding = local_bindings_.find(node);
        binding != local_bindings_.end())
      return binding->second;
    auto variable = ffi::GetRef<Var>(node).as_or_throw<PrimVar>();
    auto resolved = analyzer_->rewrite_simplify(variable);
    if (resolved.same_as(variable))
      return ffi::Unchanged();
    active_.insert(node);
    auto result = Normalize(resolved);
    active_.erase(node);
    return result;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const LetNode *node,
                                     ffi::InplaceMode) final {
    if (SideEffect(node->value) > CallEffectKind::kPure) {
      has_effectful_let = true;
      PrimExpr original = ffi::GetRef<PrimExpr>(node);
      PrimVar replacement("tilelang_effectful_let", original.ty());
      opaque.emplace_back(original, replacement);
      return replacement;
    }
    PrimExpr value = Normalize(node->value);
    ffi::Optional<PrimExpr> previous;
    if (auto binding = local_bindings_.find(node->var.get());
        binding != local_bindings_.end())
      previous = binding->second;
    local_bindings_.insert_or_assign(node->var.get(), value);
    PrimExpr result = Normalize(node->body);
    if (previous)
      local_bindings_.insert_or_assign(node->var.get(), *previous);
    else
      local_bindings_.erase(node->var.get());
    return result;
  }

  PrimExpr Normalize(const PrimExpr &expression) {
    return Mutate(expression, ffi::InplaceMode::kDisallow)
        .ValueOrUnchanged(expression);
  }

  PrimExpr Restore(const PrimExpr &expression) const {
    return ffi::make_object<OpaqueRestorer>(opaque)
        ->Mutate(expression, ffi::InplaceMode::kDisallow)
        .ValueOrUnchanged(expression);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const CastNode *node,
                                     ffi::InplaceMode) final {
    PrimExpr value = Normalize(node->value);
    PrimType target = node->ty.as_or_throw<PrimType>();
    if (tl::IsCustomDType(target) || tl::IsCustomDType(value.ty()))
      return prim::Cast(target, value, node->loc);
    if (auto broadcast = value.as<BroadcastNode>())
      return prim::Broadcast(
          Normalize(prim::Cast(target.WithLanes(1), broadcast->value)),
          broadcast->lanes, node->loc);
    if (auto inner = value.as<CastNode>();
        inner && IsInteger(inner->value.ty()) &&
        IsInteger(inner->ty.as_or_throw<PrimType>()) &&
        Fits(analyzer_->const_int_bound(inner->value),
             inner->ty.as_or_throw<PrimType>()))
      value = inner->value;
    if (IsBoolean(target))
      return value != tl::MakeConst(value.ty(), 0);
    if (!IsInteger(value.ty()) || !IsInteger(target))
      return tl::CastValue(value, target, node->loc);
    PrimExpr simplified = analyzer_->rewrite_simplify(value);
    if (simplified.as<IntImmNode>())
      return tl::CastValue(simplified, target, node->loc);
    if (Lossless(value.ty(), target) ||
        Fits(analyzer_->const_int_bound(value), target))
      return tl::CastValue(value, target, node->loc);
    if (arithmetic_casts_ && target.bits() < 63) {
      int bits = std::max(value.ty().bits(), target.bits() + 1);
      bits = bits <= 32 ? 32 : bits <= 64 ? 64 : bits;
      PrimType compute =
          value.ty().MatchesCode(kDLUInt) && value.ty().bits() >= 64
              ? value.ty()
              : PrimType::Int(bits, value.ty().lanes());
      PrimExpr widened = tl::CastValue(value, compute, node->loc);
      PrimExpr modulus =
          tl::MakeConst(compute, ffi::BigInt(1) << target.bits());
      PrimExpr wrapped = tvm::floormod(widened, modulus);
      if (target.MatchesCode(kDLInt)) {
        if (compute.MatchesCode(kDLUInt)) {
          compute = PrimType::Int(64, target.lanes());
          wrapped = prim::Cast(compute, wrapped);
          modulus = tl::MakeConst(compute, ffi::BigInt(1) << target.bits());
        }
        PrimExpr sign =
            tl::MakeConst(compute, ffi::BigInt(1) << (target.bits() - 1));
        wrapped = prim::Select(wrapped >= sign, wrapped - modulus, wrapped);
      }
      return prim::Cast(target, wrapped, node->loc);
    }
    PrimExpr original = prim::Cast(target, value, node->loc);
    for (const auto &[previous, replacement] : opaque)
      if (ffi::StructuralEqual::Equal(previous, original))
        return replacement;
    PrimVar replacement("tilelang_integer_cast", target);
    opaque.emplace_back(original, replacement);
    return replacement;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const FloorModNode *node,
                                     ffi::InplaceMode mode) final {
    auto result = ExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(ffi::GetRef<PrimExpr>(node));
    auto modulo = result.as<FloorModNode>();
    if (!modulo)
      return result;
    auto divisor = modulo->b.as<IntImmNode>();
    if (auto nested = modulo->a.as<FloorModNode>();
        nested && divisor && divisor->value != 0) {
      auto inner = nested->b.as<IntImmNode>();
      if (inner && inner->value != 0 && inner->value % divisor->value == 0)
        return tvm::floormod(nested->a, modulo->b);
    }
    std::vector<PrimExpr> terms;
    std::function<void(PrimExpr)> flatten = [&](PrimExpr value) {
      if (auto add = value.as<AddNode>()) {
        flatten(add->a);
        flatten(add->b);
      } else {
        terms.push_back(value);
      }
    };
    flatten(modulo->a);
    for (size_t index = 0; index < terms.size(); ++index) {
      auto product = terms[index].as<MulNode>();
      if (!product || terms.size() < 2)
        continue;
      PrimExpr residual = tl::MakeConst(modulo->a.ty(), 0);
      for (size_t other = 0; other < terms.size(); ++other)
        if (other != index)
          residual = residual + terms[other];
      if ((ffi::StructuralEqual::Equal(product->a, modulo->b) ||
           ffi::StructuralEqual::Equal(product->b, modulo->b)) &&
          analyzer_->CanProve(modulo->b > 0))
        return tvm::floormod(residual, modulo->b);
      if (!divisor || divisor->value <= 0)
        continue;
      const IntImmNode *factor = product->b.as<IntImmNode>();
      PrimExpr variable = product->a;
      if (!factor) {
        factor = product->a.as<IntImmNode>();
        variable = product->b;
      }
      if (factor && factor->value > 0 && divisor->value % factor->value == 0 &&
          analyzer_->CanProve(residual >= 0 &&
                              residual < ffi::GetRef<IntImm>(factor))) {
        PrimExpr reduced_modulus =
            IntImm(modulo->b.ty(), divisor->value / factor->value);
        return tvm::floormod(variable, reduced_modulus) *
                   ffi::GetRef<IntImm>(factor) +
               residual;
      }
    }
    return result;
  }

private:
  sym::AnalyzerObj *analyzer_;
  bool arithmetic_casts_;
  std::unordered_set<const VarNode *> active_;
  std::unordered_map<const VarNode *, PrimExpr> local_bindings_;
};

PrimExpr RunSimplifier(AnalyzerRef analyzer, const PrimExpr &expression,
                       int mode, int steps = 2) {
  if (HasCustomType(expression))
    return expression;
  auto normalizer = ffi::make_object<QueryNormalizer>(analyzer.value, false);
  PrimExpr input = normalizer->Normalize(expression);
  if (normalizer->has_effectful_let)
    return expression;
  if (HasCustomType(input))
    return normalizer->Restore(input);
  PrimExpr result = mode == 1   ? analyzer.value->rewrite_simplify(input)
                    : mode == 2 ? analyzer.value->canonical_simplify(input)
                                : analyzer.value->Simplify(input, steps);
  return normalizer->Restore(result);
}

class BoundOperands : public tvm::ExprMutator {
public:
  BoundOperands(AnalyzerRef source, PrimExpr root,
                sym::ConstIntBoundAnalyzer::BoundMapType *bounds)
      : source_(source), root_(std::move(root)), bounds_(bounds) {}
  using ExprMutator::Mutate;
  sym::Analyzer working;

  ffi::UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                                    ffi::InplaceMode mode) final {
    if (auto expression = input.as<PrimExpr>();
        expression && !expression->same_as(root_) &&
        !expression->as<IntImmNode>() && !expression->as<FloatImmNode>()) {
      auto bound = ConstIntBoundOf(source_, *expression, bounds_);
      PrimVar replacement("tilelang_bound_operand", expression->ty());
      working->const_int_bound.Update(replacement, bound);
      return replacement;
    }
    return ExprMutator::Mutate(input, mode);
  }

private:
  AnalyzerRef source_;
  PrimExpr root_;
  sym::ConstIntBoundAnalyzer::BoundMapType *bounds_;
};

bool NeedsExtendedBounds(const PrimExpr &expression) {
  return ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
             expression,
             [](const PrimExpr &value) -> ffi::Expected<ffi::WalkResult> {
               if (value.as<CastNode>() || value.as<BitwiseOrNode>() ||
                   value.as<BitwiseXorNode>())
                 return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(value));
               return ffi::WalkResult::Advance();
             })
      .has_value();
}

class VectorScalarizer : public tl::DataTypeLegalizer {
public:
  explicit VectorScalarizer(const sym::Analyzer &analyzer)
      : analyzer_(analyzer) {}
  using DataTypeLegalizer::Mutate_;
  std::unordered_set<const VarNode *> unknown;

  PrimExpr Rewrite(const PrimExpr &expression) {
    return Mutate(expression, ffi::InplaceMode::kDisallow)
        .ValueOrUnchanged(expression);
  }

  ffi::UnchangedOr<Expr> Mutate_(const VarNode *node, ffi::InplaceMode) final {
    auto type = node->ty.as<PrimType>();
    if (!type || type->IsScalar())
      return ffi::Unchanged();
    auto variable = ffi::GetRef<Var>(node).as_or_throw<PrimVar>();
    auto resolved = analyzer_->rewrite_simplify(variable);
    if (def_region_kind() == kTVMFFIDefRegionKindNone &&
        !resolved.same_as(variable) && !active_.count(node)) {
      active_.insert(node);
      PrimExpr result = Rewrite(resolved);
      active_.erase(node);
      return result;
    }
    if (auto found = variables_.Get(variable))
      return *found;
    PrimVar scalar(node->name, type->WithLanes(1));
    variables_.Set(variable, scalar);
    unknown.insert(scalar.get());
    return scalar;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const RampNode *node,
                                     ffi::InplaceMode) final {
    if (!node->lanes.as<IntImmNode>()) {
      PrimVar scalar("tilelang_scalable_lane", node->base.ty());
      unknown.insert(scalar.get());
      return scalar;
    }
    PrimVar lane("tilelang_lane", node->base.ty());
    analyzer_->Bind(lane, Range(tl::MakeConst(lane.ty(), 0),
                                tl::CastValue(node->lanes, lane.ty())));
    return Rewrite(node->base) + Rewrite(node->stride) * lane;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const BroadcastNode *node,
                                     ffi::InplaceMode) final {
    return Rewrite(node->value);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const CastNode *node,
                                     ffi::InplaceMode) final {
    return tl::CastValue(Rewrite(node->value),
                         node->ty.as_or_throw<PrimType>().WithLanes(1));
  }

private:
  sym::Analyzer analyzer_;
  ffi::Map<Var, PrimVar> variables_;
  std::unordered_set<const VarNode *> active_;
};

sym::IntSet EvaluateIntSet(AnalyzerRef analyzer, const PrimExpr &expression,
                           ffi::Optional<ffi::Map<Var, sym::IntSet>> domains) {
  if (HasCustomType(expression))
    return sym::IntSet::Everything();
  if (auto cast = expression.as<CastNode>();
      cast && IsInteger(expression.ty())) {
    auto input = EvaluateIntSet(analyzer, cast->value, domains);
    if (input.IsNothing())
      return input;
    PrimType type = expression.ty().WithLanes(1);
    if (input.IsSinglePoint())
      return sym::IntSet::SinglePoint(tl::CastValue(input.min(), type));
    if (IsBoolean(type))
      return sym::IntSet::Interval(tl::MakeConst(type, 0),
                                   tl::MakeConst(type, 1));
    auto [minimum, maximum] = Limits(type);
    if (!Lossless(cast->value.ty(), type)) {
      auto lower = input.min().as<IntImmNode>();
      auto upper = input.max().as<IntImmNode>();
      if (!lower || !upper || lower->value < minimum || upper->value > maximum)
        return sym::IntSet::Interval(IntImm(type, minimum),
                                     IntImm(type, maximum));
    }
    return sym::IntSet::Interval(tl::CastValue(input.min(), type),
                                 tl::CastValue(input.max(), type));
  }
  if (expression.ty().IsScalar()) {
    sym::Analyzer working = ffi::GetRef<sym::Analyzer>(analyzer.value);
    bool extended = NeedsExtendedBounds(expression);
    if (domains && extended) {
      working = analyzer.value->Clone();
      for (const auto &[variable, values] : *domains) {
        if (values.IsNothing())
          return values;
        if (values.HasLowerBound() && values.HasUpperBound()) {
          auto lower = ConstIntBoundOf(analyzer, values.min());
          auto upper = ConstIntBoundOf(analyzer, values.max());
          working->const_int_bound.Update(
              variable, sym::ConstIntBound(lower->min_value, upper->max_value),
              true);
        }
      }
    }
    auto normalizer = ffi::make_object<QueryNormalizer>(working.get());
    PrimExpr normalized = normalizer->Normalize(expression);
    if (!normalizer->opaque.empty() || HasCustomType(normalized))
      return sym::IntSet::Everything();
    auto result = domains ? working->int_set(normalized, *domains)
                          : working->int_set(normalized);
    if (extended && IsInteger(expression.ty()) && !result.IsNothing()) {
      auto bounds = ConstIntBoundOf(working, expression);
      if (Fits(bounds, expression.ty())) {
        auto lower = tl::MakeConst(expression.ty(), bounds->min_value);
        auto upper = tl::MakeConst(expression.ty(), bounds->max_value);
        if (result.HasLowerBound())
          lower = Simplify(working, tvm::max(lower, result.min()));
        if (result.HasUpperBound())
          upper = Simplify(working, tvm::min(upper, result.max()));
        return sym::IntSet::Interval(lower, upper);
      }
    }
    return result;
  }
  if (IsBoolean(expression.ty()))
    return sym::IntSet::Interval(
        tl::MakeConst(expression.ty().WithLanes(1), 0),
        tl::MakeConst(expression.ty().WithLanes(1), 1));
  sym::Analyzer working = analyzer.value->Clone();
  if (domains)
    for (const auto &[variable, values] : *domains)
      working->int_set.Update(variable, values, true);
  auto scalarizer = ffi::make_object<VectorScalarizer>(working);
  PrimExpr scalar =
      expression.ty().IsScalar() ? expression : scalarizer->Rewrite(expression);
  auto normalizer = ffi::make_object<QueryNormalizer>(working.get());
  auto result = working->int_set(normalizer->Normalize(scalar));
  auto has_unknown = [&](const PrimExpr &bound) {
    return ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
               bound,
               [&](const Var &variable) -> ffi::Expected<ffi::WalkResult> {
                 if (scalarizer->unknown.count(variable.get()))
                   return ffi::WalkResult::Interrupt(
                       ffi::VisitInterrupt(variable));
                 return ffi::WalkResult::Advance();
               })
        .has_value();
  };
  if (!result.min().ty().IsScalar() || !result.max().ty().IsScalar() ||
      has_unknown(result.min()) || has_unknown(result.max()) ||
      !normalizer->opaque.empty())
    return sym::IntSet::Everything();
  return result;
}

} // namespace

PrimExpr Simplify(AnalyzerRef analyzer, const PrimExpr &expression, int steps) {
  return RunSimplifier(analyzer, expression, 0, steps);
}

PrimExpr RewriteSimplify(AnalyzerRef analyzer, const PrimExpr &expression) {
  return RunSimplifier(analyzer, expression, 1);
}

PrimExpr CanonicalSimplify(AnalyzerRef analyzer, const PrimExpr &expression) {
  return RunSimplifier(analyzer, expression, 2);
}

sym::ConstIntBound
ConstIntBoundOf(AnalyzerRef analyzer, const PrimExpr &expression,
                sym::ConstIntBoundAnalyzer::BoundMapType *bounds) {
  auto finish = [&](sym::ConstIntBound result) {
    if (bounds)
      bounds->insert_or_assign(expression, result);
    return result;
  };
  if (HasCustomType(expression))
    return finish(Everything(expression.ty()));
  if (auto conversion = expression.as<CastNode>();
      conversion && IsInteger(conversion->value.ty()) &&
      IsInteger(expression.ty())) {
    auto source = ConstIntBoundOf(analyzer, conversion->value, bounds);
    auto type = expression.ty();
    if (IsBoolean(type)) {
      if (source->min_value > 0 || source->max_value < 0)
        return finish(sym::ConstIntBound(1, 1));
      if (source->min_value == 0 && source->max_value == 0)
        return finish(sym::ConstIntBound(0, 0));
      return finish(sym::ConstIntBound(0, 1));
    }
    if (Fits(source, type))
      return finish(source);
    if (Finite(source)) {
      ffi::BigInt modulus = ffi::BigInt(1) << type.bits();
      auto wrap = [&](int64_t value) {
        ffi::BigInt result = ffi::BigInt(value) & (modulus - 1);
        if (type.MatchesCode(kDLInt) && result >= (modulus >> 1))
          result -= modulus;
        return result;
      };
      ffi::BigInt lower = wrap(source->min_value);
      ffi::BigInt upper = wrap(source->max_value);
      if (upper - lower == ffi::BigInt(source->max_value) - source->min_value &&
          lower > kNegativeInfinity && upper < kPositiveInfinity)
        return finish(sym::ConstIntBound(static_cast<int64_t>(lower),
                                         static_cast<int64_t>(upper)));
    }
    return finish(Everything(type));
  }
  ffi::Optional<PrimExpr> left, right;
  if (auto bitwise = expression.as<BitwiseOrNode>()) {
    left = bitwise->a;
    right = bitwise->b;
  } else if (auto bitwise = expression.as<BitwiseXorNode>()) {
    left = bitwise->a;
    right = bitwise->b;
  }
  if (left) {
    auto first = ConstIntBoundOf(analyzer, *left, bounds);
    auto second = ConstIntBoundOf(analyzer, *right, bounds);
    if (Finite(first) && Finite(second) && first->min_value >= 0 &&
        second->min_value >= 0) {
      auto largest =
          static_cast<uint64_t>(first->max_value | second->max_value);
      int64_t mask =
          static_cast<int64_t>((uint64_t{1} << std::bit_width(largest)) - 1);
      return finish(sym::ConstIntBound(
          0, std::min(mask, Everything(expression.ty())->max_value)));
    }
    return finish(Everything(expression.ty()));
  }
  if (NeedsExtendedBounds(expression) && !expression.as<VarNode>()) {
    auto operands =
        ffi::make_object<BoundOperands>(analyzer, expression, bounds);
    auto rewritten = operands->Mutate(expression).ValueOrUnchanged(expression);
    return finish(operands->working->const_int_bound(rewritten));
  }
  return finish(analyzer.value->const_int_bound(expression, bounds));
}

sym::ModularSet ModularSetOf(AnalyzerRef analyzer, const PrimExpr &expression) {
  if (HasCustomType(expression))
    return sym::ModularSet(1, 0);
  if (auto conversion = expression.as<CastNode>();
      conversion && IsInteger(conversion->value.ty()) &&
      IsInteger(expression.ty())) {
    auto source = ModularSetOf(analyzer, conversion->value);
    if (source->coeff == 0) {
      auto converted = tl::CastValue(
          tl::MakeConst(conversion->value.ty(), source->base), expression.ty());
      if (auto constant = converted.as<IntImmNode>())
        if (auto value = constant->value.as<int64_t>())
          return sym::ModularSet(0, *value);
      return sym::ModularSet(1, 0);
    }
    if (IsBoolean(expression.ty()))
      return sym::ModularSet(1, 0);
    if (Lossless(conversion->value.ty(), expression.ty()) ||
        Fits(ConstIntBoundOf(analyzer, conversion->value), expression.ty()))
      return source;
    int64_t coefficient =
        expression.ty().bits() < 63
            ? std::gcd(source->coeff, int64_t{1} << expression.ty().bits())
            : int64_t{1} << std::countr_zero(
                  static_cast<uint64_t>(source->coeff));
    int64_t base = source->base % coefficient;
    return sym::ModularSet(coefficient, base < 0 ? base + coefficient : base);
  }
  auto normalizer = ffi::make_object<QueryNormalizer>(analyzer.value);
  PrimExpr normalized = normalizer->Normalize(expression);
  return HasCustomType(normalized) ? sym::ModularSet(1, 0)
                                   : analyzer.value->modular_set(normalized);
}

sym::IntSet IntSetOf(AnalyzerRef analyzer, const PrimExpr &expression) {
  return EvaluateIntSet(analyzer, expression, std::nullopt);
}

sym::IntSet IntSetOf(AnalyzerRef analyzer, const PrimExpr &expression,
                     const ffi::Map<Var, sym::IntSet> &domains) {
  return EvaluateIntSet(analyzer, expression, domains);
}

bool CanProve(AnalyzerRef analyzer, const PrimExpr &condition,
              sym::ProofStrength strength) {
  if (HasCustomType(condition))
    return false;
  auto normalizer = ffi::make_object<QueryNormalizer>(analyzer.value);
  PrimExpr normalized = normalizer->Normalize(condition);
  return !HasCustomType(normalized) &&
         analyzer.value->CanProve(normalized, strength);
}

bool CanProveEqual(AnalyzerRef analyzer, const PrimExpr &first,
                   const PrimExpr &second) {
  return CanProve(analyzer, first == second);
}

bool CanProveGreaterEqual(AnalyzerRef analyzer, const PrimExpr &expression,
                          int64_t lower_bound) {
  if (IsInteger(expression.ty()) &&
      ffi::BigInt(lower_bound) > Limits(expression.ty()).second)
    return false;
  return ConstIntBoundOf(analyzer, expression)->min_value >= lower_bound ||
         CanProve(analyzer,
                  expression >= tl::MakeConst(expression.ty(), lower_bound));
}

bool CanProveLess(AnalyzerRef analyzer, const PrimExpr &expression,
                  int64_t upper_bound) {
  if (IsInteger(expression.ty()) &&
      ffi::BigInt(upper_bound) <= Limits(expression.ty()).first)
    return false;
  return ConstIntBoundOf(analyzer, expression)->max_value < upper_bound ||
         CanProve(analyzer,
                  expression < tl::MakeConst(expression.ty(), upper_bound));
}

bool CanProveLessEqualThanSymbolicShapeValue(AnalyzerRef analyzer,
                                             const PrimExpr &expression,
                                             const PrimExpr &shape) {
  if (HasCustomType(expression) || HasCustomType(shape))
    return false;
  auto normalizer = ffi::make_object<QueryNormalizer>(analyzer.value);
  auto first = normalizer->Normalize(expression);
  auto second = normalizer->Normalize(shape);
  return analyzer.value->CanProveLessEqualThanSymbolicShapeValue(first, second);
}

bool Z3CanProve(AnalyzerRef analyzer, const PrimExpr &condition) {
  if (HasCustomType(condition))
    return false;
  auto normalizer = ffi::make_object<QueryNormalizer>(analyzer.value);
  auto normalized = normalizer->Normalize(condition);
  return !HasCustomType(normalized) &&
         analyzer.value->z3_prover.CanProve(normalized);
}

ffi::String Z3GetModel(AnalyzerRef analyzer, const PrimExpr &condition) {
  auto normalizer = ffi::make_object<QueryNormalizer>(analyzer.value);
  auto normalized = normalizer->Normalize(condition);
  return analyzer.value->z3_prover.GetModel(normalized);
}

void Bind(AnalyzerRef analyzer, const Var &variable, const PrimExpr &value,
          bool allow_override) {
  auto normalizer = ffi::make_object<QueryNormalizer>(analyzer.value);
  auto normalized = normalizer->Normalize(value);
  analyzer.value->const_int_bound.Update(
      variable, ConstIntBoundOf(analyzer, value), allow_override);
  analyzer.value->modular_set.Update(variable, ModularSetOf(analyzer, value),
                                     allow_override);
  analyzer.value->int_set.Update(variable, IntSetOf(analyzer, value),
                                 allow_override);
  auto simplified = Simplify(analyzer, value);
  analyzer.value->rewrite_simplify.Update(variable, simplified, allow_override);
  analyzer.value->canonical_simplify.Update(variable, simplified,
                                            allow_override);
  if (!HasCustomType(normalized) && normalizer->opaque.empty() &&
      value.ty().IsScalar()) {
    analyzer.value->known_comparisons.Bind(variable, normalized,
                                           allow_override);
    analyzer.value->z3_prover.Bind(variable, normalized, allow_override);
    return;
  }
  PrimVar unknown("tilelang_unknown_binding",
                  variable->ty.as_or_throw<PrimType>());
  analyzer.value->known_comparisons.Bind(variable, unknown, allow_override);
  analyzer.value->z3_prover.Bind(variable, unknown, allow_override);
}

void Bind(AnalyzerRef analyzer, const Var &variable, const Range &range,
          bool allow_override) {
  auto normalizer = ffi::make_object<QueryNormalizer>(analyzer.value);
  auto begin = normalizer->Normalize(range->begin);
  auto extent = normalizer->Normalize(range->extent);
  if (normalizer->opaque.empty() && !HasCustomType(begin) &&
      !HasCustomType(extent)) {
    if (prim::IsOne(extent))
      Bind(analyzer, variable, range->begin, allow_override);
    else
      analyzer.value->Bind(variable, Range::FromMinExtent(begin, extent),
                           allow_override);
    return;
  }
  auto type = variable->ty.as_or_throw<PrimType>();
  analyzer.value->const_int_bound.Update(variable, Everything(type),
                                         allow_override);
  analyzer.value->modular_set.Update(variable, sym::ModularSet(1, 0),
                                     allow_override);
  analyzer.value->int_set.Update(variable, sym::IntSet::Everything(),
                                 allow_override);
  analyzer.value->rewrite_simplify.Update(
      variable, variable.as_or_throw<PrimVar>(), allow_override);
  analyzer.value->canonical_simplify.Update(
      variable, variable.as_or_throw<PrimVar>(), allow_override);
  PrimVar unknown("tilelang_unknown_range", type);
  analyzer.value->known_comparisons.Bind(variable, unknown, allow_override);
  analyzer.value->z3_prover.Bind(variable, unknown, allow_override);
}

void Bind(AnalyzerRef analyzer, const ffi::Map<Var, Range> &ranges,
          bool allow_override) {
  for (const auto &[variable, range] : ranges)
    Bind(analyzer, variable, range, allow_override);
}

PrimExpr NormalizeConstraint(AnalyzerRef analyzer, const PrimExpr &condition) {
  auto normalizer = ffi::make_object<QueryNormalizer>(analyzer.value);
  auto normalized = normalizer->Normalize(condition);
  return normalizer->opaque.empty() && !HasCustomType(normalized)
             ? normalized
             : tl::MakeConst(PrimType::Bool(), 1);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.sym.Simplify",
           [](sym::Analyzer analyzer, PrimExpr expression, int steps) {
             return Simplify(analyzer, expression, steps);
           })
      .def("tl.sym.RewriteSimplify",
           [](sym::Analyzer analyzer, PrimExpr expression) {
             return RewriteSimplify(analyzer, expression);
           })
      .def("tl.sym.CanonicalSimplify",
           [](sym::Analyzer analyzer, PrimExpr expression) {
             return CanonicalSimplify(analyzer, expression);
           })
      .def("tl.sym.ConstIntBound",
           [](sym::Analyzer analyzer, PrimExpr expression) {
             return ConstIntBoundOf(analyzer, expression);
           })
      .def("tl.sym.ModularSet",
           [](sym::Analyzer analyzer, PrimExpr expression) {
             return ModularSetOf(analyzer, expression);
           })
      .def("tl.sym.IntSet",
           [](sym::Analyzer analyzer, PrimExpr expression,
              ffi::Optional<ffi::Map<Var, sym::IntSet>> domains) {
             return domains ? IntSetOf(analyzer, expression, *domains)
                            : IntSetOf(analyzer, expression);
           })
      .def("tl.sym.CanProve",
           [](sym::Analyzer analyzer, PrimExpr condition, int strength) {
             return CanProve(analyzer, condition,
                             static_cast<sym::ProofStrength>(strength));
           })
      .def("tl.sym.CanProveEqual",
           [](sym::Analyzer analyzer, PrimExpr first, PrimExpr second) {
             return CanProveEqual(analyzer, first, second);
           })
      .def("tl.sym.Bind",
           [](sym::Analyzer analyzer, Var variable, ffi::Any value,
              bool allow_override) {
             if (auto range = value.as<Range>())
               Bind(analyzer, variable, *range, allow_override);
             else
               Bind(analyzer, variable, value.cast<PrimExpr>(), allow_override);
           })
      .def("tl.sym.NormalizeConstraint",
           [](sym::Analyzer analyzer, PrimExpr condition) {
             return NormalizeConstraint(analyzer, condition);
           });
}

} // namespace tvm::tl::arith
