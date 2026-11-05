/*!
 * \file layout/utils.cc
 * \brief Some arith tools for layout & fragment inference
 *
 */

#include "utils.h"
#include "ir/analyzer.h"
#include "support/check.h"
#include "tvm/sym/iter_affine_map.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>

#include "expr_utils.h"
#include <algorithm>
#include <limits>
#include <map>
#include <sstream>
#include <tvm/ir/prim/op.h>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;
using namespace sym;

namespace {

// A split selects the radix window [lower, lower * extent). It distributes
// across a boundary only when that boundary divides (or is divisible by) both
// ends. Keep adjacent digits fused when a window crosses their radix, e.g.
// (i * 6 + j) % 4; independently taking i % 4 and j % 4 would lose carries.
bool RadixAligned(int64_t boundary, int64_t factor) {
  return boundary % factor == 0 || factor % boundary == 0;
}

bool FitsRadixLiteral(const PrimExpr &original, int64_t value) {
  PrimType type = original.ty();
  int bits = type.bits() - (type.code() == kDLInt ? 1 : 0);
  return value >= 0 && (bits >= 63 || (value >> bits) == 0);
}

// Several disjoint windows can refer to the same fused mark. Downstream
// collectors visit that shared source once; cloning it per window would count
// its logical axes more than once. Keep unchanged nodes and memoize rewrites
// for this query, including groups shared by two different radix windows.
struct RadixExposureContext {
  std::unordered_map<const IterSumExprObj *, IterSumExpr> sums;
  std::unordered_map<const IterMarkObj *, IterMark> marks;
  std::unordered_map<const IterMarkObj *,
                     std::map<std::pair<size_t, size_t>, IterMark>>
      groups;
};

IterSumExpr ExposeRadixDigits(const IterSumExpr &sum,
                              RadixExposureContext *context);

Array<IterSplitExpr> ExposeRadixDigits(const IterSplitExpr &original,
                                       RadixExposureContext *context) {
  const auto *nested = original->source->source.as<IterSumExprObj>();
  if (!nested)
    return {original};
  IterMark source_mark = original->source;
  auto cached_mark = context->marks.find(source_mark.get());
  if (cached_mark != context->marks.end()) {
    source_mark = cached_mark->second;
  } else {
    IterSumExpr source =
        ExposeRadixDigits(GetRef<IterSumExpr>(nested), context);
    if (!source.same_as(GetRef<IterSumExpr>(nested)))
      source_mark = IterMark(source, original->source->extent);
    context->marks.emplace(original->source.get(), source_mark);
  }
  IterSumExpr source = source_mark->source.as_or_throw<IterSumExpr>();
  IterSplitExpr split = original;
  if (!source_mark.same_as(original->source))
    split = IterSplitExpr(source_mark, original->lower_factor, original->extent,
                          original->scale);
  auto lower = TryConstInt(split->lower_factor);
  auto extent = TryConstInt(split->extent);
  auto multiplier = TryConstInt(split->scale);
  auto mark_extent = TryConstInt(split->source->extent);
  if (!IsZero(source->base) || !lower || !extent || !multiplier ||
      !mark_extent || *lower <= 0 || *extent <= 0 || *multiplier <= 0 ||
      *lower > std::numeric_limits<int64_t>::max() / *extent)
    return {split};
  const int64_t upper = *lower * *extent;

  struct Digit {
    IterSplitExpr split;
    int64_t scale;
    int64_t extent;
  };
  std::vector<Digit> digits;
  for (const IterSplitExpr &child : source->args) {
    auto scale = TryConstInt(child->scale);
    auto count = TryConstInt(child->extent);
    if (!scale || !count || *scale <= 0 || *count <= 0)
      return {split};
    digits.push_back({child, *scale, *count});
  }
  std::sort(digits.begin(), digits.end(),
            [](const Digit &a, const Digit &b) { return a.scale < b.scale; });
  int64_t radix = 1;
  for (const Digit &digit : digits) {
    if (digit.scale != radix ||
        radix > std::numeric_limits<int64_t>::max() / digit.extent)
      return {split};
    radix *= digit.extent;
  }
  if (digits.empty() || radix != *mark_extent)
    return {split};

  Array<IterSplitExpr> result;
  size_t end = digits.size();
  while (end > 0) {
    size_t begin = end - 1;
    while (begin > 0 && !(RadixAligned(digits[begin].scale, *lower) &&
                          RadixAligned(digits[begin].scale, upper)))
      --begin;
    const int64_t scale = digits[begin].scale;
    const int64_t group_end = digits[end - 1].scale * digits[end - 1].extent;
    const int64_t group_extent = group_end / scale;
    if (scale < upper && group_end > *lower) {
      const int64_t new_lower = std::max<int64_t>(1, *lower / scale);
      // A partial final digit needs a padding guard. Keep the native fused
      // split when division would truncate its extent; otherwise the new
      // modulo would wrap that final digit back to zero.
      if (group_extent % new_lower != 0)
        return {split};
      const int64_t new_extent =
          std::min(group_extent, upper / scale) / new_lower;
      const int64_t scale_factor = std::max<int64_t>(1, scale / *lower);
      if (scale_factor > std::numeric_limits<int64_t>::max() / *multiplier ||
          new_extent <= 0)
        return {split};
      const int64_t new_scale = scale_factor * *multiplier;
      if (!FitsRadixLiteral(split->lower_factor, new_lower) ||
          !FitsRadixLiteral(split->extent, new_extent) ||
          !FitsRadixLiteral(split->scale, new_scale))
        return {split};
      IterMark mark = split->source;
      int64_t combined_lower = new_lower;
      if (end - begin == 1 && group_extent % (new_lower * new_extent) == 0) {
        const IterSplitExpr &child = digits[begin].split;
        auto child_lower = TryConstInt(child->lower_factor);
        if (!child_lower || *child_lower <= 0 ||
            *child_lower > std::numeric_limits<int64_t>::max() / new_lower)
          return {split};
        combined_lower *= *child_lower;
        if (!FitsRadixLiteral(split->lower_factor, combined_lower))
          return {split};
        mark = child->source;
      } else {
        if (begin == 0 && end == digits.size())
          return {split};
        if (!FitsRadixLiteral(split->source->extent, group_extent))
          return {split};
        auto &groups = context->groups[source_mark.get()];
        auto key = std::make_pair(begin, end);
        auto cached_group = groups.find(key);
        if (cached_group != groups.end()) {
          mark = cached_group->second;
        } else {
          Array<IterSplitExpr> group;
          for (size_t i = end; i > begin; --i) {
            const auto &digit = digits[i - 1];
            group.push_back(IterSplitExpr(
                digit.split->source, digit.split->lower_factor,
                digit.split->extent,
                MakeConst(digit.split->scale.ty(), digit.scale / scale)));
          }
          mark = IterMark(IterSumExpr(group, MakeConst(source.ty(), 0)),
                          MakeConst(split->source->extent.ty(), group_extent));
          groups.emplace(key, mark);
        }
      }
      result.push_back(IterSplitExpr(
          mark, MakeConst(split->lower_factor.ty(), combined_lower),
          MakeConst(split->extent.ty(), new_extent),
          MakeConst(split->scale.ty(), new_scale)));
    }
    end = begin;
  }
  return result;
}

IterSumExpr ExposeRadixDigits(const IterSumExpr &sum,
                              RadixExposureContext *context) {
  auto cached = context->sums.find(sum.get());
  if (cached != context->sums.end())
    return cached->second;
  Array<IterSplitExpr> result;
  bool changed = false;
  for (const IterSplitExpr &split : sum->args) {
    auto exposed = ExposeRadixDigits(split, context);
    changed |= exposed.size() != 1 || !exposed[0].same_as(split);
    result.insert(result.end(), exposed.begin(), exposed.end());
  }
  IterSumExpr rewritten = changed ? IterSumExpr(result, sum->base) : sum;
  context->sums.emplace(sum.get(), rewritten);
  return rewritten;
}

} // namespace

IterSumExpr NormalizeLayoutIterSum(const PrimExpr &expr,
                                   const Map<PrimVar, Range> &input_iters,
                                   sym::AnalyzerObj *analyzer) {
  RadixExposureContext context;
  return ExposeRadixDigits(
      sym::NormalizeToIterSum(expr, input_iters,
                              ffi::GetRef<sym::Analyzer>(analyzer)),
      &context);
}

Fragment ComputeReducerLayout(const Fragment &src_layout, int dim) {
  ICHECK_GE(dim, 0) << "ReduceOp: reduction axis must be non-negative";
  ICHECK_LT(dim, src_layout->InputDim())
      << "ReduceOp: reduction axis exceeds the source layout rank";
  PrimExpr src_rep_extent = src_layout->ReplicateExtent();
  PrimExpr reduce_extent = src_layout->InputShape()[dim];
  Array<PrimExpr> fwd;
  for (size_t i = 0; i + 1 < src_layout->InputDim(); ++i) {
    fwd.push_back(InputPlaceholder(i));
  }
  fwd.insert(fwd.begin() + dim,
             FloorMod(ReplicationPlaceholder(), reduce_extent));
  PrimExpr thread = src_layout->ForwardThread(
      fwd, FloorDiv(ReplicationPlaceholder(), reduce_extent));

  Array<PrimExpr> shape = src_layout->InputShape();
  shape.erase(shape.begin() + dim);
  if (shape.empty()) {
    shape.push_back(1);
  }
  return Fragment(shape, {}, thread, reduce_extent * src_rep_extent,
                  std::nullopt)
      ->CondenseReplicateVar()
      ->BindThreadRange(src_layout->ThreadRange());
}

std::optional<int64_t> EvaluateConstantInteger(const PrimExpr &expr) {
  if (auto constant = TryConstInt(expr))
    return constant;

  // Evaluate native primitive nodes with arbitrary-precision intermediates.
  // A layout enumeration must never use C++ signed overflow as its oracle.
  auto finish = [&](const ffi::BigInt &value) -> std::optional<int64_t> {
    PrimType type = expr.ty();
    if (!type.IsScalar() || (!type.MatchesCode(DLDataTypeCode::kDLInt) &&
                             !type.MatchesCode(DLDataTypeCode::kDLUInt) &&
                             !type.MatchesCode(DLDataTypeCode::kDLBool)))
      return std::nullopt;
    if (type.MatchesCode(DLDataTypeCode::kDLUInt)) {
      ffi::BigInt mask = (ffi::BigInt(1) << type.bits()) - 1;
      return (value & mask).as<int64_t>();
    }
    if (type.MatchesCode(DLDataTypeCode::kDLBool)) {
      if (value != 0 && value != 1)
        return std::nullopt;
    } else {
      ffi::BigInt limit = ffi::BigInt(1) << (type.bits() - 1);
      if (value < -limit || value >= limit)
        return std::nullopt;
    }
    return value.as<int64_t>();
  };
  auto binary = [&](const PrimExpr &a, const PrimExpr &b,
                    auto evaluate) -> std::optional<int64_t> {
    auto lhs = EvaluateConstantInteger(a), rhs = EvaluateConstantInteger(b);
    if (!lhs || !rhs)
      return std::nullopt;
    return evaluate(ffi::BigInt(*lhs), ffi::BigInt(*rhs));
  };
  if (const auto *op = expr.as<AddObj>()) {
    return binary(op->a, op->b, [&](auto a, auto b) { return finish(a + b); });
  } else if (const auto *op = expr.as<SubObj>()) {
    return binary(op->a, op->b, [&](auto a, auto b) { return finish(a - b); });
  } else if (const auto *op = expr.as<MulObj>()) {
    return binary(op->a, op->b, [&](auto a, auto b) { return finish(a * b); });
  } else if (const auto *op = expr.as<FloorDivObj>()) {
    return binary(op->a, op->b, [&](auto a, auto b) -> std::optional<int64_t> {
      if (!b)
        return std::nullopt;
      return finish(ffi::floordiv(a, b));
    });
  } else if (const auto *op = expr.as<FloorModObj>()) {
    return binary(op->a, op->b, [&](auto a, auto b) -> std::optional<int64_t> {
      if (!b)
        return std::nullopt;
      return finish(ffi::floormod(a, b));
    });
  } else if (const auto *op = expr.as<BitwiseXorObj>()) {
    return binary(op->a, op->b, [&](auto a, auto b) { return finish(a ^ b); });
  } else if (const auto *op = expr.as<BitwiseAndObj>()) {
    return binary(op->a, op->b, [&](auto a, auto b) { return finish(a & b); });
  } else if (const auto *op = expr.as<BitwiseOrObj>()) {
    return binary(op->a, op->b, [&](auto a, auto b) { return finish(a | b); });
  } else if (const auto *op = expr.as<LShiftObj>()) {
    return binary(op->a, op->b, [&](auto a, auto b) -> std::optional<int64_t> {
      if (b < 0 || b >= expr.ty().bits())
        return std::nullopt;
      return finish(a << b);
    });
  } else if (const auto *op = expr.as<RShiftObj>()) {
    return binary(op->a, op->b, [&](auto a, auto b) -> std::optional<int64_t> {
      if (b < 0 || b >= expr.ty().bits())
        return std::nullopt;
      return finish(a >> b);
    });
  } else if (const auto *op = expr.as<BitwiseNotObj>()) {
    if (auto value = EvaluateConstantInteger(op->a))
      return finish(~ffi::BigInt(*value));
  }
  return std::nullopt;
}

bool CanProveDivisible(const PrimExpr &lhs, const PrimExpr &rhs) {
  const auto *clhs = lhs.as<IntImmObj>();
  const auto *crhs = rhs.as<IntImmObj>();
  if (crhs && crhs->value == 0) {
    return false;
  } else if (clhs && crhs) {
    return clhs->value % crhs->value == 0;
  }

  return false;
}

/*!
 * \brief Collector that collects the outgoing split reference of each IterMark.
 *
 *  These out-going splits can then be used to check if the iterators are
 * independent.
 */
class IterMarkSplitCollector {
public:
  // mark all IterMarks that are visited.
  std::unordered_set<IterMark, ObjectPtrHash, ObjectPtrEqual> visited_;
  // each iter mark to its outgoing splits that are referenced.
  std::unordered_map<IterMark, std::vector<IterSplitExpr>, ObjectPtrHash,
                     ObjectPtrEqual>
      mark2splits_;
  /*!
   * \brief Collect all mark2splits recursively from indices.
   * \param indices The iterator of interest.
   */
  void Collect(const Array<IterSumExpr> &indices) {
    for (IterSumExpr sum_expr : indices) {
      for (IterSplitExpr split : sum_expr->args) {
        this->CollectInternal(split->source);
        mark2splits_[split->source].push_back(split);
      }
    }
  }

  void CollectInternal(const IterMark &mark) {
    if (visited_.count(mark))
      return;
    visited_.insert(mark);
    if (auto *op = mark->source.as<IterSumExprObj>()) {
      for (IterSplitExpr split : op->args) {
        this->CollectInternal(split->source);
        mark2splits_[split->source].push_back(split);
      }
    }
  }
};

Array<IterSplitExpr> get_unused_iters(const IterMark &mark,
                                      const std::vector<IterSplitExpr> &splits,
                                      AnalyzerObj *analyzer) {
  PrimExpr expected_lower_factor = MakeConst(mark->source.ty(), 1);
  std::vector<bool> used(splits.size(), false);
  std::vector<IterSplitExpr> results;
  size_t i = 0;
  for (; i < splits.size();) {
    size_t j = 0;
    size_t lowest = splits.size();
    for (; j < splits.size(); ++j) {
      if (used[j])
        continue;
      if (!used[j] && analyzer->CanProveEqual(splits[j]->lower_factor,
                                              expected_lower_factor)) {
        break;
      }
      if (lowest == splits.size() ||
          CanProveDivisible(splits[lowest]->lower_factor,
                            splits[j]->lower_factor)) {
        lowest = j;
      }
    }
    if (j == splits.size()) {
      ICHECK(lowest != splits.size());
      ICHECK(CanProveDivisible(splits[lowest]->lower_factor,
                               expected_lower_factor))
          << " Cannot prove divisible for " << splits[lowest]->lower_factor
          << " and " << expected_lower_factor;
      results.emplace_back(
          mark, expected_lower_factor,
          analyzer->Simplify(
              FloorDiv(splits[lowest]->lower_factor, expected_lower_factor)),
          1);
      expected_lower_factor = splits[lowest]->lower_factor;
    } else {
      used[j] = true;
      i++;
      expected_lower_factor =
          analyzer->Simplify(splits[j]->lower_factor * splits[j]->extent);
    }
  }
  // Iter split normalization may over-approximate the original mark span.
  // Treat over-coverage as fully covered instead of synthesizing a zero-extent
  // leftover iterator, which later normalizes into division by zero.
  bool covers_full_iter =
      analyzer->CanProveEqual(expected_lower_factor, mark->extent) ||
      analyzer->CanProve(expected_lower_factor > mark->extent);
  if (!covers_full_iter) {
    results.emplace_back(
        mark, expected_lower_factor,
        analyzer->Simplify(FloorDiv(mark->extent, expected_lower_factor)), 1);
  }
  return results;
}

struct IterExprPP {
  // std::vector<std::pair<std::string, PrimExpr>> marks;
  Map<String, PrimExpr> marks;
  std::string data;

  IterExprPP(const PrimExpr &expr) { data = Visit(expr); }

  IterExprPP(const IterMark &mark) { data = Visit_(mark.get()); }

  std::string Visit(const PrimExpr &expr) {
    if (auto *sum = expr.as<IterSumExprObj>()) {
      return Visit_(sum);
    } else if (auto *split = expr.as<IterSplitExprObj>()) {
      return Visit_(split);
    } else if (auto *var = expr.as<VarObj>()) {
      return var->name;
    } else {
      std::stringstream ss;
      ss << "<UNKNOWN: " << expr << ">";
      return ss.str();
    }
  }

  std::string Visit_(const IterMarkObj *op) {
    std::stringstream ss;
    ss << "(";
    ss << Visit(op->source);
    ss << ")";
    auto res = ss.str();
    marks.Set(res, op->extent);
    return res;
  }

  std::string Visit_(const IterSumExprObj *op) {
    std::stringstream ss;
    bool first = true;
    for (const auto args : op->args) {
      if (!first) {
        ss << " + ";
      } else {
        first = false;
      }
      ss << Visit_(args.get());
    }
    return ss.str();
  }

  std::string Visit_(const IterSplitExprObj *op) {
    std::stringstream ss;
    ss << Visit_(op->source.get());
    if (!IsOne(op->lower_factor)) {
      ss << " / " << op->lower_factor;
    }
    ss << " % " << op->extent;
    if (!IsOne(op->scale)) {
      ss << " * " << op->scale;
    }
    return ss.str();
  }

  friend std::ostream &operator<<(std::ostream &os, const IterExprPP &pp) {
    os << "IterExpr(\n";
    os << "  expr=" << pp.data << "\n";
    os << "  iter_mark_extents=";
    if (pp.marks.empty()) {
      os << "{}\n";
    } else {
      os << "{\n";
      for (const auto &[k, v] : pp.marks) {
        os << "    " << k << ": " << v << ",\n";
      }
      os << "  }\n";
    }
    os << ")";
    return os;
  }
};

// Heuristic: detect per-iterator gaps ("unused" pieces) even when the iterator
// appears in fused forms across multiple index expressions. We first normalize
// every index into IterSumExpr, collect all splits per source PrimVar, then
// consolidate them to avoid misclassifying a used split as unused.
Array<IterSplitExpr>
DivideUnusedIterators(const Array<PrimExpr> &exprs,
                      const Array<LayoutIterVar> input_iters,
                      AnalyzerObj *analyzer) {
  auto iter_sum = exprs.Map([&](const PrimExpr &e) {
    return NormalizeLayoutIterSum(e, ToVMap(input_iters), analyzer);
  });
  IterMarkSplitCollector collector;
  collector.Collect(iter_sum);

  std::unordered_map<IterMark, std::vector<IterSplitExpr>, StructuralHash,
                     StructuralEqual>
      mark_splits;
  std::vector<IterMark> mark_order;

  // Step. 1: force add all input_iters to marks (some may not appear in
  // collector)
  for (auto &iter : input_iters) {
    IterMark mark(iter->var, iter->dom->extent);
    mark_splits[mark] = {};
    mark_order.push_back(mark);
  }

  // Step. 2: add all collected marks and their splits
  for (auto &mark : collector.visited_) {
    if (!mark_splits.count(mark)) {
      mark_splits[mark] = {};
      mark_order.push_back(mark);
    }
    for (const auto &splits : collector.mark2splits_[mark]) {
      mark_splits[mark].push_back(splits);
    }
  }

  Array<IterSplitExpr> results;
  // Step. 3: process marks in order and collect complement
  for (const auto &mark : mark_order) {
    const auto &existing_splits = mark_splits.at(mark);
    auto complement_splits = get_unused_iters(mark, existing_splits, analyzer);
    results.insert(results.end(), complement_splits.rbegin(),
                   complement_splits.rend());
  }

  return results;
}

PrimExpr MakeFlattenedExpression(const Array<sym::IterSplitExpr> &splits) {
  Array<sym::IterSplitExpr> lists;
  PrimExpr scale = 1;
  for (int i = splits.size() - 1; i >= 0; i--) {
    auto scaled_split = sym::IterSplitExpr(
        splits[i]->source, splits[i]->lower_factor, splits[i]->extent, scale);
    lists.push_back(scaled_split);
    scale *= splits[i]->extent;
  }
  return sym::NormalizeIterMapToExpr(sym::IterSumExpr(lists, 0));
}

class IterSumMutator {
public:
  IterSumMutator(const Map<IterSplitExpr, IterSplitExpr> &replace_map)
      : replace_map_(replace_map) {}

  // override the original mutate function.
  IterSumExpr Mutate(const IterSumExpr &iter_sum) {
    Array<IterSplitExpr> args;
    for (const auto &split : iter_sum->args) {
      if (replace_map_.count(split)) {
        args.push_back(replace_map_[split]);
      } else {
        auto split_ = IterSplitExpr(Mutate(split->source), split->lower_factor,
                                    split->extent, split->scale);
        args.push_back(split_);
      }
    }
    return IterSumExpr(args, iter_sum->base);
  }

  IterMark Mutate(const IterMark &mark) {
    if (auto *op = mark->source.as<IterSumExprObj>()) {
      return IterMark(Mutate(GetRef<IterSumExpr>(op)), mark->extent);
    } else {
      return mark;
    }
  }

private:
  Map<IterSplitExpr, IterSplitExpr> replace_map_;
};

std::pair<PrimExpr, LayoutIterVar>
CompressIterator(const PrimExpr &expr, const Array<LayoutIterVar> input_iters,
                 const PrimVar &var, sym::AnalyzerObj *analyzer) {
  auto iter_sum = NormalizeLayoutIterSum(expr, ToVMap(input_iters), analyzer);
  IterMarkSplitCollector collector;
  collector.Collect({iter_sum});
  Optional<IterMark> mark;
  for (const IterMark &m : collector.visited_) {
    auto v = m->source.as<PrimVar>();
    if (v && v.value().same_as(var)) {
      mark = m;
      break;
    }
  }
  std::vector<tvm::sym::IterSplitExpr> splits;
  if (mark.has_value()) {
    splits = collector.mark2splits_[mark.value()];
  }

  PrimExpr extent = 1;
  for (const auto &split : splits) {
    extent *= split->extent;
  }
  extent = analyzer->Simplify(extent);

  auto new_var = PrimVar(var->name, var->ty);
  auto new_iter_var = LayoutIterVar(Range(0, extent), new_var);
  auto new_mark = IterMark(new_var, extent);
  PrimExpr scale = 1;
  Map<IterSplitExpr, IterSplitExpr> replace_map;
  for (const auto &split : splits) {
    auto rescaled =
        sym::IterSplitExpr(new_mark, scale, split->extent, split->scale);
    replace_map.Set(split, rescaled);
    scale *= split->extent;
  }

  IterSumMutator mutator(replace_map);
  PrimExpr reaplced =
      analyzer->Simplify(NormalizeIterMapToExpr(mutator.Mutate(iter_sum)));

  return {reaplced, new_iter_var};
}

Array<LayoutIterVar> ToIterVars(const Map<PrimVar, Range> &vmap) {
  Array<LayoutIterVar> result;
  for (const auto &[var, range] : vmap) {
    result.push_back(LayoutIterVar(range, var));
  }
  return result;
}

Map<PrimVar, Range> ToVMap(const Array<LayoutIterVar> &ivs) {
  Map<PrimVar, Range> result;
  for (const auto &iv : ivs) {
    result.Set(iv->var, iv->dom);
  }
  return result;
}

// ProveFragmentContains checks whether the threads that access elements of a
// smaller fragment (small_frag) are a subset of the threads that access
// elements of a larger fragment (large_frag) for any given loop index. This
// function ensures that if the small fragment's layout corresponds to the loop
// itself, accessing the large fragment's elements is valid. Additionally, if
// small is updated to large, the originally valid access remains valid. The
// proof is performed by:
//
// 1. Defining a variable `rep_small` to represent the replicate index of the
//    small fragment that is being checked.
// 2. Using the `small_frag_indices` and `rep_small` to derive the thread
//    accessing the element in the small fragment.
// 3. Using `large_frag_indices` to derive the physical index of the large
//    fragment along with the thread information, and then feeding these into
//    the inverse of the large fragment to obtain the logical index and
//    replicate index.
// 4. Verifying the mapping by checking whether the computed thread using the
//    inverse layout corresponds to the original thread calculated for the small
//    fragment. If they don't match, this indicates that the inverse layout's
//    domain does not include the thread and thus the access is invalid.
// Thanks @huanqicao for contributing this algorithm.
bool ProveFragmentContains(Fragment small_frag, Fragment large_frag,
                           Array<PrimExpr> small_frag_indices,
                           Array<PrimExpr> large_frag_indices,
                           const Analyzer &analyzer, bool check_forward_index) {
  // ThreadRange.begin is an origin, as in PartitionLoop: forward_thread is
  // relative to that origin. Compare absolute owners, while checking each
  // fragment's own relative domain. Equal raw maps with different origins do
  // not imply containment.
  auto proof = analyzer->Clone();
  auto indices_in_domain = [&](const Fragment &fragment,
                               const Array<PrimExpr> &indices) {
    if (indices.size() != fragment->InputShape().size())
      return false;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (!proof->CanProve(indices[i] >= 0) ||
          !proof->CanProve(indices[i] < fragment->InputShape()[i]))
        return false;
    }
    return true;
  };
  if (!indices_in_domain(small_frag, small_frag_indices) ||
      !indices_in_domain(large_frag, large_frag_indices))
    return false;

  // When check_forward_index is true, verify that the physical indices
  // (forward index) of both fragments are equal. This is required when
  // validating loop layout against buffer fragment, as code generation
  // needs to correctly derive buffer physical indices from loop layout.
  if (check_forward_index) {
    auto small_physical = small_frag->Forward(small_frag_indices);
    auto large_physical = large_frag->Forward(large_frag_indices);
    // Dimension mismatch means they are not equal.
    if (small_physical.size() != large_physical.size()) {
      return false;
    }
    // Check each physical index component for equality.
    for (size_t i = 0; i < small_physical.size(); i++) {
      if (!proof->CanProveEqual(small_physical[i], large_physical[i])) {
        return false;
      }
    }
  }

  PrimVar rep_small("__checking_frag_contains_rep",
                    small_frag->ReplicateExtent().ty());
  if (!proof->CanProve(small_frag->ReplicateExtent() > 0) ||
      !proof->CanProve(large_frag->ReplicateExtent() > 0))
    return false;
  BindAnalyzer(proof, rep_small,
               Range::FromMinExtent(IntImm(rep_small.ty(), 0),
                                    small_frag->ReplicateExtent()));
  PrimExpr small_thread =
      small_frag->ForwardThread(small_frag_indices, rep_small);
  auto relative_thread_in_domain = [&](const Fragment &fragment,
                                       const PrimExpr &thread) {
    if (!proof->CanProve(thread >= 0))
      return false;
    if (auto range = fragment->ThreadRange())
      return proof->CanProve(thread < range->extent);
    return true;
  };
  if (!relative_thread_in_domain(small_frag, small_thread))
    return false;
  PrimExpr absolute_thread = small_thread;
  if (auto range = small_frag->ThreadRange())
    absolute_thread = absolute_thread + range->begin;
  PrimExpr large_thread = absolute_thread;
  if (auto range = large_frag->ThreadRange())
    large_thread = large_thread - range->begin;
  large_thread = proof->Simplify(large_thread);
  if (!relative_thread_in_domain(large_frag, large_thread))
    return false;

  // A bounded replica witness proves owner inclusion without requiring a
  // rectangular inverse. This also handles fully-replicated fragments: their
  // actual range must contain the small fragment, not just have the right tag.
  auto replica_is_witness = [&](const PrimExpr &replica) {
    return replica.ty() == large_frag->ReplicateExtent().ty() &&
           proof->CanProve(replica >= 0) &&
           proof->CanProve(replica < large_frag->ReplicateExtent()) &&
           proof->CanProveEqual(large_thread, large_frag->ForwardThread(
                                                  large_frag_indices, replica));
  };
  if (replica_is_witness(rep_small))
    return true;
  auto first_owner = large_frag->ForwardThread(
      large_frag_indices, MakeConst(large_frag->ReplicateExtent().ty(), 0));
  if (large_thread.ty() == first_owner.ty() &&
      large_thread.ty() == large_frag->ReplicateExtent().ty() &&
      replica_is_witness(proof->Simplify(large_thread - first_owner)))
    return true;

  // Get physical index and thread for large_frag.
  auto large_frag_physical_and_thread = large_frag->Forward(large_frag_indices);
  // The inverse consumes the large fragment's relative thread coordinate.
  large_frag_physical_and_thread.push_back(large_thread);
  // Get the inverse of the large fragment.
  Layout inv_large_frag;
  try {
    inv_large_frag = large_frag->Inverse();
  } catch (const NormalizeIterException &) {
    // A candidate with a ragged physical domain need not have a bijective
    // rectangular inverse. This does not prove containment, but must not abort
    // the caller's search for another valid loop or reducer layout.
    return false;
  } catch (const Error &) {
    // Unsupported iter-map scales may fail in InverseAffineIterMap after
    // normalization. A missing inverse is no proof of containment.
    return false;
  }
  // Compute logical index and replicate index using inverse layout.
  auto inv_large_frag_logical_and_rep =
      inv_large_frag->Forward(large_frag_physical_and_thread);
  if (inv_large_frag_logical_and_rep.size() != large_frag_indices.size() + 1)
    return false;
  for (size_t i = 0; i < large_frag_indices.size(); ++i) {
    if (!proof->CanProveEqual(inv_large_frag_logical_and_rep[i],
                              large_frag_indices[i]))
      return false;
  }

  // Extract replicate index from the result.
  auto inv_large_frag_rep =
      inv_large_frag_logical_and_rep[inv_large_frag_logical_and_rep.size() - 1];

  return replica_is_witness(inv_large_frag_rep);
}

} // namespace tl
} // namespace tvm
