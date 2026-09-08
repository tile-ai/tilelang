/*!
 * \file layout/utils.cc
 * \brief Some arith tools for layout & fragment inference
 *
 */

#include "utils.h"
#include "support/check.h"
#include "tvm/arith/iter_affine_map.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr_functor.h>

#include <limits>
#include <sstream>
#include <string>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using namespace arith;

namespace {

class IntegerExpressionEvaluator final
    : public ExprFunctor<std::optional<int64_t>(const PrimExpr &)> {
public:
  IntegerExpressionEvaluator() = default;

  IntegerExpressionEvaluator(const std::vector<Var> *variables,
                             const std::vector<int64_t> *values)
      : variables_(variables), values_(values) {
    ICHECK_EQ(variables_->size(), values_->size());
  }

private:
  using Result = std::optional<int64_t>;

  // Floored division/modulo (rounds toward -inf), matching FloorDiv/FloorMod.
  static int64_t FloorDiv(int64_t lhs, int64_t rhs) {
    int64_t quotient = lhs / rhs;
    int64_t remainder = lhs % rhs;
    return quotient -
           ((remainder != 0) && ((remainder < 0) != (rhs < 0)) ? 1 : 0);
  }

  static int64_t FloorMod(int64_t lhs, int64_t rhs) {
    int64_t remainder = lhs % rhs;
    return remainder != 0 && ((remainder < 0) != (rhs < 0)) ? remainder + rhs
                                                            : remainder;
  }

  static bool IsInvalidDivision(int64_t lhs, int64_t rhs) {
    return rhs == 0 ||
           (lhs == std::numeric_limits<int64_t>::min() && rhs == -1);
  }

  static Result ExactCast(int64_t value, DataType dtype) {
    if (!dtype.is_scalar()) {
      return std::nullopt;
    }
    if (dtype.is_bool()) {
      return value != 0;
    }
    if (dtype.is_int()) {
      int bits = dtype.bits();
      if (bits <= 0 || bits > 64) {
        return std::nullopt;
      }
      if (bits == 64) {
        return value;
      }
      int64_t limit = int64_t{1} << (bits - 1);
      if (value < -limit || value >= limit) {
        return std::nullopt;
      }
      return value;
    }
    if (dtype.is_uint()) {
      int bits = dtype.bits();
      if (bits <= 0 || bits > 64 || value < 0) {
        return std::nullopt;
      }
      if (bits < 63 && value >= (int64_t{1} << bits)) {
        return std::nullopt;
      }
      return value;
    }
    return std::nullopt;
  }

  template <typename Node, typename F>
  Result EvaluateBinary(const Node *op, F &&combine) {
    Result lhs = VisitExpr(op->a);
    Result rhs = VisitExpr(op->b);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return combine(*lhs, *rhs);
  }

  Result VisitExpr_(const IntImmNode *op) final { return op->value; }

  Result VisitExpr_(const VarNode *op) final {
    if (variables_ == nullptr || values_ == nullptr) {
      return std::nullopt;
    }
    Var var = GetRef<Var>(op);
    for (size_t i = 0; i < variables_->size(); ++i) {
      if ((*variables_)[i].same_as(var)) {
        return (*values_)[i];
      }
    }
    return std::nullopt;
  }

  Result VisitExpr_(const AddNode *op) final {
    return EvaluateBinary(op,
                          [](int64_t lhs, int64_t rhs) { return lhs + rhs; });
  }

  Result VisitExpr_(const SubNode *op) final {
    return EvaluateBinary(op,
                          [](int64_t lhs, int64_t rhs) { return lhs - rhs; });
  }

  Result VisitExpr_(const MulNode *op) final {
    return EvaluateBinary(op,
                          [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
  }

  Result VisitExpr_(const DivNode *op) final {
    return EvaluateBinary(op, [](int64_t lhs, int64_t rhs) -> Result {
      if (IsInvalidDivision(lhs, rhs)) {
        return std::nullopt;
      }
      return lhs / rhs;
    });
  }

  Result VisitExpr_(const ModNode *op) final {
    return EvaluateBinary(op, [](int64_t lhs, int64_t rhs) -> Result {
      if (IsInvalidDivision(lhs, rhs)) {
        return std::nullopt;
      }
      return lhs % rhs;
    });
  }

  Result VisitExpr_(const FloorDivNode *op) final {
    return EvaluateBinary(op, [](int64_t lhs, int64_t rhs) -> Result {
      if (IsInvalidDivision(lhs, rhs)) {
        return std::nullopt;
      }
      return FloorDiv(lhs, rhs);
    });
  }

  Result VisitExpr_(const FloorModNode *op) final {
    return EvaluateBinary(op, [](int64_t lhs, int64_t rhs) -> Result {
      if (IsInvalidDivision(lhs, rhs)) {
        return std::nullopt;
      }
      return FloorMod(lhs, rhs);
    });
  }

  Result VisitExpr_(const MinNode *op) final {
    return EvaluateBinary(
        op, [](int64_t lhs, int64_t rhs) { return lhs < rhs ? lhs : rhs; });
  }

  Result VisitExpr_(const MaxNode *op) final {
    return EvaluateBinary(
        op, [](int64_t lhs, int64_t rhs) { return lhs > rhs ? lhs : rhs; });
  }

  Result VisitExpr_(const EQNode *op) final {
    return EvaluateBinary(op,
                          [](int64_t lhs, int64_t rhs) { return lhs == rhs; });
  }

  Result VisitExpr_(const NENode *op) final {
    return EvaluateBinary(op,
                          [](int64_t lhs, int64_t rhs) { return lhs != rhs; });
  }

  Result VisitExpr_(const LTNode *op) final {
    return EvaluateBinary(op,
                          [](int64_t lhs, int64_t rhs) { return lhs < rhs; });
  }

  Result VisitExpr_(const LENode *op) final {
    return EvaluateBinary(op,
                          [](int64_t lhs, int64_t rhs) { return lhs <= rhs; });
  }

  Result VisitExpr_(const GTNode *op) final {
    return EvaluateBinary(op,
                          [](int64_t lhs, int64_t rhs) { return lhs > rhs; });
  }

  Result VisitExpr_(const GENode *op) final {
    return EvaluateBinary(op,
                          [](int64_t lhs, int64_t rhs) { return lhs >= rhs; });
  }

  Result VisitExpr_(const AndNode *op) final {
    return EvaluateBinary(op, [](int64_t lhs, int64_t rhs) {
      return static_cast<bool>(lhs) && static_cast<bool>(rhs);
    });
  }

  Result VisitExpr_(const OrNode *op) final {
    return EvaluateBinary(op, [](int64_t lhs, int64_t rhs) {
      return static_cast<bool>(lhs) || static_cast<bool>(rhs);
    });
  }

  Result VisitExpr_(const CastNode *op) final {
    Result value = VisitExpr(op->value);
    return value ? ExactCast(*value, op->dtype) : std::nullopt;
  }

  Result VisitExpr_(const NotNode *op) final {
    Result value = VisitExpr(op->a);
    return value ? Result(!static_cast<bool>(*value)) : std::nullopt;
  }

  Result VisitExpr_(const SelectNode *op) final {
    Result condition = VisitExpr(op->condition);
    if (!condition) {
      return std::nullopt;
    }
    return VisitExpr(*condition ? op->true_value : op->false_value);
  }

  Result VisitExpr_(const CallNode *op) final {
    if (op->args.size() == 2) {
      Result lhs = VisitExpr(op->args[0]);
      Result rhs = VisitExpr(op->args[1]);
      if (!lhs || !rhs) {
        return std::nullopt;
      }
      if (op->op.same_as(builtin::bitwise_xor())) {
        return *lhs ^ *rhs;
      }
      if (op->op.same_as(builtin::bitwise_and())) {
        return *lhs & *rhs;
      }
      if (op->op.same_as(builtin::bitwise_or())) {
        return *lhs | *rhs;
      }
      if (op->op.same_as(builtin::shift_left())) {
        if (*lhs < 0 || *rhs < 0 || *rhs >= 64 ||
            *lhs > (std::numeric_limits<int64_t>::max() >> *rhs)) {
          return std::nullopt;
        }
        return *lhs << *rhs;
      }
      if (op->op.same_as(builtin::shift_right())) {
        if (*lhs < 0 || *rhs < 0 || *rhs >= 64) {
          return std::nullopt;
        }
        return *lhs >> *rhs;
      }
    } else if (op->args.size() == 1 && op->op.same_as(builtin::bitwise_not())) {
      if (Result value = VisitExpr(op->args[0])) {
        return ~*value;
      }
    }
    return std::nullopt;
  }

  Result VisitExprDefault_(const ffi::Object *op) final { return std::nullopt; }

  const std::vector<Var> *variables_{nullptr};
  const std::vector<int64_t> *values_{nullptr};
};

} // namespace

std::optional<int64_t> EvaluateConstantInteger(const PrimExpr &expr) {
  return IntegerExpressionEvaluator()(expr);
}

bool CanProveDivisible(const PrimExpr &lhs, const PrimExpr &rhs) {
  const auto *clhs = lhs.as<IntImmNode>();
  const auto *crhs = rhs.as<IntImmNode>();
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
    if (auto *op = mark->source.as<IterSumExprNode>()) {
      for (IterSplitExpr split : op->args) {
        this->CollectInternal(split->source);
        mark2splits_[split->source].push_back(split);
      }
    }
  }
};

Array<IterSplitExpr> get_unused_iters(const IterMark &mark,
                                      const std::vector<IterSplitExpr> &splits,
                                      Analyzer *analyzer) {
  PrimExpr expected_lower_factor = make_const(mark->source->dtype, 1);
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
    if (auto *sum = expr.as<IterSumExprNode>()) {
      return Visit_(sum);
    } else if (auto *split = expr.as<IterSplitExprNode>()) {
      return Visit_(split);
    } else if (auto *var = expr.as<VarNode>()) {
      return var->name_hint;
    } else {
      std::stringstream ss;
      ss << "<UNKNOWN: " << expr << ">";
      return ss.str();
    }
  }

  std::string Visit_(const IterMarkNode *op) {
    std::stringstream ss;
    ss << "(";
    ss << Visit(op->source);
    ss << ")";
    auto res = ss.str();
    marks.Set(res, op->extent);
    return res;
  }

  std::string Visit_(const IterSumExprNode *op) {
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

  std::string Visit_(const IterSplitExprNode *op) {
    std::stringstream ss;
    ss << Visit_(op->source.get());
    if (!is_one(op->lower_factor)) {
      ss << " / " << op->lower_factor;
    }
    ss << " % " << op->extent;
    if (!is_one(op->scale)) {
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
// every index into IterSumExpr, collect all splits per source Var, then
// consolidate them to avoid misclassifying a used split as unused.
Array<IterSplitExpr> DivideUnusedIterators(const Array<PrimExpr> &exprs,
                                           const Array<IterVar> input_iters,
                                           Analyzer *analyzer) {
  auto iter_sum = exprs.Map([&](const PrimExpr &e) {
    return NormalizeToIterSum(e, ToVMap(input_iters), analyzer);
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

PrimExpr MakeFlattenedExpression(const Array<arith::IterSplitExpr> &splits) {
  Array<arith::IterSplitExpr> lists;
  PrimExpr scale = 1;
  for (int i = splits.size() - 1; i >= 0; i--) {
    auto scaled_split = arith::IterSplitExpr(
        splits[i]->source, splits[i]->lower_factor, splits[i]->extent, scale);
    lists.push_back(scaled_split);
    scale *= splits[i]->extent;
  }
  return arith::NormalizeIterMapToExpr(arith::IterSumExpr(lists, 0));
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
    if (auto *op = mark->source.as<IterSumExprNode>()) {
      return IterMark(Mutate(GetRef<IterSumExpr>(op)), mark->extent);
    } else {
      return mark;
    }
  }

private:
  Map<IterSplitExpr, IterSplitExpr> replace_map_;
};

std::pair<PrimExpr, IterVar> CompressIterator(const PrimExpr &expr,
                                              const Array<IterVar> input_iters,
                                              const Var &var,
                                              arith::Analyzer *analyzer) {
  auto iter_sum =
      arith::NormalizeToIterSum(expr, ToVMap(input_iters), analyzer);
  IterMarkSplitCollector collector;
  collector.Collect({iter_sum});
  IterMark mark;
  for (const IterMark &m : collector.visited_) {
    auto v = m->source.as<Var>();
    if (v && v.value().same_as(var)) {
      mark = m;
      break;
    }
  }
  std::vector<tvm::arith::IterSplitExpr> splits;
  if (mark.defined()) {
    splits = collector.mark2splits_[mark];
  }

  PrimExpr extent = 1;
  for (const auto &split : splits) {
    extent *= split->extent;
  }
  extent = analyzer->Simplify(extent);

  auto new_var = Var(var->name_hint, var->type_annotation);
  auto new_iter_var = IterVar(Range(0, extent), new_var, IterVarType::kDataPar);
  auto new_mark = IterMark(new_var, extent);
  PrimExpr scale = 1;
  Map<IterSplitExpr, IterSplitExpr> replace_map;
  for (const auto &split : splits) {
    auto rescaled =
        arith::IterSplitExpr(new_mark, scale, split->extent, split->scale);
    replace_map.Set(split, rescaled);
    scale *= split->extent;
  }

  IterSumMutator mutator(replace_map);
  PrimExpr reaplced =
      analyzer->Simplify(NormalizeIterMapToExpr(mutator.Mutate(iter_sum)));

  return {reaplced, new_iter_var};
}

Array<IterVar> ToIterVars(const Map<Var, Range> &vmap) {
  Array<IterVar> result;
  for (const auto &[var, range] : vmap) {
    result.push_back(IterVar(range, var, IterVarType::kDataPar));
  }
  return result;
}

Map<Var, Range> ToVMap(const Array<IterVar> &ivs) {
  Map<Var, Range> result;
  for (const auto &iv : ivs) {
    result.Set(iv->var, iv->dom);
  }
  return result;
}

std::optional<std::string> GetForwardMapBijectionError(
    const Array<PrimExpr> &physical_coordinates,
    const Array<IterVar> &logical_domain, arith::Analyzer *analyzer,
    const std::string &description, bool require_zero_based) {
  ICHECK(analyzer != nullptr);

  // Only variables in logical_domain vary during this check.  Other symbols
  // (for example an enclosing serial-loop iterator) are parameters fixed for
  // one invocation of the Parallel operation.
  auto scoped_analyzer = analyzer->Clone();
  PrimExpr domain_volume = Integer(1);
  for (const IterVar &iter_var : logical_domain) {
    scoped_analyzer->Bind(iter_var->var, iter_var->dom, true);
    PrimExpr extent = analyzer->Simplify(iter_var->dom->extent);
    if (!analyzer->CanProve(extent > 0)) {
      std::ostringstream os;
      os << description << " has a logical dimension whose positive extent "
         << "cannot be proved: " << iter_var->dom;
      return os.str();
    }
    domain_volume = analyzer->Simplify(domain_volume * extent);
  }

  PrimExpr bounding_volume = Integer(1);
  std::ostringstream ranges;
  for (size_t i = 0; i < physical_coordinates.size(); ++i) {
    PrimExpr coordinate = analyzer->Simplify(physical_coordinates[i]);
    arith::IntSet coordinate_set = scoped_analyzer->int_set(coordinate);
    if (coordinate_set.IsEverything()) {
      std::ostringstream os;
      os << description << " does not form a rectangle: physical coordinate "
         << i << " has an unbounded range " << coordinate_set;
      return os.str();
    }
    PrimExpr min_value = analyzer->Simplify(coordinate_set.min());
    PrimExpr max_value = analyzer->Simplify(coordinate_set.max());
    if (require_zero_based && !analyzer->CanProveEqual(min_value, 0)) {
      std::ostringstream os;
      os << description << " is not zero-based in physical dimension " << i
         << ": minimum is " << min_value;
      return os.str();
    }
    PrimExpr extent = analyzer->Simplify(max_value - min_value + 1);
    bounding_volume = analyzer->Simplify(bounding_volume * extent);
    if (i != 0) {
      ranges << ", ";
    }
    ranges << '[' << min_value << ", " << max_value << ']';
  }

  domain_volume = analyzer->Simplify(domain_volume);
  bounding_volume = analyzer->Simplify(bounding_volume);
  if (!analyzer->CanProveEqual(domain_volume, bounding_volume)) {
    std::ostringstream os;
    os << description << " does not form a rectangle: " << domain_volume
       << " logical points map inside bounding ranges " << ranges.str()
       << " with volume " << bounding_volume;
    return os.str();
  }

  // Layout's constructor requires zero-based domains.  Normalize nonzero loop
  // minima before using the common injectivity checker; foreign symbols remain
  // untouched and therefore act as fixed parameters.
  Array<IterVar> normalized_domain;
  Map<Var, PrimExpr> normalization;
  for (size_t i = 0; i < logical_domain.size(); ++i) {
    const IterVar &iter_var = logical_domain[i];
    Var normalized_var("__tl_bijection_i" + std::to_string(i),
                       iter_var->var.dtype());
    normalized_domain.push_back(IterVar(Range(0, iter_var->dom->extent),
                                        normalized_var, IterVarType::kDataPar));
    normalization.Set(iter_var->var, normalized_var + iter_var->dom->min);
  }
  Array<PrimExpr> normalized_coordinates =
      Substitute(physical_coordinates, normalization);
  arith::IterMapResult injectivity =
      Layout(normalized_domain, normalized_coordinates)->DetectInjective();
  if (!injectivity->errors.empty()) {
    std::ostringstream os;
    os << description
       << " does not form a rectangle: the forward map is not provably "
          "one-to-one. Details: "
       << injectivity->errors;
    return os.str();
  }
  return std::nullopt;
}

namespace {

constexpr int64_t kMaxEnumeratedFragmentPoints = int64_t{1} << 30;
constexpr int64_t kMaxEnumeratedContainmentChecks = int64_t{1} << 20;

enum class FragmentBijectionStatus { kValid, kInvalid, kUnavailable };

struct FragmentBijectionResult {
  FragmentBijectionStatus status;
  std::string detail;
};

FragmentBijectionResult EnumerateFragmentBijection(const Fragment &fragment,
                                                   arith::Analyzer *analyzer) {
  Array<PrimExpr> logical_shape = fragment->InputShape();
  logical_shape.push_back(fragment->ReplicateExtent());
  auto scoped_analyzer = analyzer->Clone();
  for (size_t i = 0; i < fragment->InputDim(); ++i) {
    scoped_analyzer->Bind(InputPlaceholder(i),
                          Range(0, fragment->InputShape()[i]), true);
  }
  scoped_analyzer->Bind(ReplicationPlaceholder(),
                        Range(0, fragment->ReplicateExtent()), true);
  arith::IntSet thread_set =
      scoped_analyzer->int_set(fragment->GetForwardThread());
  if (thread_set.IsEverything()) {
    return {FragmentBijectionStatus::kUnavailable, ""};
  }
  PrimExpr thread_min = analyzer->Simplify(thread_set.min());
  PrimExpr thread_extent =
      analyzer->Simplify(thread_set.max() - thread_min + 1);
  Array<PrimExpr> physical_shape{thread_extent};
  Array<PrimExpr> output_shape = fragment->OutputShape();
  physical_shape.insert(physical_shape.end(), output_shape.begin(),
                        output_shape.end());

  bool unavailable = false;
  bool exceeds_limit = false;
  std::string invalid_extent;
  auto collect_extents = [&](const Array<PrimExpr> &shape,
                             const char *domain_name,
                             std::vector<int64_t> *extents, int64_t *volume) {
    *volume = 1;
    extents->reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      PrimExpr simplified = analyzer->Simplify(shape[i]);
      std::optional<int64_t> extent = EvaluateConstantInteger(simplified);
      if (!extent) {
        unavailable = true;
        return;
      }
      if (*extent <= 0) {
        std::ostringstream os;
        os << "the fragment forward map has a non-positive " << domain_name
           << " extent in dimension " << i << ": " << *extent;
        invalid_extent = os.str();
        return;
      }
      if (*extent > kMaxEnumeratedFragmentPoints / *volume) {
        exceeds_limit = true;
        return;
      }
      *volume *= *extent;
      extents->push_back(*extent);
    }
  };

  std::vector<int64_t> logical_extents;
  std::vector<int64_t> physical_extents;
  int64_t logical_volume = 1;
  int64_t physical_volume = 1;
  collect_extents(logical_shape, "logical", &logical_extents, &logical_volume);
  if (!invalid_extent.empty()) {
    return {FragmentBijectionStatus::kInvalid, invalid_extent};
  }
  if (!unavailable && !exceeds_limit) {
    collect_extents(physical_shape, "physical", &physical_extents,
                    &physical_volume);
  }
  if (!invalid_extent.empty()) {
    return {FragmentBijectionStatus::kInvalid, invalid_extent};
  }
  if (exceeds_limit) {
    LOG(WARNING) << "Skipping exhaustive fragment-bijection enumeration above "
                    "the limit of "
                 << kMaxEnumeratedFragmentPoints
                 << " points; falling back to symbolic proof. Fragment: "
                 << fragment->DebugOutput();
    return {FragmentBijectionStatus::kUnavailable, ""};
  }
  if (unavailable) {
    return {FragmentBijectionStatus::kUnavailable, ""};
  }
  if (logical_volume != physical_volume) {
    std::ostringstream os;
    os << "the fragment forward map does not form a rectangle: "
       << logical_volume << " logical points map into a physical rectangle "
       << "with volume " << physical_volume;
    return {FragmentBijectionStatus::kInvalid, os.str()};
  }
  std::optional<int64_t> thread_min_value =
      EvaluateConstantInteger(analyzer->Simplify(thread_min));
  if (!thread_min_value) {
    return {FragmentBijectionStatus::kUnavailable, ""};
  }

  Array<PrimExpr> forward_coordinates{fragment->GetForwardThread()};
  Array<PrimExpr> forward_indices = fragment->GetForwardIndex();
  forward_coordinates.insert(forward_coordinates.end(), forward_indices.begin(),
                             forward_indices.end());
  std::vector<uint64_t> occupied(
      static_cast<size_t>((physical_volume + 63) / 64), uint64_t{0});
  std::vector<int64_t> logical_coordinate(logical_extents.size());
  std::vector<Var> logical_variables;
  logical_variables.reserve(logical_extents.size());
  for (size_t i = 0; i < fragment->InputDim(); ++i) {
    logical_variables.push_back(InputPlaceholder(i));
  }
  logical_variables.push_back(ReplicationPlaceholder());
  IntegerExpressionEvaluator evaluator(&logical_variables, &logical_coordinate);

  for (int64_t linear = 0; linear < logical_volume; ++linear) {
    int64_t residual = linear;
    for (size_t rev = logical_extents.size(); rev > 0; --rev) {
      size_t i = rev - 1;
      logical_coordinate[i] = residual % logical_extents[i];
      residual /= logical_extents[i];
    }

    int64_t physical_linear = 0;
    for (size_t i = 0; i < forward_coordinates.size(); ++i) {
      std::optional<int64_t> coordinate = evaluator(forward_coordinates[i]);
      if (!coordinate) {
        return {FragmentBijectionStatus::kUnavailable, ""};
      }
      if (i == 0) {
        *coordinate -= *thread_min_value;
      }
      if (*coordinate < 0 || *coordinate >= physical_extents[i]) {
        std::ostringstream os;
        os << "the fragment forward map does not form a rectangle: physical "
              "coordinate "
           << i << " evaluates to " << *coordinate << " outside [0, "
           << physical_extents[i] << ')';
        return {FragmentBijectionStatus::kInvalid, os.str()};
      }
      physical_linear = physical_linear * physical_extents[i] + *coordinate;
    }

    uint64_t mask = uint64_t{1} << (physical_linear % 64);
    uint64_t &word = occupied[static_cast<size_t>(physical_linear / 64)];
    if ((word & mask) != 0) {
      std::ostringstream os;
      os << "the fragment forward map does not form a rectangle: multiple "
            "logical points map to physical cell "
         << physical_linear;
      return {FragmentBijectionStatus::kInvalid, os.str()};
    }
    word |= mask;
  }
  // The logical and physical volumes are equal, so an injective in-bounds map
  // also visits every cell in the physical rectangle exactly once.
  return {FragmentBijectionStatus::kValid, ""};
}

enum class FragmentContainmentStatus { kValid, kInvalid, kUnavailable };

FragmentContainmentStatus
EnumerateFragmentContains(const Fragment &small_frag,
                          const Fragment &large_frag,
                          const Array<PrimExpr> &small_frag_indices,
                          const Array<PrimExpr> &large_frag_indices,
                          arith::Analyzer *analyzer, bool check_forward_index) {
  ICHECK(analyzer != nullptr);
  if (small_frag_indices.size() != small_frag->InputDim() ||
      large_frag_indices.size() != large_frag->InputDim()) {
    return FragmentContainmentStatus::kUnavailable;
  }

  std::vector<Var> logical_variables;
  std::vector<int64_t> logical_extents;
  logical_variables.reserve(small_frag_indices.size() + 2);
  logical_extents.reserve(small_frag_indices.size());
  int64_t logical_volume = 1;
  for (size_t i = 0; i < small_frag_indices.size(); ++i) {
    const auto *var_node = small_frag_indices[i].as<VarNode>();
    std::optional<int64_t> extent = EvaluateConstantInteger(
        analyzer->Simplify(small_frag->InputShape()[i]));
    if (var_node == nullptr || !extent || *extent <= 0 ||
        *extent > kMaxEnumeratedContainmentChecks / logical_volume) {
      return FragmentContainmentStatus::kUnavailable;
    }
    Var var = GetRef<Var>(var_node);
    if (std::any_of(logical_variables.begin(), logical_variables.end(),
                    [&](const Var &other) { return other.same_as(var); })) {
      return FragmentContainmentStatus::kUnavailable;
    }
    logical_variables.push_back(var);
    logical_extents.push_back(*extent);
    logical_volume *= *extent;
  }

  std::optional<int64_t> small_replicas = EvaluateConstantInteger(
      analyzer->Simplify(small_frag->ReplicateExtent()));
  std::optional<int64_t> large_replicas = EvaluateConstantInteger(
      analyzer->Simplify(large_frag->ReplicateExtent()));
  if (!small_replicas || !large_replicas || *small_replicas <= 0 ||
      *large_replicas <= 0 ||
      *small_replicas > kMaxEnumeratedContainmentChecks / logical_volume ||
      *large_replicas > kMaxEnumeratedContainmentChecks /
                            (logical_volume * *small_replicas)) {
    return FragmentContainmentStatus::kUnavailable;
  }

  std::vector<int64_t> large_shape;
  large_shape.reserve(large_frag->InputDim());
  for (const PrimExpr &extent_expr : large_frag->InputShape()) {
    std::optional<int64_t> extent =
        EvaluateConstantInteger(analyzer->Simplify(extent_expr));
    if (!extent || *extent <= 0) {
      return FragmentContainmentStatus::kUnavailable;
    }
    large_shape.push_back(*extent);
  }

  Var small_rep("__tl_contains_small_rep",
                small_frag->ReplicateExtent()->dtype);
  Var large_rep("__tl_contains_large_rep",
                large_frag->ReplicateExtent()->dtype);
  PrimExpr small_thread =
      small_frag->ForwardThread(small_frag_indices, small_rep);
  PrimExpr large_thread =
      large_frag->ForwardThread(large_frag_indices, large_rep);
  Array<PrimExpr> small_physical = small_frag->Forward(small_frag_indices);
  Array<PrimExpr> large_physical = large_frag->Forward(large_frag_indices);
  if (check_forward_index && small_physical.size() != large_physical.size()) {
    return FragmentContainmentStatus::kInvalid;
  }

  logical_variables.push_back(small_rep);
  logical_variables.push_back(large_rep);
  std::vector<int64_t> values(logical_variables.size(), 0);
  IntegerExpressionEvaluator evaluator(&logical_variables, &values);

  for (int64_t linear = 0; linear < logical_volume; ++linear) {
    int64_t residual = linear;
    for (size_t rev = logical_extents.size(); rev > 0; --rev) {
      size_t i = rev - 1;
      values[i] = residual % logical_extents[i];
      residual /= logical_extents[i];
    }

    for (size_t i = 0; i < large_frag_indices.size(); ++i) {
      std::optional<int64_t> index = evaluator(large_frag_indices[i]);
      if (!index) {
        return FragmentContainmentStatus::kUnavailable;
      }
      if (*index < 0 || *index >= large_shape[i]) {
        return FragmentContainmentStatus::kInvalid;
      }
    }
    if (check_forward_index) {
      for (size_t i = 0; i < small_physical.size(); ++i) {
        std::optional<int64_t> small_index = evaluator(small_physical[i]);
        std::optional<int64_t> large_index = evaluator(large_physical[i]);
        if (!small_index || !large_index) {
          return FragmentContainmentStatus::kUnavailable;
        }
        if (*small_index != *large_index) {
          return FragmentContainmentStatus::kInvalid;
        }
      }
    }

    for (int64_t small_replica = 0; small_replica < *small_replicas;
         ++small_replica) {
      values[logical_extents.size()] = small_replica;
      std::optional<int64_t> expected_thread = evaluator(small_thread);
      if (!expected_thread) {
        return FragmentContainmentStatus::kUnavailable;
      }
      bool found_owner = false;
      for (int64_t large_replica = 0; large_replica < *large_replicas;
           ++large_replica) {
        values[logical_extents.size() + 1] = large_replica;
        std::optional<int64_t> owner_thread = evaluator(large_thread);
        if (!owner_thread) {
          return FragmentContainmentStatus::kUnavailable;
        }
        if (*owner_thread == *expected_thread) {
          found_owner = true;
          break;
        }
      }
      if (!found_owner) {
        return FragmentContainmentStatus::kInvalid;
      }
    }
  }
  return FragmentContainmentStatus::kValid;
}

} // namespace

std::optional<std::string>
GetFragmentBijectionError(const Fragment &fragment, arith::Analyzer *analyzer) {
  ICHECK(fragment.defined());
  ICHECK(analyzer != nullptr);

  FragmentBijectionResult enumeration =
      EnumerateFragmentBijection(fragment, analyzer);
  if (enumeration.status == FragmentBijectionStatus::kValid) {
    return std::nullopt;
  }
  if (enumeration.status == FragmentBijectionStatus::kInvalid) {
    return enumeration.detail;
  }

  Array<IterVar> logical_domain;
  for (size_t i = 0; i < fragment->InputDim(); ++i) {
    Range domain(0, fragment->InputShape()[i]);
    logical_domain.push_back(
        IterVar(domain, InputPlaceholder(i), IterVarType::kDataPar));
  }
  Range replicate_domain(0, fragment->ReplicateExtent());
  logical_domain.push_back(IterVar(replicate_domain, ReplicationPlaceholder(),
                                   IterVarType::kDataPar));

  Array<PrimExpr> physical_coordinates{fragment->GetForwardThread()};
  Array<PrimExpr> forward_indices = fragment->GetForwardIndex();
  physical_coordinates.insert(physical_coordinates.end(),
                              forward_indices.begin(), forward_indices.end());

  auto scoped_analyzer = analyzer->Clone();
  for (const IterVar &iter_var : logical_domain) {
    scoped_analyzer->Bind(iter_var->var, iter_var->dom, true);
  }
  arith::IntSet thread_set =
      scoped_analyzer->int_set(fragment->GetForwardThread());
  if (thread_set.IsEverything()) {
    return "the fragment forward map has an unbounded thread range";
  }
  physical_coordinates.Set(
      0, analyzer->Simplify(fragment->GetForwardThread() - thread_set.min()));

  return GetForwardMapBijectionError(physical_coordinates, logical_domain,
                                     analyzer, "the fragment forward map",
                                     /*require_zero_based=*/true);
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
                           Analyzer &analyzer, bool check_forward_index) {
  // When check_forward_index is true, verify that the physical indices
  // (forward index) of both fragments are equal. This is required when
  // validating loop layout against buffer fragment, as code generation
  // needs to correctly derive buffer physical indices from loop layout.
  bool large_physical_is_fully_replicated = large_frag->IsCompletedReplicated();
  if (large_physical_is_fully_replicated) {
    return true; // fully replicated fragments are always compatible
  }

  if (check_forward_index) {
    auto small_physical = small_frag->Forward(small_frag_indices);
    auto large_physical = large_frag->Forward(large_frag_indices);
    // Dimension mismatch means they are not equal.
    if (small_physical.size() != large_physical.size()) {
      return false;
    }
    // Check each physical index component for equality.
    for (size_t i = 0; i < small_physical.size(); i++) {
      auto diff = analyzer.Simplify(small_physical[i] - large_physical[i]);
      if (!analyzer.CanProve(diff == 0)) {
        return false;
      }
    }
  }

  Var rep_small("__checking_frag_contains_rep");
  analyzer.Bind(rep_small,
                Range(IntImm(small_frag->ReplicateExtent()->dtype, 0),
                      small_frag->ReplicateExtent()),
                true); // Bind the replicate extent of small_frag.
  PrimExpr small_thread_base =
      small_frag->ThreadRange().defined()
          ? small_frag->ThreadRange()->min
          : make_zero(small_frag->GetForwardThread()->dtype);
  PrimExpr large_thread_base =
      large_frag->ThreadRange().defined()
          ? large_frag->ThreadRange()->min
          : make_zero(large_frag->GetForwardThread()->dtype);
  // Compare physical participant ids.  Fragment forward-thread expressions are
  // normalized to their own ThreadRange, whose minimum may differ for a sliced
  // Parallel operation.
  auto thread = small_frag->ForwardThread(small_frag_indices, rep_small) +
                small_thread_base;

  // Get physical index and thread for large_frag.
  auto large_frag_physical_and_thread = large_frag->Forward(large_frag_indices);
  // The large fragment inverse consumes a thread coordinate normalized to its
  // own participant range.
  large_frag_physical_and_thread.push_back(thread - large_thread_base);
  // Get the inverse of the large fragment.
  auto inv_large_frag = large_frag->Inverse();
  // Compute logical index and replicate index using inverse layout.
  auto inv_large_frag_logical_and_rep =
      inv_large_frag->Forward(large_frag_physical_and_thread);

  // Extract replicate index from the result.
  auto inv_large_frag_rep =
      inv_large_frag_logical_and_rep[inv_large_frag_logical_and_rep.size() - 1];

  // Calculate thread based on the logical index and replicate index.
  auto check_thread =
      large_frag->ForwardThread(large_frag_indices, inv_large_frag_rep) +
      large_thread_base;

  // Simplify the difference between the threads.
  auto diff = analyzer.Simplify(thread - check_thread);
  // If the difference is zero, the threads match and the access is valid.
  if (analyzer.CanProve(diff == 0)) {
    return true;
  }

  // Symbolic inversion can be inconclusive for valid many-to-one accesses,
  // such as `fragment[i, j // 32]`. Check the ownership relation directly:
  // every thread selected by the small fragment must appear in the forward
  // thread image of the accessed large-fragment element. Keep this fallback
  // deliberately bounded because it evaluates every logical point and pair of
  // replica coordinates.
  return EnumerateFragmentContains(small_frag, large_frag, small_frag_indices,
                                   large_frag_indices, &analyzer,
                                   check_forward_index) ==
         FragmentContainmentStatus::kValid;
}

} // namespace tl
} // namespace tvm
