/*!
 * \file layout/utils.cc
 * \brief Some arith tools for layout & fragment inference
 *
 */

#include "utils.h"

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tir;
using namespace arith;

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
          FloorDiv(splits[lowest]->lower_factor, expected_lower_factor), 1);
      expected_lower_factor = splits[lowest]->lower_factor;
    } else {
      used[j] = true;
      i++;
      expected_lower_factor = splits[j]->lower_factor * splits[j]->extent;
    }
  }
  bool match_full_iter =
      analyzer->CanProveEqual(expected_lower_factor, mark->extent);
  if (!match_full_iter) {
    results.emplace_back(mark, expected_lower_factor,
                         FloorDiv(mark->extent, expected_lower_factor), 1);
  }
  return results;
}

// Heuristic: detect per-iterator gaps ("unused" pieces) even when the iterator
// appears in fused forms across multiple index expressions. We first normalize
// every index into IterSumExpr, collect all splits per source Var, then
// consolidate them to avoid misclassifying a used split as unused.
Array<IterSplitExpr> DivideUnusedIterators(const Array<PrimExpr> &exprs,
                                           const Array<IterVar> input_iters,
                                           Analyzer *analyzer) {
  auto iter_sum = exprs.Map([&](const auto &e) {
    return NormalizeToIterSum(e, ToVMap(input_iters), analyzer);
  });
  IterMarkSplitCollector collector;
  collector.Collect(iter_sum);
  Array<IterSplitExpr> results;

  for (const IterMark &mark : collector.visited_) {
    if (!mark->source.as<Var>()) {
      std::ostringstream oss;
      oss << "Not a normalized iterator: " << mark;
      throw NormalizeIterException(oss.str());
    }
  }

  for (const IterVar &iter : input_iters) {
    // Merge splits from all IterMark that share the same source Var as `iter`.
    std::vector<IterSplitExpr> merged_splits;
    for (const IterMark &mark : collector.visited_) {
      auto vexpr = mark->source.as<Var>();
      if (vexpr && vexpr.value().same_as(iter->var)) {
        auto it = collector.mark2splits_.find(mark);
        if (it != collector.mark2splits_.end()) {
          const auto &vec = it->second;
          merged_splits.insert(merged_splits.end(), vec.begin(), vec.end());
        }
      }
    }

    if (!merged_splits.empty()) {
      // Use a unified mark (Var + full extent) to compute the missing pieces
      // so that fused usages are honored as "used" and not reintroduced.
      IterMark unified_mark(iter->var, iter->dom->extent);
      auto splits = get_unused_iters(unified_mark, merged_splits, analyzer);
      // Put the small axis last for a flattened ordering.
      results.insert(results.end(), splits.rbegin(), splits.rend());
    } else if (!is_one(iter->dom->extent)) {
      auto mark = IterMark(iter->var, iter->dom->extent);
      auto split = IterSplitExpr(mark, 1, iter->dom->extent, 1);
      results.push_back(split);
    }
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
      return IterMark(Mutate(tvm::ffi::GetRef<IterSumExpr>(op)), mark->extent);
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
    ICHECK(m->source.as<Var>()) << "Not a normalized iterator: " << mark;
    if (m->source.as<Var>().value().same_as(var)) {
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

} // namespace tl
} // namespace tvm
