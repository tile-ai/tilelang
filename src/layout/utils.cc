/*!
 * \file layout/utils.cc
 * \brief Some arith tools for layout & fragment inference
 *
 */

#include "utils.h"
#include "tvm/arith/iter_affine_map.h"
#include "tvm/ffi/container/map.h"
#include "tvm/node/functor.h"
#include "tvm/node/repr_printer.h"
#include "tvm/node/structural_equal.h"

#include <sstream>
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
          analyzer->Simplify(
              FloorDiv(splits[lowest]->lower_factor, expected_lower_factor)),
          1);
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
    results.emplace_back(
        mark, expected_lower_factor,
        analyzer->Simplify(FloorDiv(mark->extent, expected_lower_factor)), 1);
  }
  return results;
}

struct IterExprPP {
  // std::vector<std::pair<std::string, PrimExpr>> marks;
  ffi::Map<ffi::String, PrimExpr> marks;
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .clear_dispatch<IterMarkNode>()
    .set_dispatch<IterMarkNode>([](const ObjectRef &obj, ReprPrinter *p) {
      auto *node = static_cast<const IterMarkNode *>(obj.get());
      IterExprPP pp(tvm::ffi::GetRef<IterMark>(node));
      p->stream << pp;
    })
    .clear_dispatch<IterSumExprNode>()
    .set_dispatch<IterSumExprNode>([](const ObjectRef &obj, ReprPrinter *p) {
      auto *node = static_cast<const IterSumExprNode *>(obj.get());
      IterExprPP pp(tvm::ffi::GetRef<IterSumExpr>(node));
      p->stream << pp;
    })
    .clear_dispatch<IterSplitExprNode>()
    .set_dispatch<IterSplitExprNode>([](const ObjectRef &obj, ReprPrinter *p) {
      auto *node = static_cast<const IterSplitExprNode *>(obj.get());
      IterExprPP pp(tvm::ffi::GetRef<IterSplitExpr>(node));
      p->stream << pp;
    });

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
      if (!is_zero(diff)) {
        return false;
      }
    }
  }

  Var rep_small("__checking_frag_contains_rep");
  analyzer.Bind(rep_small,
                Range(IntImm(small_frag->ReplicateExtent()->dtype, 0),
                      small_frag->ReplicateExtent()),
                true); // Bind the replicate extent of small_frag.
  // Derive thread for small_frag.
  auto thread = small_frag->ForwardThread(small_frag_indices, rep_small);

  // Get physical index and thread for large_frag.
  auto large_frag_physical_and_thread = large_frag->Forward(large_frag_indices);
  // Add small_frag's thread to the large fragment's thread info.
  large_frag_physical_and_thread.push_back(thread);
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
      large_frag->ForwardThread(large_frag_indices, inv_large_frag_rep);

  // Simplify the difference between the threads.
  auto diff = analyzer.Simplify(thread - check_thread);
  // If the difference is zero, the threads match and the access is valid.
  return is_zero(diff);
}

} // namespace tl
} // namespace tvm
