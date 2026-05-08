/*!
 * \file layout/utils.h
 * \brief Some arith tools for layout & fragment inference
 *
 */

#ifndef TVM_TL_LAYOUT_UTILS_H_
#define TVM_TL_LAYOUT_UTILS_H_

#include <tvm/arith/iter_affine_map.h>

#include "../support/ffi_aliases.h"
#include "layout.h"

namespace tvm {
namespace tl {

using namespace tir;

class NormalizeIterException : public std::exception {
public:
  const char *what() const noexcept override { return msg_.c_str(); }
  NormalizeIterException(const std::string &msg) : msg_(msg) {}

private:
  std::string msg_;
};

/*!
 * \brief Collect the IterSplit that is not used in expr.
 *
 *  If the expr is (x // 2) and x is in Range(4),
 *  than the result should be (x % 2)
 */
Array<arith::IterSplitExpr>
DivideUnusedIterators(const Array<PrimExpr> &exprs,
                      const Array<IterVar> input_iters,
                      arith::Analyzer *analyzer);

/*!
 * \brief Compress the iterator var, remove the unused part of the var not
 * present in the expr
 *
 *  Returns the compressed IterVar as well as the Updated iter sum expression.
 */
std::pair<PrimExpr, IterVar> CompressIterator(const PrimExpr &expr,
                                              const Array<IterVar> input_iters,
                                              const Var &var,
                                              arith::Analyzer *analyzer);

/*!
 * \brief Convert the iter splits returned by DivideUnusedIterators into
 * flattened expression
 *
 */
PrimExpr MakeFlattenedExpression(const Array<arith::IterSplitExpr> &splits);

/*!
 * \brief Convert an Array of IterVar to a Map object
 *
 */
Map<Var, Range> ToVMap(const Array<IterVar> &ivs);

/*!
 * \brief Convert a Map object to an Array of IterVar
 *
 */
Array<IterVar> ToIterVars(const Map<Var, Range> &vmap);

/*!
 * \brief Check whether the threads that access elements of a smaller fragment
 *        are a subset of the threads that access elements of a larger fragment.
 *
 * This function ensures that if the small fragment's layout corresponds to the
 * loop itself, accessing the large fragment's elements is valid. Additionally,
 * if small is updated to large, the originally valid access remains valid.
 *
 * \param small_frag The smaller fragment to check
 * \param large_frag The larger fragment to check against
 * \param small_frag_indices The indices used to access small_frag
 * \param large_frag_indices The indices used to access large_frag
 * \param analyzer The analyzer for simplification
 * \param check_forward_index Whether to also check physical index equality
 * \return true if small_frag's threads are contained in large_frag's threads
 */
bool ProveFragmentContains(Fragment small_frag, Fragment large_frag,
                           Array<PrimExpr> small_frag_indices,
                           Array<PrimExpr> large_frag_indices,
                           arith::Analyzer &analyzer,
                           bool check_forward_index = false);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_UTILS_H_
