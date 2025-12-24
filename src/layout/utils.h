/*!
 * \file layout/utils.h
 * \brief Some arith tools for layout & fragment inference
 *
 */

#ifndef TVM_TL_LAYOUT_UTILS_H_
#define TVM_TL_LAYOUT_UTILS_H_

#include <tvm/arith/iter_affine_map.h>

#include "../support/ffi_aliases.h"

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

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_UTILS_H_
