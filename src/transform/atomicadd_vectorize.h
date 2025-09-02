/*!
 * \file atomicadd_vectorize.h
 * \brief A tool to automatically vectorize a for atomicadd
 */

#ifndef TVM_TL_ATOMICADD_VECTORIZE_H_
#define TVM_TL_ATOMICADD_VECTORIZE_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

For VectorizeAtomicAdd(const For &for_node, const Var &thread_var,
                       const Range &thread_bounds, int compute_capability);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ATOMICADD_VECTORIZE_H_