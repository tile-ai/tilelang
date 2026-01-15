/*!
 * \file atomicadd_vectorize.h
 * \brief Vectorization pass for atomic add operations.
 */

#ifndef TVM_TL_ATOMICADD_VECTORIZE_H_
#define TVM_TL_ATOMICADD_VECTORIZE_H_

#include "../op/builtin.h"
#include "../target/utils.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Vectorize atomic add operations inside vectorized loops.
 *
 * This function detects atomic_add_elem_op inside ForKind::kVectorized loops
 * and converts them to vectorized versions (atomic_addx2_elem_op or
 * atomic_addx4_elem_op) based on the loop extent and data type.
 *
 * \param for_node The For loop to process.
 * \return The transformed For loop.
 */
For VectorizeAtomicAdd(const For &for_node);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ATOMICADD_VECTORIZE_H_
