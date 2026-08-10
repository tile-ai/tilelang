/*!
 * \file tl/transform/reducer/vectorize_reducer.h
 * \brief Reducer-aware vectorization after physical layout materialization.
 */

#ifndef TVM_TL_TRANSFORM_REDUCER_VECTORIZE_REDUCER_H_
#define TVM_TL_TRANSFORM_REDUCER_VECTORIZE_REDUCER_H_

#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

/*!
 * \brief Vectorize planned reducer updates using their materialized physical
 * access patterns, then remove all reducer-update markers.
 */
tirx::Stmt VectorizeReducerUpdates(tirx::Stmt stmt, Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_REDUCER_VECTORIZE_REDUCER_H_
