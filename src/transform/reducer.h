/*!
 * \file tl/transform/reducer.h
 * \brief Verification and materialization passes for deferred reducers.
 */

#ifndef TVM_TL_TRANSFORM_REDUCER_H_
#define TVM_TL_TRANSFORM_REDUCER_H_

#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

/*! \brief Verify first-class reducer lifecycle and access legality. */
TVM_DLL tvm::transform::Pass VerifyReducerEpochs();

/*! \brief Materialize opaque reducer handles as full per-thread local arrays.
 */
TVM_DLL tvm::transform::Pass PlanAndMaterializeReducers();

/*! \brief Check that all reducer-only IR has been consumed before codegen. */
TVM_DLL tvm::transform::Pass VerifyReducerLowered();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_REDUCER_H_
