/*!
 * \file tl/transform/reducer/reducer_metadata.h
 * \brief Shared metadata and semantic verification helpers for reducers.
 */

#ifndef TVM_TL_TRANSFORM_REDUCER_REDUCER_METADATA_H_
#define TVM_TL_TRANSFORM_REDUCER_REDUCER_METADATA_H_

#include "op/deferred_reducer.h"

#include <tvm/tirx/function.h>

#include <unordered_map>

namespace tvm {
namespace tl {

using ReducerInfoMap =
    std::unordered_map<tirx::Var, ReducerInfo, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;
using ReducerBufferMap =
    std::unordered_map<tirx::Var, tirx::Buffer, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;

/*! \brief Reducer definitions and their corresponding logical allocations. */
struct ReducerMetadata {
  ReducerInfoMap info;
  ReducerBufferMap buffers;
};

/*! \brief Collect and validate reducer definitions and allocations. */
ReducerMetadata CollectReducerMetadata(const tirx::PrimFunc &func);

/*! \brief Verify reducer lifecycle, accesses, and contribution expressions. */
void VerifyReducerEpochSemantics(const tirx::PrimFunc &func,
                                 const ReducerMetadata &metadata);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_REDUCER_REDUCER_METADATA_H_
