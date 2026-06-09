/*!
 * \file tl/cuda/target_utils.h
 * \brief CUDA target attribute helpers.
 */

#ifndef TVM_TL_CUDA_TARGET_UTILS_H_
#define TVM_TL_CUDA_TARGET_UTILS_H_

#include <tvm/runtime/data_type.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsCuda(Target target);
bool TargetIsCuTeDSL(Target target);

bool TargetIsVolta(Target target);
bool TargetIsTuring(Target target);
bool TargetIsAmpere(Target target);
bool TargetIsHopper(Target target);
bool TargetIsSm100(Target target);
bool TargetIsSM120(Target target);

bool TargetCudaHasAsyncCopy(Target target);
int TargetCudaGetWarpSize(Target target);
bool TargetHasLdmatrix(Target target);
bool TargetHasStmatrix(Target target);
bool TargetHasTmem(Target target);
bool TargetHasBulkCopy(Target target);
bool TargetSupportVectorize256(Target target);
bool TargetHasSMVersionGE(Target target, int version);

bool IsCudaVectorizableFP8(DataType dtype);
bool IsCudaVectorizableCast(DataType from_ty, DataType target_ty);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_CUDA_TARGET_UTILS_H_
