/*!
 * \file tl/cuda/target_utils.h
 * \brief CUDA target attribute helpers.
 */

#ifndef TVM_TL_CUDA_TARGET_UTILS_H_
#define TVM_TL_CUDA_TARGET_UTILS_H_

#include <tvm/ir/type.h>
#include <tvm/target/target.h>

#include "support/base.h"

namespace tvm {
namespace tl {

TILELANG_DLL bool TargetIsCuda(Target target);
TILELANG_DLL bool TargetIsCuTeDSL(Target target);

TILELANG_DLL bool TargetIsVolta(Target target);
TILELANG_DLL bool TargetIsTuring(Target target);
TILELANG_DLL bool TargetIsAmpere(Target target);
TILELANG_DLL bool TargetIsHopper(Target target);
TILELANG_DLL bool TargetIsSm100(Target target);
TILELANG_DLL bool TargetIsSM120(Target target);

TILELANG_DLL bool TargetCudaHasAsyncCopy(Target target);
TILELANG_DLL int TargetCudaGetWarpSize(Target target);
TILELANG_DLL bool TargetHasLdmatrix(Target target);
TILELANG_DLL bool TargetHasStmatrix(Target target, bool is_m16n8 = false);
TILELANG_DLL bool TargetHasTmem(Target target);
TILELANG_DLL bool TargetHasBulkCopy(Target target);
TILELANG_DLL bool TargetSupportsNamedBarrier(Target target);
TILELANG_DLL bool TargetSupportVectorize256(Target target);
TILELANG_DLL bool TargetHasSMVersionGE(Target target, int version);

TILELANG_DLL bool IsCudaVectorizableFP8(PrimType dtype);
TILELANG_DLL bool IsCudaVectorizableCast(PrimType from_ty, PrimType target_ty);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_CUDA_TARGET_UTILS_H_
