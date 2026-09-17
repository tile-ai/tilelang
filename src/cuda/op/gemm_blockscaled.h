/*!
 * \file tl/cuda/op/gemm_blockscaled.h
 * \brief CUDA instruction selection for block-scaled GEMM.
 */

#ifndef TVM_TL_CUDA_OP_GEMM_BLOCKSCALED_H_
#define TVM_TL_CUDA_OP_GEMM_BLOCKSCALED_H_

#include "op/gemm_blockscaled.h"

namespace tvm {
namespace tl {
namespace cuda {

/*! \brief Select a block-scaled instruction without a dense GEMM fallback. */
ffi::String SelectBlockScaledGemmInst(const GemmBlockScaled &op, int block_size,
                                      const Target &target);

} // namespace cuda
} // namespace tl
} // namespace tvm

#endif // TVM_TL_CUDA_OP_GEMM_BLOCKSCALED_H_
