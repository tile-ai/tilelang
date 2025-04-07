// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/builtin.h
 * \brief Builtin intrinsics.
 *
 */

#ifndef TVM_TL_OP_BUILTIN_H_
#define TVM_TL_OP_BUILTIN_H_

#include "op.h"
#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

static constexpr const char *kDisableTMALower = "tl.disable_tma_lower";

static constexpr const char *kConfigIndexBitwidth = "tl.config_index_bitwidth";

/*!
 * \brief tvm intrinsics for TMADescriptor creation for tiled load
 *
 * CuTensorMap* CreateTMADescriptorOp(data_type, rank, global_addr,
 * global_shape..., global_stride..., smem_box..., smem_stride..., interleave,
 * swizzle, l2_promotion, oob_fill)
 *
 */
const Op &CreateTMADescriptorOp();

/*!
 * \brief tvm intrinsics for TMADescriptor creation for image to column load
 *
 * CuTensorMap* CreateTMAIm2ColDescriptorOp(data_type, rank, global_addr,
 * global_shape..., global_stride..., elem_stride..., lower_corner...,
 * upper_corner..., smme_box_pixel, smem_box_channel, interleave, swizzle,
 * l2_promotion, oob_fill)
 *
 */
const Op &CreateTMAIm2ColDescriptorOp();

/*!
 * \brief Create a list of mbarrier with num_threads
 *
 * CreateListofMBarrierOp(num_threads0, num_threads1, ...)
 *
 */
const Op &CreateListofMBarrierOp();

/*!
 * \brief Get the mbarrier with barrier_id
 *
 * int64_t* GetMBarrier(barrier_id)
 *
 */
const Op &GetMBarrierOp();

/*!
 * \brief tvm intrinsics for loading data from global tensor descriptor to
 * shared memory
 *
 * TMALoadOp(descriptor, mbarrier, smem_data, coord_0, coord_1, ...)
 *
 */
const Op &TMALoadOp();

/*!
 * \brief tvm intrinsics for loading image from global tensor to columns in
 * shared memory
 *
 * TMALoadOp(descriptor, mbarrier, smem_data, coord_0, coord_1, ...,
 * image_offset, ...)
 *
 */
const Op &TMALoadIm2ColOp();

/*!
 * \brief tvm intrinsics for storing data from shared memory to global tensor
 * descriptor
 *
 * TMAStoreOp(descriptor, smem_data, coord_0, coord_1, ...)
 *
 */
const Op &TMAStoreOp();

/*!
 * \brief tvm intrinsics for mbarrier wait with parity bit
 *
 * MBarrierWaitParity(mbarrier, parity)
 *
 */
const Op &MBarrierWaitParity();

/*!
 * \brief tvm intrinsics for mbarrier expect tx
 *
 * MBarrierExpectTX(mbarrier, transaction_bytes)
 *
 */
const Op &MBarrierExpectTX();

/*!
 * \brief tvm intrinsics for ldmatrix
 *
 * LDMatrixOp(transposed, num, shared_addr, local_addr)
 *
 */
const Op &LDMatrixOp();

/*!
 * \brief tvm intrinsics for stmatrix
 *
 * LDMatrixOp(transposed, num, shared_addr, int32_values...)
 *
 */
const Op &STMatrixOp();

/*!
 * \brief Pack two b16 value into a b32 value
 *
 * int32 PackB16Op(b16_value, b16_value)
 *
 */
const Op &PackB16Op();

/*!
 * \brief Similar to __syncthreads(), but can be used to sync partial threads
 *
 * SyncThreadsPartialOp(num_partial_threads or mbarrier)
 *
 */
const Op &SyncThreadsPartialOp();

/*!
 * \brief Issue a shared memory fence for async operations
 *
 * FenceProxyAsync()
 *
 */
const Op &FenceProxyAsyncOp();

/*!
 * \brief Indicate arrival of warp issuing TMA_STORE
 *
 * TMAStoreArrive()
 *
 */
const Op &TMAStoreArrive();

/*!
 * \brief Wait for TMA_STORE to finish
 *
 * TMAStoreWait()
 *
 */
const Op &TMAStoreWait();

/*!
 * \brief Set reg hint for warp-specialized branched
 *
 * SetMaxNRegInc(num_reg, is_inc)
 *
 */
const Op &SetMaxNReg();

/*!
 * \brief No set reg hint for warp-specialized branched
 *
 * NoSetMaxNReg()
 *
 */
const Op &NoSetMaxNReg();

/*!
 * \brief Wait the previous wgmma to finish
 *
 * WaitWgmma(num_mma)
 *
 */
const Op &WaitWgmma();

/*!
 * \brief tvm intrinsic for amd matrix core mfma instructions.
 *
 *  void tvm_mfma(StringImm shape, StringImm A_layout, StringImm B_layout,
 *               StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *               Var multiplicand_a, Expr a_index,
 *               Var multiplicand_b, Expr b_index,
 *               Var accumulator, Expr c_index);
 */
TVM_DLL const Op &tvm_mfma();

/*!
 * \brief tvm intrinsic for storing the result of AMD MFMA into a destination
 * pointer.
 *
 *        There is no real instruction that does that, but we want to hide
 * details of complex index manipulation behind this intrinsic to simplify TIR
 * lowering passes (e.g. LowerWarpMemory) like cuda ptx backend does.
 *
 * void tvm_mfma_store(IntImm m, IntImm n, Var dst_ptr, Var src_ptr, Expr
 * src_offset, Var dst_stride);
 */
TVM_DLL const Op &tvm_mfma_store();

/*!
 * \brief tvm intrinsic for amd rdna matrix core instructions.
 *
 *  void tvm_rdna_wmma(StringImm shape, StringImm A_layout, StringImm B_layout,
 *               StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *               Var multiplicand_a, Expr a_index,
 *               Var multiplicand_b, Expr b_index,
 *               Var accumulator, Expr c_index);
 */
TVM_DLL const Op &tvm_rdna_wmma();

/*!
 * \brief tvm intrinsic for storing the result of AMD RDNA WMMA into a
 * destination pointer.
 *
 *        There is no real instruction that does that, but we want to hide
 * details of complex index manipulation behind this intrinsic to simplify TIR
 * lowering passes (e.g. LowerWarpMemory) like cuda ptx backend does.
 *
 * void tvm_rdna_wmma_store(IntImm m, IntImm n, Var dst_ptr, Var src_ptr, Expr
 * src_offset, Var dst_stride);
 */
TVM_DLL const Op &tvm_rdna_wmma_store();

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_BUILTIN_H_