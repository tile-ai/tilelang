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
static constexpr const char *kDebugMergeSharedMemoryAllocations =
    "tl.debug_merge_shared_memory_allocations";
static constexpr const char *kDisableTMALower = "tl.disable_tma_lower";
static constexpr const char *kDisableSafeMemoryLegalize =
    "tl.disable_safe_memory_legalize";
static constexpr const char *kDisableWarpSpecialized =
    "tl.disable_warp_specialized";
static constexpr const char *kConfigIndexBitwidth = "tl.config_index_bitwidth";

/*!
 * \brief Whether to disable dynamic tail split
 *
 * kDisableDynamicTailSplit = "tl.disable_dynamic_tail_split"
 *
 */
static constexpr const char *kDisableDynamicTailSplit =
    "tl.disable_dynamic_tail_split";

/*!
 * \brief The size of the vectorized dimension in buffer, designed by user
 *
 * For example, if the vectorized dimension is 128 bits and the dtype of buffer
 * A[m, k] is float16, the size of the vectorized dimension (i.e. k) in buffer A
 * should be divisible by 8 (8 = 128 / 16).
 *
 * kDynamicAlignment = "tl.dynamic_alignment"
 *
 */
static constexpr const char *kDynamicAlignment = "tl.dynamic_alignment";

/*!
 * \brief tvm intrinsics for TMADescriptor creation for tiled load
 *
 * CuTensorMap* create_tma_descriptor(data_type, rank, global_addr,
 * global_shape..., global_stride..., smem_box..., smem_stride..., interleave,
 * swizzle, l2_promotion, oob_fill)
 *
 */
const Op &create_tma_descriptor();

/*!
 * \brief tvm intrinsics for TMADescriptor creation for image to column load
 *
 * CuTensorMap* create_tma_im2col_descriptor(data_type, rank, global_addr,
 * global_shape..., global_stride..., elem_stride..., lower_corner...,
 * upper_corner..., smme_box_pixel, smem_box_channel, interleave, swizzle,
 * l2_promotion, oob_fill)
 *
 */
const Op &create_tma_im2col_descriptor();

/*!
 * \brief Create a list of mbarrier with num_threads
 *
 * create_list_of_mbarrier(num_threads0, num_threads1, ...)
 *
 */
const Op &create_list_of_mbarrier();

/*!
 * \brief Get the mbarrier with barrier_id
 *
 * int64_t* GetMBarrier(barrier_id)
 *
 */
const Op &get_mbarrier();

/*!
 * \brief tvm intrinsics for loading data from global tensor descriptor to
 * shared memory
 *
 * tma_load(descriptor, mbarrier, smem_data, coord_0, coord_1, ...)
 *
 */
const Op &tma_load();

/*!
 * \brief tvm intrinsics for loading image from global tensor to columns in
 * shared memory
 *
 * tma_load(descriptor, mbarrier, smem_data, coord_0, coord_1, ...,
 * image_offset, ...)
 *
 */
const Op &tma_load_im2col();

/*!
 * \brief tvm intrinsics for storing data from shared memory to global tensor
 * descriptor
 *
 * tma_store(descriptor, smem_data, coord_0, coord_1, ...)
 *
 */
const Op &tma_store();

/*!
 * \brief tvm intrinsics for mbarrier wait with parity bit
 *
 * mbarrier_wait_parity(mbarrier, parity)
 *
 */
const Op &mbarrier_wait_parity();

/*!
 * \brief tvm intrinsics for mbarrier expect tx
 *
 * mbarrier_expect_tx(mbarrier, transaction_bytes)
 *
 */
const Op &mbarrier_expect_tx();

/*!
 * \brief tvm intrinsics for ldmatrix
 *
 * ptx_ldmatirx(transposed, num, shared_addr, local_addr)
 *
 */
const Op &ptx_ldmatirx();

/*!
 * \brief tvm intrinsics for stmatrix
 *
 * ptx_ldmatirx(transposed, num, shared_addr, int32_values...)
 *
 */
const Op &ptx_stmatirx();

/*!
 * \brief Pack two b16 value into a b32 value
 *
 * int32 pack_b16(b16_value, b16_value)
 *
 */
const Op &pack_b16();

/*!
 * \brief Similar to __syncthreads(), but can be used to sync partial threads
 *
 * sync_thread_partial(num_partial_threads or mbarrier)
 *
 */
const Op &sync_thread_partial();

/*!
 * \brief Issue a shared memory fence for async operations
 *
 * FenceProxyAsync()
 *
 */
const Op &fence_proxy_async();

/*!
 * \brief Indicate arrival of warp issuing TMA_STORE
 *
 * tma_store_arrive()
 *
 */
const Op &tma_store_arrive();

/*!
 * \brief Wait for TMA_STORE to finish
 *
 * tma_store_wait()
 *
 */
const Op &tma_store_wait();

/*!
 * \brief Set reg hint for warp-specialized branched
 *
 * SetMaxNRegInc(num_reg, is_inc)
 *
 */
const Op &set_max_nreg();

/*!
 * \brief No set reg hint for warp-specialized branched
 *
 * no_set_max_nreg()
 *
 */
const Op &no_set_max_nreg();

/*!
 * \brief Wait the previous wgmma to finish
 *
 * wait_wgmma(num_mma)
 *
 */
const Op &wait_wgmma();

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