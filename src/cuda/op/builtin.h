/*!
 * \file tl/cuda/op/builtin.h
 * \brief CUDA-specific TileLang intrinsic Ops and compiler attributes.
 */

#ifndef TVM_TL_CUDA_OP_BUILTIN_H_
#define TVM_TL_CUDA_OP_BUILTIN_H_

#include "op/builtin.h"

namespace tvm {
namespace tl {

namespace attr {

// Warp-specialization annotations consumed only by CUDA lowering.
static constexpr const char *kWarpSpecializationScope =
    "kWarpSpecializationScope";
static constexpr const char *kCustomWarpSpecialization =
    "kCustomWarpSpecialization";

// A CUDA TMA descriptor cannot encode a pointer bound inside device code.
static constexpr const char *kTmaDescriptorBaseIsDeviceBound =
    "tma_descriptor_base_is_device_bound";

// Minimum CUDA thread blocks per SM for __launch_bounds__ emission.
static constexpr const char *kMinBlocksPerSM = "tl.min_blocks_per_sm";

} // namespace attr

// PrimFunc attribute set when CUDA TMA operations are generated.
static constexpr const char *kHasTMA = "tl.has_tma";

// CUDA-specific pass configuration keys. Their string values remain stable
// because they are part of the Python PassContext interface.
static constexpr const char *kDisableWarpSpecialized =
    "tl.disable_warp_specialized";
static constexpr const char *kEnableAutoSchedule = "tl.enable_auto_schedule";
static constexpr const char *kDisableTMALower = "tl.disable_tma_lower";
static constexpr const char *kPtxasRegisterUsageLevel =
    "tl.ptxas_register_usage_level";
static constexpr const char *kDisableVectorize256 = "tl.disable_vectorize_256";
static constexpr const char *kDisableWGMMA = "tl.disable_wgmma";
static constexpr const char *kDisableShuffleElect = "tl.disable_shuffle_elect";
static constexpr const char *kEnableLowerLDGSTG = "tl.enable_lower_ldgstg";
static constexpr const char *kEnableLowerLDGSTGPredicated =
    "tl.enable_lower_ldgstg_predicated";

// fast math related op
// __exp(x) - fast exponential
TVM_DLL const Op &__exp();
// __exp10(x) - fast base-10 exponential
TVM_DLL const Op &__exp10();
// __log(x) - fast natural logarithm
TVM_DLL const Op &__log();
// __log2(x) - fast base-2 logarithm
TVM_DLL const Op &__log2();
// __log10(x) - fast base-10 logarithm
TVM_DLL const Op &__log10();
// __tan(x) - fast tangent
TVM_DLL const Op &__tan();
// __cos(x) - fast cosine
TVM_DLL const Op &__cos();
// __sin(x) - fast sine
TVM_DLL const Op &__sin();
// fast_rcp(x) - approximate reciprocal
TVM_DLL const Op &fast_rcp();
// max_nan(x, y) - max with CUDA __hmax_nan semantics for fp16/bf16
TVM_DLL const Op &max_nan();
// min_nan(x, y) - min with CUDA __hmin_nan semantics for fp16/bf16
TVM_DLL const Op &min_nan();

// high precision with IEEE-compliant.
// ieee_add(x, y, rounding_mode) - IEEE-compliant addition
TVM_DLL const Op &ieee_add();
// ieee_sub(x, y, rounding_mode) - IEEE-compliant subtraction
TVM_DLL const Op &ieee_sub();
// ieee_mul(x, y, rounding_mode) - IEEE-compliant multiplication
TVM_DLL const Op &ieee_mul();
// ieee_fmaf(x, y, z, rounding_mode) - IEEE-compliant fused multiply-add
TVM_DLL const Op &ieee_fmaf();
// ieee_frcp(x, rounding_mode) - IEEE-compliant reciprocal
TVM_DLL const Op &ieee_frcp();
// ieee_fsqrt(x, rounding_mode) - IEEE-compliant square root
TVM_DLL const Op &ieee_fsqrt();
// ieee_frsqrt(x) - IEEE-compliant reciprocal square root (rn only)
TVM_DLL const Op &ieee_frsqrt();
// ieee_fdiv(x, y, rounding_mode) - IEEE-compliant division
TVM_DLL const Op &ieee_fdiv();
TVM_DLL const Op &max2_nan();
TVM_DLL const Op &min2_nan();

// random op
TVM_DLL const Op &rng_init();
TVM_DLL const Op &rng_rand();
TVM_DLL const Op &rng_rand_float();

/*!
 * \brief Return the sentinel dtype used for CUDA tensor-map parameters.
 */
DataType CuTensorMapType();

/*!
 * \brief tvm intrinsics for TMADescriptor creation for tiled load
 *
 * CuTensorMap* create_tma_descriptor(data_type, rank, global_addr,
 * global_shape..., global_stride..., smem_box..., smem_stride..., interleave,
 * swizzle, l2_promotion, oob_fill)
 *
 */
TVM_DLL const Op &create_tma_descriptor();

/*!
 * \brief tvm intrinsics for TMADescriptor creation for image to column load
 *
 * CuTensorMap* create_tma_im2col_descriptor(data_type, rank, global_addr,
 * global_shape..., global_stride..., elem_stride..., lower_corner...,
 * upper_corner..., smme_box_pixel, smem_box_channel, interleave, swizzle,
 * l2_promotion, oob_fill)
 *
 */
TVM_DLL const Op &create_tma_im2col_descriptor();

/*!
 * \brief tvm intrinsic for prefetching a TMA descriptor on Hopper.
 *
 * prefetch_tma_descriptor(descriptor)
 *
 */
TVM_DLL const Op &prefetch_tma_descriptor();

/*!
 * \brief tvm intrinsics for loading data from global tensor descriptor to
 * shared memory
 *
 * tma_load(descriptor, mbarrier, smem_data, coord_0, coord_1, ...)
 *
 */
TVM_DLL const Op &tma_load();

/*!
 * \brief tvm intrinsics for loading image from global tensor to columns in
 * shared memory
 *
 * tma_load(descriptor, mbarrier, smem_data, coord_0, coord_1, ...,
 * image_offset, ...)
 *
 */
TVM_DLL const Op &tma_load_im2col();

/*!
 * \brief TMA multicast load from a tensor descriptor to cluster shared memory.
 *
 * tma_load_multicast(descriptor, mbarrier, smem_data, multicast_mask,
 *                    coord_0, coord_1, ..., eviction_policy)
 */
TVM_DLL const Op &tma_load_multicast();

/*!
 * \brief tvm intrinsics for storing data from shared memory to global tensor
 * descriptor
 *
 * tma_store(descriptor, smem_data, coord_0, coord_1, ...)
 *
 */
TVM_DLL const Op &tma_store();

/*!
 * \brief tvm intrinsics for tile::gather4 TMA load (sm_90+).
 *
 * Loads four rows from a 2D global tensor (described by a tiled CUtensorMap)
 * into a shared memory tile. The four rows can be at arbitrary indices.
 *
 *   tma_load_gather4(descriptor, mbarrier, smem_data, col,
 *                    row0, row1, row2, row3, eviction_policy)
 *
 * The descriptor must be encoded with rank=2 and box dim along axis 1 = 1
 * (the four-row pack is implicit in the gather4 PTX mode).
 */
TVM_DLL const Op &tma_load_gather4();

/*!
 * \brief tvm intrinsics for tile::scatter4 TMA store (sm_90+).
 *
 * Stores four shared-memory rows back to four arbitrary rows of a 2D global
 * tensor (described by a tiled CUtensorMap).
 *
 *   tma_store_scatter4(descriptor, smem_data, col,
 *                      row0, row1, row2, row3, eviction_policy)
 */
TVM_DLL const Op &tma_store_scatter4();

/*!
 * \brief tvm intrinsics for barrier initialization fence
 *
 * ptx_fence_barrier_init()
 *
 */
const Op &ptx_fence_barrier_init();

/*
 * \brief tvm intrinsics for cluster barrier arrive
 *
 * ptx_arrive_cluster_barrier(mbarrier, cta_id)
 *
 */
TVM_DLL const Op &ptx_arrive_cluster_barrier();

/*!
 * \brief tvm intrinsic for ptx tensor core wgmma instructions.
 *
 *  void ptx_wgmma_ss(StringImm accum_dtype, StringImm wgmma_prefix, bool
 * a_is_k_major, bool b_is_k_major, StringImm a_dtype_abbrv, StringImm
 * b_dtype_abbrv, StringImm accum_dtype_abbrv, Var A_descriptor, PrimExpr
 * A_offset, Var B_descriptor, Var B_offset, Var C_data, Var C_offset, bool
 * scale_out, bool scale_in_a, bool scale_in_b);
 */
TVM_DLL const Op &ptx_wgmma_ss();

/*!
 * \brief tvm intrinsics for ptx tensor core wgmma instructions.
 *
 *  void ptx_wgmma_rs(StringImm accum_dtype, StringImm wgmma_prefix,
 * bool b_is_k_major, StringImm a_dtype_abbrv, StringImm b_dtype_abbrv,
 * StringImm accum_dtype_abbrv, Var A_descriptor, PrimExpr A_offset, Var
 * B_descriptor, Var B_offset, Var C_data, Var C_offset, bool scale_out,
 * bool scale_in_a, bool scale_in_b);
 */
TVM_DLL const Op &ptx_wgmma_rs();

/*!
 * \brief tvm intrinsic for sparse ptx wgmma shared-shared instructions.
 */
TVM_DLL const Op &ptx_wgmma_sp_ss();

/*!
 * \brief tvm intrinsic for sparse ptx wgmma register-shared instructions.
 */
TVM_DLL const Op &ptx_wgmma_sp_rs();

/*!
 * \brief tvm intrinsic for ptx tensor core mma with block scaling on SM120a.
 */
TVM_DLL const Op &ptx_mma_block_scale();

/*!
 * \brief tvm intrinsic for tcgen05 mma shared-shared instructions.
 */
TVM_DLL const Op &ptx_tcgen05_mma_ss();

/*!
 * \brief tvm intrinsic for tcgen05 mma tensor-shared instructions.
 */
TVM_DLL const Op &ptx_tcgen05_mma_ts();

/*!
 * \brief tvm intrinsic for tcgen05 block-scaled mma shared-shared instructions.
 */
TVM_DLL const Op &ptx_tcgen05_mma_blockscaled_ss();

/*!
 * \brief tvm intrinsic for tcgen05 copy warpx4 (smem to tmem).
 */
TVM_DLL const Op &ptx_tcgen05_cp_warpx4();

/*!
 * \brief tvm intrinsic for scale factor warp transpose in shared memory.
 */
TVM_DLL const Op &ptx_tcgen05_sf_warp_transpose();

/*!
 * \brief Frontend TMEM deallocation marker.
 *
 * deallocate_tmem(tmem_buffer_data)
 *
 * This op is produced by the TileLang Python frontend and must be lowered by
 * LowerSharedTmem into ptx_deallocate_tensor_memory(access_ptr, num_cols).
 */
TVM_DLL const Op &deallocate_tmem();

/*!
 * \brief tvm intrinsics for initializing tensor memory
 *
 * ptx_init_tensor_memory(tmem_buffer, num_cols)
 *
 */
TVM_DLL const Op &ptx_init_tensor_memory();

/*!
 * \brief tvm intrinsics for deallocating tensor memory
 *
 * tmem_deallocate(tmem_buffer)
 *
 */
TVM_DLL const Op &ptx_deallocate_tensor_memory();

/*!
 * \brief tvm intrinsic for ptx tensor core mma instructions on SM70.
 *
 *  void ptx_mma_sm70(StringImm shape, StringImm A_layout, StringImm B_layout,
 *                    StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *                    Var multiplicand_a, Expr a_index,
 *                    Var multiplicand_b, Expr b_index,
 *                    Var accumulator, Expr c_index, bool saturate);
 */
TVM_DLL const Op &ptx_mma_sm70();

/*!
 * \brief tvm intrinsics for ldmatrix
 *
 * ptx_ldmatrix(transposed, num, shared_addr, local_addr)
 *
 */
TVM_DLL const Op &ptx_ldmatrix();

/*!
 * \brief tvm intrinsic for ptx async copy barrier using
 * cp.async.mbarrier.arrive.noinc
 *
 *  This op is used to represent a ptx async copy barrier operation in tilelang.
 */
TVM_DLL const Op &ptx_cp_async_barrier_noinc();

/*!
 * \brief TileLang intrinsic for zeroing shared memory with st.bulk.
 *
 * ptx_st_bulk_shared(smem_data, bytes, init_val)
 *
 */
TVM_DLL const Op &ptx_st_bulk_shared();

/*!
 * \brief Pack four b8 value into a b32 value
 *
 * int32 pack_b8x4(b8_value, b8_value, b8_value, b8_value)
 *
 */
TVM_DLL const Op &pack_b8x4();

/*!
 * \brief Issue a shared memory fence for async operations
 *
 * FenceProxyAsync()
 *
 */
TVM_DLL const Op &fence_proxy_async();

/*!
 * \brief Indicate arrival of warp issuing TMA_STORE
 *
 * tma_store_arrive()
 *
 */
TVM_DLL const Op &tma_store_arrive();

/*!
 * \brief Wait for TMA_STORE to finish
 *
 * tma_store_wait()
 *
 */
TVM_DLL const Op &tma_store_wait();

/*!
 * \brief Set reg hint for warp-specialized branched
 *
 * SetMaxNRegInc(num_reg, is_inc)
 *
 */
TVM_DLL const Op &set_max_nreg();

/*!
 * \brief Arrive at a warpgroup fence for WGMMA sequences
 *
 * warpgroup_arrive()
 *
 */
TVM_DLL const Op &warpgroup_arrive();

/*!
 * \brief Commit the current warpgroup batch for WGMMA sequences
 *
 * warpgroup_commit_batch()
 *
 */
TVM_DLL const Op &warpgroup_commit_batch();

/*!
 * \brief Wait for the warpgroup batch identified by num_mma
 *
 * warpgroup_wait(num_mma)
 *
 */
TVM_DLL const Op &warpgroup_wait();

/*!
 * \brief Fence accumulator operand registers for upcoming WGMMA operations
 *
 * warpgroup_fence_operand(dtype, ptr, offset, num_regs)
 *
 */
TVM_DLL const Op &warpgroup_fence_operand();

/*!
 * \brief Return the canonical lane index for the calling thread.
 *
 * get_lane_idx([warp_size])
 *
 */
TVM_DLL const Op &get_lane_idx();

/*!
 * \brief Return the canonical warp index, assuming converged threads.
 *
 * get_warp_idx_sync([warp_size])
 *
 */
TVM_DLL const Op &get_warp_idx_sync();

/*!
 * \brief Return the canonical warp index without synchronizing the warp.
 *
 * get_warp_idx([warp_size])
 *
 */
TVM_DLL const Op &get_warp_idx();

/*!
 * \brief Return the canonical warp group index for converged threads.
 *
 * get_warp_group_idx([warp_size, warps_per_group])
 *
 */
TVM_DLL const Op &get_warp_group_idx();

/*!
 * \brief Cluster barrier arrive with relaxed ordering
 *
 * cluster_arrive_relaxed()
 *
 */
TVM_DLL const Op &cluster_arrive_relaxed();

/*!
 * \brief Cluster barrier arrive
 *
 * cluster_arrive()
 *
 */
TVM_DLL const Op &cluster_arrive();

/*!
 * \brief Cluster barrier wait
 *
 * cluster_wait()
 *
 */
TVM_DLL const Op &cluster_wait();

/*!
 * \brief Cluster barrier arrive + wait (full sync)
 *
 * cluster_sync()
 *
 */
TVM_DLL const Op &cluster_sync();

/*!
 * \brief Return the 1-D rank of the calling CTA within its cluster
 *
 * int block_rank_in_cluster()
 *
 */
TVM_DLL const Op &block_rank_in_cluster();

/*!
 * \brief Issue a Blackwell cluster launch control query that writes a 16-byte
 * response into shared memory and signals completion on the given mbarrier.
 *
 * clc_try_cancel(result_ptr, mbar_ptr)
 *
 */
TVM_DLL const Op &clc_try_cancel();

/*!
 * \brief Cluster-wide multicast variant of cluster launch control query.
 *
 * clc_try_cancel_multicast(result_ptr, mbar_ptr)
 *
 */
TVM_DLL const Op &clc_try_cancel_multicast();

/*!
 * \brief Return 1 when a CLC response represents a successful cancellation.
 *
 * int32 clc_is_canceled(result_ptr)
 *
 */
TVM_DLL const Op &clc_is_canceled();

/*!
 * \brief Return the x coordinate of the first CTA in a successful CLC response.
 *
 * uint32 clc_get_first_ctaid_x(result_ptr)
 *
 */
TVM_DLL const Op &clc_get_first_ctaid_x();

/*!
 * \brief Return the y coordinate of the first CTA in a successful CLC response.
 *
 * uint32 clc_get_first_ctaid_y(result_ptr)
 *
 */
TVM_DLL const Op &clc_get_first_ctaid_y();

/*!
 * \brief Return the z coordinate of the first CTA in a successful CLC response.
 *
 * uint32 clc_get_first_ctaid_z(result_ptr)
 *
 */
TVM_DLL const Op &clc_get_first_ctaid_z();

/*!
 * \brief CTA named barrier one-sided arrive (bar.arrive).
 *
 * Signals that the calling threads have arrived at the named barrier without
 * waiting for other participants.  Useful in warp-specialized producer/consumer
 * pipelines where one side must signal readiness/free-buffer state without
 * blocking, while the other side waits with bar.sync / T.sync_threads().
 *
 * named_barrier_arrive(barrier_id, thread_count)
 *   barrier_id   - named barrier index (0-15)
 *   thread_count - total number of participating threads
 *
 * Lowers to: asm volatile("bar.arrive %0, %1;" : : "r"(id), "r"(cnt));
 */
TVM_DLL const Op &named_barrier_arrive();

/*!
 * \brief Programmatic dependency trigger.
 *
 * pdl_trigger()
 *
 */
TVM_DLL const Op &pdl_trigger();

/*!
 * \brief Programmatic grid dependency synchronization.
 *
 * pdl_sync()
 *
 */
TVM_DLL const Op &pdl_sync();

/*!
 * \brief tilelang intrinsic for shuffle elect.
 *
 *  This op is used to represent a shuffle elect operation in tilelang.
 */
TVM_DLL const Op &tl_shuffle_elect();

/*!
 * \brief tilelang intrinsic for initializing a descriptor buffer for
 * wgmma/utcmma.
 *
 *  This op is used to represent a descriptor initialization operation in
 * tilelang.
 */
TVM_DLL const Op &initialize_wgmma_descriptor();

/*!
 * \brief tilelang intrinsic for initializing a descriptor buffer for
 * tcgen05 mma.
 */
TVM_DLL const Op &initialize_tcgen05_descriptor();

/*!
 * \brief tilelang intrinsic for committing UMMA (TCGEN05) barrier arrive.
 *
 *  This op wraps the device-side arrive used to signal completion of MMA work
 *  to a shared-memory mbarrier. It mirrors CUTLASS's umma_arrive.
 */
TVM_DLL const Op &tcgen05_mma_arrive();

/*!
 * \brief tilelang intrinsic for lowered TCGEN05 tensor-memory load.
 *
 *  Internal lowering op used by LowerTmemCopy to represent
 *  `tl::tcgen05_ld_*` calls without routing through `call_extern`.
 */
TVM_DLL const Op &tcgen05_ld();

/*!
 * \brief tilelang intrinsic for lowered TCGEN05 tensor-memory store.
 *
 *  Internal lowering op used by LowerTmemCopy to represent
 *  `tl::tcgen05_st_*` calls without routing through `call_extern`.
 */
TVM_DLL const Op &tcgen05_st();

/*!
 * \brief TCGEN05 fence before a thread-block-wide sync (__syncthreads /
 * bar.sync). Matches PTX \c tcgen05.fence::before_thread_sync (DeepGEMM /
 * Blackwell UMMA sequencing).
 */
TVM_DLL const Op &tcgen05_before_thread_sync();

/*!
 * \brief TCGEN05 fence after a thread-block-wide sync. Matches PTX \c
 * tcgen05.fence::after_thread_sync.
 */
TVM_DLL const Op &tcgen05_after_thread_sync();

/*!
 * \brief tilelang intrinsic for setting the start address of a descriptor
 * buffer for wgmma/utcmma.
 *
 *  This op is used to represent a descriptor start address setting operation in
 * tilelang.
 */

TVM_DLL const Op &increase_descriptor_offset();

/*!
 * \brief tilelang intrinsic for assert on device.
 *
 *  This op is used to represent an assert on device
 */
TVM_DLL const Op &device_assert();

/*!
 * \brief tilelang intrinsic for assert on device with additional message.
 *
 *  This op is used to represent an assert on device with additional message.
 */
TVM_DLL const Op &device_assert_with_msg();

/*!
 * \brief tilelang intrinsic for CUDA find-first-set bit (__ffs / __ffsll).
 *
 *  Returns the one-based position of the least significant set bit, or 0 when
 *  the input is zero. CUDA codegen emits `__ffs` for 32-bit integer inputs and
 *  `__ffsll` for 64-bit integer inputs.
 *
 *  Usage from TVMScript:
 *    lane = T.__ffs(mask) - 1
 */
TVM_DLL const Op &__ffs();

/*!
 * \brief tilelang intrinsic for CUDA find-nth-set bit (__fns).
 *
 *  Returns the zero-based position of the offset-th set bit in mask starting
 *  from base, or 0xFFFFFFFF when not found. CUDA codegen emits `__fns`.
 *
 *  Usage from TVMScript:
 *    lane = T.__fns(mask, 0, k + 1)
 */
TVM_DLL const Op &__fns();

/*!
 * \brief tilelang intrinsic for global memory load with 32-bit vector width.
 *
 *  This op loads 32 bits (4 bytes) from global memory using explicit
 *  PTX ld.global instructions for performance-sensitive loads.
 *
 *  Usage from TVMScript:
 *    y[i] = T.ldg32(x, i)
 */
TVM_DLL const Op &ldg32();

/*!
 * \brief tilelang intrinsic for global memory load with 64-bit vector width.
 *
 *  This op loads 64 bits (8 bytes) from global memory using explicit
 *  PTX ld.global.v2 instructions for vectorized loads.
 *
 *  Usage from TVMScript:
 *    y[i] = T.ldg64(x, i)
 */
TVM_DLL const Op &ldg64();

/*!
 * \brief tilelang intrinsic for global memory load with 128-bit vector width.
 *
 *  This op loads 128 bits (16 bytes) from global memory using explicit
 *  PTX ld.global.v4 or ld.global.v2.s64 instructions for wide vectorized loads.
 *
 *  Usage from TVMScript:
 *    y[i] = T.ldg128(x, i)
 */
TVM_DLL const Op &ldg128();

/*!
 * \brief tilelang intrinsic for shared memory load with 32-bit vector width.
 *
 * This op loads 32 bits (4 bytes) from shared memory and returns uint32.
 */
TVM_DLL const Op &lds32();

/*!
 * \brief tilelang intrinsic for shared memory load with 64-bit vector width.
 *
 * This op loads 64 bits (8 bytes) from shared memory and returns uint32x2.
 */
TVM_DLL const Op &lds64();

/*!
 * \brief tilelang intrinsic for shared memory load with 128-bit vector width.
 *
 * This op loads 128 bits (16 bytes) from shared memory and returns uint32x4.
 */
TVM_DLL const Op &lds128();

/*!
 * \brief tilelang intrinsic for global memory load with 256-bit vector width.
 *
 *  This op loads 256 bits (32 bytes) from global memory using explicit
 *  PTX ld.global.v4.s64 instructions for maximum vectorized loads.
 *  Requires CUDA 12.9+ for native support; older versions use two 128-bit
 * loads.
 *
 *  Usage from TVMScript:
 *    y[i] = T.ldg256(x, i)
 */
TVM_DLL const Op &ldg256();

/*!
 * \brief tilelang intrinsic for global memory store with 32-bit vector width.
 *
 *  This op stores 32 bits (4 bytes) to global memory using explicit
 *  PTX st.global instructions for performance-sensitive stores.
 *
 *  Usage from TVMScript:
 *    T.stg32(y, i, value)
 */
TVM_DLL const Op &stg32();

/*!
 * \brief tilelang intrinsic for global memory store with 64-bit vector width.
 *
 *  This op stores 64 bits (8 bytes) to global memory using explicit
 *  PTX st.global.v2 instructions for vectorized stores.
 *
 *  Usage from TVMScript:
 *    T.stg64(y, i, value)
 */
TVM_DLL const Op &stg64();

/*!
 * \brief tilelang intrinsic for global memory store with 128-bit vector width.
 *
 *  This op stores 128 bits (16 bytes) to global memory using explicit
 *  PTX st.global.v4 instructions for wide vectorized stores.
 *
 *  Usage from TVMScript:
 *    T.stg128(y, i, value)
 */
TVM_DLL const Op &stg128();

/*!
 * \brief tilelang intrinsic for shared memory store with 32-bit vector width.
 *
 * This op stores a uint32 value to shared memory.
 */
TVM_DLL const Op &sts32();

/*!
 * \brief tilelang intrinsic for shared memory store with 64-bit vector width.
 *
 * This op stores a uint32x2 value to shared memory.
 */
TVM_DLL const Op &sts64();

/*!
 * \brief tilelang intrinsic for shared memory store with 128-bit vector width.
 *
 * This op stores a uint32x4 value to shared memory.
 */
TVM_DLL const Op &sts128();

/*!
 * \brief tilelang intrinsic for global memory store with 256-bit vector width.
 *
 *  This op stores 256 bits (32 bytes) to global memory using explicit
 *  PTX st.global.v4.s64 instructions for maximum vectorized stores.
 *  Requires CUDA 12.9+ for native support; older versions use two 128-bit
 * stores.
 *
 *  Usage from TVMScript:
 *    T.stg256(y, i, value)
 */
TVM_DLL const Op &stg256();

/*!
 * \brief Elementwise shared::cluster store via cooperative groups.
 */
TVM_DLL const Op &ptx_cluster_store();

/*!
 * \brief Bulk async shared::cluster store to another CTA.
 *
 * tma_store_cluster(dst_ptr, src_ptr, dst_cta, size_bytes, bar_ref)
 */
TVM_DLL const Op &tma_store_cluster();

/*!
 * \brief Mark a buffer version index generated by MultiVersionBufferRewriter.
 *
 * This compiler-internal intrinsic preserves the provenance of synthetic
 * pipeline stage indices until warp specialization assigns branch-local
 * transaction counters. It must be removed before code generation.
 */
TVM_DLL const Op &mvb_stage_index();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_CUDA_OP_BUILTIN_H_
