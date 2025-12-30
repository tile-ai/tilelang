/*!
 * \file tl/op/builtin.h
 * \brief Builtin intrinsics.
 *
 */

#ifndef TVM_TL_OP_BUILTIN_H_
#define TVM_TL_OP_BUILTIN_H_

#include "operator.h"
#include <tvm/ir/transform.h>

namespace tvm {
/*!
 * \brief Create the TVM intrinsic that initializes a PTX fence barrier.
 *
 * Initializes a PTX fence-style barrier used to coordinate asynchronous memory
 * operations (for example, TMA/TMA_STORE). Returns the Op representing this
 * intrinsic for use in TIR lowering and code generation.
 *
 */
namespace tl {

namespace attr {
static constexpr const char *kSafeValueMap = "safe_value_map";
static constexpr const char *kWarpSpecializationScope =
    "kWarpSpecializationScope";
static constexpr const char *kCustomWarpSpecialization =
    "kCustomWarpSpecialization";
static constexpr const char *kLocalVarInit = "tl.local_var_init";
// A PrimFunc-level attribute carrying a list of handle Vars
// that must NOT be marked with the restrict qualifier in codegen.
// Type: Array<tir::Var>
static constexpr const char *kNonRestrictParams = "tl.non_restrict_params";
} // namespace attr

static constexpr const char *kDebugMergeSharedMemoryAllocations =
    "tl.debug_merge_shared_memory_allocations";
static constexpr const char *kDisableTMALower = "tl.disable_tma_lower";
static constexpr const char *kDisableSafeMemoryLegalize =
    "tl.disable_safe_memory_legalize";
static constexpr const char *kDisableWarpSpecialized =
    "tl.disable_warp_specialized";
static constexpr const char *kConfigIndexBitwidth = "tl.config_index_bitwidth";
static constexpr const char *kEnableAggressiveSharedMemoryMerge =
    "tl.enable_aggressive_shared_memory_merge";
static constexpr const char *kDisableFastMath = "tl.disable_fast_math";
static constexpr const char *kEnableFastMath = "tl.enable_fast_math";
static constexpr const char *kPtxasRegisterUsageLevel =
    "tl.ptxas_register_usage_level";
static constexpr const char *kEnablePTXASVerboseOutput =
    "tl.enable_ptxas_verbose_output";
static constexpr const char *kDisableVectorize256 = "tl.disable_vectorize_256";
static constexpr const char *kDisableWGMMA = "tl.disable_wgmma";
static constexpr const char *kDisableShuffleElect = "tl.disable_shuffle_elect";
static constexpr const char *kStorageRewriteDetectInplace =
    "tl.storage_rewrite_detect_inplace";
static constexpr const char *kASTPrintEnable = "tl.ast_print_enable";
static constexpr const char *kLayoutVisualizationEnable =
    "tl.layout_visualization_enable";
static constexpr const char *kLayoutVisualizationFormats =
    "tl.layout_visualization_formats";
static constexpr const char *kDeviceCompileFlags = "tl.device_compile_flags";

/*!
 * \brief Whether to disable thread storage synchronization
 *
 * When enabled, disables the automatic insertion of thread synchronization
 * barriers (e.g., __syncthreads()) for shared memory access coordination.
 * This can be useful for performance optimization in cases where manual
 * synchronization is preferred or when synchronization is not needed.
 *
 * kDisableThreadStorageSync = "tl.disable_thread_storage_sync"
 *
 */
static constexpr const char *kDisableThreadStorageSync =
    "tl.disable_thread_storage_sync";

/*!
 * \brief Force inline Let bindings during simplification.
 *
 * kForceLetInline = "tl.force_let_inline"
 *
 */
static constexpr const char *kForceLetInline = "tl.force_let_inline";

/*!
 * \brief Get the type of the CUDA tensor map
 *
 * DataType cuTensorMapType()
 *
 */
DataType cuTensorMapType();

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

// random op
TVM_DLL const Op &rng_init();
TVM_DLL const Op &rng_rand();

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
 * \brief Create a list of mbarrier with num_threads
 *
 * create_list_of_mbarrier(num_threads0, num_threads1, ...)
 *
 */
TVM_DLL const Op &create_list_of_mbarrier();

/*!
 * \brief Get the mbarrier with barrier_id
 *
 * int64_t* GetMBarrier(barrier_id)
 *
 */
TVM_DLL const Op &get_mbarrier();

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
 * \brief tvm intrinsics for storing data from shared memory to global tensor
 * descriptor
 *
 * tma_store(descriptor, smem_data, coord_0, coord_1, ...)
 *
 */
TVM_DLL const Op &tma_store();

/*!
 * \brief tvm intrinsics for barrier initialization fence
 *
 * ptx_fence_barrier_init()
 *
 */
const Op &ptx_fence_barrier_init();

/*!
 * \brief tvm intrinsics for mbarrier wait with parity bit
 *
 * mbarrier_wait_parity(mbarrier, parity)
 *
 */
TVM_DLL const Op &mbarrier_wait_parity();

/*!
 * \brief tvm intrinsics for mbarrier expect tx
 *
 * mbarrier_expect_tx(mbarrier, transaction_bytes)
 *
 */
TVM_DLL const Op &mbarrier_expect_tx();

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
 * \brief tvm intrinsic for tcgen05 mma shared-shared instructions.
 */
TVM_DLL const Op &ptx_tcgen05_mma_ss();

/*!
 * \brief tvm intrinsic for tcgen05 mma tensor-shared instructions.
 */
TVM_DLL const Op &ptx_tcgen05_mma_ts();

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
 * \brief tvm intrinsics for stmatrix
 *
 * ptx_ldmatrix(transposed, num, shared_addr, int32_values...)
 *
 */
TVM_DLL const Op &ptx_stmatrix();

/*!
 * \brief tvm intrinsic for ptx async copy barrier using
 * cp.async.mbarrier.arrive.noinc
 *
 *  This op is used to represent a ptx async copy barrier operation in tilelang.
 */
TVM_DLL const Op &ptx_cp_async_barrier_noinc();

/*!
 * \brief Pack two b16 value into a b32 value
 *
 * int32 pack_b16(b16_value, b16_value)
 *
 */
TVM_DLL const Op &pack_b16();

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
 * \brief No set reg hint for warp-specialized branched
 *
 * no_set_max_nreg()
 *
 */
TVM_DLL const Op &no_set_max_nreg();

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
 * \brief Wait the previous wgmma to finish
 *
 * wait_wgmma(num_mma)
 *
 */
TVM_DLL const Op &wait_wgmma();

/*!
 * \brief Synchronize all threads in a grid
 *
 * sync_grid()
 *
 */
TVM_DLL const Op &sync_grid();

/*!
 * \brief tvm intrinsic for loop continue
 *
 * loop_break()
 *
 */
TVM_DLL const Op &loop_break();

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

/*!
 * \brief tilelang intrinsic for general matrix multiplication (GEMM).
 *
 *  This op is used to represent a generic GEMM operation in tilelang.
 */
TVM_DLL const Op &tl_gemm();

/*!
 * \brief tilelang intrinsic for sparse matrix multiplication (GEMM with
 * sparsity).
 *
 *  This op is used to represent a sparse GEMM operation in tilelang.
 */
TVM_DLL const Op &tl_gemm_sp();

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
 * \brief tilelang intrinsic for setting the start address of a descriptor
 * buffer for wgmma/utcmma.
 *
 *  This op is used to represent a descriptor start address setting operation in
 * tilelang.
 */

TVM_DLL const Op &increase_descriptor_offset();

/*!
 * \brief tilelang intrinsic for element-wise atomic addition.
 *
 *  This op is used to represent an element-wise atomic add operation in
 * tilelang.
 */
TVM_DLL const Op &atomicadd_elem_op();

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
 * \brief tilelang intrinsic for warp reduction sum.
 */
TVM_DLL const Op &warp_reduce_sum();

/*!
 * \brief tilelang intrinsic for warp reduction max.
 */
TVM_DLL const Op &warp_reduce_max();

/*!
 * \brief tilelang intrinsic for warp reduction min.
 */
TVM_DLL const Op &warp_reduce_min();

/*!
 * \brief tilelang intrinsic for warp reduction bitand.
 */
TVM_DLL const Op &warp_reduce_bitand();

/*!
 * \brief tilelang intrinsic for warp reduction bitor.
 */
TVM_DLL const Op &warp_reduce_bitor();

/*!
 * \brief tilelang intrinsic for CUDA read-only cache load (__ldg).
 *
 *  This op allows users to explicitly request a non-coherent cached load
 *  from global memory on CUDA by emitting `__ldg(&ptr[idx])` for 32-bit
 *  element types on supported architectures. It provides a direct way to
 *  leverage the read-only data cache for performance-sensitive loads when
 *  the compiler cannot infer `const __restrict__` automatically.
 *
 *  Usage from TVMScript:
 *    y[i] = T.__ldg(x[i])
 *
 *  The op takes one argument preferred as a BufferLoad identifying the
 *  source element; alternatively, backends may support passing a Buffer and
 *  index expression.
 */
TVM_DLL const Op &__ldg();

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_BUILTIN_H_
