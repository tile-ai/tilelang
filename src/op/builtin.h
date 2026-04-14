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
// Loop annotation key controlling whether PTX async-copy rewriting is enabled
// in the annotated loop subtree. Value should be Bool (False/True).
static constexpr const char *kLoopPreferAsync = "parallel_prefer_async";
// Loop annotation key controlling whether async commit/wait should be omitted
// for injected cp.async in this parallel loop subtree. Value should be Bool.
static constexpr const char *kParallelAsyncWithoutAsyncCommitWait =
    "parallel_async_without_async_commit_wait";
// Copy-op annotation key controlling whether cp.async commit/wait are managed
// by an enclosing transform (e.g. software pipeline / warp specialization).
// Value should be IntImm/Bool-like truthy scalar.
static constexpr const char *kAsyncCopyNoImplicitCommitWait =
    "no_implicit_async_commit_wait";
// Tile-op annotation key carrying an explicit mbarrier parity expression.
// Pipeline transforms set this on ops whose lowering would otherwise infer
// parity from surrounding loop context.
static constexpr const char *kPipelineMbarPhaseExpr =
    "tl.pipeline_mbar_phase_expr";
static constexpr const char *kLocalVarInit = "tl.local_var_init";
// A PrimFunc-level attribute carrying a list of handle Vars
// that must NOT be marked with the restrict qualifier in codegen.
// Type: Array<tir::Var>
static constexpr const char *kNonRestrictParams = "tl.non_restrict_params";
// A PrimFunc-level attribute carrying the minimum number of thread blocks
// per SM (multiprocessor).  When present it is emitted as the second
// argument of __launch_bounds__(maxThreads, minBlocksPerMultiprocessor).
// Type: Integer
static constexpr const char *kMinBlocksPerSM = "tl.min_blocks_per_sm";
// lexical_alloc_scope may first appear as a Block annotation, requesting that
// LowerOpaqueBlock materialize a lexical scope boundary for that block subtree.
// After LowerOpaqueBlock, the same key appears as an AttrStmt marker that
// generates a C/CUDA lexical scope `{ ... }` in codegen. Allocations nested
// inside this scope cannot be hoisted past the boundary by StorageRewrite,
// giving the underlying compiler accurate variable lifetime information for
// register allocation.
static constexpr const char *kLexicalAllocScope = "lexical_alloc_scope";
} // namespace attr

inline Optional<PrimExpr>
GetAnnotatedMbarPhaseExpr(const Map<String, ObjectRef> &annotations) {
  if (auto val = annotations.Get(attr::kPipelineMbarPhaseExpr)) {
    if (val.value()->IsInstance<PrimExprNode>()) {
      return Downcast<PrimExpr>(val.value());
    }
    LOG(FATAL) << "Annotation `" << attr::kPipelineMbarPhaseExpr
               << "` expects a PrimExpr value, but got "
               << val.value().GetTypeKey();
  }
  return Optional<PrimExpr>();
}

static constexpr const char *kDebugMergeSharedMemoryAllocations =
    "tl.debug_merge_shared_memory_allocations";
// PrimFunc attribute: set by LowerTileOp to indicate TMA operations were
// actually generated.  Read by OptimizeForTarget to pick the right pipeline.
static constexpr const char *kHasTMA = "tl.has_tma";
static constexpr const char *kDisableSafeMemoryLegalize =
    "tl.disable_safe_memory_legalize";
static constexpr const char *kDisableWarpSpecialized =
    "tl.disable_warp_specialized";
static constexpr const char *kConfigIndexBitwidth = "tl.config_index_bitwidth";
// Deprecated pass config, temporarily re-enabled. Prevents plain T.copy()
// from auto-lowering to TMA store. Will be removed in v0.1.10.
static constexpr const char *kDisableTMALower = "tl.disable_tma_lower";
static constexpr const char *kEnableAggressiveSharedMemoryMerge =
    "tl.enable_aggressive_shared_memory_merge";
static constexpr const char *kDisableFastMath = "tl.disable_fast_math";
static constexpr const char *kEnableFastMath = "tl.enable_fast_math";
static constexpr const char *kPtxasRegisterUsageLevel =
    "tl.ptxas_register_usage_level";
static constexpr const char *kEnablePTXASVerboseOutput =
    "tl.enable_ptxas_verbose_output";
static constexpr const char *kDisableVectorize256 = "tl.disable_vectorize_256";
static constexpr const char *kEnableAsyncCopy = "tl.enable_async_copy";
static constexpr const char *kEnableVectorizePlannerVerbose =
    "tl.enable_vectorize_planner_verbose";
static constexpr const char *kDisableWGMMA = "tl.disable_wgmma";
static constexpr const char *kDisableShuffleElect = "tl.disable_shuffle_elect";
static constexpr const char *kDisableLoopUnswitching =
    "tl.disable_loop_unswitching";
// Allow loop unswitching even when the else-version of the loop body is
// non-trivial (has side effects). Default: false (conservative).
static constexpr const char *kLoopUnswitchingAllowNonTrivialElse =
    "tl.loop_unswitching_allow_non_trivial_else";

/*!
 * \brief Enable lowering non-predicated global load/store to ldg/stg intrinsics
 *
 * When enabled, transforms regular (non-predicated) global memory loads and
 * stores to explicit ldg/stg intrinsics for potentially better performance.
 * Default: OFF (disabled)
 *
 * kEnableLowerLDGSTG = "tl.enable_lower_ldgstg"
 */
static constexpr const char *kEnableLowerLDGSTG = "tl.enable_lower_ldgstg";

/*!
 * \brief Enable lowering predicated global load/store to ldg/stg intrinsics
 *
 * When enabled (set to true), predicated loads (if_then_else with else=0) and
 * predicated stores (IfThenElse with store in then case) will be lowered
 * to predicated ldg/stg intrinsics.
 * Default: OFF (predicated lowering is disabled by default)
 *
 * kEnableLowerLDGSTGPredicated = "tl.enable_lower_ldgstg_predicated"
 */
static constexpr const char *kEnableLowerLDGSTGPredicated =
    "tl.enable_lower_ldgstg_predicated";
static constexpr const char *kStorageRewriteDetectInplace =
    "tl.storage_rewrite_detect_inplace";
static constexpr const char *kASTPrintEnable = "tl.ast_print_enable";
static constexpr const char *kLayoutVisualizationEnable =
    "tl.layout_visualization_enable";
static constexpr const char *kLayoutVisualizationFormats =
    "tl.layout_visualization_formats";
static constexpr const char *kDeviceCompileFlags = "tl.device_compile_flags";
static constexpr const char *kDisableDataRaceCheck =
    "tl.disable_data_race_check";

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
 * \brief Disable out of bound warning in LegalizeSafeMemoryAccess pass.
 *
 * kDisableOutOfBoundWarning = "tl.disable_out_of_bound_warning"
 *
 */
static constexpr const char *kDisableOutOfBoundWarning =
    "tl.disable_out_of_bound_warning";

/*!
 * \brief Enable dumping IR during lowering between passes.
 *
 * kEnableDumpIR = "tl.enable_dump_ir"
 *
 */
static constexpr const char *kEnableDumpIR = "tl.enable_dump_ir";
static constexpr const char *kDumpIRDir = "tl.dump_ir_path";

/*!
 * \brief Get the type of the CUDA tensor map
 *
 * DataType cuTensorMapType()
 *
 */
DataType cuTensorMapType();

/*!
 * \brief TileLang intrinsic for carrying pointer access metadata in frontend.
 *
 * Unlike `tir.builtin.tvm_access_ptr`, this op keeps a `BufferLoad` argument so
 * downstream analysis can recover the referenced `Buffer` (and its strides /
 * scope), while also carrying the access mask required by synchronization and
 * safety checks.
 *
 * The frontend is expected to lower this op to `tir.builtin.tvm_access_ptr`
 * once the additional metadata is no longer needed.
 *
 * access_ptr(base_load, extent, rw_mask)
 *
 * - base_load: BufferLoad whose indices denote the base element address.
 * - extent: 1D extent in elements (same meaning as tvm_access_ptr arg3).
 * - rw_mask: 1=read, 2=write, 3=read-write.
 */
TVM_DLL const Op &access_ptr();

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

// Packed x2 element-wise math (float32x2, bfloat16x2, float16x2)
TVM_DLL const Op &add2();
TVM_DLL const Op &sub2();
TVM_DLL const Op &mul2();
TVM_DLL const Op &fma2();
TVM_DLL const Op &max2();
TVM_DLL const Op &min2();
TVM_DLL const Op &abs2();

// random op
TVM_DLL const Op &rng_init();
TVM_DLL const Op &rng_rand();
TVM_DLL const Op &rng_rand_float();

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

/*
 * \brief tvm intrinsics for cluster barrier arrive
 *
 * ptx_arrive_cluster_barrier(mbarrier, cta_id)
 *
 */
TVM_DLL const Op &ptx_arrive_cluster_barrier();

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
 * \brief TileLang intrinsic for PTX async copy from global to shared memory
 *
 * ptx_cp_async(dst_access_ptr, src_access_ptr, bytes)
 * ptx_cp_async(dst_access_ptr, src_access_ptr, bytes, predicate)
 *
 */
TVM_DLL const Op &ptx_cp_async();

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
 * \brief Synchronize all threads in a grid
 *
 * sync_grid()
 *
 */
TVM_DLL const Op &sync_grid();

/*!
 * \brief Synchronize all threads in a warp
 *
 * sync_warp()
 *
 */
TVM_DLL const Op &sync_warp();

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
 * \brief Warp-vote: non-zero if ANY active lane in the mask has a non-zero
 * predicate. Lowers to `__any_sync(mask, predicate)` on CUDA and
 * `__any(predicate)` on HIP (mask is ignored on HIP).
 *
 * int32 any_sync(mask, predicate)
 */
TVM_DLL const Op &any_sync();

/*!
 * \brief Warp-vote: non-zero only if ALL active lanes in the mask have a
 * non-zero predicate. Lowers to `__all_sync(mask, predicate)` on CUDA and
 * `__all(predicate)` on HIP (mask is ignored on HIP).
 *
 * int32 all_sync(mask, predicate)
 */
TVM_DLL const Op &all_sync();

/*!
 * \brief Warp-ballot: bitmask of lanes in the mask with non-zero predicate.
 *
 * CUDA: `__ballot_sync(mask, predicate)` returns `uint32`; the codegen
 * zero-extends the result to `uint64`.
 * HIP: `__ballot(predicate)` returns `uint64` natively, covering all 64
 * lanes of the wavefront. Mask is ignored on HIP.
 *
 * uint64 ballot_sync(mask, predicate)
 */
TVM_DLL const Op &ballot_sync();

/*!
 * \brief Full-warp / full-wavefront ballot. Equivalent to
 * `ballot_sync(0xFFFFFFFF, predicate)`.
 *
 * uint64 ballot(predicate)
 */
TVM_DLL const Op &ballot();

/*!
 * \brief Bitmask of currently active (non-exited) lanes. Lowers to
 * `__activemask()` (zero-extended to `uint64`) on CUDA and `__ballot(1)` on
 * HIP.
 *
 * uint64 activemask()
 */
TVM_DLL const Op &activemask();

/*!
 * \brief Block barrier that returns the number of threads whose predicate
 * evaluates to non-zero. Lowers to `__syncthreads_count(predicate)` on both
 * CUDA and HIP.
 *
 * int32 syncthreads_count(predicate)
 */
TVM_DLL const Op &syncthreads_count();

/*!
 * \brief Block barrier that returns non-zero only if ALL threads have a
 * non-zero predicate. Lowers to `__syncthreads_and(predicate)` on both
 * CUDA and HIP.
 *
 * int32 syncthreads_and(predicate)
 */
TVM_DLL const Op &syncthreads_and();

/*!
 * \brief Block barrier that returns non-zero if ANY thread has a non-zero
 * predicate. Lowers to `__syncthreads_or(predicate)` on both CUDA and HIP.
 *
 * int32 syncthreads_or(predicate)
 */
TVM_DLL const Op &syncthreads_or();

/*!
 * \brief Warp shuffle: broadcast `value` from `src_lane` within each subgroup
 * of `width` lanes. Lowers to `__shfl_sync(mask, value, src_lane, width)` on
 * CUDA and `__shfl(value, src_lane, width)` on HIP. The dtype of the result
 * matches the dtype of `value`.
 *
 * T shfl_sync(mask, value, src_lane, width)
 */
TVM_DLL const Op &shfl_sync();

/*!
 * \brief Warp shuffle (XOR-swap variant). Lowers to `__shfl_xor_sync` on CUDA
 * and `__shfl_xor` on HIP.
 *
 * T shfl_xor_sync(mask, value, lane_mask, width)
 */
TVM_DLL const Op &shfl_xor_sync();

/*!
 * \brief Warp shuffle (shift-down variant). Lowers to `__shfl_down_sync` on
 * CUDA and `__shfl_down` on HIP.
 *
 * T shfl_down_sync(mask, value, delta, width)
 */
TVM_DLL const Op &shfl_down_sync();

/*!
 * \brief Warp shuffle (shift-up variant). Lowers to `__shfl_up_sync` on CUDA
 * and `__shfl_up` on HIP.
 *
 * T shfl_up_sync(mask, value, delta, width)
 */
TVM_DLL const Op &shfl_up_sync();

/*!
 * \brief Warp match-any: returns a mask of lanes in `mask` whose `value`
 * equals the calling lane's value. Lowers to `__match_any_sync` on CUDA
 * (compute capability >= 7.0). Not supported on HIP.
 *
 * uint32 match_any_sync(mask, value)
 */
TVM_DLL const Op &match_any_sync();

/*!
 * \brief Warp match-all: returns `mask` if all lanes in `mask` agree on
 * `value`, else 0. Lowers to `__match_all_sync` on CUDA (compute capability
 * >= 7.0, the trailing `int*` predicate output is discarded via an
 * immediately-invoked lambda). Not supported on HIP.
 *
 * uint32 match_all_sync(mask, value)
 */
TVM_DLL const Op &match_all_sync();

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
 *  This op wraps a templated `tl::gemm_*<...>` call into the generated device
 *  code. Python-side lowering backends that want to delegate to the C++
 *  template implementations in `src/tl_templates/<target>/gemm*.h` can emit a
 *  call to this builtin directly via
 *    T.call_intrin("handle", "tl.tl_gemm", op_instance_str, A_ptr, B_ptr,
 * C_ptr) where `op_instance_str` is the fully-instantiated `tl::gemm_ss<M, N,
 * K, ...>` template string.
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
 * \brief tilelang intrinsic for element-wise atomic addition.
 *
 *  This op is used to represent an element-wise atomic add operation in
 * tilelang.
 */
TVM_DLL const Op &atomic_add_elem_op();

/*!
 * \brief tilelang intrinsic for element-wise atomic addition with return value.
 *
 *  This op is used to represent an element-wise atomic add operation in
 * tilelang that returns the previous value.
 */
TVM_DLL const Op &atomic_add_ret_elem_op();

/*!
 * \brief tilelang intrinsic for vectorized (x2) atomic addition.
 *
 *  This op is used to represent a vectorized atomic add operation (2 elements)
 * in tilelang.
 */
TVM_DLL const Op &atomic_addx2_elem_op();

/*!
 * \brief tilelang intrinsic for vectorized (x4) atomic addition.
 *
 *  This op is used to represent a vectorized atomic add operation (4 elements)
 * in tilelang.
 */
TVM_DLL const Op &atomic_addx4_elem_op();

/*!
 * \brief tilelang intrinsic for atomic load.
 *
 *  This op is used to represent an atomic load operation in tilelang.
 */
TVM_DLL const Op &atomic_load_elem_op();

/*!
 * \brief tilelang intrinsic for atomic store.
 *
 *  This op is used to represent an atomic store operation in tilelang.
 */
TVM_DLL const Op &atomic_store_elem_op();

/*!
 * \brief tilelang intrinsic for element-wise atomic maximum.
 *
 *  This op is used to represent an element-wise atomic max operation in
 * tilelang.
 */
TVM_DLL const Op &atomic_max_elem_op();

/*!
 * \brief tilelang intrinsic for element-wise atomic maximum with return value.
 *
 *  This op is used to represent an element-wise atomic max operation in
 * tilelang that returns the previous value.
 */
TVM_DLL const Op &atomic_max_ret_elem_op();

/*!
 * \brief tilelang intrinsic for element-wise atomic minimum.
 *
 *  This op is used to represent an element-wise atomic min operation in
 * tilelang.
 */
TVM_DLL const Op &atomic_min_elem_op();

/*!
 * \brief tilelang intrinsic for element-wise atomic minimum with return value.
 *
 *  This op is used to represent an element-wise atomic min operation in
 * tilelang that returns the previous value.
 */
TVM_DLL const Op &atomic_min_ret_elem_op();

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

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_BUILTIN_H_
