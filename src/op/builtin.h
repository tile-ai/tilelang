/*!
 * \file tl/op/builtin.h
 * \brief Backend-neutral and cross-backend TileLang intrinsic Ops.
 */

#ifndef TVM_TL_OP_BUILTIN_H_
#define TVM_TL_OP_BUILTIN_H_

#include "operator.h"

#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>

namespace tvm {
namespace tl {

namespace attr {

static constexpr const char *kSafeValueMap = "safe_value_map";

// Async-copy annotations shared by CUDA and ROCm lowering.
static constexpr const char *kLoopPreferAsync = "parallel_prefer_async";
static constexpr const char *kParallelAsyncWithoutAsyncCommitWait =
    "parallel_async_without_async_commit_wait";
static constexpr const char *kAsyncCopyNoImplicitCommitWait =
    "no_implicit_async_commit_wait";

// Pipeline annotation carrying an explicit mbarrier parity expression.
static constexpr const char *kPipelineMbarPhaseExpr =
    "tl.pipeline_mbar_phase_expr";

static constexpr const char *kLocalVarInit = "tl.local_var_init";
static constexpr const char *kNonRestrictParams = "tl.non_restrict_params";
static constexpr const char *kLexicalAllocScope = "lexical_alloc_scope";

} // namespace attr

inline ffi::Optional<PrimExpr> GetAnnotatedMbarPhaseExpr(
    const ffi::Map<ffi::String, ffi::ObjectRef> &annotations) {
  if (auto val = annotations.Get(attr::kPipelineMbarPhaseExpr)) {
    if (val.value()->IsInstance<PrimExprNode>()) {
      return Downcast<PrimExpr>(val.value());
    }
    LOG(FATAL) << "Annotation `" << attr::kPipelineMbarPhaseExpr
               << "` expects a PrimExpr value, but got "
               << val.value().GetTypeKey();
  }
  return ffi::Optional<PrimExpr>();
}

// Backend-neutral pass configuration and PrimFunc attribute keys.
static constexpr const char *kDebugMergeSharedMemoryAllocations =
    "tl.debug_merge_shared_memory_allocations";
static constexpr const char *kSmemAlignmentMap = "tl.smem_alignment_map";
static constexpr const char *kDisableSafeMemoryLegalize =
    "tl.disable_safe_memory_legalize";
static constexpr const char *kConfigIndexBitwidth = "tl.config_index_bitwidth";
static constexpr const char *kEnableAggressiveSharedMemoryMerge =
    "tl.enable_aggressive_shared_memory_merge";
static constexpr const char *kDisableSharedMemoryReuse =
    "tl.disable_shared_memory_reuse";
static constexpr const char *kDisableFastMath = "tl.disable_fast_math";
static constexpr const char *kEnableFastMath = "tl.enable_fast_math";
static constexpr const char *kEnableAsyncCopy = "tl.enable_async_copy";
// Force the canonical FullParticipant baseline for every reducer epoch,
// disabling narrow physical plans (compact storage / sub-block collectives).
//
// This switch is NOT a workaround for expected narrow-plan bugs — a narrow
// plan whose structural proofs succeed must be semantically correct. It
// exists because the baseline is the reducer design's reference lowering
// (proposal: every physical-plan optimization must be independently
// switchable back to the same canonical semantics), which gives us:
//   * differential testing: for any kernel, forced-baseline and auto plan
//     selection must agree numerically — the standing acceptance test for
//     every future planner extension (dst steering, multi-step collectives,
//     narrow-plan seed/batch);
//   * a field escape hatch: if a narrow plan ever miscompiles, one config
//     line restores the proof-free lowering while preserving a repro;
//   * plan-choice A/B measurement (registers, collective width, latency).
static constexpr const char *kReducerForceBaseline =
    "tl.reducer_force_baseline";
static constexpr const char *kEnableReducerPlanVerbose =
    "tl.enable_reducer_plan_verbose";
// The cost model that ranks free-mode layout attempts, by name:
// "io-aware" (default) scores estimated global-memory access cost —
// vector width / coalescing of every fragment<->global copy, weighted by
// bytes moved — with register count as the tiebreak; "register-count" is
// the legacy total-register-slots-only ordering.
static constexpr const char *kLayoutCostModel = "tl.layout_cost_model";
static constexpr const char *kEnableVectorizePlannerVerbose =
    "tl.enable_vectorize_planner_verbose";
static constexpr const char *kDisableLoopUnswitching =
    "tl.disable_loop_unswitching";
static constexpr const char *kLoopUnswitchingAllowNonTrivialElse =
    "tl.loop_unswitching_allow_non_trivial_else";
static constexpr const char *kIfStmtBindingInlineReplayableBinds =
    "tl.if_stmt_binding_inline_replayable_binds";
static constexpr const char *kStorageRewriteDetectInplace =
    "tl.storage_rewrite_detect_inplace";
static constexpr const char *kASTPrintEnable = "tl.ast_print_enable";
static constexpr const char *kLayoutVisualizationEnable =
    "tl.layout_visualization_enable";
static constexpr const char *kLayoutVisualizationFormats =
    "tl.layout_visualization_formats";
static constexpr const char *kDeviceCompileFlags = "tl.device_compile_flags";
/*! \brief Emit #line directives in generated C-family source from TIR spans,
 * mapping generated statements back to their Python source lines. Default:
 * false. */
static constexpr const char *kEmitLineDirectives = "tl.emit_line_directives";
static constexpr const char *kDisableDataRaceCheck =
    "tl.disable_data_race_check";
/*! \brief Disable the buffer-initialization check.
 *
 * The check warns when a non-global-scope buffer is read before anything
 * writes it. It is enabled by default.
 */
static constexpr const char *kDisableBufferInitCheck =
    "tl.disable_buffer_init_check";
static constexpr const char *kDisableThreadStorageSync =
    "tl.disable_thread_storage_sync";
static constexpr const char *kForceLetInline = "tl.force_let_inline";
static constexpr const char *kDisableOutOfBoundWarning =
    "tl.disable_out_of_bound_warning";
static constexpr const char *kEnableDumpIR = "tl.enable_dump_ir";
static constexpr const char *kDumpIRDir = "tl.dump_ir_path";
static constexpr const char *kPassProfile = "tl.pass_profile";
static constexpr const char *kPassProfileThresholdMs =
    "tl.pass_profile_threshold_ms";

/*!
 * \brief Call a TVM-FFI packed function with an existing argument array and
 * result slot.
 *
 * tvm_ffi_call_with_result(func_name, args, num_args, result)
 *
 * This is an internal host-codegen intrinsic.  Unlike tvm_call_packed, the
 * caller owns the already-populated TVMFFIAny argument array and provides the
 * result slot directly.  It is used by the callee-allocated output wrapper to
 * assemble multiple environment-allocated tensors into an ffi.Array without
 * routing their shapes or handles back through Python.
 */
TVM_DLL const Op &tvm_ffi_call_with_result();

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

/*!
 * \brief Tile memory region descriptor: a transport-only bridge that carries
 * a BufferRegion (plus an access mask) through Call args.
 *
 * Why tl.region instead of passing BufferRegion directly?
 * - When a BufferRegion is passed as a call argument through call_intrin/FFI,
 *   the Python->C++ conversion lowers it to a BufferLoad(indices), encoding a
 *   contiguous interval as Ramp(base, stride, lanes).
 * - Ramp lanes may only be a constant or vscale*k, so a dynamic extent
 *   (e.g. H1 - H0) cannot be encoded as lanes, and BufferLoad carries no
 *   per-axis extents, so downstream tile operators (tl.copy, tl.reduce, ...)
 *   cannot losslessly recover dynamic extents from a BufferLoad alone.
 * - tl.region packs buffer + mins (BufferLoad indices) + explicit extents
 *   into Call args; the backend reconstructs a BufferRegion faithfully via
 *   NormalizeToBufferRegion / NormalizeToAccessRegion (op/utils.h).
 *
 * region(BufferLoad(buffer, [min_0, ..., min_{n-1}]), access_mask,
 *        extent_0, ..., extent_{n-1})
 *
 * - args[0]: BufferLoad whose indices are the per-axis minima.
 * - args[1]: constant int access mask (1=read, 2=write, 3=read-write).
 *   Transport metadata only; it does not affect lowering.
 * - args[2 + i]: extent of axis i (may be a dynamic PrimExpr).
 */
TVM_DLL const Op &region();

// Packed x2 element-wise math (float32x2, bfloat16x2, float16x2)
TVM_DLL const Op &add2();
TVM_DLL const Op &sub2();
TVM_DLL const Op &mul2();
TVM_DLL const Op &fma2();
TVM_DLL const Op &max2();
TVM_DLL const Op &min2();
TVM_DLL const Op &abs2();

// These historical PTX-named IR markers are shared by CUDA and ROCm
// lowerings. Keep their registered names stable for frontend compatibility.

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
 * \brief tvm intrinsics for stmatrix
 *
 * ptx_ldmatrix(transposed, num, shared_addr, int32_values...)
 *
 */
TVM_DLL const Op &ptx_stmatrix();

/*!
 * \brief TileLang intrinsic for PTX async copy from global to shared memory
 *
 * ptx_cp_async(dst_access_ptr, src_access_ptr, num_elems)
 * ptx_cp_async(dst_access_ptr, src_access_ptr, num_elems, predicate)
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
 * \brief Annotation-only producer reg dealloc hint for warp specialization
 *
 * annotate_producer_reg_dealloc(num_reg)
 *
 */
TVM_DLL const Op &annotate_producer_reg_dealloc();

/*!
 * \brief Annotation-only consumer reg alloc hint for warp specialization
 *
 * annotate_consumer_reg_alloc(num_reg)
 *
 */
TVM_DLL const Op &annotate_consumer_reg_alloc();

/*!
 * \brief No set reg hint for warp-specialized branched
 *
 * no_set_max_nreg()
 *
 */
TVM_DLL const Op &no_set_max_nreg();

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
 * \brief Synchronize all threads in a warp
 *
 * sync_warp()
 *
 */
TVM_DLL const Op &sync_warp();

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
 * \brief tilelang intrinsic for vectorized (x2) atomic addition with return
 * value.
 *
 *  This op is used to represent a vectorized atomic add operation (2 elements)
 * in tilelang that returns the previous packed value.
 */
TVM_DLL const Op &atomic_addx2_ret_elem_op();

/*!
 * \brief tilelang intrinsic for vectorized (x4) atomic addition.
 *
 *  This op is used to represent a vectorized atomic add operation (4 elements)
 * in tilelang.
 */
TVM_DLL const Op &atomic_addx4_elem_op();

/*!
 * \brief tilelang intrinsic for vectorized (x4) atomic addition with return
 * value.
 *
 *  This op is used to represent a vectorized atomic add operation (4 elements)
 * in tilelang that returns the previous packed value.
 */
TVM_DLL const Op &atomic_addx4_ret_elem_op();

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
 * \brief tilelang intrinsic for element-wise atomic bitwise-or.
 *
 *  This op is used to represent an element-wise atomic or operation in
 * tilelang.
 */
TVM_DLL const Op &atomic_or_elem_op();

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
 * \brief tilelang intrinsic for CUDA/HIP read-only cache load (__ldg).
 *
 *  This op allows users to explicitly request a non-coherent cached load
 *  from global memory by emitting `__ldg(&ptr[idx])`. It provides a direct way
 *  to leverage the read-only data cache for performance-sensitive loads when
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

#endif // TVM_TL_OP_BUILTIN_H_
