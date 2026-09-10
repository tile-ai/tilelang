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

// Annotation on the tilelang_root block recording the SIMT thread-block
// extents requested by T.Kernel(threads=...). It is a launch hint: SIMT
// backends materialize it as threadIdx.* thread_extent scopes, other
// backends ignore it.
static constexpr const char *kLaunchThreads = "tl.launch_threads";

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
// "register-count" (default) uses total fragment register slots;
// "io-aware" scores estimated global-memory access cost — vector width /
// coalescing of every fragment<->global copy, weighted by bytes moved —
// with register count as the tiebreak.
static constexpr const char *kLayoutCostModel = "tl.layout_cost_model";
static constexpr const char *kEnableVectorizePlannerVerbose =
    "tl.enable_vectorize_planner_verbose";
static constexpr const char *kEnableAutoSchedule = "tl.enable_auto_schedule";
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

/*!
 * \brief Placeholder for the thread index along one launch axis.
 *
 * T.Kernel binds each thread variable as `LetStmt(tx, launch_thread_idx(axis))`
 * so the kernel body can reference a thread index before the target is known.
 * tl.MaterializeKernelLaunch replaces the binding with a real threadIdx.*
 * thread_extent scope on SIMT backends and rejects any use on backends
 * without SIMT. It must never reach codegen.
 *
 * int32 launch_thread_idx(axis)
 */
TVM_DLL const Op &launch_thread_idx();

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
 * \brief Ascend pipeline barrier intrinsic.
 *
 * ascend_pipe_barrier(pipe_t_string)
 *
 */
TVM_DLL const Op &ascend_pipe_barrier();

// RNG ops. #2855 registers these under the CUDA dialect, but this fork also
// consumes them from backend-neutral code (loop_vectorize) and the Ascend
// codegen/SIMT RNG path, so declare them here too (registration stays in
// cuda/op/builtin.cc; a redeclaration is harmless).
TVM_DLL const Op &rng_init();
TVM_DLL const Op &rng_rand();
TVM_DLL const Op &rng_rand_float();

// device_assert lowers through the Ascend codegen (toolkit assert()) as well as
// CUDA; #2855 files it under the CUDA dialect, so redeclare for asc codegen.
TVM_DLL const Op &device_assert();
TVM_DLL const Op &device_assert_with_msg();

// Ascend SIMD raw CCE intrinsics (T.simd.* API).
// TIR ops registered as tl.simd.<name>. Emit raw CCE intrinsics
// (vadd/vld/vst/vfcvt/...) from __clang_cce_vector_intrinsics.h.
// Fragment registers use vector_<T> types; masks are vector_bool.

// -- Mask
TVM_DLL const Op &simd_pset();
TVM_DLL const Op &simd_pge();
TVM_DLL const Op &simd_pand();
TVM_DLL const Op &simd_por();
TVM_DLL const Op &simd_pxor();
TVM_DLL const Op &simd_pnot();
TVM_DLL const Op &simd_psel();

// -- Load / Store
TVM_DLL const Op &simd_pld();
TVM_DLL const Op &simd_pst();
TVM_DLL const Op &simd_vld();
TVM_DLL const Op &simd_vld2(); // dual-dest load (e.g. DINTLV_B16) → vec_pair
TVM_DLL const Op &simd_vsts();
TVM_DLL const Op &simd_vsstb();

// -- Binary arithmetic
TVM_DLL const Op &simd_vadd();
TVM_DLL const Op &simd_vaddc();
TVM_DLL const Op &simd_vsubc();
TVM_DLL const Op &simd_vaddcs();
TVM_DLL const Op &simd_vsubcs();
TVM_DLL const Op &simd_vmull();
TVM_DLL const Op &simd_vlrelu();
TVM_DLL const Op &simd_vprelu();
TVM_DLL const Op &simd_update_mask();
TVM_DLL const Op &simd_ppack();
TVM_DLL const Op &simd_punpack();
TVM_DLL const Op &simd_pintlv();
TVM_DLL const Op &simd_pdintlv();
TVM_DLL const Op &simd_vunpack();
TVM_DLL const Op &simd_vusqz();
TVM_DLL const Op &simd_vsub();
TVM_DLL const Op &simd_vmul();
TVM_DLL const Op &simd_vmula();
TVM_DLL const Op &simd_vmadd();
TVM_DLL const Op &simd_vaxpy();
TVM_DLL const Op &simd_vdiv();
TVM_DLL const Op &simd_vmax();
TVM_DLL const Op &simd_vmin();
TVM_DLL const Op &simd_vand();
TVM_DLL const Op &simd_vor();
TVM_DLL const Op &simd_vxor();
TVM_DLL const Op &simd_vshl();
TVM_DLL const Op &simd_vshr();

// -- Unary
TVM_DLL const Op &simd_vexp();
TVM_DLL const Op &simd_vln();
TVM_DLL const Op &simd_vsqrt();
TVM_DLL const Op &simd_vabs();
TVM_DLL const Op &simd_vneg();
TVM_DLL const Op &simd_vrelu();
TVM_DLL const Op &simd_vnot();

// -- Broadcast
TVM_DLL const Op &simd_vdup();
TVM_DLL const Op &simd_vdupv(); // 5-arg vector→vector lane-N broadcast

// -- Cross-lane reduction
TVM_DLL const Op &simd_vcpadd();
TVM_DLL const Op &simd_vcadd();
TVM_DLL const Op &simd_vcmax();
TVM_DLL const Op &simd_vcmin();
TVM_DLL const Op &simd_vcgadd();
TVM_DLL const Op &simd_vcgmax();
TVM_DLL const Op &simd_vcgmin();
TVM_DLL const Op &simd_vsqz();
TVM_DLL const Op &simd_dhistv2();
TVM_DLL const Op &simd_chistv2();

// -- Index ramp / compare
TVM_DLL const Op &simd_vci();
TVM_DLL const Op &simd_vcmp();
TVM_DLL const Op &simd_vcmps();

// -- Register permutation
TVM_DLL const Op &simd_vintlv();
TVM_DLL const Op &simd_vgatherb();
TVM_DLL const Op &simd_vgather2();
TVM_DLL const Op &simd_vscatter();
TVM_DLL const Op &simd_vdintlv();
TVM_DLL const Op &simd_pair_get();
TVM_DLL const Op &simd_vpack();

// -- Type conversion
TVM_DLL const Op &simd_vcvt();

// -- Special
TVM_DLL const Op &simd_vsel();
TVM_DLL const Op &simd_vselr();
TVM_DLL const Op &simd_vmaxs();
TVM_DLL const Op &simd_vmins();
TVM_DLL const Op &simd_vmuls();
TVM_DLL const Op &simd_vadds();
TVM_DLL const Op &simd_vshls();
TVM_DLL const Op &simd_vshrs();
TVM_DLL const Op &simd_mem_bar();
TVM_DLL const Op &simd_vexpdif();
TVM_DLL const Op &simd_vabsdif();

// Ascend VMI virtual vector intrinsics (T.vmi.* API).
// TIR ops registered as tl.vmi.<name>. These mirror PTODSL's public pto.vmi
// surface.
//
// ABI notes:
// - The TIR positional operands only carry dataflow values such as
//   ptr/offset/values/mask. Python keyword-only API parameters like size,
//   to_dtype, dist_mode, group, pmode, and order are lowered as call
//   annotations instead of extra positional operands.
// - vload(ptr, offset) with annotations {size, to_dtype?, stride?,
//   block_stride?, repeat_stride?, dist_mode?, group?}
// - vstore(values, destination_ptr, offset, mask?) with annotations
//   {stride?, block_stride?, repeat_stride?, dist_mode?, group?, pmode?}
// - create_mask(active_lanes) with annotations {size, group?}
// - vci(base) with annotations {size, order?}
// - vbrc(value) with annotations {size, group?}
// - vintlv(lhs, rhs, mask) / vdintlv(lhs, rhs, mask)
// - pair_get(pair, index) is an internal pure helper for tuple-style unpacking
//   of multi-result VMI calls.

// -- Load / Store / Predicates
TVM_DLL const Op &vmi_vload();
TVM_DLL const Op &vmi_vstore();
TVM_DLL const Op &vmi_create_mask();

// -- Index / Broadcast / Rearrange
TVM_DLL const Op &vmi_vci();
TVM_DLL const Op &vmi_vbrc();
TVM_DLL const Op &vmi_vintlv();
TVM_DLL const Op &vmi_vdintlv();
TVM_DLL const Op &vmi_pair_get();

// -- Binary arithmetic / bitwise / shifts
TVM_DLL const Op &vmi_vadd();
TVM_DLL const Op &vmi_vsub();
TVM_DLL const Op &vmi_vmul();
TVM_DLL const Op &vmi_vdiv();
TVM_DLL const Op &vmi_vmax();
TVM_DLL const Op &vmi_vmin();
TVM_DLL const Op &vmi_vand();
TVM_DLL const Op &vmi_vor();
TVM_DLL const Op &vmi_vxor();
TVM_DLL const Op &vmi_vshl();
TVM_DLL const Op &vmi_vshr();

// -- Unary / math
TVM_DLL const Op &vmi_vabs();
TVM_DLL const Op &vmi_vneg();
TVM_DLL const Op &vmi_vrelu();
TVM_DLL const Op &vmi_vexp();
TVM_DLL const Op &vmi_vln();
TVM_DLL const Op &vmi_vsqrt();
TVM_DLL const Op &vmi_vnot();

// -- Vector-scalar arithmetic / shifts
TVM_DLL const Op &vmi_vadds();
TVM_DLL const Op &vmi_vmuls();
TVM_DLL const Op &vmi_vmaxs();
TVM_DLL const Op &vmi_vmins();
TVM_DLL const Op &vmi_vshls();
TVM_DLL const Op &vmi_vshrs();

// -- Compare / select / reduce / convert
TVM_DLL const Op &vmi_vcmp();
TVM_DLL const Op &vmi_vcmps();
TVM_DLL const Op &vmi_vsel();
TVM_DLL const Op &vmi_vselr();
TVM_DLL const Op &vmi_vcadd();
TVM_DLL const Op &vmi_vcmax();
TVM_DLL const Op &vmi_vcmin();
TVM_DLL const Op &vmi_vcvt();
TVM_DLL const Op &vmi_vinterpret_cast();

// -- SFU / irregular / histogram
TVM_DLL const Op &vmi_vexpdif();
TVM_DLL const Op &vmi_vaxpy();
TVM_DLL const Op &vmi_vlrelu();
TVM_DLL const Op &vmi_vprelu();
TVM_DLL const Op &vmi_vmull();
TVM_DLL const Op &vmi_vmula();
TVM_DLL const Op &vmi_vdhist();
TVM_DLL const Op &vmi_vchist();
TVM_DLL const Op &vmi_vgather();
TVM_DLL const Op &vmi_vgatherb();
TVM_DLL const Op &vmi_vscatter();

/*!
 * \brief Ascend SetFlag intrinsic for pipeline synchronization.
 *
 * ascend_set_flag(hard_event_string, event_id)
 *
 */
TVM_DLL const Op &ascend_set_flag();

/*!
 * \brief Ascend WaitFlag intrinsic for pipeline synchronization.
 *
 * ascend_wait_flag(hard_event_string, event_id)
 *
 */
TVM_DLL const Op &ascend_wait_flag();

/*!
 * \brief Ascend thread fence intrinsic.
 *
 * ascend_threadfence()
 *
 */
TVM_DLL const Op &ascend_threadfence();

/*!
 * \brief Ascend CrossCoreSetFlag intrinsic for cross-core synchronization.
 *
 * ascend_cross_core_set_flag(mode_id_string, pipe_string, flag_id)
 *
 */
TVM_DLL const Op &ascend_cross_core_set_flag();

/*!
 * \brief Ascend CrossCoreWaitFlag intrinsic for cross-core synchronization.
 *
 * ascend_cross_core_wait_flag(flag_id)
 *
 */
TVM_DLL const Op &ascend_cross_core_wait_flag();

/*!
 * \brief Ascend DMA copy from GM to UBuf.
 *
 * ascend_copy_gm_to_ubuf(dst, src, sid, nBurst, burstLen,
 * leftPadding, rightPadding, dataSelect, l2CacheCtl,
 * burstSrcStride, burstDstStride)
 *
 */
TVM_DLL const Op &ascend_copy_gm_to_ubuf();

/*!
 * \brief Ascend set padding fill value for subsequent padded MTE copies.
 *
 * ascend_set_copy_pad_value(value)
 *
 * The single argument is a typed scalar PrimExpr; its dtype selects the
 * AscendC::SetPadValue<T> template instantiation. Configures stateful hardware
 * pad value consumed by a following padded GM -> UB/L1 copy.
 *
 */
TVM_DLL const Op &ascend_set_copy_pad_value();

/*!
 * \brief Ascend DMA copy from UBuf to GM.
 *
 * ascend_copy_ubuf_to_gm(dst, src, sid, burst_num, burst_len, l2_cache_ctl,
 * burst_dst_stride, burst_src_stride)
 *
 */
TVM_DLL const Op &ascend_copy_ubuf_to_gm();

/*!
 * \brief Ascend DMA copy from GM to CBuf (L1).
 *
 * ascend_copy_gm_to_cbuf(dst, src, sid, loop1_src_stride, l2_cache_ctrl,
 * n_value, d_value, loop4_src_stride, smallc0_en, transpose, dst_n_value,
 * physical_dtype)
 *
 * When transpose=0 emits copy_gm_to_cbuf_multi_nd2nz (row-major → NZ).
 * When transpose!=0 emits copy_gm_to_cbuf_multi_dn2nz (col-major → NZ),
 * which transposes the N/D mapping — useful for NN matmul on Ascend.
 */
TVM_DLL const Op &ascend_copy_gm_to_cbuf();

/*!
 * \brief Fill an L1 (CBuf) NZ matrix region with a scalar value.
 *
 * ascend_fill_l1(dst, byte_offset, raw_value, repeat_times, block_num,
 *                dst_gap, fill_word_bits)
 *
 * Emits create_cbuf_matrix at dst + byte_offset. repeat_times is the number of
 * fill iterations; block_num is the number of 32-byte blocks written by each
 * iteration; dst_gap is the number of skipped 32-byte blocks between adjacent
 * iterations. fill_word_bits selects a raw uint16_t or uint32_t destination
 * view, and raw_value carries the repeated element bit pattern.
 */
TVM_DLL const Op &ascend_fill_l1();

/*!
 * \brief Ascend load from CBuf (L1) to CA (L0A).
 *
 * ascend_load_cbuf_to_ca(dst, src, mStartPosition, kStartPosition,
 * mStep, kStep, srcStride, dstStride, transpose)
 *
 * Optionally 16 args when an MX scale-factor companion load is attached; the
 * extra args drive a following load_cbuf_to_ca_mx:
 *   [9]  sf_ptr, [10] sf_x_start,
 *   [11] sf_y_start (y is contiguous fractal direction),
 *   [12] sf_x_step,
 *   [13] sf_y_step,
 *   [14] sf_src_stride, [15] sf_dst_stride.
 */
TVM_DLL const Op &ascend_load_cbuf_to_ca();

/*!
 * \brief Ascend load from CBuf (L1) to CB (L0B).
 *
 * ascend_load_cbuf_to_cb(dst, src, mStartPosition, kStartPosition,
 * mStep, kStep, srcStride, dstStride, transpose)
 *
 * Optionally 16 args when an MX scale-factor companion load is attached; the
 * extra args drive a following load_cbuf_to_cb_mx:
 *   [9]  sf_ptr, [10] sf_x_start,
 *   [11] sf_y_start (y is contiguous fractal direction),
 *   [12] sf_x_step,
 *   [13] sf_y_step,
 *   [14] sf_src_stride, [15] sf_dst_stride.
 */
TVM_DLL const Op &ascend_load_cbuf_to_cb();

/*!
 * \brief Ascend copy matrix from CC (L0C) to UBuf.
 *
 * ascend_copy_matrix_cc_to_ub(dst, src, sid, n_size, m_size,
 * loop_dst_stride, loop_src_stride, dual_dst_ctl, sub_blockid, clip_relu_pre,
 * unit_flag_ctl, quant_pre, relu_pre, split_en, NZ2ND_en, quant_post,
 * relu_post, clip_relu_post, loop_enhance_en, eltwise_op, eltwise_antq_en,
 * loop_enhance_merge_en, C0_pad_en, wino_post_en, broadcast_en, NZ2DN_en)
 *
 */
TVM_DLL const Op &ascend_copy_matrix_cc_to_ub();

/*!
 * \brief Ascend DMA copy from UBuf to CBuf (L1).
 *
 * ascend_copy_ubuf_to_cbuf(dst, src, sub_blockid, burst_num, burst_len,
 * src_gap, dst_gap)
 *
 */
TVM_DLL const Op &ascend_copy_ubuf_to_cbuf();

/*!
 * \brief Ascend ND→NZ scatter (SimdVF) from UB ND tile to UB NZ tile.
 *
 * ascend_nd2nz_scatter(src_ub, tmp_nz_ub, rows, cols,
 *                      dst_dtype_str, src_dtype_str)
 *
 */
TVM_DLL const Op &ascend_nd2nz_scatter();

/*!
 * \brief Ascend post-scatter UB→L1 raw DMA with NZ-fractal stride correction.
 *
 * ascend_nd2nz_post_copy(dst_l1, src_ub, rows, cols, full_rows, dst_dtype_str)
 *
 */
TVM_DLL const Op &ascend_nd2nz_post_copy();

TVM_DLL const Op &ascend_copy_matrix_cc_to_gm();

/*!
 * \brief Ascend Cube MAD (matrix multiply-add) instruction.
 *
 * ascend_mad(dst, src_a, src_b, M, K, N, unit_flag_ctrl, gemv_ctrl, BTbuf_ctrl,
 * zero_Cmatrix)
 *
 */
TVM_DLL const Op &ascend_mad();

/*!
 * \brief Ascend Cube block-scaled MAD (MXFP8) instruction.
 *
 * ascend_mad_mx(dst, src_a, src_b, M, K, N, unit_flag_ctrl, gemv_ctrl,
 * BTbuf_ctrl, zero_Cmatrix)
 *
 * Same arguments as ascend_mad. The per-block scale factors are read from the
 * L0A/L0B MX scale registers that must be loaded beforehand (e.g. via
 * T.copy(l1_data, l0, sf=l1_sf) which emits load_cbuf_to_ca_mx/cb_mx).
 */
TVM_DLL const Op &ascend_mad_mx();

/*!
 * \brief Ascend GEMM with L1-scoped inputs (auto sub-K pipeline).
 *
 * ascend_gemm_l1(cc_ptr, cbuf_a_ptr, cbuf_b_ptr, M, K, N, tile_k_sub,
 * trans_b, clear_accum, dtype_str)
 *
 */
TVM_DLL const Op &ascend_gemm_l1();

/*!
 * \brief Ascend block-scaled GEMM with L1-scoped inputs (MXFP8).
 *
 * ascend_blockscaled_gemm_l1(cc_ptr, a_ptr, b_ptr, sfa_ptr, sfb_ptr,
 * M, K, N, tile_k_sub, trans_b, clear_accum, in_dtype_str, sf_dtype_str,
 * accum_dtype_str, buf_offset, sf_k_offset, sf_nz_stride, unit_flag_ctrl)
 *
 * Uses blockscaled_gemm.h template with mad_mx instruction.
 */
TVM_DLL const Op &ascend_blockscaled_gemm_l1();

/*!
 * \brief Ascend scalar GM read bypassing dcache.
 *
 * ascend_read_gm_bypass_dcache(address_of(BufferLoad))
 *
 * Replaces a scalar BufferLoad from a global buffer that has writes
 * elsewhere.  The codegen emits tl::read_gm_bypass_dcache(ptr).
 */
TVM_DLL const Op &ascend_read_gm_bypass_dcache();

/*!
 * \brief Ascend scalar GM write bypassing dcache.
 *
 * ascend_write_gm_bypass_dcache(address_of(BufferLoad), value)
 *
 * Replaces a scalar BufferStore to a global buffer that has writes
 * elsewhere.  The codegen emits tl::write_gm_bypass_dcache(ptr, val).
 */
TVM_DLL const Op &ascend_write_gm_bypass_dcache();

/*!
 * \brief Ascend get_buf for pipe buffer acquisition.
 *
 * ascend_get_buf(pipe_string, buf_id, mode)
 *
 */
TVM_DLL const Op &ascend_get_buf();

/*!
 * \brief Ascend rls_buf for pipe buffer release.
 *
 * ascend_rls_buf(pipe_string, buf_id, mode)
 *
 */
TVM_DLL const Op &ascend_rls_buf();

/*!
 * \brief Ascend set HF32 mode for fp32 matmul.
 *
 * ascend_set_hf32_mode(mode_int)
 *   mode_int: 0=disable, 1=enable nearest_zero, 2=enable nearest_even
 *
 */
TVM_DLL const Op &ascend_set_hf32_mode();

/*!
 * \brief Ascend arm a hardware store-mode atomic op for GM stores.
 *
 * ascend_set_atomic(op_str, typed_zero)
 *   op_str:     "add" | "max" | "min"
 *   typed_zero: a constant whose dtype selects the accumulate type T
 *               (float / half / int16 / int32 / int8 / bfloat16).
 *
 * Emits AscendC::SetAtomic{Add,Max,Min}<T>(). Arm once; subsequent L0C->GM /
 * UB->GM stores reduce into GM in hardware. Clear with
 * ascend_set_atomic_none().
 */
TVM_DLL const Op &ascend_set_atomic();

/*!
 * \brief Ascend clear the store-mode atomic flag (restore plain store).
 *
 * ascend_set_atomic_none()
 */
TVM_DLL const Op &ascend_set_atomic_none();

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
