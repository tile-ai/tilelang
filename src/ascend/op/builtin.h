/*!
 * \file tl/ascend/op/builtin.h
 * \brief Ascend-specific TileLang intrinsic Ops.
 */

#ifndef TVM_TL_ASCEND_OP_BUILTIN_H_
#define TVM_TL_ASCEND_OP_BUILTIN_H_

#include "op/builtin.h"

namespace tvm {
namespace tl {

// Ascend-specific pass configuration keys. Their string values remain stable
// because they are part of the Python PassContext interface.
static constexpr const char *kEnableAutoSchedule = "tl.enable_auto_schedule";

/*!
 * \brief Marker op for user-declared auto-schedule conflict facts.
 *
 * One no-op marker carries both conflict polarities as statements:
 *   Evaluate(Call(tl.conflict_hint, a, b, level, cross, group, is_conflict))
 * The marker is consumed by NormalizeConflictHints before AutoSchedule.
 * Registered as `kPure` so that, when auto-schedule is disabled and the marker
 * is left unconsumed, RemoveNoOp drops it -- it never reaches codegen. It has
 * no `TLOpBuilder`: ParseOperator returns an empty TileOperator, which every
 * tile-op consumer skips via `.defined()`.
 */
TVM_DLL const Op &conflict_hint();

/*!
 * \brief Ascend pipeline barrier intrinsic.
 *
 * ascend_pipe_barrier(pipe_t_string)
 *
 */
TVM_DLL const Op &ascend_pipe_barrier();

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
 * Codegen emits asc_fill_l1 at dst + byte_offset. repeat_times is the number
 * of fill iterations; block_num is the number of 32-byte blocks written by
 * each iteration; dst_gap is the number of skipped 32-byte blocks between
 * adjacent iterations. fill_word_bits selects a raw uint16_t or uint32_t
 * destination view, and raw_value carries the repeated element bit pattern.
 */
TVM_DLL const Op &ascend_fill_l1();

/*!
 * \brief Ascend load from CBuf (L1) to CA (L0A).
 *
 * ascend_load_cbuf_to_ca(dst, src, mStartPosition, kStartPosition,
 * mStep, kStep, srcStride, dstStride, transpose)
 */
TVM_DLL const Op &ascend_load_cbuf_to_ca();

/*!
 * \brief Ascend load from CBuf (L1) to CB (L0B).
 *
 * ascend_load_cbuf_to_cb(dst, src, mStartPosition, kStartPosition,
 * mStep, kStep, srcStride, dstStride, transpose)
 */
TVM_DLL const Op &ascend_load_cbuf_to_cb();

/*!
 * \brief Ascend standalone MX scale-factor load into the L0A slot shadow of a
 *        data tile.
 *
 * ascend_load_ca_sf(dst_data_ptr, sf_ptr, x_start, y_start, x_step, y_step,
 * src_stride, dst_stride)
 *
 * `dst_data_ptr` addresses the L0A DATA tile whose MX scale slots are
 * written: the hardware keys the slot positions to that address
 * (asc_copy_l12l0a_mx consumes it in 16-byte units; codegen appends the /16).
 * x/y are the L1-source fractal coordinates of the scales, exactly as in the
 * 16-arg companion form of ascend_load_cbuf_to_ca.
 */
TVM_DLL const Op &ascend_load_ca_sf();

/*!
 * \brief Ascend standalone MX scale-factor load into the L0B slot shadow of a
 *        data tile. Argument protocol mirrors ascend_load_ca_sf.
 */
TVM_DLL const Op &ascend_load_cb_sf();

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
 * \brief Set the preferred M/N result traversal direction for Ascend MAD.
 * ascend_set_mmad_direction(direction), where direction is "m" or "n".
 */
TVM_DLL const Op &ascend_set_mmad_direction();

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

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_BUILTIN_H_
