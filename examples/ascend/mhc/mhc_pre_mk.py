# SPDX-License-Identifier: MIT
# Copyright (c) 2026 DeepSeek. Portions adapted for shifted mHC.
# Copyright (c) 2026 Q4Kernels contributors.
"""K-split mHC Pre and the SIMD math shared by the pure-M implementation.

Each M/K pair reads its residual partition once for projection, RMS and
weighted Apply. A publication barrier orders workspace writes before the
batched K reduction and register-resident Sinkhorn normalization.
"""

import tilelang
from tilelang import tvm
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout


def dense_ub_view(buffer, shape, element_offset=0):
    """Describe a dense FP32 subregion without allocating another UB buffer."""
    return tvm.tirx.decl_buffer(
        shape, buffer.dtype, data=buffer.data, elem_offset=buffer.elem_offset + element_offset, scope=buffer.scope()
    )


PUBLICATION_BIND_SOURCE = "#define MHC_AS_BIND(...) ({ __VA_ARGS__; 0; })\n"
BATCHED_DMA_SOURCE = r"""
#ifndef MHC_BATCHED_DMA_SOURCE
#define MHC_BATCHED_DMA_SOURCE
__aicore__ inline void mhc_begin_raw_k_loop(
    __ubuf__ float* dst_footprint, __gm__ float* src_footprint,
    uint64_t groups, uint64_t src_k_stride_bytes,
    uint64_t dst_k_stride_bytes) {
    // These pointers declare the complete atomic Task's repeated-copy effects
    // to TileLang. The ordinary T.copy below issues the actual MTE2 descriptor.
    (void)dst_footprint;
    (void)src_footprint;
    asc_set_gm2ub_loop1_stride(src_k_stride_bytes, dst_k_stride_bytes);
    asc_set_gm2ub_loop2_stride(0, 0);
    asc_set_gm2ub_loop_size(groups, 1);
}
__aicore__ inline void mhc_end_raw_k_loop() {
    // Match AscendC ResetLoopModePara: size 1 disables both loop strides.
    asc_set_gm2ub_loop_size(1, 1);
}
#endif
"""


@T.macro
def copy_raw_k_partials(
    raw_read, raw_temp_ub, num_tokens, aiv_id, tile, row_offset, physical_rows, *, block_m: int, k_groups: int, coefficient_rows: int
):
    """Copy all K planes with one descriptor plus loop-mode setup/reset.

    ``physical_rows`` must be positive and fit both the source half-tile and
    the destination coefficient pack. The source pointer's span includes its
    K gaps for dependency analysis, while the descriptor only reads 24 valid
    FP32 values per row. Keep setup, copy and reset inside this single Task:
    an opaque external call alone has no TileLang MTE2 resource classification.
    """
    half_rows = block_m // 2
    token_tiles = T.ceildiv(num_tokens, block_m)
    source_k_stride = 2 * token_tiles * half_rows * 32
    destination_k_stride = coefficient_rows * 32
    source_span = (k_groups - 1) * source_k_stride + (physical_rows - 1) * 32 + 24
    destination_span = (k_groups - 1) * destination_k_stride + (physical_rows - 1) * 32 + 24
    with T.Task():
        T.evaluate(
            T.call_extern(
                "void",
                "mhc_begin_raw_k_loop",
                T.access_ptr(raw_temp_ub[0, 0, 0], "w", extent=destination_span),
                T.access_ptr(raw_read[0, aiv_id, tile, row_offset, 0], "r", extent=source_span),
                T.uint64(k_groups),
                T.uint64(source_k_stride) * T.uint64(4),
                T.uint64(destination_k_stride) * T.uint64(4),
            )
        )
        T.copy(raw_read[0, aiv_id, tile, row_offset : row_offset + physical_rows, 0:24], raw_temp_ub[0, 0:physical_rows, 0:24])
        T.evaluate(T.call_extern("void", "mhc_end_raw_k_loop"))


@T.macro
def copy_square_k_partials(
    square_read, square_temp_ub, num_tokens, aiv_id, tile, row_offset, physical_rows, *, block_m: int, k_groups: int
):
    """Collapse the two unit source axes into one ordinary strided 2D DMA."""
    half_rows = block_m // 2
    token_tiles = T.ceildiv(num_tokens, block_m)
    sq_read = T.reshape(square_read, (k_groups, square_read.shape[1] * square_read.shape[2] * square_read.shape[3]))
    start = (aiv_id * token_tiles + tile) * half_rows + row_offset
    T.copy(sq_read[0:k_groups, start : start + physical_rows], square_temp_ub[0:k_groups, 0:physical_rows])


@T.macro
def prepare_prefix(
    x_ub,
    x_nz_ub,
    previous_ub,
    square_ub,
    y_acc_ub,
    row_offset,
    column_offset,
    stream,
    row_count,
    output_ub=None,
    *,
    first_stream: bool,
    chunks: int,
    nz_rows: int,
    write_output: bool = False,
):
    """Convert one static K prefix, accumulate RMS/Apply, and write NZ.

    Select the prefix and first/later stream before entering this VF. A static
    chunk count and a uint16 row loop let Bisheng generate VLOOP instead of
    scalar branches and repeated address arithmetic inside the vector body.
    Keep the branches as separate scheduled VFs: enclosing all alternatives
    in one Task currently makes AutoSchedule serialize the NZ staging buffer.
    """
    with T.SimdVF():
        S.mem_bar("VST_VLD")
        full = S.pset(32)
        one_lane = S.pset(32, "PAT_VL1")
        nz_stride = T.int32(nz_rows << 16)
        for row_u16 in T.Serial(T.uint16(0), T.uint16(row_count)):
            row_index = T.Cast("int32", row_u16)
            physical_row = row_offset + row_index
            old_square = S.vld(square_ub[physical_row], dist="BRC_B32")
            weight = S.vld(previous_ub[physical_row, stream], dist="BRC_B32")
            square_lanes = S.alloc_var(T.float32)
            for chunk in T.unroll(chunks, explicit=True):
                column = column_offset + chunk * 64
                nz_column = chunk * 64
                converted = S.vcvt(S.vld(x_ub[row_index, chunk * 64], dist="UNPK_B16"), T.float32, part=0)
                nz_pointer = S.make_ubuf_ptr(T.access_ptr(x_nz_ub[physical_row, nz_column], "w", 1, 64), "float32")
                S.vsstb(converted, nz_pointer, nz_stride, full)
                squared_chunk = S.vmul(converted, converted, full)
                if chunk == 0:
                    square_lanes = squared_chunk
                else:
                    square_lanes = S.vadd(square_lanes, squared_chunk, full)
                product = S.vmul(converted, weight, full)
                if first_stream:
                    S.vsts(y_acc_ub[physical_row, column], product, full)
                else:
                    combined = S.vadd(S.vld(y_acc_ub[physical_row, column]), product, full)
                    if write_output:
                        S.vsts(output_ub[row_index, chunk * 64], S.vcvt(combined, T.bfloat16), full, "PK_B32")
                    else:
                        S.vsts(y_acc_ub[physical_row, column], combined, full)
            total_square = S.vadd(old_square, S.vcadd(square_lanes, full), one_lane)
            S.vsts(square_ub[physical_row], total_square, one_lane, "ONEPT_B32")


def _sum4_pairwise(a, b, c, d, mask):
    return S.vadd(S.vadd(a, b, mask), S.vadd(c, d, mask), mask)


def _sum4_serial(a, b, c, d, mask):
    return S.vadd(S.vadd(S.vadd(a, b, mask), c, mask), d, mask)


@T.macro
def _pre_or_post(raw_ub, sq_ub, scale_ub, base_ub, output_ub, rows: int, valid_rows, hidden: int, is_post: int):
    flat_scale = dense_ub_view(scale_ub, (24,))
    for batch in T.unroll((rows + 63) // 64, explicit=True):
        with T.SimdVF():
            full = S.pset(32)
            active = T.min(T.min(64, rows - batch * 64), T.max(valid_rows - batch * 64, 0))
            mask = S.update_mask(active, 32)
            lane = S.vci(T.int32(0), T.int32)
            input_index = T.reinterpret(S.vshls(lane, T.int32(5), full), "uint32x64")
            output_index = T.reinterpret(S.vshls(lane, T.int32(3), full), "uint32x64")
            one = S.vdup(T.float32(1), T.float32, full)
            zero = S.vdup(T.float32(0), T.float32, full)
            rms_inv = S.vdiv(
                one,
                S.vsqrt(S.vadds(S.vmuls(S.vld(sq_ub[batch * 64]), T.float32(1.0 / (4 * hidden)), mask), T.float32(1e-20), mask), mask),
                mask,
                precision="ftz_true",
            )
            scale = S.vld(flat_scale[is_post * 1], dist="BRC_B32")
            value = S.alloc_var(T.float32)
            for channel in T.unroll(4, explicit=True):
                value = S.vgather2(
                    T.access_ptr(raw_ub[batch * 64, channel + is_post * 4], "r", extent=(rows - batch * 64) * 32 - channel - is_post * 4),
                    input_index,
                    mask,
                )
                value = S.vadd(
                    S.vmul(S.vmul(value, rms_inv, mask), scale, mask), S.vld(base_ub[channel + is_post * 4], dist="BRC_B32"), mask
                )
                value = S.vdiv(one, S.vadds(S.vexpdif(zero, value, mask), T.float32(1), mask), mask, precision="ftz_true")
                if is_post:
                    value = S.vmuls(value, T.float32(2), mask)
                else:
                    value = S.vadds(value, T.float32(1e-06), mask)
                S.vscatter(
                    value, T.access_ptr(output_ub[batch * 64, channel], "w", extent=(rows - batch * 64) * 8 - channel), output_index, mask
                )


@T.macro
def coefficients(raw_ub, sq_ub, scale_ub, base_ub, pre_ub, post_ub, comb_ub, rows: int, valid_rows, hidden: int):
    """Compute pre/post and twenty-round Sinkhorn from FP32 raw dot/sumsq.

    Storage: raw[rows,32], sq[ceil(rows/64)*64], alpha[3,8], bias[64],
    pre/post[rows,8], comb[rows,16]. Gather/scatter masks are clamped to both
    valid_rows and each physical chunk, including zero-row AIV tails.
    The three scale values occupy contiguous offsets 0, 1 and 2 after DMA.
    """
    flat_scale = dense_ub_view(scale_ub, (24,))
    _pre_or_post(raw_ub, sq_ub, scale_ub, base_ub, pre_ub, rows, valid_rows, hidden, 0)
    _pre_or_post(raw_ub, sq_ub, scale_ub, base_ub, post_ub, rows, valid_rows, hidden, 1)
    for batch in T.unroll((rows + 63) // 64, explicit=True):
        with T.SimdVF():
            full = S.pset(32)
            active = T.min(T.min(64, rows - batch * 64), T.max(valid_rows - batch * 64, 0))
            mask = S.update_mask(active, 32)
            lane = S.vci(T.int32(0), T.int32)
            input_index = T.reinterpret(S.vshls(lane, T.int32(5), full), "uint32x64")
            output_index = T.reinterpret(S.vshls(lane, T.int32(4), full), "uint32x64")
            one = S.vdup(T.float32(1), T.float32, full)
            rms_inv = S.vdiv(
                one,
                S.vsqrt(S.vadds(S.vmuls(S.vld(sq_ub[batch * 64]), T.float32(1.0 / (4 * hidden)), mask), T.float32(1e-20), mask), mask),
                mask,
                precision="ftz_true",
            )
            scale = S.vld(flat_scale[2], dist="BRC_B32")
            m00 = S.alloc_var(T.float32)
            m00 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 8], "r", extent=(rows - batch * 64) * 32 - 8), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[8], dist="BRC_B32"),
                mask,
            )
            m01 = S.alloc_var(T.float32)
            m01 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 9], "r", extent=(rows - batch * 64) * 32 - 9), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[9], dist="BRC_B32"),
                mask,
            )
            m02 = S.alloc_var(T.float32)
            m02 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 10], "r", extent=(rows - batch * 64) * 32 - 10), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[10], dist="BRC_B32"),
                mask,
            )
            m03 = S.alloc_var(T.float32)
            m03 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 11], "r", extent=(rows - batch * 64) * 32 - 11), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[11], dist="BRC_B32"),
                mask,
            )
            m10 = S.alloc_var(T.float32)
            m10 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 12], "r", extent=(rows - batch * 64) * 32 - 12), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[12], dist="BRC_B32"),
                mask,
            )
            m11 = S.alloc_var(T.float32)
            m11 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 13], "r", extent=(rows - batch * 64) * 32 - 13), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[13], dist="BRC_B32"),
                mask,
            )
            m12 = S.alloc_var(T.float32)
            m12 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 14], "r", extent=(rows - batch * 64) * 32 - 14), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[14], dist="BRC_B32"),
                mask,
            )
            m13 = S.alloc_var(T.float32)
            m13 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 15], "r", extent=(rows - batch * 64) * 32 - 15), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[15], dist="BRC_B32"),
                mask,
            )
            m20 = S.alloc_var(T.float32)
            m20 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 16], "r", extent=(rows - batch * 64) * 32 - 16), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[16], dist="BRC_B32"),
                mask,
            )
            m21 = S.alloc_var(T.float32)
            m21 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 17], "r", extent=(rows - batch * 64) * 32 - 17), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[17], dist="BRC_B32"),
                mask,
            )
            m22 = S.alloc_var(T.float32)
            m22 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 18], "r", extent=(rows - batch * 64) * 32 - 18), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[18], dist="BRC_B32"),
                mask,
            )
            m23 = S.alloc_var(T.float32)
            m23 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 19], "r", extent=(rows - batch * 64) * 32 - 19), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[19], dist="BRC_B32"),
                mask,
            )
            m30 = S.alloc_var(T.float32)
            m30 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 20], "r", extent=(rows - batch * 64) * 32 - 20), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[20], dist="BRC_B32"),
                mask,
            )
            m31 = S.alloc_var(T.float32)
            m31 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 21], "r", extent=(rows - batch * 64) * 32 - 21), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[21], dist="BRC_B32"),
                mask,
            )
            m32 = S.alloc_var(T.float32)
            m32 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 22], "r", extent=(rows - batch * 64) * 32 - 22), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[22], dist="BRC_B32"),
                mask,
            )
            m33 = S.alloc_var(T.float32)
            m33 = S.vadd(
                S.vmul(
                    S.vmul(
                        S.vgather2(T.access_ptr(raw_ub[batch * 64, 23], "r", extent=(rows - batch * 64) * 32 - 23), input_index, mask),
                        rms_inv,
                        mask,
                    ),
                    scale,
                    mask,
                ),
                S.vld(base_ub[23], dist="BRC_B32"),
                mask,
            )
            maximum = S.alloc_var(T.float32)
            total = S.alloc_var(T.float32)
            maximum = S.vmax(S.vmax(m00, m01, mask), S.vmax(m02, m03, mask), mask)
            m00 = S.vexpdif(m00, maximum, mask)
            m01 = S.vexpdif(m01, maximum, mask)
            m02 = S.vexpdif(m02, maximum, mask)
            m03 = S.vexpdif(m03, maximum, mask)
            total = _sum4_pairwise(m00, m01, m02, m03, mask)
            m00 = S.vadds(S.vdiv(m00, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m01 = S.vadds(S.vdiv(m01, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m02 = S.vadds(S.vdiv(m02, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m03 = S.vadds(S.vdiv(m03, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            maximum = S.vmax(S.vmax(m10, m11, mask), S.vmax(m12, m13, mask), mask)
            m10 = S.vexpdif(m10, maximum, mask)
            m11 = S.vexpdif(m11, maximum, mask)
            m12 = S.vexpdif(m12, maximum, mask)
            m13 = S.vexpdif(m13, maximum, mask)
            total = _sum4_pairwise(m10, m11, m12, m13, mask)
            m10 = S.vadds(S.vdiv(m10, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m11 = S.vadds(S.vdiv(m11, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m12 = S.vadds(S.vdiv(m12, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m13 = S.vadds(S.vdiv(m13, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            maximum = S.vmax(S.vmax(m20, m21, mask), S.vmax(m22, m23, mask), mask)
            m20 = S.vexpdif(m20, maximum, mask)
            m21 = S.vexpdif(m21, maximum, mask)
            m22 = S.vexpdif(m22, maximum, mask)
            m23 = S.vexpdif(m23, maximum, mask)
            total = _sum4_pairwise(m20, m21, m22, m23, mask)
            m20 = S.vadds(S.vdiv(m20, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m21 = S.vadds(S.vdiv(m21, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m22 = S.vadds(S.vdiv(m22, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m23 = S.vadds(S.vdiv(m23, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            maximum = S.vmax(S.vmax(m30, m31, mask), S.vmax(m32, m33, mask), mask)
            m30 = S.vexpdif(m30, maximum, mask)
            m31 = S.vexpdif(m31, maximum, mask)
            m32 = S.vexpdif(m32, maximum, mask)
            m33 = S.vexpdif(m33, maximum, mask)
            total = _sum4_pairwise(m30, m31, m32, m33, mask)
            m30 = S.vadds(S.vdiv(m30, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m31 = S.vadds(S.vdiv(m31, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m32 = S.vadds(S.vdiv(m32, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            m33 = S.vadds(S.vdiv(m33, total, mask, precision="ftz_true"), T.float32(1e-06), mask)
            total = S.vadds(_sum4_serial(m00, m10, m20, m30, mask), T.float32(1e-06), mask)
            m00 = S.vdiv(m00, total, mask, precision="ftz_true")
            m10 = S.vdiv(m10, total, mask, precision="ftz_true")
            m20 = S.vdiv(m20, total, mask, precision="ftz_true")
            m30 = S.vdiv(m30, total, mask, precision="ftz_true")
            total = S.vadds(_sum4_serial(m01, m11, m21, m31, mask), T.float32(1e-06), mask)
            m01 = S.vdiv(m01, total, mask, precision="ftz_true")
            m11 = S.vdiv(m11, total, mask, precision="ftz_true")
            m21 = S.vdiv(m21, total, mask, precision="ftz_true")
            m31 = S.vdiv(m31, total, mask, precision="ftz_true")
            total = S.vadds(_sum4_serial(m02, m12, m22, m32, mask), T.float32(1e-06), mask)
            m02 = S.vdiv(m02, total, mask, precision="ftz_true")
            m12 = S.vdiv(m12, total, mask, precision="ftz_true")
            m22 = S.vdiv(m22, total, mask, precision="ftz_true")
            m32 = S.vdiv(m32, total, mask, precision="ftz_true")
            total = S.vadds(_sum4_serial(m03, m13, m23, m33, mask), T.float32(1e-06), mask)
            m03 = S.vdiv(m03, total, mask, precision="ftz_true")
            m13 = S.vdiv(m13, total, mask, precision="ftz_true")
            m23 = S.vdiv(m23, total, mask, precision="ftz_true")
            m33 = S.vdiv(m33, total, mask, precision="ftz_true")
            for _iteration in T.serial(19):
                total = S.vadds(_sum4_pairwise(m00, m01, m02, m03, mask), T.float32(1e-06), mask)
                m00 = S.vdiv(m00, total, mask, precision="ftz_true")
                m01 = S.vdiv(m01, total, mask, precision="ftz_true")
                m02 = S.vdiv(m02, total, mask, precision="ftz_true")
                m03 = S.vdiv(m03, total, mask, precision="ftz_true")
                total = S.vadds(_sum4_pairwise(m10, m11, m12, m13, mask), T.float32(1e-06), mask)
                m10 = S.vdiv(m10, total, mask, precision="ftz_true")
                m11 = S.vdiv(m11, total, mask, precision="ftz_true")
                m12 = S.vdiv(m12, total, mask, precision="ftz_true")
                m13 = S.vdiv(m13, total, mask, precision="ftz_true")
                total = S.vadds(_sum4_pairwise(m20, m21, m22, m23, mask), T.float32(1e-06), mask)
                m20 = S.vdiv(m20, total, mask, precision="ftz_true")
                m21 = S.vdiv(m21, total, mask, precision="ftz_true")
                m22 = S.vdiv(m22, total, mask, precision="ftz_true")
                m23 = S.vdiv(m23, total, mask, precision="ftz_true")
                total = S.vadds(_sum4_pairwise(m30, m31, m32, m33, mask), T.float32(1e-06), mask)
                m30 = S.vdiv(m30, total, mask, precision="ftz_true")
                m31 = S.vdiv(m31, total, mask, precision="ftz_true")
                m32 = S.vdiv(m32, total, mask, precision="ftz_true")
                m33 = S.vdiv(m33, total, mask, precision="ftz_true")
                total = S.vadds(_sum4_serial(m00, m10, m20, m30, mask), T.float32(1e-06), mask)
                m00 = S.vdiv(m00, total, mask, precision="ftz_true")
                m10 = S.vdiv(m10, total, mask, precision="ftz_true")
                m20 = S.vdiv(m20, total, mask, precision="ftz_true")
                m30 = S.vdiv(m30, total, mask, precision="ftz_true")
                total = S.vadds(_sum4_serial(m01, m11, m21, m31, mask), T.float32(1e-06), mask)
                m01 = S.vdiv(m01, total, mask, precision="ftz_true")
                m11 = S.vdiv(m11, total, mask, precision="ftz_true")
                m21 = S.vdiv(m21, total, mask, precision="ftz_true")
                m31 = S.vdiv(m31, total, mask, precision="ftz_true")
                total = S.vadds(_sum4_serial(m02, m12, m22, m32, mask), T.float32(1e-06), mask)
                m02 = S.vdiv(m02, total, mask, precision="ftz_true")
                m12 = S.vdiv(m12, total, mask, precision="ftz_true")
                m22 = S.vdiv(m22, total, mask, precision="ftz_true")
                m32 = S.vdiv(m32, total, mask, precision="ftz_true")
                total = S.vadds(_sum4_serial(m03, m13, m23, m33, mask), T.float32(1e-06), mask)
                m03 = S.vdiv(m03, total, mask, precision="ftz_true")
                m13 = S.vdiv(m13, total, mask, precision="ftz_true")
                m23 = S.vdiv(m23, total, mask, precision="ftz_true")
                m33 = S.vdiv(m33, total, mask, precision="ftz_true")
            S.vscatter(m00, T.access_ptr(comb_ub[batch * 64, 0], "w", extent=(rows - batch * 64) * 16 - 0), output_index, mask)
            S.vscatter(m01, T.access_ptr(comb_ub[batch * 64, 1], "w", extent=(rows - batch * 64) * 16 - 1), output_index, mask)
            S.vscatter(m02, T.access_ptr(comb_ub[batch * 64, 2], "w", extent=(rows - batch * 64) * 16 - 2), output_index, mask)
            S.vscatter(m03, T.access_ptr(comb_ub[batch * 64, 3], "w", extent=(rows - batch * 64) * 16 - 3), output_index, mask)
            S.vscatter(m10, T.access_ptr(comb_ub[batch * 64, 4], "w", extent=(rows - batch * 64) * 16 - 4), output_index, mask)
            S.vscatter(m11, T.access_ptr(comb_ub[batch * 64, 5], "w", extent=(rows - batch * 64) * 16 - 5), output_index, mask)
            S.vscatter(m12, T.access_ptr(comb_ub[batch * 64, 6], "w", extent=(rows - batch * 64) * 16 - 6), output_index, mask)
            S.vscatter(m13, T.access_ptr(comb_ub[batch * 64, 7], "w", extent=(rows - batch * 64) * 16 - 7), output_index, mask)
            S.vscatter(m20, T.access_ptr(comb_ub[batch * 64, 8], "w", extent=(rows - batch * 64) * 16 - 8), output_index, mask)
            S.vscatter(m21, T.access_ptr(comb_ub[batch * 64, 9], "w", extent=(rows - batch * 64) * 16 - 9), output_index, mask)
            S.vscatter(m22, T.access_ptr(comb_ub[batch * 64, 10], "w", extent=(rows - batch * 64) * 16 - 10), output_index, mask)
            S.vscatter(m23, T.access_ptr(comb_ub[batch * 64, 11], "w", extent=(rows - batch * 64) * 16 - 11), output_index, mask)
            S.vscatter(m30, T.access_ptr(comb_ub[batch * 64, 12], "w", extent=(rows - batch * 64) * 16 - 12), output_index, mask)
            S.vscatter(m31, T.access_ptr(comb_ub[batch * 64, 13], "w", extent=(rows - batch * 64) * 16 - 13), output_index, mask)
            S.vscatter(m32, T.access_ptr(comb_ub[batch * 64, 14], "w", extent=(rows - batch * 64) * 16 - 14), output_index, mask)
            S.vscatter(m33, T.access_ptr(comb_ub[batch * 64, 15], "w", extent=(rows - batch * 64) * 16 - 15), output_index, mask)


@T.macro
def _zero_fp32(buffer, elements: int):
    """Clear aligned UB storage with SIMD stores rather than scalar fill loops."""
    assert elements % 64 == 0
    flat = dense_ub_view(buffer, (elements,))
    with T.SimdVF():
        full = S.pset(32)
        zero = S.vdup(T.float32(0), T.float32, full)
        for chunk in T.Serial(elements // 64):
            S.vsts(flat[chunk * 64], zero, full)


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def mhc_pre_mk(
    x,
    prev,
    phi,
    alpha,
    bias,
    y,
    pre,
    post,
    comb,
    raw_partial,
    sq_partial,
    raw_read,
    sq_read,
    *,
    hidden: int,
    block_m: int,
    block_k: int = 192,
    num_aic_cores: int = 28,
    m_groups: int,
    k_groups: int,
    split_d: int,
    coeff_rows: int = 64,
    ub_k: int = 64,
    mad_k: int = 32,
    full_m_tiles: bool = False,
):
    """Prepare L1 groups from UB tiles, then reduce within one MIX.

    Projection partials are partitioned as
    ``raw_partial[k_groups,2,ceil(N/block_m),block_m/2,32]``; only their first
    24 columns are meaningful. Square sums are partitioned as
    ``sq_partial[k_groups,2,ceil(N/block_m),block_m/2]`` so the reduction
    preserves each physical AIV's row ownership and needs no AIV0/AIV1 relay.
    Both workspaces are fresh per invocation and overwritten before reading.

    ``full_m_tiles=True`` requires runtime N divisible by ``block_m``. Every
    active job then has complete Vector rows, so input staging can omit both
    padding and its conditional write dependency. The entry point derives this
    specialization from the actual runtime N.
    """
    assert hidden in (4096, 5120, 7168)
    assert block_m % 16 == 0 and 16 <= block_m <= 256
    assert block_k > 0 and block_k % 64 == 0
    assert split_d > 0 and split_d % 64 == 0
    assert ub_k > 0 and ub_k % 64 == 0 and (ub_k <= block_k)
    assert mad_k > 0 and mad_k % 32 == 0 and (mad_k <= block_k)
    assert block_m * mad_k * 4 * 2 <= 65536
    assert 32 * mad_k * 4 * 2 <= 65536
    assert hidden % 64 == 0
    assert k_groups == (hidden + split_d - 1) // split_d
    assert 0 < m_groups * k_groups <= num_aic_cores
    assert 8 <= coeff_rows <= 64 and coeff_rows % 8 == 0
    T.func_attr({"tir.noalias": False})
    n = T.dynamic("num_tokens")
    x: T.Tensor[(n, 4, hidden), T.bfloat16]
    prev: T.Tensor[(n, 4), T.float32]
    phi: T.Tensor[(24, 4 * hidden), T.float32]
    alpha: T.Tensor[(3,), T.float32]
    bias: T.Tensor[(24,), T.float32]
    y: T.Tensor[(n, hidden), T.bfloat16]
    pre: T.Tensor[(n, 4, 1), T.float32]
    post: T.Tensor[(n, 4, 1), T.float32]
    comb: T.Tensor[(n, 4, 4), T.float32]
    raw_partial: T.Tensor[(k_groups, 2, T.ceildiv(n, block_m), block_m // 2, 32), T.float32]
    sq_partial: T.Tensor[(k_groups, 2, T.ceildiv(n, block_m), block_m // 2), T.float32]
    # Read formals deliberately alias the two workspace output tensors.
    raw_read: T.Tensor[(k_groups, 2, T.ceildiv(n, block_m), block_m // 2, 32), T.float32]
    sq_read: T.Tensor[(k_groups, 2, T.ceildiv(n, block_m), block_m // 2), T.float32]
    rows = block_m // 2
    sq_rows = (rows + 63) // 64 * 64
    nz_rows = rows + 1
    coefficient_rows = coeff_rows
    coefficient_square_rows = (coefficient_rows + 63) // 64 * 64
    partial_batches = k_groups
    nz_columns = ub_k
    raw_elements = coefficient_rows * 32
    raw_temp_elements = partial_batches * raw_elements
    sq_elements = coefficient_square_rows
    sq_temp_elements = partial_batches * sq_elements
    scale_offset = raw_elements + raw_temp_elements + sq_elements + sq_temp_elements
    base_offset = scale_offset + 24
    pre_offset = base_offset + 64
    post_offset = pre_offset + coefficient_rows * 8
    comb_offset = post_offset + coefficient_rows * 8
    phase2_elements = comb_offset + coefficient_rows * 16
    with T.MixedKernel(num_aic_cores) as (core_id, aiv_id):
        T.import_source(PUBLICATION_BIND_SOURCE + BATCHED_DMA_SOURCE)
        flat_x = T.reshape(x, (n, 4 * hidden))
        flat_pre = T.reshape(pre, (n, 4))
        flat_post = T.reshape(post, (n, 4))
        flat_comb = T.reshape(comb, (n, 16))
        m_group = core_id // k_groups
        k_group = core_id % k_groups
        x_ub = T.alloc_shared((rows, ub_k), T.bfloat16)
        x_nz_ub = T.alloc_shared((nz_rows, nz_columns), T.float32)
        T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})
        previous_ub = T.alloc_shared((rows, 8), T.float32)
        square_ub = T.alloc_shared((sq_rows,), T.float32)
        # Stage two reuses yAcc storage after the publication handoff.
        phase_arena_ub = T.alloc_shared((max(rows * block_k, phase2_elements),), T.float32)
        y_acc_ub = dense_ub_view(phase_arena_ub, (rows, block_k))
        T.annotate_buffer_versions({phase_arena_ub: 1})
        y_ub = T.alloc_shared((rows, ub_k), T.bfloat16)
        projection_ub = T.alloc_shared((rows, 32), T.float32)
        x_l1 = T.alloc_l1((block_m, block_k), T.float32)
        phi_l1 = T.alloc_l1((32, block_k), T.float32)
        x_l0 = T.alloc_l0a((block_m, mad_k), T.float32)
        phi_l0 = T.alloc_l0b((32, mad_k), T.float32)
        projection_l0 = T.alloc_l0c((block_m, 32), T.float32)
        T.annotate_buffer_versions({x_ub: 2, x_nz_ub: 2, x_l1: 2, phi_l1: 2, x_l0: 2, phi_l0: 2, y_ub: 2})
        raw_ub = dense_ub_view(phase_arena_ub, (coefficient_rows, 32))
        raw_temp_ub = dense_ub_view(phase_arena_ub, (partial_batches, coefficient_rows, 32), raw_elements)
        sq_ub = dense_ub_view(phase_arena_ub, (coefficient_square_rows,), raw_elements + raw_temp_elements)
        sq_temp_ub = dense_ub_view(
            phase_arena_ub, (partial_batches, coefficient_square_rows), raw_elements + raw_temp_elements + sq_elements
        )
        scale_ub = dense_ub_view(phase_arena_ub, (3, 8), scale_offset)
        base_ub = dense_ub_view(phase_arena_ub, (64,), base_offset)
        pre_ub = dense_ub_view(phase_arena_ub, (coefficient_rows, 8), pre_offset)
        post_ub = dense_ub_view(phase_arena_ub, (coefficient_rows, 8), post_offset)
        comb_ub = dense_ub_view(phase_arena_ub, (coefficient_rows, 16), comb_offset)
        T.set_hf32_mode("nearest_zero")
        jobs = T.ceildiv(T.max(T.ceildiv(n, block_m) - m_group, 0), m_groups)
        last_token_base = (m_group + (jobs - 1) * m_groups) * block_m
        if core_id < m_groups * k_groups:
            windows = T.ceildiv(T.min(split_d, hidden - k_group * split_d), block_k)
            for job in T.Serial(jobs):
                token_base = (m_group + job * m_groups) * block_m
                vector_base = token_base + aiv_id * rows
                valid_rows = T.max(T.min(n - vector_base, rows), 0)
                _zero_fp32(square_ub, sq_rows)
                _zero_fp32(previous_ub, rows * 8)
                if valid_rows > 0:
                    T.copy(prev[vector_base : vector_base + valid_rows, :], previous_ub[0:valid_rows, 0:4])
                for window in T.Serial(windows):
                    d_base = k_group * split_d + window * block_k
                    current_k = T.min(block_k, hidden - d_base)
                    current_k = T.min(current_k, split_d - window * block_k)
                    for stream in T.Serial(4):
                        k_base = stream * hidden + d_base
                        T.copy(phi[0:24, k_base : k_base + current_k], phi_l1[0:24, 0:current_k])
                        for row_tile in T.Serial(T.ceildiv(rows, rows)):
                            row_offset = row_tile * rows
                            tile_rows = T.min(rows, rows - row_offset)
                            copy_rows = T.max(T.min(valid_rows - row_offset, tile_rows), 0)
                            for column_tile in T.Serial(T.ceildiv(current_k, ub_k)):
                                column_offset = column_tile * ub_k
                                copy_k = T.min(ub_k, current_k - column_offset)
                                if full_m_tiles:
                                    T.copy(
                                        flat_x[
                                            vector_base + row_offset : vector_base + row_offset + tile_rows,
                                            k_base + column_offset : k_base + column_offset + copy_k,
                                        ],
                                        x_ub[0:tile_rows, 0:copy_k],
                                    )
                                else:
                                    if copy_rows < tile_rows:
                                        with T.SimdVF():
                                            mask = S.pset(16, "PAT_VL64")
                                            zero = S.vdup(T.bfloat16(0), T.bfloat16, mask)
                                            for row in T.Serial(tile_rows):
                                                for chunk in T.Serial(ub_k // 64):
                                                    S.vsts(x_ub[row, chunk * 64], zero, mask, "NORM_B16", extent=64)
                                    if copy_rows > 0:
                                        T.copy(
                                            flat_x[
                                                vector_base + row_offset : vector_base + row_offset + copy_rows,
                                                k_base + column_offset : k_base + column_offset + copy_k,
                                            ],
                                            x_ub[0:copy_rows, 0:copy_k],
                                        )
                                for prepared_prefix in T.unroll(64, ub_k + 1, 64, explicit=True):
                                    if copy_k == prepared_prefix:
                                        if stream == 0:
                                            prepare_prefix(
                                                x_ub,
                                                x_nz_ub,
                                                previous_ub,
                                                square_ub,
                                                y_acc_ub,
                                                row_offset,
                                                column_offset,
                                                stream,
                                                tile_rows,
                                                first_stream=True,
                                                chunks=prepared_prefix // 64,
                                                nz_rows=nz_rows,
                                            )
                                        elif stream == 3:
                                            prepare_prefix(
                                                x_ub,
                                                x_nz_ub,
                                                previous_ub,
                                                square_ub,
                                                y_acc_ub,
                                                row_offset,
                                                column_offset,
                                                stream,
                                                tile_rows,
                                                y_ub,
                                                first_stream=False,
                                                chunks=prepared_prefix // 64,
                                                nz_rows=nz_rows,
                                                write_output=True,
                                            )
                                        else:
                                            prepare_prefix(
                                                x_ub,
                                                x_nz_ub,
                                                previous_ub,
                                                square_ub,
                                                y_acc_ub,
                                                row_offset,
                                                column_offset,
                                                stream,
                                                tile_rows,
                                                first_stream=False,
                                                chunks=prepared_prefix // 64,
                                                nz_rows=nz_rows,
                                            )
                                for prefix in T.unroll(64, ub_k + 1, 64, explicit=True):
                                    if copy_k == prefix:
                                        T.dual_copy(x_nz_ub[0:rows, 0:prefix], x_l1[0:block_m, column_offset : column_offset + prefix])
                                # Keep stream and row guards as separate scheduling scopes.
                                if stream == 3:  # noqa: SIM102
                                    if copy_rows > 0:
                                        T.copy(
                                            y_ub[0:copy_rows, 0:copy_k],
                                            y[
                                                vector_base + row_offset : vector_base + row_offset + copy_rows,
                                                d_base + column_offset : d_base + column_offset + copy_k,
                                            ],
                                        )
                        mads = T.ceildiv(current_k, mad_k)
                        for mad in T.Serial(mads):
                            mad_offset = mad * mad_k
                            effective_k = T.min(mad_k, current_k - mad_offset)
                            T.copy(x_l1[:, mad_offset : mad_offset + effective_k], x_l0[:, 0:effective_k])
                            T.copy(phi_l1[0:24, mad_offset : mad_offset + effective_k], phi_l0[0:24, 0:effective_k])
                            T.gemm(
                                x_l0[:, 0:effective_k],
                                phi_l0[0:24, 0:effective_k],
                                projection_l0[0:block_m, 0:24],
                                transpose_B=True,
                                clear_accum=window == 0 and stream == 0 and (mad == 0),
                                unit_flag_ctrl=T.Select(window == windows - 1 and stream == 3 and (mad == mads - 1), 3, 2),
                            )
                T.dual_copy(projection_l0, projection_ub, unit_flag_ctrl=3)
                if job < jobs - 1:
                    T.copy(projection_ub[0:rows, 0:24], raw_partial[k_group, aiv_id, token_base // block_m, 0:rows, 0:24])
                    T.copy(square_ub[0:rows], sq_partial[k_group, aiv_id, token_base // block_m, 0:rows])
        # Every physical pair joins publication, including idle MK pairs.
        # The global handoff closes both workspaces and the Apply arena.
        with T.PerCoreTask():
            if core_id < m_groups * k_groups and jobs > 0:
                with T.Task():
                    T.copy(projection_ub[0:rows, 0:24], raw_partial[k_group, aiv_id, last_token_base // block_m, 0:rows, 0:24])
                    T.copy(square_ub[0:rows], sq_partial[k_group, aiv_id, last_token_base // block_m, 0:rows])
            T.ascend_sync_inter_arrive("PIPE_MTE3", 1)
            T.ascend_sync_inter_wait("PIPE_MTE3", 1)
            _publication_drain = T.call_extern("int32", "MHC_AS_BIND", T.ascend_pipe_barrier("PIPE_MTE3"))
            _publication_mte2 = T.call_extern("int32", "MHC_AS_BIND", T.ascend_set_flag("MTE3_MTE2", 7))
            _publication_scalar_ready = T.call_extern("int32", "MHC_AS_BIND", T.ascend_set_flag("MTE3_S", 7))
            _publication_scalar_wait = T.call_extern("int32", "MHC_AS_BIND", T.ascend_wait_flag("MTE3_S", 7))
            _publication_vector = T.call_extern("int32", "MHC_AS_BIND", T.ascend_set_flag("S_V", 7))
        # Reuse the Apply arena only after DMA and Vector observe publication.
        with T.Stage(0):
            with T.Task():
                T.ascend_wait_flag("MTE3_MTE2", 7)
                T.copy(bias[0:24], base_ub[0:24])
                T.copy(alpha[0:3], scale_ub[0, 0:3])
            with T.Task():
                T.ascend_wait_flag("S_V", 7)
                _zero_fp32(raw_ub, coefficient_rows * 32)
            packs_per_half = (rows + coefficient_rows - 1) // coefficient_rows
            coefficient_tasks = T.ceildiv(n, block_m) * packs_per_half
            coefficient_jobs = T.ceildiv(T.max(coefficient_tasks - core_id, 0), num_aic_cores)
            for job in T.Serial(coefficient_jobs):
                task = core_id + job * num_aic_cores
                tile = task // packs_per_half
                coefficient_row_offset = task % packs_per_half * coefficient_rows
                token_base = tile * block_m + aiv_id * rows + coefficient_row_offset
                physical_rows = T.min(coefficient_rows, rows - coefficient_row_offset)
                valid_rows = T.max(T.min(physical_rows, n - token_base), 0)
                _zero_fp32(raw_ub, coefficient_rows * 32)
                _zero_fp32(sq_ub, coefficient_square_rows)
                copy_raw_k_partials(
                    raw_read,
                    raw_temp_ub,
                    n,
                    aiv_id,
                    tile,
                    coefficient_row_offset,
                    physical_rows,
                    block_m=block_m,
                    k_groups=k_groups,
                    coefficient_rows=coefficient_rows,
                )
                copy_square_k_partials(
                    sq_read, sq_temp_ub, n, aiv_id, tile, coefficient_row_offset, physical_rows, block_m=block_m, k_groups=k_groups
                )
                # Only the 24 projection channels are meaningful in each row.
                with T.SimdVF():
                    full = S.pset(32)
                    lane = S.vci(T.int32(0), T.int32)
                    column = S.vand(lane, S.vdup(T.int32(31), T.int32, full), full)
                    raw_mask = S.vcmps(column, T.int32(24), full, "lt")
                    square_mask = S.update_mask(physical_rows, 32)
                    destination = dense_ub_view(raw_ub, (coefficient_rows * 32,))
                    source = dense_ub_view(raw_temp_ub, (partial_batches, coefficient_rows * 32))
                    for chunk in T.Serial(coefficient_rows * 32 // 64):
                        accumulated = S.alloc_var(T.float32)
                        accumulated = S.vdup(T.float32(0), T.float32, full)
                        for split in T.Serial(k_groups):
                            accumulated = S.vadd(accumulated, S.vld(source[split, chunk * 64]), raw_mask)
                        S.vsts(destination[chunk * 64], accumulated, full)
                    squared = S.alloc_var(T.float32)
                    squared = S.vdup(T.float32(0), T.float32, full)
                    for split in T.Serial(k_groups):
                        squared = S.vadd(squared, S.vld(sq_temp_ub[split, 0]), square_mask)
                    S.vsts(sq_ub[0], squared, full)
                coefficients(raw_ub, sq_ub, scale_ub, base_ub, pre_ub, post_ub, comb_ub, coefficient_rows, valid_rows, hidden)
                if valid_rows > 0:
                    T.copy(pre_ub[0:valid_rows, 0:4], flat_pre[token_base : token_base + valid_rows, :])
                    T.copy(post_ub[0:valid_rows, 0:4], flat_post[token_base : token_base + valid_rows, :])
                    T.copy(comb_ub[0:valid_rows, :], flat_comb[token_base : token_base + valid_rows, :])
