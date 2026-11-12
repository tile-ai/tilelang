# SPDX-License-Identifier: MIT
# Copyright (c) 2026 DeepSeek. Portions adapted for shifted mHC.
# Copyright (c) 2026 Q4Kernels contributors.
"""M-only mHC Pre: one MIX launch, one residual pass, no GM partials.

Each core pair owns complete token tiles and the full projection K dimension.
Residual conversion, RMS statistics and weighted layer input share the same
load. Projection reaches UB through FIXP before register-resident Sinkhorn.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

from .mhc_pre_mk import (
    PUBLICATION_BIND_SOURCE,
    coefficients,
    dense_ub_view,
    prepare_prefix,
)


@T.macro
def prepare_tail_prefix(
    x_ub,
    x_nz_ub,
    previous_ub,
    square_ub,
    y_acc_ub,
    row_offset,
    column_offset,
    stream,
    row_count,
    valid_rows,
    output_ub=None,
    *,
    first_stream: bool,
    chunks: int,
    nz_rows: int,
    write_output: bool = False,
):
    """Prepare valid rows and zero only the NZ rows consumed as padding.

    Invalid X/previous/yAcc rows are never loaded. The separate uint16 loops
    keep both regions vectorizable without a branch inside the valid-row body.
    Every physical row of the NZ prefix is defined before its L1 handoff.
    """
    with T.SimdVF():
        S.mem_bar("VST_VLD")
        full = S.pset(32)
        one_lane = S.pset(32, "PAT_VL1")
        nz_stride = T.int32(nz_rows << 16)
        for row_u16 in T.Serial(T.uint16(0), T.Cast("uint16", valid_rows)):
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
            # Reduce once per UB prefix, keeping the scalar result in lane 0.
            # Other lanes are never consumed, so no broadcast is necessary.
            total_square = S.vadd(old_square, S.vcadd(square_lanes, full), one_lane)
            S.vsts(square_ub[physical_row], total_square, one_lane, "ONEPT_B32")
        padding_zero = S.vdup(T.float32(0), T.float32, full)
        for padding_row_u16 in T.Serial(T.Cast("uint16", valid_rows), T.uint16(row_count)):
            padding_row = row_offset + T.Cast("int32", padding_row_u16)
            for padding_chunk in T.unroll(chunks, explicit=True):
                padding_column = padding_chunk * 64
                padding_nz_pointer = S.make_ubuf_ptr(T.access_ptr(x_nz_ub[padding_row, padding_column], "w", 1, 64), "float32")
                S.vsstb(padding_zero, padding_nz_pointer, nz_stride, full)


@T.macro
def _zero(buffer, elements: int):
    flat = dense_ub_view(buffer, (elements,))
    with T.SimdVF():
        mask = S.pset(32)
        zero = S.vdup(T.float32(0), T.float32, mask)
        for index in T.Serial(elements // 64):
            S.vsts(flat[index * 64], zero, mask)


@T.macro
def _tile(
    flat_x,
    prev,
    phi,
    alpha,
    bias,
    y,
    flat_pre,
    flat_post,
    flat_comb,
    x_ub,
    x_nz_ub,
    previous_ub,
    square_ub,
    y_acc_ub,
    y_ub,
    raw_ub,
    scale_ub,
    base_ub,
    pre_ub,
    post_ub,
    comb_ub,
    x_l1,
    phi_l1,
    x_l0,
    phi_l0,
    projection_l0,
    token_base,
    aiv_id,
    *,
    hidden: int,
    block_m: int,
    block_k_l1: int,
    ub_k: int,
    mad_k: int,
    full_tile: bool,
):
    rows = block_m // 2
    sq_rows = (rows + 63) // 64 * 64
    vector_base = token_base + aiv_id * rows
    valid_rows = T.max(T.min(flat_x.shape[0] - vector_base, rows), 0)
    _zero(square_ub, sq_rows)
    if full_tile:
        T.copy(prev[vector_base : vector_base + rows, :], previous_ub[:, 0:4])
    else:
        _zero(previous_ub, rows * 8)
        if valid_rows > 0:
            T.copy(prev[vector_base : vector_base + valid_rows, :], previous_ub[0:valid_rows, 0:4])

    for window in T.Serial(T.ceildiv(hidden, block_k_l1)):
        d_base = window * block_k_l1
        current_k = T.min(block_k_l1, hidden - d_base)
        for stream in T.Serial(4):
            k_base = stream * hidden + d_base
            T.copy(phi[0:24, k_base : k_base + current_k], phi_l1[0:24, 0:current_k])
            for column_tile in T.Serial(T.ceildiv(current_k, ub_k)):
                column_offset = column_tile * ub_k
                copy_k = T.min(ub_k, current_k - column_offset)
                if full_tile:
                    # No conditional zero writer in the complete-tile pipeline.
                    T.copy(
                        flat_x[vector_base : vector_base + rows, k_base + column_offset : k_base + column_offset + copy_k],
                        x_ub[:, 0:copy_k],
                    )
                else:
                    # Keep the MTE2 task on an empty AIV half as well. Its
                    # ready/free credits cycle even when no GM rows are read.
                    # NZ padding is produced by the following VF, so X has no
                    # per-prefix zero writer that blocks DMA prefetch.
                    with T.Task():
                        if valid_rows > 0:
                            T.copy(
                                flat_x[vector_base : vector_base + valid_rows, k_base + column_offset : k_base + column_offset + copy_k],
                                x_ub[0:valid_rows, 0:copy_k],
                            )

                for prefix in T.unroll(64, ub_k + 1, 64, explicit=True):
                    if copy_k == prefix:
                        if not full_tile:
                            if stream == 0:
                                prepare_tail_prefix(
                                    x_ub,
                                    x_nz_ub,
                                    previous_ub,
                                    square_ub,
                                    y_acc_ub,
                                    0,
                                    column_offset,
                                    stream,
                                    rows,
                                    valid_rows,
                                    first_stream=True,
                                    chunks=prefix // 64,
                                    nz_rows=rows + 1,
                                )
                            else:
                                if stream == 3:
                                    prepare_tail_prefix(
                                        x_ub,
                                        x_nz_ub,
                                        previous_ub,
                                        square_ub,
                                        y_acc_ub,
                                        0,
                                        column_offset,
                                        stream,
                                        rows,
                                        valid_rows,
                                        y_ub,
                                        first_stream=False,
                                        chunks=prefix // 64,
                                        nz_rows=rows + 1,
                                        write_output=True,
                                    )
                                else:
                                    prepare_tail_prefix(
                                        x_ub,
                                        x_nz_ub,
                                        previous_ub,
                                        square_ub,
                                        y_acc_ub,
                                        0,
                                        column_offset,
                                        stream,
                                        rows,
                                        valid_rows,
                                        first_stream=False,
                                        chunks=prefix // 64,
                                        nz_rows=rows + 1,
                                    )
                        elif stream == 0:
                            prepare_prefix(
                                x_ub,
                                x_nz_ub,
                                previous_ub,
                                square_ub,
                                y_acc_ub,
                                0,
                                column_offset,
                                stream,
                                rows,
                                first_stream=True,
                                chunks=prefix // 64,
                                nz_rows=rows + 1,
                            )
                        else:
                            if stream == 3:
                                prepare_prefix(
                                    x_ub,
                                    x_nz_ub,
                                    previous_ub,
                                    square_ub,
                                    y_acc_ub,
                                    0,
                                    column_offset,
                                    stream,
                                    rows,
                                    y_ub,
                                    first_stream=False,
                                    chunks=prefix // 64,
                                    nz_rows=rows + 1,
                                    write_output=True,
                                )
                            else:
                                prepare_prefix(
                                    x_ub,
                                    x_nz_ub,
                                    previous_ub,
                                    square_ub,
                                    y_acc_ub,
                                    0,
                                    column_offset,
                                    stream,
                                    rows,
                                    first_stream=False,
                                    chunks=prefix // 64,
                                    nz_rows=rows + 1,
                                )
                # Keep the MTE3 handoff ahead of the following output task:
                # READY may release Cube as soon as the last prefix arrives.
                with T.Stage(0):
                    for nz_prefix in T.unroll(64, ub_k + 1, 64, explicit=True):
                        if copy_k == nz_prefix:
                            T.dual_copy(x_nz_ub[0:rows, 0:nz_prefix], x_l1[:, column_offset : column_offset + nz_prefix])

                if stream == 3:
                    # Only the last real output descriptor owns the local
                    # phase transition. Invalid AIV halves still drain/set
                    # their gates, while their GM descriptor is skipped.
                    with T.PerCoreTask():
                        if full_tile:
                            with T.Task():
                                T.copy(
                                    y_ub[:, 0:copy_k],
                                    y[vector_base : vector_base + rows, d_base + column_offset : d_base + column_offset + copy_k],
                                )
                        else:
                            # Both AIVs prepare a padded Y tile. Keep the
                            # consumer task present on an empty half as well:
                            # only its descriptor is conditional, so the
                            # ready/free credits still cycle across windows.
                            with T.Task():
                                if valid_rows > 0:
                                    T.copy(
                                        y_ub[0:valid_rows, 0:copy_k],
                                        y[vector_base : vector_base + valid_rows, d_base + column_offset : d_base + column_offset + copy_k],
                                    )
                        # Preserve the staged task's nested dynamic guards.
                        if window == T.ceildiv(hidden, block_k_l1) - 1:  # noqa: SIM102
                            if column_tile == T.ceildiv(current_k, ub_k) - 1:
                                _drain = T.call_extern("int32", "MHC_AS_BIND", T.ascend_pipe_barrier("PIPE_MTE3"))
                                _dma_release = T.call_extern("int32", "MHC_AS_BIND", T.ascend_set_flag("MTE3_MTE2", 7))
                                _scalar_release = T.call_extern("int32", "MHC_AS_BIND", T.ascend_set_flag("MTE3_S", 7))
                                _scalar_wait = T.call_extern("int32", "MHC_AS_BIND", T.ascend_wait_flag("MTE3_S", 7))
                                _vector_release = T.call_extern("int32", "MHC_AS_BIND", T.ascend_set_flag("S_V", 7))

            for mad in T.Serial(T.ceildiv(current_k, mad_k)):
                mad_offset = mad * mad_k
                effective_k = T.min(mad_k, current_k - mad_offset)
                T.copy(x_l1[:, mad_offset : mad_offset + effective_k], x_l0[:, 0:effective_k])
                T.copy(phi_l1[0:24, mad_offset : mad_offset + effective_k], phi_l0[0:24, 0:effective_k])
                T.gemm(
                    x_l0[:, 0:effective_k],
                    phi_l0[0:24, 0:effective_k],
                    projection_l0[:, 0:24],
                    transpose_B=True,
                    clear_accum=(window == 0 and stream == 0 and mad == 0),
                    unit_flag_ctrl=T.Select(
                        window == T.ceildiv(hidden, block_k_l1) - 1 and stream == 3 and mad == T.ceildiv(current_k, mad_k) - 1, 3, 2
                    ),
                )

    T.dual_copy(projection_l0, raw_ub, unit_flag_ctrl=3)
    # These writes reuse yAcc only after the local final Apply DMA drains.
    # No global grid barrier or projection/RMS GM handoff is involved.
    with T.Task():
        T.ascend_wait_flag("MTE3_MTE2", 7)
        T.copy(bias[0:24], base_ub[0:24])
        T.copy(alpha[0:3], scale_ub[0, 0:3])
    with T.Task():
        T.ascend_wait_flag("S_V", 7)
        coefficients(raw_ub, square_ub, scale_ub, base_ub, pre_ub, post_ub, comb_ub, rows, valid_rows, hidden)
    if valid_rows > 0:
        T.copy(pre_ub[0:valid_rows, 0:4], flat_pre[vector_base : vector_base + valid_rows, :])
        T.copy(post_ub[0:valid_rows, 0:4], flat_post[vector_base : vector_base + valid_rows, :])
        T.copy(comb_ub[0:valid_rows, :], flat_comb[vector_base : vector_base + valid_rows, :])


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def mhc_pre_large(
    x,
    prev,
    phi,
    alpha,
    bias,
    y,
    pre,
    post,
    comb,
    *,
    hidden: int,
    block_m: int,
    block_k_l1: int,
    mad_k: int,
    num_aic_cores: int,
    full_m_tiles: bool,
):
    """Whole-M tile ownership; fixed physical halves also cover arbitrary tails.

    Full tiles use their own padding-free macro expansion. A final partial M
    tile executes a separate expansion with zero padded rows. It is partitioned
    at fixed M/2 rather than ASC's dynamic ceil(actual_M/2); only valid GM rows
    are accessed. FP32 HF32-nearest-zero projection, independent FP32 Mul/Add
    Apply, BF16 RNE and the element-major coefficients are preserved.
    """
    assert hidden in (4096, 5120, 7168)
    ub_k = 64
    assert block_m % 16 == 0 and 96 <= block_m <= 208
    assert block_k_l1 > 0 and block_k_l1 % 128 == 0
    assert mad_k in (32, 64) and mad_k <= ub_k
    assert block_m * mad_k * 8 <= 65536
    assert 32 * mad_k * 8 <= 65536
    assert num_aic_cores > 0
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
    rows = block_m // 2
    sq_rows = (rows + 63) // 64 * 64
    phase2_elements = 88 + rows * 32

    with T.MixedKernel(num_aic_cores) as (core_id, aiv_id):
        T.import_source(PUBLICATION_BIND_SOURCE)
        flat_x = T.reshape(x, (n, 4 * hidden))
        flat_pre = T.reshape(pre, (n, 4))
        flat_post = T.reshape(post, (n, 4))
        flat_comb = T.reshape(comb, (n, 16))
        x_ub = T.alloc_shared((rows, ub_k), T.bfloat16)
        x_nz_ub = T.alloc_shared((rows + 1, ub_k), T.float32)
        T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})
        previous_ub = T.alloc_shared((rows, 8), T.float32)
        square_ub = T.alloc_shared((sq_rows,), T.float32)
        raw_ub = T.alloc_shared((rows, 32), T.float32)
        arena = T.alloc_shared((max(rows * block_k_l1, phase2_elements),), T.float32)
        T.annotate_buffer_versions({arena: 1})
        y_acc_ub = dense_ub_view(arena, (rows, block_k_l1))
        scale_ub = dense_ub_view(arena, (3, 8))
        base_ub = dense_ub_view(arena, (64,), 24)
        pre_ub = dense_ub_view(arena, (rows, 8), 88)
        post_ub = dense_ub_view(arena, (rows, 8), 88 + rows * 8)
        comb_ub = dense_ub_view(arena, (rows, 16), 88 + rows * 16)
        y_ub = T.alloc_shared((rows, ub_k), T.bfloat16)
        x_l1 = T.alloc_l1((block_m, block_k_l1), T.float32)
        phi_l1 = T.alloc_l1((32, block_k_l1), T.float32)
        x_l0 = T.alloc_l0a((block_m, mad_k), T.float32)
        phi_l0 = T.alloc_l0b((32, mad_k), T.float32)
        projection_l0 = T.alloc_l0c((block_m, 32), T.float32)
        # The tail DMA and VF use the same valid-row range. Each generation
        # initializes every X element that is read; an empty half has no X
        # reads. This explicit sparse-write contract keeps both physical slots.
        x_versions = 2 if full_m_tiles else (2, "write_counter")
        T.annotate_buffer_versions({x_ub: x_versions, x_nz_ub: 2, y_ub: 2, x_l1: 2, phi_l1: 2, x_l0: 2, phi_l0: 2})
        T.set_hf32_mode("nearest_zero")
        # Complete tiles never share their loop with a conditional zero writer.
        full_jobs = T.ceildiv(T.max(n // block_m - core_id, 0), num_aic_cores)
        for job in T.Serial(full_jobs):
            token_base = (core_id + job * num_aic_cores) * block_m
            _tile(
                flat_x,
                prev,
                phi,
                alpha,
                bias,
                y,
                flat_pre,
                flat_post,
                flat_comb,
                x_ub,
                x_nz_ub,
                previous_ub,
                square_ub,
                y_acc_ub,
                y_ub,
                raw_ub,
                scale_ub,
                base_ub,
                pre_ub,
                post_ub,
                comb_ub,
                x_l1,
                phi_l1,
                x_l0,
                phi_l0,
                projection_l0,
                token_base,
                aiv_id,
                hidden=hidden,
                block_m=block_m,
                block_k_l1=block_k_l1,
                ub_k=ub_k,
                mad_k=mad_k,
                full_tile=True,
            )
        # Keep compile-time specialization outside the runtime tail guard.
        if not full_m_tiles:  # noqa: SIM102
            if n % block_m != 0 and core_id == (n // block_m) % num_aic_cores:
                _tile(
                    flat_x,
                    prev,
                    phi,
                    alpha,
                    bias,
                    y,
                    flat_pre,
                    flat_post,
                    flat_comb,
                    x_ub,
                    x_nz_ub,
                    previous_ub,
                    square_ub,
                    y_acc_ub,
                    y_ub,
                    raw_ub,
                    scale_ub,
                    base_ub,
                    pre_ub,
                    post_ub,
                    comb_ub,
                    x_l1,
                    phi_l1,
                    x_l0,
                    phi_l0,
                    projection_l0,
                    (n // block_m) * block_m,
                    aiv_id,
                    hidden=hidden,
                    block_m=block_m,
                    block_k_l1=block_k_l1,
                    ub_k=ub_k,
                    mad_k=mad_k,
                    full_tile=False,
                )
