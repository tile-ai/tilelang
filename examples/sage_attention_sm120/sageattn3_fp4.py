"""SageAttention3 FP4 raw-core TileLang kernel for NVIDIA SM120."""

from __future__ import annotations

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver

from examples.sage_attention_sm120.tir_helpers import (
    mma_m16n32k64_blockscale_f32,
    pack_cast2_u32,
)


_LOG2_E = 1.4426950408889634
_F32_NEG_INF = -3.4028234663852886e38
_SAGE3_PV_INV_SCALE_LOG2 = -11.392317422778762  # log2(1 / (448 * 6))
_SAGE3_FP4_INV_SCALE_LOG2 = -2.584962500721156  # log2(1 / 6)


@T.macro
def _sage3_load_q_scale_word_shared(SFQ_words, row, ko):
    row_local = row % 64
    return T.lds32(SFQ_words[ko, row // 64, row_local % 16, row_local // 16])


@T.macro
def _sage3_load_k_scale_word_shared_stage(SFK_words, stage, physical_col, ko):
    physical_local = physical_col % 64
    return T.lds32(SFK_words[stage, ko, physical_col // 64, physical_local % 16, physical_local // 16])


@T.macro
def _sage3_pack_fp4_pair_byte(v0, v1):
    return pack_cast2_u32(T.float4_e2m1fn, T.uint8, v0, v1) & T.Cast("uint32", 0xFF)


@T.macro
def _sage3_load_v_scale_word_stage(SFV_words, stage, dim, ko):
    dim_local = dim % 64
    return T.lds32(SFV_words[stage, ko, dim // 64, dim_local % 16, dim_local // 16])


@T.macro
def _sage3_ldmatrix_x4_u8_q_rowmajor_qregs(tile, row_tile, ko, q_regs, q_regs_ko, q_regs_mi, lane):
    matrix_group = lane >> 3
    row_in_matrix = lane & 0x7
    half = matrix_group >> 1
    row_pair = matrix_group & 0x1
    row = row_tile * 16 + row_pair * 8 + row_in_matrix
    col = ko * 64 + half * 32
    T.ptx_ldmatrix(
        T.bool(False),
        4,
        T.access_ptr(tile[row, col], "r", extent=32),
        T.access_ptr(q_regs[q_regs_ko, q_regs_mi, 0], "w", extent=4),
    )


@T.macro
def _sage3_ldmatrix_x4_u8_b_rowmajor_stage_into(tile, stage, n32, ko, tile_half, regs, reg_offset, lane):
    matrix_group = lane >> 3
    row_in_matrix = lane & 0x7
    reg_word = tile_half * 4 + matrix_group
    e = reg_word >> 1
    half = reg_word & 0x1
    row = n32 * 32 + e * 8 + row_in_matrix
    col = ko * 64 + half * 32
    T.ptx_ldmatrix(
        T.bool(False),
        4,
        T.access_ptr(tile[stage, row, col], "r", extent=32),
        T.access_ptr(regs[reg_offset], "w", extent=4),
    )


@tilelang.jit
def sage3_packed_fp4_attention_raw_kernel(
    query_tokens: int,
    kv_tokens: int,
    valid_k_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    dtype: str = "bfloat16",
    block_n: int = 128,
):
    """SageAttention3 raw-core ABI implemented in pure TileLang.

    External inputs intentionally mirror ``fp4attn_cuda.fwd``:

    - ``Q``/``K``: uint8 packed FP4, shape ``[1,H,N,D/2]``.
    - ``Vt``: uint8 packed transposed FP4, shape ``[1,H,D,N/2]``.
    - ``SFQ``/``SFK``/``SFVt``: e4m3 scale tensors with Sage3's physical
      swizzled layout but the same public 4D shapes.
    - ``DeltaS``: float32 block-mean correction, shape
      ``[1,H,ceil(query_tokens/128),kv_tokens]``.
    - ``Out``/``LSE``: the raw-core padded output tensors returned by
      ``fp4attn_cuda.fwd``.

    This version keeps the raw input/output contract and Sage3 scale/K layouts,
    uses SM120 register FP4 blockscale MMA for both QK and PV, and has one
    load path: packed FP4 global views are TMA-loaded into CUTLASS-compatible
    packed FP4 shared memory and consumed with LDSM.

    Like the official kernel, the QK accumulator stays in Sage3's permuted
    ("physical") K-token order end to end: K rows are LDSM-loaded row-major
    (conflict-free), DeltaS seeds the accumulator using the physical->logical
    column mapping, and each thread then owns 8 logically-consecutive P
    columns so FP4 quantization of P is thread-local with no cross-lane
    shuffles.
    """
    if query_tokens <= 0 or kv_tokens <= 0 or valid_k_tokens <= 0:
        raise ValueError(f"query_tokens, kv_tokens, and valid_k_tokens must be positive; got {query_tokens}, {kv_tokens}, {valid_k_tokens}")
    if valid_k_tokens > kv_tokens:
        raise ValueError(f"valid_k_tokens must be <= kv_tokens, got {valid_k_tokens} > {kv_tokens}")
    if q_heads <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError(f"q_heads, kv_heads, and head_dim must be positive; got {q_heads}, {kv_heads}, {head_dim}")
    if q_heads != kv_heads:
        raise ValueError(f"SageAttn3 raw core currently requires q_heads == kv_heads, got {q_heads}, {kv_heads}")
    if head_dim != 128:
        raise ValueError(f"sage3_packed_fp4_attention_raw fixes head_dim=128, got {head_dim}")
    if dtype != "bfloat16":
        raise ValueError(f"sage3_packed_fp4_attention_raw currently supports bfloat16 output, got {dtype}")
    if block_n != 128:
        raise ValueError(f"sage3_packed_fp4_attention_raw fixes block_n=128, got {block_n}")
    if query_tokens % 128 != 0 or kv_tokens % 128 != 0:
        raise ValueError(
            f"SageAttn3 raw inputs must be preprocessed/padded to 128-token blocks; got query_tokens={query_tokens}, kv_tokens={kv_tokens}"
        )
    if kv_tokens % 16 != 0:
        raise ValueError(f"kv_tokens must be a multiple of 16 for V scale groups, got {kv_tokens}")

    num_qh = q_heads
    num_kvh = kv_heads
    q_tokens = query_tokens
    k_tokens = kv_tokens
    k_valid = valid_k_tokens
    block_M = 128
    block_N = block_n
    q_blocks = T.ceildiv(q_tokens, block_M)
    kv_blocks = T.ceildiv(k_valid, block_N)
    kv_stage0_blocks = (kv_blocks + 1) // 2
    kv_stage1_blocks = kv_blocks // 2
    qk_scale_cols = head_dim // 16
    v_scale_cols = k_tokens // 16
    scale = (1.0 / head_dim) ** 0.5 * _LOG2_E
    accum_dtype = "float32"
    has_k_tail = (k_valid % block_N) != 0
    full_kv_blocks = k_valid // block_N
    compute_threads = 256
    warp_count = compute_threads // 32
    warp_M_tiles = 8 // warp_count
    warp_M = warp_M_tiles * 16
    warp_N16_tiles = 8
    warp_N32_tiles = 4
    qk_acc_count = warp_M_tiles * warp_N32_tiles * 16
    launch_sms = driver.get_num_sms()
    if launch_sms <= 0:
        raise ValueError(f"driver.get_num_sms() must be positive, got {launch_sms}")

    @T.prim_func
    def main(
        Q: T.Tensor((1, num_qh, q_tokens, 64), "uint8"),
        K: T.Tensor((1, num_kvh, k_tokens, 64), "uint8"),
        Vt: T.Tensor((1, num_kvh, 128, k_tokens // 2), "uint8"),
        SFQ: T.Tensor((1, num_qh, q_tokens, qk_scale_cols), "float8_e4m3fn"),
        SFK: T.Tensor((1, num_kvh, k_tokens, qk_scale_cols), "float8_e4m3fn"),
        SFVt: T.Tensor((1, num_kvh, 128, v_scale_cols), "float8_e4m3fn"),
        DeltaS: T.Tensor((1, num_qh, q_blocks, k_tokens), "float32"),
        Out: T.Tensor((1, num_qh, q_tokens, 128), "bfloat16"),
        LSE: T.Tensor((1, num_qh, q_tokens), "float32"),
    ):
        T.annotate_compile_flags(["--use_fast_math"])
        Q_fp4 = T.view(Q, (1, num_qh, q_tokens, 128), dtype=T.float4_e2m1fn)
        K_fp4 = T.view(K, (1, num_kvh, k_tokens, 128), dtype=T.float4_e2m1fn)
        Vt_fp4 = T.view(Vt, (1, num_kvh, 128, k_tokens), dtype=T.float4_e2m1fn)
        SFQ_tma = T.view(
            SFQ,
            (num_qh, qk_scale_cols // 4, q_tokens // 64, 16, 8),
            dtype=T.uint16,
            strides=(
                (q_tokens // 64) * ((qk_scale_cols // 4) * 16 * 8),
                16 * 8,
                (qk_scale_cols // 4) * 16 * 8,
                8,
                1,
            ),
        )
        SFK_tma = T.view(
            SFK,
            (num_kvh, qk_scale_cols // 4, k_tokens // 64, 16, 8),
            dtype=T.uint16,
            strides=(
                (k_tokens // 64) * ((qk_scale_cols // 4) * 16 * 8),
                16 * 8,
                (qk_scale_cols // 4) * 16 * 8,
                8,
                1,
            ),
        )
        SFV_tma = T.view(
            SFVt,
            (num_kvh, v_scale_cols // 4, 128 // 64, 16, 8),
            dtype=T.uint16,
            strides=(
                (128 // 64) * ((v_scale_cols // 4) * 16 * 8),
                16 * 8,
                (v_scale_cols // 4) * 16 * 8,
                8,
                1,
            ),
        )

        with T.Kernel(launch_sms, threads=384) as block_id:
            Q_shared = T.alloc_shared((block_M, 128), T.float4_e2m1fn)
            T.annotate_layout({Q_shared: tilelang.layout.make_sm120_fp4_smem_layout(Q_shared)})
            SFQ_shared_storage = T.alloc_shared((qk_scale_cols // 4, block_M // 64, 16, 4, 4), "uint8")
            SFQ_shared_tma = T.view(SFQ_shared_storage, (qk_scale_cols // 4, block_M // 64, 16, 8), dtype=T.uint16)
            SFQ_shared_words = T.view(SFQ_shared_storage, (qk_scale_cols // 4, block_M // 64, 16, 4), dtype=T.uint32)
            K_shared = T.alloc_shared((2, block_N, 128), T.float4_e2m1fn)
            T.annotate_layout({K_shared: tilelang.layout.make_sm120_fp4_smem_layout(K_shared)})
            SFK_shared_storage = T.alloc_shared((2, qk_scale_cols // 4, block_N // 64, 16, 4, 4), "uint8")
            SFK_shared_tma = T.view(
                SFK_shared_storage,
                (2, qk_scale_cols // 4, block_N // 64, 16, 8),
                dtype=T.uint16,
            )
            SFK_shared_words = T.view(
                SFK_shared_storage,
                (2, qk_scale_cols // 4, block_N // 64, 16, 4),
                dtype=T.uint32,
            )
            DS_shared = T.alloc_shared((2, block_N), "float32")
            Vt_shared = T.alloc_shared((2, 128, block_N), T.float4_e2m1fn)
            T.annotate_layout({Vt_shared: tilelang.layout.make_sm120_fp4_smem_layout(Vt_shared)})
            SFV_shared_storage = T.alloc_shared((2, (block_N // 16) // 4, 128 // 64, 16, 4, 4), "uint8")
            SFV_shared_tma = T.view(
                SFV_shared_storage,
                (2, (block_N // 16) // 4, 128 // 64, 16, 8),
                dtype=T.uint16,
            )
            SFV_shared_words = T.view(
                SFV_shared_storage,
                (2, (block_N // 16) // 4, 128 // 64, 16, 4),
                dtype=T.uint32,
            )
            O_shared = T.alloc_shared((block_M, 128), "bfloat16")
            q_ready = T.alloc_barrier(32)
            q_empty = T.alloc_barrier(256)
            kv_full = T.alloc_barrier([32, 32])
            kv_empty = T.alloc_barrier([256, 256])
            o_ready = T.alloc_barrier(256)
            o_empty = T.alloc_barrier(32)
            tx = T.get_thread_binding()
            producer_tx = tx - 256
            producer_warp_role = producer_tx >> 5
            consumer_tx = tx
            lane = consumer_tx & 31
            warp = consumer_tx >> 5
            g = lane >> 2
            sublane = lane & 0x3
            scale_lane = consumer_tx & 31
            scale_warp = consumer_tx >> 5
            scale_g = scale_lane >> 2
            scale_sublane = scale_lane & 0x3
            with T.ws(2):
                T.annotate_producer_reg_dealloc(24)
                if producer_warp_role == 0:
                    for qb, qh in T.Persistent(
                        [q_blocks, num_qh],
                        launch_sms,
                        block_id,
                        group_size=1,
                    ):
                        tile_linear = qh * q_blocks + qb
                        tile_iter = (tile_linear - block_id) // launch_sms
                        tile_phase = tile_iter & 1

                        T.mbarrier_wait_parity(q_empty, tile_phase ^ 1)
                        T.tma_copy(
                            Q_fp4[0, qh, qb * block_M : (qb + 1) * block_M, 0:128],
                            Q_shared,
                            barrier=q_ready,
                            leader_scope_threads=32,
                        )
                        T.tma_copy(
                            SFQ_tma[
                                qh,
                                0 : qk_scale_cols // 4,
                                qb * (block_M // 64) : (qb + 1) * (block_M // 64),
                                0:16,
                                0:8,
                            ],
                            SFQ_shared_tma,
                            barrier=q_ready,
                            leader_scope_threads=32,
                        )
                        T.mbarrier_arrive(q_ready)

                        for kb in T.serial(kv_blocks, unroll_factor=1):
                            stage = kb & 1
                            stage_phase0 = (tile_iter * kv_stage0_blocks + (kb >> 1)) & 1
                            stage_phase1 = (tile_iter * kv_stage1_blocks + (kb >> 1)) & 1
                            phase = T.if_then_else(stage == 0, stage_phase0, stage_phase1)
                            T.mbarrier_wait_parity(kv_empty[stage], phase ^ 1)
                            T.tma_copy(
                                DeltaS[0, qh, qb, kb * block_N : (kb + 1) * block_N],
                                DS_shared[stage, 0:block_N],
                                barrier=kv_full[stage],
                                leader_scope_threads=32,
                            )
                            T.tma_copy(
                                SFK_tma[
                                    qh,
                                    0 : qk_scale_cols // 4,
                                    kb * (block_N // 64) : (kb + 1) * (block_N // 64),
                                    0:16,
                                    0:8,
                                ],
                                SFK_shared_tma[stage, 0 : qk_scale_cols // 4, 0 : block_N // 64, 0:16, 0:8],
                                barrier=kv_full[stage],
                                leader_scope_threads=32,
                            )
                            T.tma_copy(
                                K_fp4[0, qh, kb * block_N : (kb + 1) * block_N, 0:128],
                                K_shared[stage, 0:block_N, 0:128],
                                barrier=kv_full[stage],
                                leader_scope_threads=32,
                            )
                            T.tma_copy(
                                SFV_tma[
                                    qh,
                                    kb * ((block_N // 16) // 4) : (kb + 1) * ((block_N // 16) // 4),
                                    0 : 128 // 64,
                                    0:16,
                                    0:8,
                                ],
                                SFV_shared_tma[stage, 0 : (block_N // 16) // 4, 0 : 128 // 64, 0:16, 0:8],
                                barrier=kv_full[stage],
                                leader_scope_threads=32,
                            )
                            T.tma_copy(
                                Vt_fp4[0, qh, 0:128, kb * block_N : (kb + 1) * block_N],
                                Vt_shared[stage, 0:128, 0:block_N],
                                barrier=kv_full[stage],
                                leader_scope_threads=32,
                            )
                            T.mbarrier_arrive(kv_full[stage])
                elif producer_warp_role == 1:
                    for qb, qh in T.Persistent(
                        [q_blocks, num_qh],
                        launch_sms,
                        block_id,
                        group_size=1,
                    ):
                        tile_linear = qh * q_blocks + qb
                        tile_iter = (tile_linear - block_id) // launch_sms
                        tile_phase = tile_iter & 1

                        T.mbarrier_wait_parity(o_ready, tile_phase)
                        T.tma_copy(
                            O_shared,
                            Out[0, qh, qb * block_M : (qb + 1) * block_M, 0:128],
                            leader_scope_threads=32,
                        )
                        T.tma_store_wait()
                        T.mbarrier_arrive(o_empty)

            with T.ws(0, 1):
                T.annotate_consumer_reg_alloc(240)
                q_regs = T.alloc_local((2, warp_M_tiles, 4), "uint32", role_scoped=True)
                a_regs = T.alloc_local((warp_M_tiles, 4), "uint32", role_scoped=True)
                b_regs = T.alloc_local((8,), "uint32", role_scoped=True)
                q_scale_frag = T.alloc_fragment((2 * 4,), T.float8_e4m3fn, role_scoped=True)
                b_scale_frag = T.alloc_fragment((4 * 4,), T.float8_e4m3fn, role_scoped=True)
                q_scale_regs = T.view(q_scale_frag, (2,), dtype=T.uint32)
                b_scale_regs = T.view(b_scale_frag, (4,), dtype=T.uint32)
                qk_acc = T.alloc_local((qk_acc_count,), accum_dtype, role_scoped=True)
                pv_acc = T.alloc_local((qk_acc_count,), accum_dtype, role_scoped=True)
                o_acc = T.alloc_local((qk_acc_count,), accum_dtype, role_scoped=True)
                row_scale_acc = T.alloc_local((warp_M_tiles, 2), accum_dtype, role_scoped=True)
                row_max_acc = T.alloc_local((warp_M_tiles, 2), accum_dtype, role_scoped=True)
                row_sum_acc = T.alloc_local((warp_M_tiles, 2), accum_dtype, role_scoped=True)
                scale_f = T.Cast(accum_dtype, scale)
                for qb, qh in T.Persistent(
                    [q_blocks, num_qh],
                    launch_sms,
                    block_id,
                    group_size=1,
                ):
                    tile_linear = qh * q_blocks + qb
                    tile_iter = (tile_linear - block_id) // launch_sms
                    tile_phase = tile_iter & 1

                    T.mbarrier_wait_parity(q_ready, tile_phase)
                    T.clear(o_acc)
                    for mi in T.unroll(warp_M_tiles):
                        row_max_acc[mi, 0] = T.Cast(accum_dtype, _F32_NEG_INF)
                        row_max_acc[mi, 1] = T.Cast(accum_dtype, _F32_NEG_INF)
                        row_sum_acc[mi, 0] = T.Cast(accum_dtype, 0.0)
                        row_sum_acc[mi, 1] = T.Cast(accum_dtype, 0.0)

                    for ko in T.unroll(2):
                        scale_slot = scale_lane & 0x3
                        scale_row_pair = scale_slot & 0x1
                        scale_row = scale_warp * warp_M + scale_g + scale_row_pair * 8
                        q_scale_regs[ko] = _sage3_load_q_scale_word_shared(SFQ_shared_words, scale_row, ko)

                        for mi in T.unroll(warp_M_tiles):
                            _sage3_ldmatrix_x4_u8_q_rowmajor_qregs(
                                Q_shared,
                                warp * warp_M_tiles + mi,
                                ko,
                                q_regs,
                                ko,
                                mi,
                                lane,
                            )

                    T.mbarrier_arrive(q_empty)

                    for kb in T.serial(kv_blocks):
                        stage = kb & 1
                        stage_phase0 = (tile_iter * kv_stage0_blocks + (kb >> 1)) & 1
                        stage_phase1 = (tile_iter * kv_stage1_blocks + (kb >> 1)) & 1
                        phase = T.if_then_else(stage == 0, stage_phase0, stage_phase1)
                        T.mbarrier_wait_parity(kv_full[stage], phase)

                        # The QK accumulator stays in Sage3's physical (permuted)
                        # column order: thread `sublane` holds logical tokens
                        # nj*32 + sublane*8 .. +7 contiguously.  DeltaS seeds the
                        # accumulator before the MMA, exactly like the official
                        # mainloop.
                        ds_vals = T.alloc_local((8,), accum_dtype, role_scoped=True)
                        for nj in T.unroll(warp_N32_tiles):
                            if has_k_tail:
                                if kb < full_kv_blocks:
                                    for jj in T.unroll(8):
                                        ds_vals[jj] = DS_shared[stage, nj * 32 + sublane * 8 + jj]
                                else:
                                    for jj in T.unroll(8):
                                        ds_vals[jj] = T.if_then_else(
                                            kb * block_N + nj * 32 + sublane * 8 + jj < k_valid,
                                            DS_shared[stage, nj * 32 + sublane * 8 + jj],
                                            T.Cast(accum_dtype, _F32_NEG_INF),
                                        )
                            else:
                                for jj in T.unroll(8):
                                    ds_vals[jj] = DS_shared[stage, nj * 32 + sublane * 8 + jj]
                            for mi in T.unroll(warp_M_tiles):
                                off = (mi * warp_N32_tiles + nj) * 16
                                for jj in T.unroll(8):
                                    qk_acc[off + (jj // 2) * 4 + (jj % 2)] = ds_vals[jj]
                                    qk_acc[off + (jj // 2) * 4 + (jj % 2) + 2] = ds_vals[jj]

                        for ko in T.unroll(2):
                            for p in T.unroll(4):
                                physical_col = p * 32 + scale_sublane * 8 + scale_g
                                b_scale_regs[p] = _sage3_load_k_scale_word_shared_stage(SFK_shared_words, stage, physical_col, ko)

                            for nj in T.unroll(warp_N32_tiles):
                                _sage3_ldmatrix_x4_u8_b_rowmajor_stage_into(K_shared, stage, nj, ko, 0, b_regs, 0, lane)
                                _sage3_ldmatrix_x4_u8_b_rowmajor_stage_into(K_shared, stage, nj, ko, 1, b_regs, 4, lane)
                                for mi in T.unroll(warp_M_tiles):
                                    mma_m16n32k64_blockscale_f32(
                                        q_regs.data,
                                        (ko * warp_M_tiles + mi) * 8,
                                        b_regs.data,
                                        0,
                                        qk_acc,
                                        (mi * warp_N32_tiles + nj) * 16,
                                        q_scale_regs[ko],
                                        b_scale_regs[nj],
                                        mi,
                                        0,
                                    )

                        # Per-16-token scale-group maxes are taken on the raw
                        # accumulator (thread's 8 values plus the sublane^1
                        # partner) and transformed analytically after exp2, so
                        # softmax needs a single pass like the official kernel.
                        absmax0 = T.alloc_local((warp_M_tiles, warp_N32_tiles), accum_dtype, role_scoped=True)
                        absmax1 = T.alloc_local((warp_M_tiles, warp_N32_tiles), accum_dtype, role_scoped=True)
                        for mi in T.unroll(warp_M_tiles):
                            for nj in T.unroll(warp_N32_tiles):
                                off = (mi * warp_N32_tiles + nj) * 16
                                m0 = T.max(
                                    T.max(T.max(qk_acc[off + 0], qk_acc[off + 1]), T.max(qk_acc[off + 4], qk_acc[off + 5])),
                                    T.max(T.max(qk_acc[off + 8], qk_acc[off + 9]), T.max(qk_acc[off + 12], qk_acc[off + 13])),
                                )
                                m1 = T.max(
                                    T.max(T.max(qk_acc[off + 2], qk_acc[off + 3]), T.max(qk_acc[off + 6], qk_acc[off + 7])),
                                    T.max(T.max(qk_acc[off + 10], qk_acc[off + 11]), T.max(qk_acc[off + 14], qk_acc[off + 15])),
                                )
                                absmax0[mi, nj] = T.max(m0, T.shfl_xor(m0, 1, width=4))
                                absmax1[mi, nj] = T.max(m1, T.shfl_xor(m1, 1, width=4))
                            max0_local = T.max(T.max(absmax0[mi, 0], absmax0[mi, 1]), T.max(absmax0[mi, 2], absmax0[mi, 3]))
                            max1_local = T.max(T.max(absmax1[mi, 0], absmax1[mi, 1]), T.max(absmax1[mi, 2], absmax1[mi, 3]))
                            max0_group = T.max(max0_local, T.shfl_xor(max0_local, 2, width=4))
                            max1_group = T.max(max1_local, T.shfl_xor(max1_local, 2, width=4))
                            prev0 = row_max_acc[mi, 0]
                            prev1 = row_max_acc[mi, 1]
                            new0 = T.max(prev0, max0_group)
                            new1 = T.max(prev1, max1_group)
                            row_scale_acc[mi, 0] = T.exp2((prev0 - new0) * scale_f)
                            row_scale_acc[mi, 1] = T.exp2((prev1 - new1) * scale_f)
                            row_max_acc[mi, 0] = new0
                            row_max_acc[mi, 1] = new1

                        for mi in T.unroll(warp_M_tiles):
                            max_scaled0 = T.alloc_var(
                                accum_dtype,
                                init=row_max_acc[mi, 0] * scale_f + T.Cast(accum_dtype, _SAGE3_PV_INV_SCALE_LOG2),
                                role_scoped=True,
                            )
                            max_scaled1 = T.alloc_var(
                                accum_dtype,
                                init=row_max_acc[mi, 1] * scale_f + T.Cast(accum_dtype, _SAGE3_PV_INV_SCALE_LOG2),
                                role_scoped=True,
                            )
                            sum0_local = T.alloc_var(accum_dtype, init=T.Cast(accum_dtype, 0.0), role_scoped=True)
                            sum1_local = T.alloc_var(accum_dtype, init=T.Cast(accum_dtype, 0.0), role_scoped=True)
                            for nj in T.unroll(warp_N32_tiles):
                                off = (mi * warp_N32_tiles + nj) * 16
                                for aa in T.unroll(4):
                                    atom = off + aa * 4
                                    p00 = T.exp2(qk_acc[atom + 0] * scale_f - max_scaled0)
                                    p01 = T.exp2(qk_acc[atom + 1] * scale_f - max_scaled0)
                                    p10 = T.exp2(qk_acc[atom + 2] * scale_f - max_scaled1)
                                    p11 = T.exp2(qk_acc[atom + 3] * scale_f - max_scaled1)
                                    qk_acc[atom + 0] = p00
                                    qk_acc[atom + 1] = p01
                                    qk_acc[atom + 2] = p10
                                    qk_acc[atom + 3] = p11
                                    sum0_local += p00 + p01
                                    sum1_local += p10 + p11
                                absmax0[mi, nj] = T.exp2(
                                    absmax0[mi, nj] * scale_f - max_scaled0 + T.Cast(accum_dtype, _SAGE3_FP4_INV_SCALE_LOG2)
                                )
                                absmax1[mi, nj] = T.exp2(
                                    absmax1[mi, nj] * scale_f - max_scaled1 + T.Cast(accum_dtype, _SAGE3_FP4_INV_SCALE_LOG2)
                                )
                            row_sum_acc[mi, 0] = row_sum_acc[mi, 0] * row_scale_acc[mi, 0] + sum0_local
                            row_sum_acc[mi, 1] = row_sum_acc[mi, 1] * row_scale_acc[mi, 1] + sum1_local

                        T.clear(pv_acc)

                        for ko in T.unroll(2):
                            # SFP word assembly: each thread owns the e4m3 scales
                            # for k-groups ko*4 + (sublane>>1) and +2; the missing
                            # byte lanes come from the sublane^2 partner.
                            sel_lo = T.Select((sublane & 1) != 0, absmax1[0, ko * 2], absmax0[0, ko * 2])
                            sel_hi = T.Select((sublane & 1) != 0, absmax1[0, ko * 2 + 1], absmax0[0, ko * 2 + 1])
                            sfp_pair = pack_cast2_u32(T.float8_e4m3fn, T.uint16, sel_lo, sel_hi)
                            sfp_own = T.alloc_var(
                                "uint32",
                                init=((sfp_pair & T.Cast("uint32", 0xFF)) | ((sfp_pair & T.Cast("uint32", 0xFF00)) << T.Cast("uint32", 8)))
                                << T.Cast("uint32", (sublane >> 1) * 8),
                                role_scoped=True,
                            )
                            scale_a = T.alloc_var(
                                "uint32",
                                init=sfp_own | T.shfl_xor(sfp_own, 2, width=4),
                                role_scoped=True,
                            )

                            for mi in T.unroll(warp_M_tiles):
                                for half in T.unroll(2):
                                    nj_pack = ko * 2 + half
                                    off = (mi * warp_N32_tiles + nj_pack) * 16
                                    inv_pmax0 = T.alloc_var(
                                        "float32",
                                        init=T.Cast("float32", 1.0) / absmax0[mi, nj_pack],
                                        role_scoped=True,
                                    )
                                    inv_pmax1 = T.alloc_var(
                                        "float32",
                                        init=T.Cast("float32", 1.0) / absmax1[mi, nj_pack],
                                        role_scoped=True,
                                    )
                                    a_regs[mi, half * 2 + 0] = (
                                        _sage3_pack_fp4_pair_byte(qk_acc[off + 0] * inv_pmax0, qk_acc[off + 1] * inv_pmax0)
                                        | (
                                            _sage3_pack_fp4_pair_byte(qk_acc[off + 4] * inv_pmax0, qk_acc[off + 5] * inv_pmax0)
                                            << T.Cast("uint32", 8)
                                        )
                                        | (
                                            _sage3_pack_fp4_pair_byte(qk_acc[off + 8] * inv_pmax0, qk_acc[off + 9] * inv_pmax0)
                                            << T.Cast("uint32", 16)
                                        )
                                        | (
                                            _sage3_pack_fp4_pair_byte(qk_acc[off + 12] * inv_pmax0, qk_acc[off + 13] * inv_pmax0)
                                            << T.Cast("uint32", 24)
                                        )
                                    )
                                    a_regs[mi, half * 2 + 1] = (
                                        _sage3_pack_fp4_pair_byte(qk_acc[off + 2] * inv_pmax1, qk_acc[off + 3] * inv_pmax1)
                                        | (
                                            _sage3_pack_fp4_pair_byte(qk_acc[off + 6] * inv_pmax1, qk_acc[off + 7] * inv_pmax1)
                                            << T.Cast("uint32", 8)
                                        )
                                        | (
                                            _sage3_pack_fp4_pair_byte(qk_acc[off + 10] * inv_pmax1, qk_acc[off + 11] * inv_pmax1)
                                            << T.Cast("uint32", 16)
                                        )
                                        | (
                                            _sage3_pack_fp4_pair_byte(qk_acc[off + 14] * inv_pmax1, qk_acc[off + 15] * inv_pmax1)
                                            << T.Cast("uint32", 24)
                                        )
                                    )

                            for p in T.unroll(4):
                                logical_dim = p * 32 + scale_sublane * 8 + scale_g
                                b_scale_regs[p] = _sage3_load_v_scale_word_stage(SFV_shared_words, stage, logical_dim, ko)

                            for nj in T.unroll(warp_N32_tiles):
                                _sage3_ldmatrix_x4_u8_b_rowmajor_stage_into(Vt_shared, stage, nj, ko, 0, b_regs, 0, lane)
                                _sage3_ldmatrix_x4_u8_b_rowmajor_stage_into(Vt_shared, stage, nj, ko, 1, b_regs, 4, lane)
                                for mi in T.unroll(warp_M_tiles):
                                    mma_m16n32k64_blockscale_f32(
                                        a_regs.data,
                                        mi * 8,
                                        b_regs.data,
                                        0,
                                        pv_acc,
                                        (mi * warp_N32_tiles + nj) * 16,
                                        scale_a,
                                        b_scale_regs[nj],
                                        mi,
                                        0,
                                    )

                        for mi in T.unroll(warp_M_tiles):
                            scale0 = row_scale_acc[mi, 0]
                            scale1 = row_scale_acc[mi, 1]
                            for nj in T.unroll(warp_N32_tiles):
                                off = (mi * warp_N32_tiles + nj) * 16
                                for p in T.unroll(4):
                                    atom = off + p * 4
                                    o_acc[atom + 0] = o_acc[atom + 0] * scale0 + pv_acc[atom + 0]
                                    o_acc[atom + 1] = o_acc[atom + 1] * scale0 + pv_acc[atom + 1]
                                    o_acc[atom + 2] = o_acc[atom + 2] * scale1 + pv_acc[atom + 2]
                                    o_acc[atom + 3] = o_acc[atom + 3] * scale1 + pv_acc[atom + 3]

                        T.mbarrier_arrive(kv_empty[stage])
                    T.mbarrier_wait_parity(o_empty, tile_phase ^ 1)
                    for mi in T.unroll(warp_M_tiles):
                        row0 = warp * warp_M + mi * 16 + g
                        row1 = row0 + 8
                        sum0_pair = T.alloc_var(
                            accum_dtype,
                            init=row_sum_acc[mi, 0] + T.shfl_xor(row_sum_acc[mi, 0], 1, width=4),
                            role_scoped=True,
                        )
                        sum1_pair = T.alloc_var(
                            accum_dtype,
                            init=row_sum_acc[mi, 1] + T.shfl_xor(row_sum_acc[mi, 1], 1, width=4),
                            role_scoped=True,
                        )
                        denom0 = T.alloc_var(
                            accum_dtype,
                            init=sum0_pair + T.shfl_xor(sum0_pair, 2, width=4),
                            role_scoped=True,
                        )
                        denom1 = T.alloc_var(
                            accum_dtype,
                            init=sum1_pair + T.shfl_xor(sum1_pair, 2, width=4),
                            role_scoped=True,
                        )
                        inv_sum0 = T.Select(
                            (denom0 != T.Cast(accum_dtype, 0.0)) and (denom0 == denom0),
                            T.Cast(accum_dtype, 1.0) / denom0,
                            T.Cast(accum_dtype, 0.0),
                        )
                        inv_sum1 = T.Select(
                            (denom1 != T.Cast(accum_dtype, 0.0)) and (denom1 == denom1),
                            T.Cast(accum_dtype, 1.0) / denom1,
                            T.Cast(accum_dtype, 0.0),
                        )
                        for nj in T.unroll(warp_N16_tiles):
                            dim0 = nj * 16 + sublane * 2
                            dim1 = dim0 + 1
                            dim2 = dim0 + 8
                            dim3 = dim2 + 1
                            off = (mi * warp_N32_tiles + (nj >> 1)) * 16
                            atom0 = off + (nj & 1) * 8
                            atom1 = atom0 + 4
                            O_shared[row0, dim0] = T.Cast("bfloat16", o_acc[atom0 + 0] * inv_sum0)
                            O_shared[row0, dim1] = T.Cast("bfloat16", o_acc[atom0 + 1] * inv_sum0)
                            O_shared[row1, dim0] = T.Cast("bfloat16", o_acc[atom0 + 2] * inv_sum1)
                            O_shared[row1, dim1] = T.Cast("bfloat16", o_acc[atom0 + 3] * inv_sum1)
                            O_shared[row0, dim2] = T.Cast("bfloat16", o_acc[atom1 + 0] * inv_sum0)
                            O_shared[row0, dim3] = T.Cast("bfloat16", o_acc[atom1 + 1] * inv_sum0)
                            O_shared[row1, dim2] = T.Cast("bfloat16", o_acc[atom1 + 2] * inv_sum1)
                            O_shared[row1, dim3] = T.Cast("bfloat16", o_acc[atom1 + 3] * inv_sum1)
                    T.mbarrier_arrive(o_ready)

    return main
