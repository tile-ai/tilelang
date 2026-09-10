"""Shared FlashAttention forward kernel builder for Ascend NPU (Cube + Vector).

Both MHA and GQA reduce to the same per-tile computation::

    Q:   [q_len, D]
    K/V: [kv_len, D]
    O:   [q_len, D]

MHA uses ``q_len == kv_len``; GQA flattens the query groups into
``q_len = S1 * G`` while K/V stay ``[S2, D]``. Group structure only affects how
the caller reshapes Q/O — the row-wise online-softmax attention is identical.

The softmax writes probabilities directly in NZ layout so the second GEMM can
consume them without an intermediate ND-to-NZ conversion. That packing path is
hardwired to ``D == 128`` (128-column ``vsstb`` stride, ``uint16x128`` merge),
so the builder pins ``head_dim`` to 128 until the path is generalized.
"""

import math
from dataclasses import dataclass

import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout
from tilelang.ascend.language import simd as S


@dataclass(frozen=True)
class FwdTiling:
    block_q: int = 128
    block_kv: int = 128
    num_stages: int = 3


def flash_attention_fwd(
    q_len,
    kv_len,
    head_dim=128,
    *,
    out_dtype="bfloat16",
    softmax_scale=None,
    num_blocks=None,
    tiling=None,
    manual_schedule=False,
):
    tiling = tiling or FwdTiling()

    # Expand config into compile-time local scalars so the hygienic T.macro
    # definitions below capture plain values, not dataclass attribute reads.
    BR = tiling.block_q
    BC = tiling.block_kv
    PIPELINE_STAGES = tiling.num_stages
    ROWS = BR // 2
    VL = 64  # Ascend fp32 SIMD lane count
    D = head_dim
    dtype = "bfloat16"
    accum_dtype = "float32"
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    # Implementation constraints of the SIMD softmax-packing path. These are
    # real limits of the current kernel, not tunables — validate explicitly
    # rather than advertise flexibility the code does not have.
    assert head_dim == 128, "The SIMD packing path requires head_dim == 128."
    assert BR % 2 == 0
    assert BC == 2 * VL
    assert q_len % BR == 0
    assert kv_len % BC == 0

    NUM_M_TILES = q_len // BR
    NUM_KV_BLOCKS = kv_len // BC
    NUM_BLOCKS = NUM_M_TILES if num_blocks is None else num_blocks
    assert NUM_BLOCKS > 0
    assert NUM_KV_BLOCKS > 0
    assert NUM_M_TILES % NUM_BLOCKS == 0, "M tiles must divide evenly across cores."
    M_TILES_PER_BLOCK = NUM_M_TILES // NUM_BLOCKS

    is_cast_out = out_dtype != accum_dtype
    assert out_dtype in (accum_dtype, "bfloat16", "float16"), (
        f"out_dtype must be {accum_dtype} (no cast) or a 16-bit type (bfloat16/float16); got {out_dtype!r}."
    )

    # -- shared T.macros (capture the compile-time locals above) -------------

    @T.macro
    def qk(kv, K, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub):
        T.copy(K[kv * BC : (kv + 1) * BC, 0:D], K_shared[0:BC, 0:D])
        T.copy(Q_shared[0:BR, 0:D], qk_a_l0)
        T.copy(K_shared[0:BC, 0:D], qk_b_l0)
        T.gemm(qk_a_l0, qk_b_l0, qk_acc_l0c, transpose_B=True, clear_accum=True)
        T.dual_copy(qk_acc_l0c, S_ub[0:ROWS, 0:BC])

    @T.macro
    def softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        with T.SimdVF(latency=735):
            softmax_full = S.pset(32, "PAT_ALL")
            softmax_one = S.pset(32, "PAT_VL1")
            softmax_b16 = S.pset(16, "PAT_VL128")
            dst_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BC),
                "bfloat16",
            )

            S.vsts(old_m_ub[0], S.vld(m_ub[0], "NORM"), softmax_full, "NORM_B32")

            for r in range(0, ROWS, 2):
                s00 = S.vmuls(S.vld(S_ub[r, 0], "NORM"), T.float32(scale), softmax_full)
                s01 = S.vmuls(S.vld(S_ub[r, VL], "NORM"), T.float32(scale), softmax_full)
                s10 = S.vmuls(S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), softmax_full)
                s11 = S.vmuls(S.vld(S_ub[r + 1, VL], "NORM"), T.float32(scale), softmax_full)
                S.vsts(S_ub[r, 0], s00, softmax_full, "NORM_B32")
                S.vsts(S_ub[r, VL], s01, softmax_full, "NORM_B32")
                S.vsts(S_ub[r + 1, 0], s10, softmax_full, "NORM_B32")
                S.vsts(S_ub[r + 1, VL], s11, softmax_full, "NORM_B32")

                max0 = S.vcmax(S.vmax(s00, s01, softmax_full), softmax_full)
                max1 = S.vcmax(S.vmax(s10, s11, softmax_full), softmax_full)
                S.vsts(m_ub[r], max0, softmax_one, "ONEPT_B32")
                S.vsts(m_ub[r + 1], max1, softmax_one, "ONEPT_B32")

            S.mem_bar("VST_VLD")
            merged_max = S.vmax(S.vld(m_ub[0], "NORM"), S.vld(old_m_ub[0], "NORM"), softmax_full)
            S.vsts(m_ub[0], merged_max, softmax_full, "NORM_B32")
            S.mem_bar("VST_VLD")

            for r in range(0, ROWS, 2):
                max0 = S.vld(m_ub[r], "BRC_B32")
                max1 = S.vld(m_ub[r + 1], "BRC_B32")
                p00, p01 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")
                e00 = S.vexpdif(p00, max0, softmax_full, 0)
                e10 = S.vexpdif(p10, max1, softmax_full, 0)
                e01 = S.vexpdif(p01, max0, softmax_full, 0)
                e11 = S.vexpdif(p11, max1, softmax_full, 0)
                row_sum0 = S.vcadd(S.vadd(e00, e01, softmax_full), softmax_full)
                row_sum1 = S.vcadd(S.vadd(e10, e11, softmax_full), softmax_full)
                S.vsts(row_sum_ub[r], row_sum0, softmax_one, "ONEPT_B32")
                S.vsts(row_sum_ub[r + 1], row_sum1, softmax_one, "ONEPT_B32")

                even_bf16_0 = S.vcvt(e00, "bfloat16", softmax_full, sat=False, part=0)
                even_bf16_1 = S.vcvt(e10, "bfloat16", softmax_full, sat=False, part=0)
                odd_bf16_0 = S.vcvt(e01, "bfloat16", softmax_full, sat=False, part=1)
                odd_bf16_1 = S.vcvt(e11, "bfloat16", softmax_full, sat=False, part=1)
                merged_bits0 = S.vor(
                    T.reinterpret(even_bf16_0, "uint16x128"),
                    T.reinterpret(odd_bf16_0, "uint16x128"),
                    softmax_b16,
                )
                merged_bits1 = S.vor(
                    T.reinterpret(even_bf16_1, "uint16x128"),
                    T.reinterpret(odd_bf16_1, "uint16x128"),
                    softmax_b16,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits0, "bfloat16x128"),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits1, "bfloat16x128"),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )

            S.mem_bar("VST_VLD")
            alpha_reg = S.vexpdif(S.vld(old_m_ub[0], "NORM"), S.vld(m_ub[0], "NORM"), softmax_full, 0)
            S.vsts(alpha_ub[0], alpha_reg, softmax_full, "NORM_B32")
            l_scaled = S.vmul(S.vld(l_ub[0], "NORM"), alpha_reg, softmax_full)
            l_updated = S.vadd(l_scaled, S.vld(row_sum_ub[0], "NORM"), softmax_full)
            S.vsts(l_ub[0], l_updated, softmax_full, "NORM_B32")

    @T.macro
    def pack_p(P_nz_ub, P_shared):
        T.dual_copy(P_nz_ub[0:ROWS, 0:BC], P_shared[0:BR, 0:BC])

    @T.macro
    def pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub):
        T.copy(V[kv * BC : (kv + 1) * BC, 0:D], V_shared[0:D, 0:BC], transpose=True)
        T.copy(P_shared[0:BR, 0:BC], pv_a_l0)
        T.copy(V_shared[0:D, 0:BC], pv_b_l0)
        T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
        T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D])

    @T.macro
    def accumulate_output(O_tmp_ub, O_ub, alpha_ub):
        with T.SimdVF(latency=318):
            rescale_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    prev = S.vld(O_ub[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    updated = S.vadd(S.vmul(alpha, prev, rescale_mask), cur, rescale_mask)
                    S.vsts(O_ub[i, c], updated, rescale_mask, "NORM_B32")

    @T.macro
    def kv_auto(
        kv,
        K,
        V,
        Q_shared,
        K_shared,
        V_shared,
        P_shared,
        qk_a_l0,
        qk_b_l0,
        pv_a_l0,
        pv_b_l0,
        qk_acc_l0c,
        pv_acc_l0c,
        S_ub,
        P_nz_ub,
        O_tmp_ub,
        O_ub,
        m_ub,
        l_ub,
        alpha_ub,
        old_m_ub,
        row_sum_ub,
    ):
        qk(kv, K, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub)
        softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        pack_p(P_nz_ub, P_shared)
        pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)
        accumulate_output(O_tmp_ub, O_ub, alpha_ub)

    @T.macro
    def kv_manual(
        kv,
        K,
        V,
        Q_shared,
        K_shared,
        V_shared,
        P_shared,
        qk_a_l0,
        qk_b_l0,
        pv_a_l0,
        pv_b_l0,
        qk_acc_l0c,
        pv_acc_l0c,
        S_ub,
        P_nz_ub,
        O_tmp_ub,
        O_ub,
        m_ub,
        l_ub,
        alpha_ub,
        old_m_ub,
        row_sum_ub,
    ):
        # Fixed-stage solution under manual per-pipe source ordering. The
        # stage vector in source task order is:
        # [0,1,1,1,3,5,6,3,6,6,6,7,7].
        with T.Stage(0):
            T.copy(K[kv * BC : (kv + 1) * BC, 0:D], K_shared[0:BC, 0:D])
        with T.Stage(1):
            T.copy(Q_shared[0:BR, 0:D], qk_a_l0)
            T.copy(K_shared[0:BC, 0:D], qk_b_l0)
            T.gemm(qk_a_l0, qk_b_l0, qk_acc_l0c, transpose_B=True, clear_accum=True)
        with T.Stage(3):
            T.dual_copy(qk_acc_l0c, S_ub[0:ROWS, 0:BC])
        with T.Stage(5):
            softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        with T.Stage(6):
            pack_p(P_nz_ub, P_shared)
        with T.Stage(3):
            T.copy(V[kv * BC : (kv + 1) * BC, 0:D], V_shared[0:D, 0:BC], transpose=True)
        with T.Stage(6):
            T.copy(P_shared[0:BR, 0:BC], pv_a_l0)
            T.copy(V_shared[0:D, 0:BC], pv_b_l0)
            T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
        with T.Stage(7):
            T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D])
            accumulate_output(O_tmp_ub, O_ub, alpha_ub)

    @T.macro
    def init_m_l(m_ub, l_ub):
        with T.SimdVF(latency=400):
            init_mask = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), accum_dtype, init_mask)
            initial_m = S.vdup(-T.infinity(accum_dtype), accum_dtype, init_mask)
            S.vsts(l_ub[0], zero, init_mask, "NORM_B32")
            S.vsts(m_ub[0], initial_m, init_mask, "NORM_B32")

    @T.macro
    def clear_O_ub(O_ub):
        with T.SimdVF(latency=142):
            clear_mask = S.pset(32, "PAT_ALL")
            output_zero = S.vdup(T.float32(0.0), accum_dtype, clear_mask)
            for i in range(ROWS):
                for c in range(0, D, VL):
                    S.vsts(O_ub[i, c], output_zero, clear_mask, "NORM_B32")

    @T.macro
    def finalize_fp32(mt, O, O_ub, l_ub):
        with T.SimdVF(latency=1780):
            normalize_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                denominator = S.vld(l_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    normalized = S.vdiv(S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask)
                    S.vsts(O_ub[i, c], normalized, normalize_mask, "NORM_B32")
        T.dual_copy(O_ub[0:ROWS, 0:D], O[mt * BR : (mt + 1) * BR, 0:D])

    @T.macro
    def finalize_cast(mt, O, O_ub, O_cast_ub, l_ub):
        with T.SimdVF(latency=1780):
            normalize_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                denominator = S.vld(l_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    normalized = S.vdiv(S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask)
                    S.vsts(O_ub[i, c], normalized, normalize_mask, "NORM_B32")
        with T.SimdVF(latency=193):
            cast_full = S.pset(32, "PAT_ALL")
            for r in range(ROWS):
                for c in range(0, D, VL):
                    ob = S.vcvt(S.vld(O_ub[r, c], "NORM"), out_dtype, cast_full)
                    S.vsts(O_cast_ub[r, c], ob, cast_full, "PK_B32")
        T.dual_copy(O_cast_ub[0:ROWS, 0:D], O[mt * BR : (mt + 1) * BR, 0:D])

    kv_step = kv_manual if manual_schedule else kv_auto

    @T.prim_func
    def main(
        Q: T.Buffer((q_len, D), dtype),
        K: T.Buffer((kv_len, D), dtype),
        V: T.Buffer((kv_len, D), dtype),
        O: T.Buffer((q_len, D), out_dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            Q_shared = T.alloc_l1((BR, D), dtype)
            K_shared = T.alloc_l1((BC, D), dtype)
            V_shared = T.alloc_l1((D, BC), dtype)
            P_shared = T.alloc_l1((BR, BC), dtype)

            qk_a_l0 = T.alloc_l0a((BR, D), dtype)
            qk_b_l0 = T.alloc_l0b((BC, D), dtype)
            pv_a_l0 = T.alloc_l0a((BR, BC), dtype)
            pv_b_l0 = T.alloc_l0b((D, BC), dtype)
            qk_acc_l0c = T.alloc_l0c((BR, BC), accum_dtype)
            pv_acc_l0c = T.alloc_l0c((BR, D), accum_dtype)

            S_ub = T.alloc_shared((ROWS, BC), accum_dtype)
            P_nz_ub = T.alloc_shared((ROWS + 1, BC), dtype)
            T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
            O_tmp_ub = T.alloc_shared((ROWS, D), accum_dtype)
            O_ub = T.alloc_shared((ROWS, D), accum_dtype)
            m_ub = T.alloc_shared((ROWS,), accum_dtype)
            l_ub = T.alloc_shared((ROWS,), accum_dtype)
            alpha_ub = T.alloc_shared((ROWS,), accum_dtype)
            old_m_ub = T.alloc_shared((ROWS,), accum_dtype)
            row_sum_ub = T.alloc_shared((ROWS,), accum_dtype)
            T.annotate_buffer_versions(
                {
                    K_shared: 3,
                    V_shared: 3,
                    qk_acc_l0c: 2,
                    pv_acc_l0c: 2,
                    S_ub: 3,
                    P_nz_ub: 2,
                    alpha_ub: 3,
                    O_tmp_ub: 2,
                }
            )

            for local_mt in T.serial(M_TILES_PER_BLOCK):
                mt = bx * M_TILES_PER_BLOCK + local_mt
                T.copy(Q[mt * BR : (mt + 1) * BR, 0:D], Q_shared[0:BR, 0:D])

                init_m_l(m_ub, l_ub)
                clear_O_ub(O_ub)

                for kv in T.Pipelined(
                    NUM_KV_BLOCKS,
                    num_stages=PIPELINE_STAGES,
                    annotations={"enable_offset": True},
                ):
                    kv_step(
                        kv,
                        K,
                        V,
                        Q_shared,
                        K_shared,
                        V_shared,
                        P_shared,
                        qk_a_l0,
                        qk_b_l0,
                        pv_a_l0,
                        pv_b_l0,
                        qk_acc_l0c,
                        pv_acc_l0c,
                        S_ub,
                        P_nz_ub,
                        O_tmp_ub,
                        O_ub,
                        m_ub,
                        l_ub,
                        alpha_ub,
                        old_m_ub,
                        row_sum_ub,
                    )

                if is_cast_out:
                    # Byte reinterpretation only: aliases O_ub, consumes no extra
                    # UB capacity or buffer versions.
                    O_cast_ub = T.view(O_ub, (ROWS * 2, D), dtype=out_dtype)
                    finalize_cast(mt, O, O_ub, O_cast_ub, l_ub)
                else:
                    finalize_fp32(mt, O, O_ub, l_ub)

    return main
