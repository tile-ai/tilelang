"""Optimized GQA FlashAttention backward builders for Ascend NPU.

The forward kernel supplies LSE. Backward launches a small Delta kernel and
then one frontend-staged AutoSchedule KV-centric mixed kernel. The mixed
kernel fuses all five backward GEMMs, accumulates dK/dV privately, and
atomically reduces BF16 dQ directly through FixPipe. Q/dO use four L1
versions; P/dS and the Vector intermediates use two. Both kernels are non-causal
and require 128x128 tiles with ``head_dim == 128``.
"""

import math

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout


def flash_attention_bwd_preprocess(q_len, head_dim=128):
    """Compute ``Delta = sum(O * dO, axis=-1)`` required by every dS tile."""

    BR = 128
    D = head_dim
    dtype = "bfloat16"
    accum_dtype = "float32"

    assert D == 128
    assert BR == 128
    assert q_len % BR == 0
    NUM_Q_TILES = q_len // BR

    @T.prim_func
    def main(
        O: T.Buffer((q_len, D), dtype),
        dO: T.Buffer((q_len, D), dtype),
        Delta: T.Buffer((q_len,), accum_dtype),
    ):
        with T.Kernel(NUM_Q_TILES) as mt:
            O_ub = T.alloc_shared((BR, D), dtype)
            dO_ub = T.alloc_shared((BR, D), dtype)
            delta_ub = T.alloc_shared((BR,), accum_dtype)

            T.copy(O[mt * BR : (mt + 1) * BR, 0:D], O_ub[0:BR, 0:D])
            T.copy(dO[mt * BR : (mt + 1) * BR, 0:D], dO_ub[0:BR, 0:D])

            with T.SimdVF(latency=1053):
                full = S.pset(32, "PAT_ALL")
                one = S.pset(32, "PAT_VL1")
                for r in range(BR):
                    o = S.vld(O_ub[r, 0], "NORM")
                    do_reg = S.vld(dO_ub[r, 0], "NORM")
                    o_even = S.vcvt(o, accum_dtype, part=0)
                    o_odd = S.vcvt(o, accum_dtype, part=1)
                    do_even = S.vcvt(do_reg, accum_dtype, part=0)
                    do_odd = S.vcvt(do_reg, accum_dtype, part=1)
                    sum_even = S.vcadd(S.vmul(o_even, do_even, full), full)
                    sum_odd = S.vcadd(S.vmul(o_odd, do_odd, full), full)
                    S.vsts(delta_ub[r], S.vadd(sum_even, sum_odd, one), one, "ONEPT_B32")

            T.copy(delta_ub[0:BR], Delta[mt * BR : (mt + 1) * BR])

    return main


def flash_attention_bwd_fused_dq_atomic_staged(
    q_len,
    kv_len,
    head_dim=128,
    *,
    softmax_scale=None,
):
    """Build a frontend-staged AutoSchedule KV-centric fused backward.

    The caller must zero dQ before launch. Stage 0 produces score/dP and
    packs P/dS; stage 1 consumes the previous query tile for dV/dK/dQ.
    UnitFlag pairs protect score, dP, and dQ L0C handoffs. The dQ FixPipe
    store converts each partial to BF16 before atomic accumulation, preserving
    the original gradient precision. AutoSchedule manages the remaining
    buffer ownership and synchronization.
    """

    BR = 128
    BC = 128
    Q_STAGES = 4
    INTERMEDIATE_STAGES = 2
    ROWS = BC // 2
    D = head_dim
    dtype = "bfloat16"
    grad_dtype = "bfloat16"
    accum_dtype = "float32"
    assert D == 128, "The backward SIMD packing path requires head_dim == 128."
    assert q_len % BR == 0
    assert kv_len % BC == 0
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    NUM_Q_TILES = q_len // BR
    NUM_KV_TILES = kv_len // BC
    assert NUM_Q_TILES >= 3, "Backward requires at least three query tiles."

    @T.macro
    def pack_p_ds(S_ub, dP_ub, lse_ub, delta_ub, P_nz_ub, dS_nz_ub):
        with T.SimdVF(latency=706):
            full = S.pset(32, "PAT_ALL")
            b16 = S.pset(16, "PAT_VL128")
            p_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BR),
                dtype,
            )
            ds_ptr = S.make_ubuf_ptr(
                T.access_ptr(dS_nz_ub[0, 0], "w", ROWS + 1, BR),
                dtype,
            )
            lse_even, lse_odd = S.vld2(lse_ub[0], "DINTLV_B32")
            delta_even, delta_odd = S.vld2(delta_ub[0], "DINTLV_B32")

            for r in range(ROWS):
                score_even, score_odd = S.vld2(S_ub[r, 0], "DINTLV_B32")
                dp_even, dp_odd = S.vld2(dP_ub[r, 0], "DINTLV_B32")

                p_even = S.vexpdif(S.vmuls(score_even, T.float32(scale), full), lse_even, full)
                p_odd = S.vexpdif(S.vmuls(score_odd, T.float32(scale), full), lse_odd, full)
                ds_even = S.vmuls(
                    S.vmul(p_even, S.vsub(dp_even, delta_even, full), full),
                    T.float32(scale),
                    full,
                )
                ds_odd = S.vmuls(
                    S.vmul(p_odd, S.vsub(dp_odd, delta_odd, full), full),
                    T.float32(scale),
                    full,
                )

                p_even_bf16 = S.vcvt(p_even, dtype, full, sat=False, part=0)
                p_odd_bf16 = S.vcvt(p_odd, dtype, full, sat=False, part=1)
                p_merged = S.vor(
                    T.reinterpret(p_even_bf16, "uint16x128"),
                    T.reinterpret(p_odd_bf16, "uint16x128"),
                    b16,
                )
                p_ptr = S.vsstb(
                    T.reinterpret(p_merged, "bfloat16x128"),
                    p_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    b16,
                    update=True,
                )

                ds_even_bf16 = S.vcvt(ds_even, dtype, full, sat=False, part=0)
                ds_odd_bf16 = S.vcvt(ds_odd, dtype, full, sat=False, part=1)
                ds_merged = S.vor(
                    T.reinterpret(ds_even_bf16, "uint16x128"),
                    T.reinterpret(ds_odd_bf16, "uint16x128"),
                    b16,
                )
                ds_ptr = S.vsstb(
                    T.reinterpret(ds_merged, "bfloat16x128"),
                    ds_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    b16,
                    update=True,
                )

    # Keep each GEMM in its own extent-one loop so AutoSchedule can distinguish
    # the L0A/L0B/L0C owners and derive their versioning protocol.
    @T.macro
    def produce_step(
        mt,
        Q,
        dO,
        LSE,
        Delta,
        K_l1,
        V_l1,
        Q_l1,
        dO_l1,
        P_l1,
        dS_l1,
        a_l0,
        b_l0,
        score_l0c,
        dp_l0c,
        S_ub,
        dP_ub,
        P_nz_ub,
        dS_nz_ub,
        lse_ub,
        delta_ub,
    ):
        with T.Stage(0):
            T.copy(Q[mt * BR : (mt + 1) * BR, 0:D], Q_l1)
            for _ in range(1):
                T.copy(K_l1, a_l0)
                T.copy(Q_l1, b_l0)
                T.gemm(
                    a_l0,
                    b_l0,
                    score_l0c,
                    transpose_B=True,
                    clear_accum=True,
                    unit_flag_ctrl=3,
                )
                T.dual_copy(score_l0c, S_ub, unit_flag_ctrl=3)
            T.copy(dO[mt * BR : (mt + 1) * BR, 0:D], dO_l1)
            for _ in range(1):
                T.copy(V_l1, a_l0)
                T.copy(dO_l1, b_l0)
                T.gemm(
                    a_l0,
                    b_l0,
                    dp_l0c,
                    transpose_B=True,
                    clear_accum=True,
                    unit_flag_ctrl=3,
                )
                T.dual_copy(dp_l0c, dP_ub[0:ROWS, 0:BR], unit_flag_ctrl=3)
            T.copy(LSE[mt * BR : (mt + 1) * BR], lse_ub[0:BR])
            T.copy(Delta[mt * BR : (mt + 1) * BR], delta_ub[0:BR])
            pack_p_ds(
                S_ub,
                dP_ub,
                lse_ub,
                delta_ub,
                P_nz_ub,
                dS_nz_ub,
            )
            T.dual_copy(P_nz_ub[0:ROWS, 0:BR], P_l1)
            T.dual_copy(dS_nz_ub[0:ROWS, 0:BR], dS_l1)

    @T.macro
    def consume_step(
        mt,
        clear_accum,
        dQ,
        K_l1,
        Q_l1,
        dO_l1,
        P_l1,
        dS_l1,
        a_l0,
        b_l0,
        dp_l0c,
        dK_l0c,
        dV_l0c,
    ):
        with T.Stage(1):
            for _ in range(1):
                T.copy(P_l1, a_l0)
                T.copy(dO_l1, b_l0, transpose=True)
                T.gemm(
                    a_l0,
                    b_l0,
                    dV_l0c,
                    transpose_B=True,
                    clear_accum=clear_accum,
                )
            for _ in range(1):
                T.copy(dS_l1, a_l0)
                T.copy(Q_l1, b_l0, transpose=True)
                T.gemm(
                    a_l0,
                    b_l0,
                    dK_l0c,
                    transpose_B=True,
                    clear_accum=clear_accum,
                )
            for _ in range(1):
                T.copy(dS_l1, a_l0, transpose=True)
                T.copy(K_l1, b_l0, transpose=True)
                T.gemm(
                    a_l0,
                    b_l0,
                    dp_l0c,
                    transpose_B=True,
                    clear_accum=True,
                    unit_flag_ctrl=3,
                )
                # FixPipe performs the BF16 conversion and atomic GM store;
                # no dQ UB tile or Vector cast is needed.
                T.copy(dp_l0c, dQ[mt * BR : (mt + 1) * BR, 0:D], unit_flag_ctrl=3)

    @T.prim_func
    def main(
        Q: T.Buffer((q_len, D), dtype),
        K: T.Buffer((kv_len, D), dtype),
        V: T.Buffer((kv_len, D), dtype),
        dO: T.Buffer((q_len, D), dtype),
        LSE: T.Buffer((q_len,), accum_dtype),
        Delta: T.Buffer((q_len,), accum_dtype),
        dQ: T.Buffer((q_len, D), grad_dtype),
        dK: T.Buffer((kv_len, D), grad_dtype),
        dV: T.Buffer((kv_len, D), grad_dtype),
    ):
        with T.Kernel(NUM_KV_TILES) as kv:
            K_l1 = T.alloc_l1((BC, D), dtype)
            V_l1 = T.alloc_l1((BC, D), dtype)
            Q_l1 = T.alloc_l1((BR, D), dtype)
            dO_l1 = T.alloc_l1((BR, D), dtype)
            P_l1 = T.alloc_l1((BC, BR), dtype)
            dS_l1 = T.alloc_l1((BC, BR), dtype)

            a_l0 = T.alloc_l0a((BC, D), dtype)
            b_l0 = T.alloc_l0b((BR, D), dtype)
            score_l0c = T.alloc_l0c((BC, BR), accum_dtype)
            dp_l0c = T.alloc_l0c((BC, BR), accum_dtype)
            dK_l0c = T.alloc_l0c((BC, D), accum_dtype)
            dV_l0c = T.alloc_l0c((BC, D), accum_dtype)

            S_ub = T.alloc_shared((ROWS, BR), accum_dtype)
            dP_ub = T.alloc_shared((ROWS, BR), accum_dtype)
            P_nz_ub = T.alloc_shared((ROWS + 1, BR), dtype)
            dS_nz_ub = T.alloc_shared((ROWS + 1, BR), dtype)
            T.annotate_layout(
                {
                    P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub),
                    dS_nz_ub: make_ascend_compact_nz_layout(dS_nz_ub),
                }
            )
            lse_ub = T.alloc_shared((BR,), accum_dtype)
            delta_ub = T.alloc_shared((BR,), accum_dtype)

            T.annotate_buffer_versions(
                {
                    # Rebalance the same 448 KiB of L1 toward input prefetch.
                    Q_l1: Q_STAGES,
                    dO_l1: Q_STAGES,
                    P_l1: INTERMEDIATE_STAGES,
                    dS_l1: INTERMEDIATE_STAGES,
                    a_l0: 2,
                    b_l0: 2,
                    S_ub: INTERMEDIATE_STAGES,
                    dP_ub: INTERMEDIATE_STAGES,
                    lse_ub: INTERMEDIATE_STAGES,
                    delta_ub: INTERMEDIATE_STAGES,
                    dp_l0c: (1, "auto"),  # Preserve multi-buffer eligibility with one version.
                    dK_l0c: 1,
                    dV_l0c: 1,
                }
            )

            T.copy(K[kv * BC : (kv + 1) * BC, 0:D], K_l1)
            T.copy(V[kv * BC : (kv + 1) * BC, 0:D], V_l1)

            T.set_atomic("add", grad_dtype)
            for mt in T.Pipelined(
                NUM_Q_TILES,
                num_stages=Q_STAGES,
                annotations={"enable_offset": True},
            ):
                produce_step(
                    mt,
                    Q,
                    dO,
                    LSE,
                    Delta,
                    K_l1,
                    V_l1,
                    Q_l1,
                    dO_l1,
                    P_l1,
                    dS_l1,
                    a_l0,
                    b_l0,
                    score_l0c,
                    dp_l0c,
                    S_ub,
                    dP_ub,
                    P_nz_ub,
                    dS_nz_ub,
                    lse_ub,
                    delta_ub,
                )
                consume_step(
                    mt,
                    mt == 0,
                    dQ,
                    K_l1,
                    Q_l1,
                    dO_l1,
                    P_l1,
                    dS_l1,
                    a_l0,
                    b_l0,
                    dp_l0c,
                    dK_l0c,
                    dV_l0c,
                )

            T.set_atomic_none()
            T.copy(dV_l0c, dV[kv * BC, 0])
            T.copy(dK_l0c, dK[kv * BC, 0])

    return main
