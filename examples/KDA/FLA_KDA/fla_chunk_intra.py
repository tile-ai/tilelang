# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from .fla_utils import autotune_cache_kwargs, exp2, prepare_chunk_indices
from .cumsum import chunk_local_cumsum

IS_TF32_SUPPORTED = False
if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32x3")
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")
# ============================================================================
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
# ============================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({"BK": BK}, num_warps=num_warps) for BK in [32, 64] for num_warps in [1, 2, 4]],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk_diag,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akk_diag.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akk_diag (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akk_diag += (bos * H + i_h) * BC

    m_tc1 = (i_tc1 + tl.arange(0, BC)) < T
    m_tc2 = (i_tc2 + tl.arange(0, BC)) < T
    m_tc3 = (i_tc3 + tl.arange(0, BC)) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    ################################################################################
    # 1. off-diagonal blocks
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_tc0), (BK, BC), (0, 1))
        p_g0 = tl.make_block_ptr(g, (K, T), (1, H * K), (i_k * BK, i_tc0), (BK, BC), (0, 1))
        b_kt0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_gt0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        b_kt1, b_gt1 = b_kt0, b_gt0
        b_kt2, b_gt2 = b_kt0, b_gt0
        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))

            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            b_kt1 = tl.trans(b_k1)
            b_gt1 = tl.trans(b_g1)

            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn1 = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_qg1 = b_q1 * b_gqn1
            b_kg1 = b_k1 * b_gqn1
            b_kgt = b_kt0 * exp2(b_gn1[:, None] - b_gt0)
            b_Aqk10 += tl.dot(b_qg1, b_kgt)
            b_Akk10 += tl.dot(b_kg1, b_kgt)

        if i_tc2 < T:
            p_q2 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))

            b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
            b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
            b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
            b_kt2 = tl.trans(b_k2)
            b_gt2 = tl.trans(b_g2)

            b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
            b_qg2 = b_q2 * b_gqn2
            b_kg2 = b_k2 * b_gqn2
            b_kgt = b_kt0 * exp2(b_gn2[:, None] - b_gt0)
            b_Aqk20 += tl.dot(b_qg2, b_kgt)
            b_Akk20 += tl.dot(b_kg2, b_kgt)

            b_kgt = b_kt1 * exp2(b_gn2[:, None] - b_gt1)
            b_Aqk21 += tl.dot(b_qg2, b_kgt)
            b_Akk21 += tl.dot(b_kg2, b_kgt)

        if i_tc3 < T:
            p_q3 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            p_g3 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
            b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
            b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)

            b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
            b_qg3 = b_q3 * b_gqn3
            b_kg3 = b_k3 * b_gqn3
            b_kgt = b_kt0 * exp2(b_gn3[:, None] - b_gt0)
            b_Aqk30 += tl.dot(b_qg3, b_kgt)
            b_Akk30 += tl.dot(b_kg3, b_kgt)

            b_kgt = b_kt1 * exp2(b_gn3[:, None] - b_gt1)
            b_Aqk31 += tl.dot(b_qg3, b_kgt)
            b_Akk31 += tl.dot(b_kg3, b_kgt)

            b_kgt = b_kt2 * exp2(b_gn3[:, None] - b_gt2)
            b_Aqk32 += tl.dot(b_qg3, b_kgt)
            b_Akk32 += tl.dot(b_kg3, b_kgt)

    ################################################################################
    # 2. save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    ################################################################################
    # 3. load diagonal Akk blocks
    ################################################################################
    p_Akk00 = tl.make_block_ptr(Akk_diag, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk_diag, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk_diag, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk_diag, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    # each diagonal block is stored contiguously: row i of block s is at Akk_diag[t=i_t*BT+s*BC+i, :BC]
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    ################################################################################
    # 4. forward substitution on diagonals
    ################################################################################
    o_i = tl.arange(0, BC)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Ai00 = -tl.where(m_A, b_Ai00, 0)
    b_Ai11 = -tl.where(m_A, b_Ai11, 0)
    b_Ai22 = -tl.where(m_A, b_Ai22, 0)
    b_Ai33 = -tl.where(m_A, b_Ai33, 0)

    # Forward substitution: load from Akk_diag (stride H*BC, columns 0:BC)
    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = -tl.load(Akk_diag + (i_tc0 + i) * H * BC + o_i)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(BC + 2, min(2 * BC, T - i_tc0)):
        b_a11 = -tl.load(Akk_diag + (i_tc0 + i) * H * BC + o_i)
        b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
        b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
    for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
        b_a22 = -tl.load(Akk_diag + (i_tc0 + i) * H * BC + o_i)
        b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
        b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
    for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
        b_a33 = -tl.load(Akk_diag + (i_tc0 + i) * H * BC + o_i)
        b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
        b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    # ################################################################################
    # # 5. compute merged inverse using off-diagonals
    # ################################################################################

    # we used tf32x3 to maintain matrix inverse's precision whenever possible.
    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION), b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION), b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION), b_Ai22, input_precision=SOLVE_TRIL_DOT_PRECISION)

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ################################################################################
    # 6. store full Akk_inv to Akk
    ################################################################################

    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
    p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [1, 2, 4, 8] for num_stages in [2, 3, 4]],
    key=["BK", "NC", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["B", "T"])
def chunk_kda_bwd_kernel_intra(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h

    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dg2 += (bos * H + i_h) * K
    db += (i_k * all + bos) * H + i_h

    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_b = tl.make_block_ptr(beta, (T,), (H,), (i_ti,), (BC,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * H * K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp2(b_gn[None, :] - b_gk)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            # [BC, BK]
            b_dq2 += tl.dot(b_dAqk, b_kg)
            b_dk2 += tl.dot(b_dAkk, b_kg)
        b_gqn = exp2(b_g - b_gn[None, :])
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * H * BT + i_i * BC
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * H * K + o_k

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC]
        b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
        b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
        # [BK]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        b_kgj = b_kj[None, :] * exp2(b_g - b_gkj[None, :])
        b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kgj, 0.0)
        b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kgj, 0.0)

        p_kj += H * K
        p_gkj += H * K
    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(dq2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (H,), (i_ti,), (BC,), (0,))

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * H * K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_b = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT + i_j * BC,), (BC,), (0,))
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            # [BC]
            b_b = tl.load(p_b, boundary_check=(0,))
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_kb = tl.load(p_k, boundary_check=(0, 1)) * b_b[:, None]
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_gkn = tl.where(m_j[:, None], exp2(b_gk - b_gn[None, :]), 0)
            b_qg = b_q * b_gkn
            b_kbg = b_kb * b_gkn
            # [BC, BK]
            b_dkt += tl.dot(b_dAqk, b_qg) + tl.dot(b_dAkk, b_kbg)
        b_dkt *= exp2(b_gn[None, :] - b_g)

    o_dA = i_ti * H * BT + i_i * BC + o_i
    p_qj = q + i_ti * H * K + o_k  # [bs, i_ti, i_h*block_h, i_k*bk:(i_k+1)*bk]
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * H * K + o_k
    p_bj = beta + i_ti * H

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dAqk = tl.load(dAqk + o_dA + j * H * BT)
        b_dAkk = tl.load(dAkk + o_dA + j * H * BT)
        # [BK,]
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_gkq = exp2(b_gkj[None, :] - b_g)
        b_dkt += tl.where(m_i, (b_dAkk[:, None] * b_kbj[None, :] + b_dAqk[:, None] * b_qj[None, :]) * b_gkq, 0.0)

        p_qj += H * K
        p_kj += H * K
        p_gkj += H * K
        p_bj += H
    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(dk2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg2 = tl.make_block_ptr(dg2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, boundary_check=(0, 1))
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))


def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor = None,
    chunk_indices: torch.LongTensor = None,
    chunk_size: int = 64,
):
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(q)
    dk2 = torch.empty_like(k)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    grid = (NK * NC, NT, B * H)
    chunk_kda_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dg=dg,
        dg2=dg2,
        db=db2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    dq = dq2
    dk = dk2
    db = db2.sum(0).add_(db)
    dg = chunk_local_cumsum(
        dg2,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    return dq, dk, db, dg


def chunk_kda_fwd_inter_solve_fused(
    q,
    k,
    gk,
    beta,
    Aqk,
    Akk_diag,
    Akk,
    scale,
    cu_seqlens: torch.LongTensor = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor = None,
):
    B, T, H, K = k.shape
    assert K <= 256
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BC = 16

    grid = (NT, B * H)
    chunk_kda_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk_diag=Akk_diag,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )
