import tilelang
import tilelang.language as T

from FLA_KDA.fla_chunk_intra import chunk_kda_fwd_inter_solve_fused
from FLA_KDA.cumsum import chunk_local_cumsum
from test_utils_kda import compare_tensors, do_bench

import torch
import torch.nn.functional as F


torch.random.manual_seed(42)


def prepare_input(
    B,
    S,
    H,
    DK,
    chunk_size,
    sub_chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
):
    q = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    k = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    beta = torch.randn(B, S, H, dtype=input_dtype).cuda()
    gk = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()  # 需要是cumsum
    gk = F.logsigmoid(gk)
    gk = chunk_local_cumsum(gk, chunk_size)

    Aqk = torch.empty(B, S, H, chunk_size, dtype=input_dtype).cuda()
    Akk_diag = torch.ones(B, S, H, sub_chunk_size, dtype=torch.float32).cuda()

    return q, k, gk, beta, Aqk, Akk_diag


def prepare_output(
    B,
    S,
    H,
    chunk_size,
    sub_chunk_size,
    output_dtype,
):
    Akk = torch.empty(B, S, H, chunk_size, dtype=output_dtype).cuda()
    return Akk


@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_chunk_kda_fwd_inter_fused(
    B,
    S,
    H,
    DK,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    sub_chunk_size,
    scale,
    block_DK=32,
    threads=32,
    num_stages=1,
):
    block_S = BS = chunk_size
    BC = sub_chunk_size
    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    GK_shape = (B, S, H, DK)
    Beta_shape = (B, S, H)
    Aqk_shape = (B, S, H, BS)
    Akk_diag_shape = (B, S, H, BC)
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

    @T.prim_func
    def kernel(
        Q: T.Tensor(Q_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        GK: T.Tensor(GK_shape, dtype=gate_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        Akk_diag: T.Tensor(Akk_diag_shape, dtype=T.float32),
        Aqk: T.Tensor(Aqk_shape, dtype=output_dtype),
        Akk: T.Tensor(Aqk_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H

            Aqk10_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Akk10_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Aqk20_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Akk20_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Aqk21_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Akk21_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Aqk30_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Akk30_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Aqk31_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Akk31_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Aqk32_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Akk32_fragment = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            Akk10_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Akk20_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Akk21_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Akk30_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Akk31_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Akk32_shared = T.alloc_shared((BC, BC), dtype=T.float32)

            K0_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            GK0_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            Q1_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            K1_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            GK1_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            Q2_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            K2_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            GK2_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            Q3_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            K3_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            GK3_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)

            Q_GK_scaled_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            K_GK_scaled_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            b_kt_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)

            b_gn1_shared = T.alloc_shared((block_DK,), dtype=T.float32)
            b_gn2_shared = T.alloc_shared((block_DK,), dtype=T.float32)
            b_gn3_shared = T.alloc_shared((block_DK,), dtype=T.float32)

            b_gqn1_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            b_gqn2_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)
            b_gqn3_shared = T.alloc_shared((BC, block_DK), dtype=T.float32)

            beta_1_shared = T.alloc_shared((BC,), dtype=T.float32)
            beta_2_shared = T.alloc_shared((BC,), dtype=T.float32)
            beta_3_shared = T.alloc_shared((BC,), dtype=T.float32)
            # Akk_inv
            Ai_00_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_10_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_11_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_20_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_21_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_22_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_30_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_31_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_32_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_33_shared = T.alloc_shared((BC, BC), dtype=T.float32)

            T.clear(Aqk10_fragment)
            T.clear(Akk10_fragment)
            T.clear(Aqk20_fragment)
            T.clear(Akk20_fragment)
            T.clear(Aqk21_fragment)
            T.clear(Akk21_fragment)
            T.clear(Aqk30_fragment)
            T.clear(Akk30_fragment)
            T.clear(Aqk31_fragment)
            T.clear(Akk31_fragment)
            T.clear(Aqk32_fragment)
            T.clear(Akk32_fragment)

            i_tc0 = bs * BS
            i_tc1 = bs * BS + BC
            i_tc2 = bs * BS + 2 * BC
            i_tc3 = bs * BS + 3 * BC

            ################################################################################
            # 1. off-diagonal blocks
            ################################################################################

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(K[bb, bs * BS : bs * BS + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], K0_shared)
                T.copy(GK[bb, bs * BS : bs * BS + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], GK0_shared)
                if i_tc1 < S:
                    T.copy(Q[bb, i_tc1 : i_tc1 + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], Q1_shared)
                    T.copy(K[bb, i_tc1 : i_tc1 + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], K1_shared)
                    T.copy(GK[bb, i_tc1 : i_tc1 + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], GK1_shared)
                    T.copy(GK[bb, i_tc1, bh, i_k * block_DK : (i_k + 1) * block_DK], b_gn1_shared)  # subblock第一个token的GK
                    for i_c1, i_k1 in T.Parallel(BC, block_DK):
                        b_gqn1_shared[i_c1, i_k1] = T.if_then_else(
                            i_tc1 + i_c1 < S, T.exp2(GK1_shared[i_c1, i_k1] - b_gn1_shared[i_k1]), 0.0
                        )
                        Q_GK_scaled_shared[i_c1, i_k1] = Q1_shared[i_c1, i_k1] * b_gqn1_shared[i_c1, i_k1]
                        K_GK_scaled_shared[i_c1, i_k1] = K1_shared[i_c1, i_k1] * b_gqn1_shared[i_c1, i_k1]
                        b_kt_shared[i_c1, i_k1] = K0_shared[i_c1, i_k1] * T.exp2(b_gn1_shared[i_k1] - GK0_shared[i_c1, i_k1])
                    T.gemm(Q_GK_scaled_shared, b_kt_shared, Aqk10_fragment, transpose_B=True)
                    T.gemm(K_GK_scaled_shared, b_kt_shared, Akk10_fragment, transpose_B=True)
                if i_tc2 < S:
                    T.copy(Q[bb, i_tc2 : i_tc2 + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], Q2_shared)
                    T.copy(K[bb, i_tc2 : i_tc2 + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], K2_shared)
                    T.copy(GK[bb, i_tc2 : i_tc2 + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], GK2_shared)
                    T.copy(GK[bb, i_tc2, bh, i_k * block_DK : (i_k + 1) * block_DK], b_gn2_shared)
                    for i_c2, i_k2 in T.Parallel(BC, block_DK):
                        b_gqn2_shared[i_c2, i_k2] = T.if_then_else(
                            i_tc2 + i_c2 < S, T.exp2(GK2_shared[i_c2, i_k2] - b_gn2_shared[i_k2]), 0.0
                        )
                        Q_GK_scaled_shared[i_c2, i_k2] = Q2_shared[i_c2, i_k2] * b_gqn2_shared[i_c2, i_k2]
                        K_GK_scaled_shared[i_c2, i_k2] = K2_shared[i_c2, i_k2] * b_gqn2_shared[i_c2, i_k2]
                        b_kt_shared[i_c2, i_k2] = K0_shared[i_c2, i_k2] * T.exp2(b_gn2_shared[i_k2] - GK0_shared[i_c2, i_k2])
                    T.gemm(Q_GK_scaled_shared, b_kt_shared, Aqk20_fragment, transpose_B=True)
                    T.gemm(K_GK_scaled_shared, b_kt_shared, Akk20_fragment, transpose_B=True)
                    for i_c3, i_k3 in T.Parallel(BC, block_DK):
                        b_kt_shared[i_c3, i_k3] = K1_shared[i_c3, i_k3] * T.exp2(b_gn2_shared[i_k3] - GK1_shared[i_c3, i_k3])
                    T.gemm(Q_GK_scaled_shared, b_kt_shared, Aqk21_fragment, transpose_B=True)
                    T.gemm(K_GK_scaled_shared, b_kt_shared, Akk21_fragment, transpose_B=True)
                if i_tc3 < S:
                    T.copy(Q[bb, i_tc3 : i_tc3 + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], Q3_shared)
                    T.copy(K[bb, i_tc3 : i_tc3 + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], K3_shared)
                    T.copy(GK[bb, i_tc3 : i_tc3 + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], GK3_shared)
                    T.copy(GK[bb, i_tc3, bh, i_k * block_DK : (i_k + 1) * block_DK], b_gn3_shared)
                    for i_c4, i_k4 in T.Parallel(BC, block_DK):
                        b_gqn3_shared[i_c4, i_k4] = T.if_then_else(
                            i_tc3 + i_c4 < S, T.exp2(GK3_shared[i_c4, i_k4] - b_gn3_shared[i_k4]), 0.0
                        )
                        Q_GK_scaled_shared[i_c4, i_k4] = Q3_shared[i_c4, i_k4] * b_gqn3_shared[i_c4, i_k4]
                        K_GK_scaled_shared[i_c4, i_k4] = K3_shared[i_c4, i_k4] * b_gqn3_shared[i_c4, i_k4]
                        b_kt_shared[i_c4, i_k4] = K0_shared[i_c4, i_k4] * T.exp2(b_gn3_shared[i_k4] - GK0_shared[i_c4, i_k4])
                    T.gemm(Q_GK_scaled_shared, b_kt_shared, Aqk30_fragment, transpose_B=True)
                    T.gemm(K_GK_scaled_shared, b_kt_shared, Akk30_fragment, transpose_B=True)
                    for i_c5, i_k5 in T.Parallel(BC, block_DK):
                        b_kt_shared[i_c5, i_k5] = K1_shared[i_c5, i_k5] * T.exp2(b_gn3_shared[i_k5] - GK1_shared[i_c5, i_k5])
                    T.gemm(Q_GK_scaled_shared, b_kt_shared, Aqk31_fragment, transpose_B=True)
                    T.gemm(K_GK_scaled_shared, b_kt_shared, Akk31_fragment, transpose_B=True)
                    for i_c6, i_k6 in T.Parallel(BC, block_DK):
                        b_kt_shared[i_c6, i_k6] = K2_shared[i_c6, i_k6] * T.exp2(b_gn3_shared[i_k6] - GK2_shared[i_c6, i_k6])
                    T.gemm(Q_GK_scaled_shared, b_kt_shared, Aqk32_fragment, transpose_B=True)
                    T.gemm(K_GK_scaled_shared, b_kt_shared, Akk32_fragment, transpose_B=True)

            ################################################################################
            # 2. save off-diagonal Aqk blocks and prepare Akk
            ################################################################################

            if i_tc1 < S:
                T.copy(Beta[bb, i_tc1 : i_tc1 + BC, bh], beta_1_shared)
                for i_c21, i_c22 in T.Parallel(BC, BC):
                    Aqk10_fragment[i_c21, i_c22] = Aqk10_fragment[i_c21, i_c22] * scale
                    Akk10_fragment[i_c21, i_c22] = Akk10_fragment[i_c21, i_c22] * beta_1_shared[i_c21]
                T.copy(Aqk10_fragment, Aqk[bb, i_tc1 : i_tc1 + BC, bh, 0:BC])
                T.copy(Akk10_fragment, Akk10_shared)
            if i_tc2 < S:
                T.copy(Beta[bb, i_tc2 : i_tc2 + BC, bh], beta_2_shared)
                for i_c23, i_c24 in T.Parallel(BC, BC):
                    Aqk20_fragment[i_c23, i_c24] = Aqk20_fragment[i_c23, i_c24] * scale
                    Aqk21_fragment[i_c23, i_c24] = Aqk21_fragment[i_c23, i_c24] * scale
                    Akk20_fragment[i_c23, i_c24] = Akk20_fragment[i_c23, i_c24] * beta_2_shared[i_c23]
                    Akk21_fragment[i_c23, i_c24] = Akk21_fragment[i_c23, i_c24] * beta_2_shared[i_c23]
                T.copy(Aqk20_fragment, Aqk[bb, i_tc2 : i_tc2 + BC, bh, 0:BC])
                T.copy(Aqk21_fragment, Aqk[bb, i_tc2 : i_tc2 + BC, bh, BC : 2 * BC])
                T.copy(Akk20_fragment, Akk20_shared)
                T.copy(Akk21_fragment, Akk21_shared)
            if i_tc3 < S:
                T.copy(Beta[bb, i_tc3 : i_tc3 + BC, bh], beta_3_shared)
                for i_c25, i_c26 in T.Parallel(BC, BC):
                    Aqk30_fragment[i_c25, i_c26] = Aqk30_fragment[i_c25, i_c26] * scale
                    Aqk31_fragment[i_c25, i_c26] = Aqk31_fragment[i_c25, i_c26] * scale
                    Aqk32_fragment[i_c25, i_c26] = Aqk32_fragment[i_c25, i_c26] * scale
                    Akk30_fragment[i_c25, i_c26] = Akk30_fragment[i_c25, i_c26] * beta_3_shared[i_c25]
                    Akk31_fragment[i_c25, i_c26] = Akk31_fragment[i_c25, i_c26] * beta_3_shared[i_c25]
                    Akk32_fragment[i_c25, i_c26] = Akk32_fragment[i_c25, i_c26] * beta_3_shared[i_c25]
                T.copy(Aqk30_fragment, Aqk[bb, i_tc3 : i_tc3 + BC, bh, 0:BC])
                T.copy(Aqk31_fragment, Aqk[bb, i_tc3 : i_tc3 + BC, bh, BC : 2 * BC])
                T.copy(Aqk32_fragment, Aqk[bb, i_tc3 : i_tc3 + BC, bh, 2 * BC : 3 * BC])
                T.copy(Akk30_fragment, Akk30_shared)
                T.copy(Akk31_fragment, Akk31_shared)
                T.copy(Akk32_fragment, Akk32_shared)

            ################################################################################
            # 3. load diagonal Akk blocks
            ################################################################################

            T.copy(Akk_diag[bb, i_tc0 : i_tc0 + BC, bh, :], Ai_00_shared)
            T.copy(Akk_diag[bb, i_tc1 : i_tc1 + BC, bh, :], Ai_11_shared)
            T.copy(Akk_diag[bb, i_tc2 : i_tc2 + BC, bh, :], Ai_22_shared)
            T.copy(Akk_diag[bb, i_tc3 : i_tc3 + BC, bh, :], Ai_33_shared)
            for i_c1, i_c2 in T.Parallel(BC, BC):
                Ai_00_shared[i_c1, i_c2] = T.if_then_else(i_c1 > i_c2, -Ai_00_shared[i_c1, i_c2], 0)
                Ai_11_shared[i_c1, i_c2] = T.if_then_else(i_c1 > i_c2, -Ai_11_shared[i_c1, i_c2], 0)
                Ai_22_shared[i_c1, i_c2] = T.if_then_else(i_c1 > i_c2, -Ai_22_shared[i_c1, i_c2], 0)
                Ai_33_shared[i_c1, i_c2] = T.if_then_else(i_c1 > i_c2, -Ai_33_shared[i_c1, i_c2], 0)

            ################################################################################
            # 4. forward substitution on diagonals
            ################################################################################
            a_00_shared = T.alloc_shared((BC,), dtype=T.float32)
            Aa_mul_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            reduce_shared = T.alloc_shared((BC,), dtype=T.float32)
            for i_i in T.Pipelined(2, T.min(BC, S - i_tc0), num_stages=num_stages):
                T.copy(Akk_diag[bb, i_tc0 + i_i, bh, :], a_00_shared)  # load row
                for i_c in T.Parallel(BC):
                    a_00_shared[i_c] = T.if_then_else(i_c < i_i, -a_00_shared[i_c], 0.0)  # mask:i_c<i_i
                for i_c2, i_c3 in T.Parallel(BC, BC):
                    Aa_mul_shared[i_c2, i_c3] = a_00_shared[i_c2] * Ai_00_shared[i_c2, i_c3]
                T.reduce_sum(Aa_mul_shared, reduce_shared, dim=0, clear=True)
                for i_c4 in T.Parallel(BC):
                    a_00_shared[i_c4] += reduce_shared[i_c4]
                for i_c5, i_c6 in T.Parallel(BC, BC):
                    Ai_00_shared[i_c5, i_c6] = T.if_then_else(i_c5 == i_i, a_00_shared[i_c6], Ai_00_shared[i_c5, i_c6])

            a_11_shared = T.alloc_shared((BC,), dtype=T.float32)
            Aa11_mul_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            for i_i in T.Pipelined(BC + 2, T.min(2 * BC, S - i_tc0), num_stages=num_stages):
                T.copy(Akk_diag[bb, i_tc0 + i_i, bh, :], a_11_shared)
                for i_c in T.Parallel(BC):
                    a_11_shared[i_c] = T.if_then_else(i_c < i_i - BC, -a_11_shared[i_c], 0.0)
                for i_c2, i_c3 in T.Parallel(BC, BC):
                    Aa11_mul_shared[i_c2, i_c3] = a_11_shared[i_c2] * Ai_11_shared[i_c2, i_c3]
                T.reduce_sum(
                    Aa11_mul_shared,
                    reduce_shared,
                    dim=0,
                )
                for i_c4 in T.Parallel(BC):
                    a_11_shared[i_c4] = reduce_shared[i_c4] + a_11_shared[i_c4]
                for i_c5, i_c6 in T.Parallel(BC, BC):
                    Ai_11_shared[i_c5, i_c6] = T.if_then_else(
                        i_c5 == (i_i - BC),
                        a_11_shared[i_c6],
                        Ai_11_shared[i_c5, i_c6],
                    )

            a_22_shared = T.alloc_shared((BC,), dtype=T.float32)
            Aa22_mul_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            for i_i in T.Pipelined(2 * BC + 2, T.min(3 * BC, S - i_tc0), num_stages=num_stages):
                T.copy(Akk_diag[bb, i_tc0 + i_i, bh, :], a_22_shared)
                for i_c in T.Parallel(BC):
                    a_22_shared[i_c] = T.if_then_else(i_c < i_i - 2 * BC, -a_22_shared[i_c], 0.0)
                for i_c2, i_c3 in T.Parallel(BC, BC):
                    Aa22_mul_shared[i_c2, i_c3] = a_22_shared[i_c2] * Ai_22_shared[i_c2, i_c3]
                T.reduce_sum(
                    Aa22_mul_shared,
                    reduce_shared,
                    dim=0,
                )
                for i_c4 in T.Parallel(BC):
                    a_22_shared[i_c4] = reduce_shared[i_c4] + a_22_shared[i_c4]
                for i_c5, i_c6 in T.Parallel(BC, BC):
                    Ai_22_shared[i_c5, i_c6] = T.if_then_else(i_c5 == (i_i - 2 * BC), a_22_shared[i_c6], Ai_22_shared[i_c5, i_c6])

            a_33_shared = T.alloc_shared((BC,), dtype=T.float32)
            Aa33_mul_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            for i_i in T.Pipelined(3 * BC + 2, T.min(4 * BC, S - i_tc0), num_stages=num_stages):
                T.copy(Akk_diag[bb, i_tc0 + i_i, bh, :], a_33_shared)
                for i_c in T.Parallel(BC):
                    a_33_shared[i_c] = T.if_then_else(i_c < i_i - 3 * BC, -a_33_shared[i_c], 0.0)
                for i_c2, i_c3 in T.Parallel(BC, BC):
                    Aa33_mul_shared[i_c2, i_c3] = a_33_shared[i_c2] * Ai_33_shared[i_c2, i_c3]
                T.reduce_sum(
                    Aa33_mul_shared,
                    reduce_shared,
                    dim=0,
                )
                for i_c4 in T.Parallel(BC):
                    a_33_shared[i_c4] = reduce_shared[i_c4] + a_33_shared[i_c4]
                for i_c5, i_c6 in T.Parallel(BC, BC):
                    Ai_33_shared[i_c5, i_c6] = T.if_then_else(
                        i_c5 == (i_i - 3 * BC),
                        a_33_shared[i_c6],
                        Ai_33_shared[i_c5, i_c6],
                    )

            for i, j in T.Parallel(BC, BC):
                Ai_00_shared[i, j] += T.if_then_else(i == j, 1.0, 0.0)
                Ai_11_shared[i, j] += T.if_then_else(i == j, 1.0, 0.0)
                Ai_22_shared[i, j] += T.if_then_else(i == j, 1.0, 0.0)
                Ai_33_shared[i, j] += T.if_then_else(i == j, 1.0, 0.0)

            ################################################################################
            # 5. compute merged inverse using off-diagonals
            ################################################################################

            Ai_10_inv_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            Ai_10_final_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            Ai_21_inv_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            Ai_21_final_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            Ai_32_inv_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            Ai_32_final_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            Ai_10_inv_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_21_inv_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai_32_inv_shared = T.alloc_shared((BC, BC), dtype=T.float32)

            # ---------- Ai_10 = - (Ai11@Akk10)@Ai00 ----------
            T.gemm(Ai_11_shared, Akk10_shared, Ai_10_inv_frag, clear_accum=True)  # [BC, BC] * [BC, BC]
            T.copy(Ai_10_inv_frag, Ai_10_inv_shared)
            T.gemm(Ai_10_inv_shared, Ai_00_shared, Ai_10_final_frag, clear_accum=True)
            for i_bc, j_bc in T.Parallel(BC, BC):
                Ai_10_final_frag[i_bc, j_bc] = -Ai_10_final_frag[i_bc, j_bc]
            T.copy(Ai_10_final_frag, Ai_10_shared)
            # ---------- Ai_21 = - (Ai22@Akk21)@Ai11 ----------
            T.gemm(Ai_22_shared, Akk21_shared, Ai_21_inv_frag, clear_accum=True)
            T.copy(Ai_21_inv_frag, Ai_21_inv_shared)
            T.gemm(Ai_21_inv_shared, Ai_11_shared, Ai_21_final_frag, clear_accum=True)
            for i_bc, j_bc in T.Parallel(BC, BC):
                Ai_21_final_frag[i_bc, j_bc] = -Ai_21_final_frag[i_bc, j_bc]
            T.copy(Ai_21_final_frag, Ai_21_shared)
            # ---------- Ai_32 = - (Ai33@Akk32)@Ai22 ----------
            T.gemm(Ai_33_shared, Akk32_shared, Ai_32_inv_frag, clear_accum=True)
            T.copy(Ai_32_inv_frag, Ai_32_inv_shared)
            T.gemm(Ai_32_inv_shared, Ai_22_shared, Ai_32_final_frag, clear_accum=True)
            for i_bc, j_bc in T.Parallel(BC, BC):
                Ai_32_final_frag[i_bc, j_bc] = -Ai_32_final_frag[i_bc, j_bc]
            T.copy(Ai_32_final_frag, Ai_32_shared)

            # ---------- Ai_20 = - Ai_22 @ ( Akk20@Ai00 + Akk21@Ai10 ) ----------
            Ai20_t0_frag = T.alloc_fragment((BC, BC), dtype=T.float32)  # Akk20 @ Ai00
            Ai20_t1_frag = T.alloc_fragment((BC, BC), dtype=T.float32)  # Akk21 @ Ai10
            Ai20_sum_shared = T.alloc_shared((BC, BC), dtype=T.float32)  # t0 + t1
            Ai20_final_frag = T.alloc_fragment((BC, BC), dtype=T.float32)

            T.gemm(Akk20_shared, Ai_00_shared, Ai20_t0_frag, clear_accum=True)
            T.gemm(Akk21_shared, Ai_10_shared, Ai20_t1_frag, clear_accum=True)

            # sum = t0 + t1
            for i_bc, j_bc in T.Parallel(BC, BC):
                Ai20_sum_shared[i_bc, j_bc] = Ai20_t0_frag[i_bc, j_bc] + Ai20_t1_frag[i_bc, j_bc]

            # final = Ai_22 @ sum
            T.gemm(Ai_22_shared, Ai20_sum_shared, Ai20_final_frag, clear_accum=True)

            # negate
            for i_bc, j_bc in T.Parallel(BC, BC):
                Ai20_final_frag[i_bc, j_bc] = -Ai20_final_frag[i_bc, j_bc]

            T.copy(Ai20_final_frag, Ai_20_shared)

            # ---------- Ai_31 = - Ai_33 @ ( Akk31@Ai11 + Akk32@Ai21 ) ----------
            Ai31_t0_frag = T.alloc_fragment((BC, BC), dtype=T.float32)  # Akk31 @ Ai11
            Ai31_t1_frag = T.alloc_fragment((BC, BC), dtype=T.float32)  # Akk32 @ Ai21
            Ai31_sum_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai31_final_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            T.gemm(Akk31_shared, Ai_11_shared, Ai31_t0_frag, clear_accum=True)
            T.gemm(Akk32_shared, Ai_21_shared, Ai31_t1_frag, clear_accum=True)
            for i_bc, j_bc in T.Parallel(BC, BC):
                Ai31_sum_shared[i_bc, j_bc] = Ai31_t0_frag[i_bc, j_bc] + Ai31_t1_frag[i_bc, j_bc]
            T.gemm(Ai_33_shared, Ai31_sum_shared, Ai31_final_frag, clear_accum=True)
            for i_bc, j_bc in T.Parallel(BC, BC):
                Ai31_final_frag[i_bc, j_bc] = -Ai31_final_frag[i_bc, j_bc]
            T.copy(Ai31_final_frag, Ai_31_shared)

            # ---------- Ai_30 = - Ai_33 @ ( Akk30@Ai00 + Akk31@Ai10 + Akk32@Ai20 ) ----------
            Ai30_t0_frag = T.alloc_fragment((BC, BC), dtype=T.float32)  # Akk30 @ Ai00
            Ai30_t1_frag = T.alloc_fragment((BC, BC), dtype=T.float32)  # Akk31 @ Ai10
            Ai30_t2_frag = T.alloc_fragment((BC, BC), dtype=T.float32)  # Akk32 @ Ai20
            Ai30_sum_shared = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai30_final_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            T.gemm(Akk30_shared, Ai_00_shared, Ai30_t0_frag, clear_accum=True)
            T.gemm(Akk31_shared, Ai_10_shared, Ai30_t1_frag, clear_accum=True)
            T.gemm(Akk32_shared, Ai_20_shared, Ai30_t2_frag, clear_accum=True)
            for i_bc, j_bc in T.Parallel(BC, BC):
                Ai30_sum_shared[i_bc, j_bc] = Ai30_t0_frag[i_bc, j_bc] + Ai30_t1_frag[i_bc, j_bc] + Ai30_t2_frag[i_bc, j_bc]
            T.gemm(Ai_33_shared, Ai30_sum_shared, Ai30_final_frag, clear_accum=True)
            for i_bc, j_bc in T.Parallel(BC, BC):
                Ai30_final_frag[i_bc, j_bc] = -Ai30_final_frag[i_bc, j_bc]
            T.copy(Ai30_final_frag, Ai_30_shared)

            T.copy(Ai_00_shared, Akk[bb, i_tc0 : i_tc0 + BC, bh, 0:BC])
            T.copy(Ai_10_shared, Akk[bb, i_tc1 : i_tc1 + BC, bh, 0:BC])
            T.copy(Ai_11_shared, Akk[bb, i_tc1 : i_tc1 + BC, bh, BC : 2 * BC])
            T.copy(Ai_20_shared, Akk[bb, i_tc2 : i_tc2 + BC, bh, 0:BC])
            T.copy(Ai_21_shared, Akk[bb, i_tc2 : i_tc2 + BC, bh, BC : 2 * BC])
            T.copy(Ai_22_shared, Akk[bb, i_tc2 : i_tc2 + BC, bh, 2 * BC : 3 * BC])
            T.copy(Ai_30_shared, Akk[bb, i_tc3 : i_tc3 + BC, bh, 0:BC])
            T.copy(Ai_31_shared, Akk[bb, i_tc3 : i_tc3 + BC, bh, BC : 2 * BC])
            T.copy(Ai_32_shared, Akk[bb, i_tc3 : i_tc3 + BC, bh, 2 * BC : 3 * BC])
            T.copy(Ai_33_shared, Akk[bb, i_tc3 : i_tc3 + BC, bh, 3 * BC : 4 * BC])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    scale,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    sub_chunk_size,
):
    q, k, gk, beta, Aqk, Akk_diag = prepare_input(
        B,
        S,
        H,
        DK,
        chunk_size,
        sub_chunk_size,
        getattr(torch, input_dtype),
        getattr(torch, output_dtype),
        getattr(torch, accum_dtype),
        getattr(torch, gate_dtype),
    )
    Aqk_ref = Aqk.clone()
    Akk_ref = prepare_output(B, S, H, chunk_size, sub_chunk_size, getattr(torch, output_dtype))
    chunk_kda_fwd_inter_solve_fused(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Aqk=Aqk_ref,
        Akk_diag=Akk_diag,
        Akk=Akk_ref,
        scale=scale,
    )
    Aqk_tilelang = Aqk.clone()
    Akk_tilelang = prepare_output(B, S, H, chunk_size, sub_chunk_size, getattr(torch, output_dtype))
    kernel = tilelang_chunk_kda_fwd_inter_fused(
        B, S, H, DK, input_dtype, output_dtype, accum_dtype, gate_dtype, chunk_size, sub_chunk_size, scale
    )
    Aqk_tilelang, Akk_tilelang = kernel(
        q,
        k,
        gk,
        beta,
        Akk_diag,
    )

    compare_tensors("Aqk", Aqk_ref, Aqk_tilelang)
    compare_tensors("Akk", Akk_ref, Akk_tilelang)
    fla_time = do_bench(
        chunk_kda_fwd_inter_solve_fused,
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Aqk=Aqk_ref,
        Akk_diag=Akk_diag,
        Akk=Akk_ref,
        scale=scale,
    )
    tilelang_time = do_bench(kernel, q, k, gk, beta, Akk_diag)
    print("fla_time:", fla_time)
    print("tilelang_time:", tilelang_time)


def main():
    run_test(
        B=1,
        S=1024 * 8,  # 32768
        H=64,
        DK=128,
        scale=1.0,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        chunk_size=64,
        sub_chunk_size=16,
    )


if __name__ == "__main__":
    main()
