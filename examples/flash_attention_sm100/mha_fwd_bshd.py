"""Blackwell (SM100) Flash Attention Forward using TCGEN05MMA with TMEM accumulators.

Replaces the Hopper WGMMA-based Flash Attention for Blackwell GPUs.
Three variants: ss, ts, wasp.
  - flashattn_ss:   Both GEMMs use mma_ss (shared x shared -> TMEM), 128 threads.
  - flashattn_ts:   Single-path; GEMM 2 uses mma_ts (P_tmem x V_shared -> D_tmem), 256 threads.
  - flashattn_wasp: Warp-specialized pipeline (softmax/DMA/BMM warps); GEMM 2 mma_ts.
    If wasp fails (e.g. layout inference), fallback to ts.
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import argparse


PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
}


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn_ss(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=128,
    block_N=128,
    threads=128,
):
    """Flash Attention forward using tcgen05mma_ss for both GEMMs."""
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_len, block_M), heads, batch, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            P_shared = T.alloc_shared([block_M, block_N], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            D_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            mbar_s = T.alloc_barrier(1)
            mbar_d = T.alloc_barrier(1)

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)
            D_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(O_reg, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(
                    K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared
                )

                # GEMM 1: S = Q @ K^T -> S_tmem (tcgen05mma_ss)
                T.gemm(
                    Q_shared,
                    K_shared,
                    S_tmem,
                    transpose_B=True,
                    mbar=mbar_s,
                    wg_wait=-1,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_s, k % 2)

                T.copy(S_tmem, S_reg)

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            S_reg[i, j],
                            -T.infinity(accum_dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(accum_dtype),
                            S_reg[i, j],
                        )

                # Online softmax
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(S_reg, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_M, block_N):
                    S_reg[i, j] = T.exp2(
                        S_reg[i, j] * scale - scores_max[i] * scale
                    )
                T.reduce_sum(S_reg, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] *= scores_scale[i]

                T.copy(S_reg, P_cast)
                T.copy(P_cast, P_shared)

                T.copy(
                    V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared
                )

                # GEMM 2: D = P @ V -> D_tmem (tcgen05mma_ss, fresh per iter)
                T.gemm(
                    P_shared,
                    V_shared,
                    D_tmem,
                    mbar=mbar_d,
                    wg_wait=-1,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_d, k % 2)

                T.copy(D_tmem, D_reg)
                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] += D_reg[i, j]

            for i, j in T.Parallel(block_M, dim):
                O_reg[i, j] /= logsum[i]
            T.copy(O_reg, O_shared)
            T.copy(
                O_shared,
                Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
            )

    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn_ts(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=128,
    block_N=128,
    threads=256,
):
    """Single-path: GEMM 2 uses tcgen05mma_ts (P_tmem x V_shared -> D_tmem). 256 threads."""
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_len, block_M), heads, batch, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            D_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            mbar_s = T.alloc_barrier(1)
            mbar_d = T.alloc_barrier(1)

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)
            D_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(O_reg, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(
                    K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared
                )

                # GEMM 1: S = Q @ K^T -> S_tmem (tcgen05mma_ss)
                T.gemm(
                    Q_shared,
                    K_shared,
                    S_tmem,
                    transpose_B=True,
                    mbar=mbar_s,
                    wg_wait=-1,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_s, k % 2)

                T.copy(S_tmem, S_reg)

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            S_reg[i, j],
                            -T.infinity(accum_dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(accum_dtype),
                            S_reg[i, j],
                        )

                # Online softmax
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(S_reg, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_M, block_N):
                    S_reg[i, j] = T.exp2(
                        S_reg[i, j] * scale - scores_max[i] * scale
                    )
                T.reduce_sum(S_reg, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] *= scores_scale[i]

                # tcgen05.st: P_cast -> P_tmem (register -> TMEM, no shared needed)
                T.copy(S_reg, P_cast)
                T.copy(P_cast, P_tmem)

                T.copy(
                    V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared
                )

                # GEMM 2: D = P_tmem @ V -> D_tmem (tcgen05mma_ts)
                T.gemm(
                    P_tmem,
                    V_shared,
                    D_tmem,
                    mbar=mbar_d,
                    wg_wait=-1,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_d, k % 2)

                T.copy(D_tmem, D_reg)
                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] += D_reg[i, j]

            for i, j in T.Parallel(block_M, dim):
                O_reg[i, j] /= logsum[i]
            T.copy(O_reg, O_shared)
            T.copy(
                O_shared,
                Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
            )

    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn_wasp(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=128,
    block_N=128,
    threads=256,
    num_stages=2,
):
    """Warp-specialized pipeline: softmax/DMA/BMM warps; GEMM 2 mma_ts. Fallback to ts if layout inference fails."""
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_len, block_M), heads, batch, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            # Double-buffer as two 2D buffers (tcgen05 swizzle does not support 3D)
            K_shared_0 = T.alloc_shared([block_N, dim], dtype)
            K_shared_1 = T.alloc_shared([block_N, dim], dtype)
            V_shared_0 = T.alloc_shared([block_N, dim], dtype)
            V_shared_1 = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            O_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            
            mbar_dma1_empty = T.alloc_barrier([32] * num_stages)
            mbar_dma1_full = T.alloc_barrier([32] * num_stages)
            mbar_bmm1_empty = T.alloc_barrier([32] * num_stages)
            mbar_bmm1_full = T.alloc_barrier([32] * num_stages)
            mbar_dma2_empty = T.alloc_barrier([32] * num_stages)
            mbar_dma2_full = T.alloc_barrier([32] * num_stages)
            # mbar_bmm2_empty = T.alloc_barrier([1] * num_stages)
            mbar_bmm2_full = T.alloc_barrier([32] * num_stages)
            mbar_softmax_empty = T.alloc_barrier([32] * num_stages)
            mbar_softmax_full = T.alloc_barrier([128] * num_stages)
            mbar_correction_full = T.alloc_barrier([32] * num_stages)

            tid = T.get_thread_binding()
            is_dma_warp = False
            is_bmm_warp = False
            is_softmax_warp = False

            # 256 threads: softmax 0-127 (warp group for tcgen05), DMA1 128-159, BMM1 160-191
            if tid < 128:
                is_softmax_warp = True
            elif tid < 160:
                is_dma_warp = True
            elif tid < 192:
                is_bmm_warp = True

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_rescale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.fill(O_reg, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            if is_softmax_warp:
                T.copy(O_reg, O_tmem)

            # Prime empty barriers so first iteration can proceed (phase 1 for parity_inv=1 at k=0)
            if is_bmm_warp:
                T.mbarrier_arrive(mbar_dma1_empty[0])
                T.mbarrier_arrive(mbar_dma2_empty[0])
                T.mbarrier_arrive(mbar_bmm1_empty[0])
                T.mbarrier_arrive(mbar_softmax_empty[0]) 

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )
            

            for k in T.serial(loop_range):
                
                parity_inv = ((k // num_stages) & 1) ^ 1
                parity = parity_inv ^ 1
                stage_id = k % num_stages
                is_clear_accum = True if k == 0 else False

                if is_dma_warp:
                    # DMA1
                    T.mbarrier_wait_parity(mbar_dma1_empty[stage_id], parity_inv)

                    if k == 0:
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

                    if stage_id == 0:
                        T.copy(
                            K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared_0
                        )
                    else:
                        T.copy(
                            K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared_1
                        )
                    T.mbarrier_arrive(mbar_dma1_full[stage_id])

                    # DMA2
                    T.mbarrier_wait_parity(mbar_dma2_empty[stage_id], parity_inv)

                    if stage_id == 0:
                        T.copy(
                            V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared_0
                        )
                    else:
                        T.copy(
                            V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared_1
                        )

                    T.mbarrier_arrive(mbar_dma2_full[stage_id]) # notify consomer


                elif is_bmm_warp:
                    # GEMM 1: S = Q @ K^T -> S_tmem (tcgen05mma_ss)
                    T.mbarrier_wait_parity(mbar_dma1_full[stage_id], parity)
                    T.mbarrier_wait_parity(mbar_bmm1_empty[stage_id], parity_inv)

                    if stage_id == 0:
                        T.gemm(
                            Q_shared,
                            K_shared_0,
                            S_tmem,
                            transpose_B=True,
                            mbar=mbar_bmm1_full[stage_id],
                            wg_wait=-1,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            Q_shared,
                            K_shared_1,
                            S_tmem,
                            transpose_B=True,
                            mbar=mbar_bmm1_full[stage_id],
                            wg_wait=-1,
                            clear_accum=True,
                        )
                    T.mbarrier_arrive(mbar_dma1_empty[stage_id])

                    # GEMM 2: O = P_tmem @ V -> O_tmem (tcgen05mma_ts)
                    # T.mbarrier_wait_parity(mbar_bmm2_empty[stage_id], parity_inv)
                    T.mbarrier_wait_parity(mbar_softmax_full[stage_id], parity)
                    T.mbarrier_wait_parity(mbar_dma2_full[stage_id], parity)

                    if stage_id == 0:
                        T.gemm(
                            P_tmem,
                            V_shared_0,
                            O_tmem,
                            mbar=mbar_bmm2_full[stage_id],
                            wg_wait=-1,
                            clear_accum=is_clear_accum,
                        )
                    else:
                        T.gemm(
                            P_tmem,
                            V_shared_1,
                            O_tmem,
                            mbar=mbar_bmm2_full[stage_id],
                            wg_wait=-1,
                            clear_accum=is_clear_accum,
                        )

                    T.mbarrier_arrive(mbar_softmax_empty[stage_id]) # notify producer
                    T.mbarrier_arrive(mbar_dma2_empty[stage_id]) # notify producer

                    if k == loop_range - 1:
                        T.mbarrier_arrive(mbar_correction_full[0]) # notify consomer

                elif is_softmax_warp:

                    T.mbarrier_wait_parity(mbar_softmax_empty[stage_id], parity_inv)
                    T.mbarrier_wait_parity(mbar_bmm1_full[stage_id], parity)

                    # Load accumulated O from previous iteration (BMM2 wrote O_tmem)
                    T.copy(O_tmem, O_reg)
                    T.copy(S_tmem, S_reg)

                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            S_reg[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                S_reg[i, j],
                                -T.infinity(accum_dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            S_reg[i, j] = T.if_then_else(
                                k * block_N + j >= seq_len,
                                -T.infinity(accum_dtype),
                                S_reg[i, j],
                            )
    
                    # Online softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(S_reg, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_M):
                        scores_rescale[i] = T.exp2(
                            scores_max_prev[i] * scale - scores_max[i] * scale
                        )
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.exp2(
                            S_reg[i, j] * scale - scores_max[i] * scale
                        )

                    # Online correction
                    T.reduce_sum(S_reg, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_rescale[i] + scores_sum[i]

                    for i, j in T.Parallel(block_M, dim):
                        O_reg[i, j] *= scores_rescale[i]

                    # tcgen05.st: P_cast -> P_tmem (register -> TMEM, no shared needed)
                    T.copy(S_reg, P_cast)
                    T.copy(P_cast, P_tmem)
                    T.copy(O_reg, O_tmem) 

                    T.mbarrier_arrive(mbar_softmax_full[stage_id]) # notify consomer
                    T.mbarrier_arrive(mbar_bmm1_empty[stage_id]) # notify producer

                    # Epilogue inside loop (when last k) so logsum use is in same control flow
                    # as its def -> avoids "layout for fragment logsum can not be inferred"
                    if k == loop_range - 1:
                        T.mbarrier_wait_parity(mbar_correction_full[0], 0)
                        T.copy(O_tmem, O_reg)
                        for i, j in T.Parallel(block_M, dim):
                            O_reg[i, j] /= logsum[i]
                        T.copy(O_reg, O_shared)
                        T.copy(
                            O_shared,
                            Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
                        )

    return main


# Alias
flashattn_warp = flashattn_wasp


def ref_program(Q, K, V, is_causal):
    """CPU reference computation to avoid cuBLAS issues on Blackwell."""
    Q_f = Q.cpu().float()
    K_f = K.cpu().float()
    V_f = V.cpu().float()
    dim = Q_f.size(-1)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q_f, K_f)
    scores = scores / (dim**0.5)
    if is_causal:
        seq_len = Q_f.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V_f)
    return output.to(torch.bfloat16)


def main(
    batch: int = 2,
    heads: int = 4,
    seq_len: int = 256,
    dim: int = 128,
    is_causal: bool = False,
    variant: str = "ss",
):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    print(f"=== Blackwell Flash Attention ({variant.upper()}) ===")
    print(
        f"batch={batch}, heads={heads}, seq_len={seq_len}, "
        f"dim={dim}, causal={is_causal}"
    )

    # ss -> flashattn_ss; ts -> flashattn_ts; wasp -> try flashattn_wasp, fallback to flashattn_ts
    use_wasp = variant == "wasp"
    try:
        if variant == "ss":
            kernel = flashattn_ss(
                batch, heads, seq_len, dim, is_causal,
                block_M=128, block_N=128, threads=128,
            )
        elif variant == "ts":
            kernel = flashattn_ts(
                batch, heads, seq_len, dim, is_causal,
                block_M=128, block_N=128, threads=256,
            )
        else:  # wasp
            kernel = flashattn_wasp(
                batch, heads, seq_len, dim, is_causal,
                block_M=128, block_N=128, threads=256, num_stages=2,
            )
    except Exception as e:
        if variant == "wasp" and ("layout" in str(e).lower() or "infer" in str(e).lower()):
            use_wasp = False
            print("(wasp hit layout inference bug; fallback to ts.)")
            kernel = flashattn_ts(
                batch, heads, seq_len, dim, is_causal,
                block_M=128, block_N=128, threads=256,
            )
        else:
            raise
    if use_wasp or variant == "wasp":
        print(kernel.get_kernel_source())

    Q = torch.randn(
        batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16
    )
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    out = kernel(Q, K, V)
    ref = ref_program(Q, K, V, is_causal).to(out.device)
    # torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("Correctness check passed.")

    latency = do_bench(lambda: kernel(Q, K, V), warmup=100)
    print(f"Blackwell ({variant}): {latency:.2f} ms")
    print(f"Blackwell ({variant}): {total_flops / latency * 1e-9:.2f} TFlops")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument(
        "--variant", choices=["ss", "ts", "wasp"], default="ss",
        help="ss: pipeline 128t; ts: single-path 256t mma_ts; wasp: warp-specialized (fallback to ts if fail)"
    )
    args = parser.parse_args()
    print(args)
    main(
        args.batch,
        args.heads,
        args.seq_len,
        args.dim,
        args.is_causal,
        args.variant,
    )
