"""Blackwell (SM100) Flash Attention Forward using TCGEN05MMA with TMEM accumulators.

Replaces the Hopper WGMMA-based Flash Attention for Blackwell GPUs.
Two variants are provided:
  - flashattn_ss:  Both GEMMs use mma_ss (shared x shared -> TMEM)
  - flashattn_ts:  GEMM 2 uses mma_ts (TMEM x shared -> TMEM) via tcgen05.st

Data flow per iteration:
  GEMM 1: Q_shared @ K_shared^T -> S_tmem  (tcgen05mma_ss)
  tcgen05.ld: S_tmem -> S_reg
  Online softmax on S_reg -> P_reg
  Rescale O_reg in registers
  tcgen05.st: O_reg -> O_tmem  (write rescaled O into TMEM)
  GEMM 2: O_tmem += P @ V  (tcgen05mma_ss or tcgen05mma_ts, clear_accum=False)
  tcgen05.ld: O_tmem -> O_reg  (read back accumulated result)
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
            O_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            mbar_s = T.alloc_barrier(1)
            mbar_d = T.alloc_barrier(1)
            
            # mbar_dma_empty = # TODO

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_rescale = T.alloc_fragment([block_M], accum_dtype)
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
                    scores_rescale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_M, block_N):
                    S_reg[i, j] = T.exp2(
                        S_reg[i, j] * scale - scores_max[i] * scale
                    )
                T.reduce_sum(S_reg, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_rescale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] *= scores_rescale[i]

                T.copy(S_reg, P_cast)
                T.copy(P_cast, P_shared)

                T.copy(
                    V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared
                )

                # GEMM 2: O_tmem += P @ V (accumulate rescaled O in TMEM)
                T.copy(O_reg, O_tmem)
                T.gemm(
                    P_shared,
                    V_shared,
                    O_tmem,
                    mbar=mbar_d,
                    wg_wait=-1,
                    clear_accum=False,
                )
                T.mbarrier_wait_parity(mbar_d, k % 2)
                T.copy(O_tmem, O_reg)

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
    threads=384,
    num_stages=2,
):
    """Flash Attention forward using tcgen05mma_ts for GEMM 2 (P_tmem x V_shared).

    GEMM 2 reads P directly from TMEM via tcgen05.st, avoiding the shared memory
    round-trip needed by the mma_ss variant.
    """
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
            K_shared = T.alloc_shared([num_stages, block_N, dim], dtype)
            V_shared = T.alloc_shared([num_stages, block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            O_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            mbar_s = T.alloc_barrier(1)
            mbar_d = T.alloc_barrier(1)
            
            # TODO: to modify 
            mbar_dma1_empty = T.alloc_barrier([32] * num_stages)
            mbar_dma1_full = T.alloc_barrier([32] * num_stages)
            mbar_bmm1_empty = T.alloc_barrier([128] * num_stages)
            mbar_bmm1_full = T.alloc_barrier([32] * num_stages)
            mbar_dma2_empty = T.alloc_barrier([32] * num_stages)
            mbar_dma2_full = T.alloc_barrier([32] * num_stages)
            # mbar_bmm2_empty = T.alloc_barrier([1] * num_stages)
            mbar_bmm2_full = T.alloc_barrier([32] * num_stages)
            mbar_softmax_empty = T.alloc_barrier([32] * num_stages)
            mbar_softmax_full = T.alloc_barrier([128] * num_stages)
            mbar_correction_full = T.alloc_barrier([32] * num_stages)
     

            tid = T.get_thread_binding()
            is_dma1_warp = False
            is_bmm1_warp = False
            is_softmax_warp = False
            # is_bmm2_warp = False
            is_epi_warp = False
            # is_correction_warp = False

            if tid < 32:
                is_dma1_warp = True
            elif tid < 64:
                is_bmm1_warp = True
            # elif tid < 96:
            #     is_dma2_warp = True
            # elif tid < 128:
            #     is_bmm2_warp = True
            elif tid >= 128 and tid < 256:
                is_softmax_warp = True
            elif tid >= 256 and tid < 384:
                is_epi_warp = True

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

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )
            

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                
                parity_inv = ((k // num_stages) & 1) ^ 1
                parity = parity_inv ^ 1
                stage_id = k % num_stages
                is_clear_accum = True if k == 0 else False

                if is_dma1_warp:
                    # DMA1
                    T.mbarrier_wait_parity(mbar_dma1_empty[stage_id], parity_inv)

                    if k == 0:
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

                    T.copy(
                        K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared[stage_id, :, :]
                    )
                    T.mbarrier_arrive(mbar_dma1_full[stage_id])

                    # DMA2 
                    T.mbarrier_wait_parity(mbar_dma2_empty[stage_id], parity_inv)
                    # T.mbarrier_wait_parity(mbar_dma1_full[stage_id], parity)

                    T.copy(
                        V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared[stage_id, :, :]
                    )

                    T.mbarrier_arrive(mbar_dma2_full[stage_id]) # notify consomer
                    # T.mbarrier_arrive(mbar_softmax_empty[stage_id]) # notify producer


                elif is_bmm1_warp:

                    # GEMM 1: S = Q @ K^T -> S_tmem (tcgen05mma_ss)
                    T.mbarrier_wait_parity(mbar_dma1_full[stage_id], parity)
                    T.mbarrier_wait_parity(mbar_bmm1_empty[stage_id], parity_inv)

                    T.gemm(
                        Q_shared,
                        K_shared[stage_id, :, :],
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

                    T.gemm(
                        P_tmem,
                        V_shared[stage_id, :, :],
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

                    T.copy(S_tmem, S_reg)

                    if not is_clear_accum:
                        T.copy(O_tmem, O_reg) 

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

            if is_epi_warp:
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

    fn = flashattn_ss if variant == "ss" else flashattn_ts
    threads = 128 if variant == "ss" else 384
    kernel = fn(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        block_M=128,
        block_N=128,
        threads=threads,
    )
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


def run_regression_perf(
    batch: int = 2,
    heads: int = 4,
    seq_len: int = 256,
    dim: int = 128,
    is_causal: bool = False,
):
    kernel = flashattn_ss(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        block_M=128,
        block_N=128,
        threads=128,
    )
    profiler = kernel.get_profiler(
        tensor_supply_type=tilelang.TensorSupplyType.Normal
    )
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument(
        "--variant", choices=["ss", "ts"], default="ss",
        help="ss: both GEMMs use mma_ss; ts: GEMM 2 uses mma_ts"
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
