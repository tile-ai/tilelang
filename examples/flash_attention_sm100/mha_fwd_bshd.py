"""Blackwell (SM100) Flash Attention Forward using TCGEN05MMA with TMEM accumulators.

Replaces the Hopper WGMMA-based Flash Attention for Blackwell GPUs.
Three variants: ss, ts, wasp.
  - flashattn (variant='ss'): Both GEMMs use mma_ss (shared x shared -> TMEM), 128 threads.
  - flashattn (variant='ts'): Single-path; GEMM 2 uses mma_ts (P_tmem x V_shared -> D_tmem), 256 threads.
  - flashattn_wasp: Warp-specialized pipeline (softmax/DMA/BMM warps); GEMM 2 mma_ts.
    If wasp fails (e.g. layout inference), fallback to ts.
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import argparse


PASS_CFG = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=128,
    block_N=128,
    variant="ss",
):
    """Flash Attention forward. variant='ss': mma_ss (128t, P via shared); 'ts': mma_ts (256t, P via TMEM)."""
    use_ts = variant == "ts"
    threads = 256 if use_ts else 128
    scale = (1.0 / dim) ** 0.5 * 1.44269504
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
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            D_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            mbar_s = T.alloc_barrier(1)
            mbar_d = T.alloc_barrier(1)

            if use_ts:
                P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            else:
                P_shared = T.alloc_shared([block_M, block_N], dtype)

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
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

                # GEMM 1: S = Q @ K^T -> S_tmem (tcgen05mma_ss)
                T.tcgen05_gemm(
                    Q_shared,
                    K_shared,
                    S_tmem,
                    transpose_B=True,
                    mbar=mbar_s,
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
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    S_reg[i, j] = T.exp2(S_reg[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(S_reg, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] *= scores_scale[i]

                T.copy(S_reg, P_cast)
                if use_ts:
                    T.copy(P_cast, P_tmem)
                    P_operand = P_tmem
                else:
                    T.copy(P_cast, P_shared)
                    P_operand = P_shared

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)

                # GEMM 2: D = P @ V -> D_tmem (ss: mma_ss; ts: mma_ts)
                T.tcgen05_gemm(
                    P_operand,
                    V_shared,
                    D_tmem,
                    mbar=mbar_d,
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


flashattn_ss = flashattn
flashattn_ts = flashattn


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
    """Warp-specialized pipeline: softmax(0-127)/DMA(128-159)/BMM(160-191); GEMM2 mma_ts."""
    scale = (1.0 / dim) ** 0.5 * 1.44269504
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
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
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
            mbar_bmm1_empty = T.alloc_barrier([128] * num_stages)
            mbar_bmm1_full = T.alloc_barrier([1] * num_stages)
            mbar_dma2_empty = T.alloc_barrier([32] * num_stages)
            mbar_dma2_full = T.alloc_barrier([32] * num_stages)
            mbar_bmm2_full = T.alloc_barrier([1] * num_stages)
            mbar_softmax_empty = T.alloc_barrier([32] * num_stages)
            mbar_softmax_full = T.alloc_barrier([128] * num_stages)
            mbar_correction_full = T.alloc_barrier([32] * num_stages)

            tid = T.get_thread_binding()

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_rescale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            if tid < 128:
                T.fill(O_reg, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.copy(O_reg, O_tmem)

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.serial(loop_range):
                parity = (k // num_stages) & 1
                parity_inv = parity ^ 1
                stage_id = k % num_stages
                is_clear_accum = k == 0

                if tid >= 128 and tid < 160:
                    T.mbarrier_wait_parity(mbar_dma1_empty[stage_id], parity_inv)

                    if k == 0:
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

                    if stage_id == 0:
                        T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared_0)
                    else:
                        T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared_1)
                    T.mbarrier_arrive(mbar_dma1_full[stage_id])

                    T.mbarrier_wait_parity(mbar_dma2_empty[stage_id], parity_inv)

                    if stage_id == 0:
                        T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared_0)
                    else:
                        T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared_1)

                    T.mbarrier_arrive(mbar_dma2_full[stage_id])

                elif tid >= 160 and tid < 192:
                    T.mbarrier_wait_parity(mbar_dma1_full[stage_id], parity)
                    T.mbarrier_wait_parity(mbar_bmm1_empty[stage_id], parity_inv)

                    if stage_id == 0:
                        T.tcgen05_gemm(
                            Q_shared,
                            K_shared_0,
                            S_tmem,
                            transpose_B=True,
                            mbar=mbar_bmm1_full[stage_id],
                            clear_accum=True,
                        )
                    else:
                        T.tcgen05_gemm(
                            Q_shared,
                            K_shared_1,
                            S_tmem,
                            transpose_B=True,
                            mbar=mbar_bmm1_full[stage_id],
                            clear_accum=True,
                        )
                    T.mbarrier_arrive(mbar_dma1_empty[stage_id])

                    T.mbarrier_wait_parity(mbar_softmax_full[stage_id], parity)
                    T.mbarrier_wait_parity(mbar_dma2_full[stage_id], parity)

                    if stage_id == 0:
                        T.tcgen05_gemm(
                            P_tmem,
                            V_shared_0,
                            O_tmem,
                            mbar=mbar_bmm2_full[stage_id],
                            clear_accum=is_clear_accum,
                        )
                    else:
                        T.tcgen05_gemm(
                            P_tmem,
                            V_shared_1,
                            O_tmem,
                            mbar=mbar_bmm2_full[stage_id],
                            clear_accum=is_clear_accum,
                        )

                    T.mbarrier_arrive(mbar_softmax_empty[stage_id])
                    T.mbarrier_arrive(mbar_dma2_empty[stage_id])

                    if k == loop_range - 1:
                        T.mbarrier_arrive(mbar_correction_full[0])

                elif tid < 128:
                    T.mbarrier_wait_parity(mbar_softmax_empty[stage_id], parity_inv)
                    T.mbarrier_wait_parity(mbar_bmm1_full[stage_id], parity)
                    if k > 0:
                        prev_stage = (k - 1) % num_stages
                        prev_parity = ((k - 1) // num_stages) & 1
                        T.mbarrier_wait_parity(mbar_bmm2_full[prev_stage], prev_parity)

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

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(S_reg, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_M):
                        scores_rescale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.exp2(S_reg[i, j] * scale - scores_max[i] * scale)

                    T.reduce_sum(S_reg, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_rescale[i] + scores_sum[i]

                    for i, j in T.Parallel(block_M, dim):
                        O_reg[i, j] *= scores_rescale[i]

                    T.copy(S_reg, P_cast)
                    T.copy(P_cast, P_tmem)
                    T.copy(O_reg, O_tmem)

                    T.mbarrier_arrive(mbar_softmax_full[stage_id])
                    T.mbarrier_arrive(mbar_bmm1_empty[stage_id])

                    if k == loop_range - 1:
                        T.mbarrier_wait_parity(mbar_correction_full[0], 0)
                        T.mbarrier_wait_parity(mbar_bmm2_full[stage_id], parity)
                        T.copy(O_tmem, O_reg)
                        for i, j in T.Parallel(block_M, dim):
                            O_reg[i, j] /= logsum[i]
                        T.copy(O_reg, O_shared)
                        T.copy(
                            O_shared,
                            Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
                        )

    return main


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
    """Run MHA forward kernel (ss / ts / wasp) and benchmark."""
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    print(f"=== Blackwell Flash Attention ({variant.upper()}) ===")
    print(f"batch={batch}, heads={heads}, seq_len={seq_len}, dim={dim}, causal={is_causal}")

    if variant in ("ss", "ts"):
        kernel = flashattn(
            batch,
            heads,
            seq_len,
            dim,
            is_causal,
            block_M=128,
            block_N=128,
            variant=variant,
        )
    else:
        kernel = flashattn_wasp(
            batch,
            heads,
            seq_len,
            dim,
            is_causal,
            block_M=128,
            block_N=128,
            threads=256,
            num_stages=2,
        )

    Q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    out = kernel(Q, K, V)
    ref = ref_program(Q, K, V, is_causal).to(out.device)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("Correctness check passed.")

    latency = do_bench(lambda: kernel(Q, K, V), warmup=100)
    print(f"Blackwell ({variant}): {latency:.2f} ms")
    print(f"Blackwell ({variant}): {total_flops / latency * 1e-9:.2f} TFlops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument(
        "--variant",
        choices=["ss", "ts", "wasp"],
        default="wasp",
        help="ss: pipeline 128t; ts: single-path 256t mma_ts; wasp: warp-specialized (fallback to ts if fail)",
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
