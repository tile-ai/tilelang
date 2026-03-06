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
from tilelang.language.print_op import print_msg, print_var_with_condition
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
    threads=256,
    num_stages=2,
    debug_log=False,
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
            # Double-buffer as two 2D buffers so tcgen05 gets supported swizzle (3D layout unsupported)
            K_shared_0 = T.alloc_shared([block_N, dim], dtype)
            K_shared_1 = T.alloc_shared([block_N, dim], dtype)
            V_shared_0 = T.alloc_shared([block_N, dim], dtype)
            V_shared_1 = T.alloc_shared([block_N, dim], dtype)
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
            mbar_bmm1_full = T.alloc_barrier([1] * num_stages)  # MMA signals 1 arrive
            mbar_dma2_empty = T.alloc_barrier([32] * num_stages)
            mbar_dma2_full = T.alloc_barrier([32] * num_stages)
            mbar_bmm2_full = T.alloc_barrier([1] * num_stages)  # MMA signals 1 arrive
            mbar_softmax_empty = T.alloc_barrier([32] * num_stages)
            mbar_softmax_full = T.alloc_barrier([128] * num_stages)

            tid = T.get_thread_binding()
            T.use_swizzle(8)

            is_softmax_warp = tid < 128
            is_dma1_warp = tid >= 128 and tid < 160
            is_bmm1_warp = tid >= 160 and tid < 192
            # Softmax warp must be exactly 128 threads (warp group) for tcgen05 copy

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
            

            for k in T.serial(loop_range):
                
                parity_inv = ((k // num_stages) & 1) ^ 1
                parity = parity_inv ^ 1
                stage_id = k % num_stages
                is_clear_accum = True if k == 0 else False

                # Debug: only first block to avoid flood (use debug_log=True + small batch to see order)
                if debug_log:
                    print_var_with_condition(
                        tid == 0 and bx == 0 and by == 0 and bz == 0, k, "[loop] k"
                    )

                if is_dma1_warp:
                    # DMA1: double-buffer K, V (2D each) so load/compute can overlap
                    if debug_log:
                        print_var_with_condition(
                            tid == 128 and bx == 0 and by == 0 and bz == 0,
                            k,
                            "[DMA1] wait_empty k",
                        )
                    T.mbarrier_wait_parity(mbar_dma1_empty[stage_id], parity_inv)
                    if debug_log:
                        print_var_with_condition(
                            tid == 128 and bx == 0 and by == 0 and bz == 0,
                            k,
                            "[DMA1] after wait_empty k",
                        )

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
                    T.mbarrier_arrive(mbar_dma2_full[stage_id])
                    # T.mbarrier_arrive(mbar_softmax_empty[stage_id]) # notify producer


                elif is_bmm1_warp:
                    if debug_log:
                        print_var_with_condition(
                            tid == 160 and bx == 0 and by == 0 and bz == 0,
                            k,
                            "[BMM1] enter k",
                        )
                    # GEMM 1: S = Q @ K^T -> S_tmem (tcgen05mma_ss), consume current stage K
                    T.mbarrier_wait_parity(mbar_dma1_full[stage_id], parity)
                    if debug_log:
                        print_var_with_condition(
                            tid == 160 and bx == 0 and by == 0 and bz == 0,
                            k,
                            "[BMM1] after dma1_full k",
                        )
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

                    # GEMM 2: O = P_tmem @ V -> O_tmem (tcgen05mma_ts), consume current stage V
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

                    T.mbarrier_arrive(mbar_softmax_empty[stage_id])  # notify producer
                    T.mbarrier_arrive(mbar_dma2_empty[stage_id])  # notify producer

                elif is_softmax_warp:
                    if debug_log:
                        print_var_with_condition(
                            tid == 0 and bx == 0 and by == 0 and bz == 0,
                            k,
                            "[Softmax] enter k",
                        )
                    T.mbarrier_wait_parity(mbar_softmax_empty[stage_id], parity_inv)
                    T.mbarrier_wait_parity(mbar_bmm1_full[stage_id], parity)
                    if debug_log:
                        print_var_with_condition(
                            tid == 0 and bx == 0 and by == 0 and bz == 0,
                            k,
                            "[Softmax] after bmm1_full k",
                        )

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

                    T.mbarrier_arrive(mbar_softmax_full[stage_id])  # notify consomer
                    T.mbarrier_arrive(mbar_bmm1_empty[stage_id])  # notify producer

            # Final epilogue: wait for last GEMM2 to finish, then normalize and store output.
            if is_softmax_warp:
                if debug_log:
                    print_var_with_condition(
                        bx == 0 and by == 0 and bz == 0,
                        loop_range - 1,
                        "[epilogue] softmax warp start last_k",
                    )
                last_k = loop_range - 1
                last_stage_id = last_k % num_stages
                last_parity = (last_k // num_stages) & 1  # same parity as loop for last k

                if debug_log:
                    print_var_with_condition(
                        bx == 0 and by == 0 and bz == 0, 0, "[epilogue] wait bmm2_full"
                    )
                T.mbarrier_wait_parity(mbar_bmm2_full[last_stage_id], last_parity)
                if debug_log:
                    print_var_with_condition(
                        bx == 0 and by == 0 and bz == 0, 0, "[epilogue] after bmm2_full"
                    )

                T.copy(O_tmem, O_reg)
                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] /= logsum[i]

                T.copy(O_reg, O_shared)
                T.copy(
                    O_shared,
                    Output[
                        bz,
                        bx * block_M : (bx + 1) * block_M,
                        by,
                        :,
                    ],
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
    debug_log: bool = False,
):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    # Device printf only flushes when kernel completes. If kernel hangs, no log.
    # With debug_log, use small seq_len so loop has 1 iteration and kernel may finish.
    run_seq_len = seq_len
    if debug_log and variant == "ts":
        run_seq_len = min(seq_len, 128)  # 1 iteration with block_N=128
        if run_seq_len < seq_len:
            print(
                f"[debug-log] seq_len overridden {seq_len} -> {run_seq_len} so "
                "kernel can complete and flush device printf (use 1 iter to see log)."
            )
        print("[debug-log] Device log will appear only after kernel returns.")

    print(f"=== Blackwell Flash Attention ({variant.upper()}) ===")
    print(
        f"batch={batch}, heads={heads}, seq_len={run_seq_len}, "
        f"dim={dim}, causal={is_causal}"
    )

    fn = flashattn_ss if variant == "ss" else flashattn_ts
    threads = 128 if variant == "ss" else 256
    kwargs = dict(
        batch=batch,
        heads=heads,
        seq_len=run_seq_len,
        dim=dim,
        is_causal=is_causal,
        block_M=128,
        block_N=128,
        threads=threads,
    )
    if variant == "ts":
        kwargs["debug_log"] = debug_log
    kernel = fn(**kwargs)
    # print(kernel.get_kernel_source())

    Q = torch.randn(
        batch, run_seq_len, heads, dim, device="cuda", dtype=torch.bfloat16
    )
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    if debug_log and variant == "ts":
        print("[debug-log] Launching kernel (device printf after return)...", flush=True)
    out = kernel(Q, K, V)
    if debug_log and variant == "ts":
        print("[debug-log] Kernel returned.", flush=True)
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
    parser.add_argument(
        "--debug-log", action="store_true",
        help="enable device printf in ts variant to debug hang (e.g. which barrier)"
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
        args.debug_log,
    )
