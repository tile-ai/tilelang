"""Blackwell (SM100) GQA forward, BSHD layout.

Q: [batch, seq_len, heads, dim], K/V: [batch, seq_len, head_kv, dim], head_kv = heads // groups.
Pipeline (default): --variant ss.
ts (optional): --variant ts (256 threads, single-path ts).
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
    groups=1,
    block_M=128,
    block_N=128,
    threads=128,
):
    """GQA forward, pipeline (ss): both GEMMs mma_ss. K/V indexed by head_kv = by // groups."""
    head_kv = heads // groups
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
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
                    K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared
                )

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
                    V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared
                )

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
    groups=1,
    block_M=128,
    block_N=128,
    threads=256,
):
    """GQA forward, warp (ts): GEMM 2 uses mma_ts. K/V indexed by by // groups."""
    head_kv = heads // groups
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
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
                    K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared
                )

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
                T.copy(P_cast, P_tmem)

                T.copy(
                    V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared
                )

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


flashattn_warp = flashattn_ts


def ref_program(Q, K, V, is_causal, groups=1):
    """CPU reference: K/V [b,s,head_kv,d], expand to heads for einsum."""
    assert Q.size(2) == K.size(2) * groups
    dim = Q.size(-1)
    K_f = K.cpu().float().repeat_interleave(groups, dim=2)
    V_f = V.cpu().float().repeat_interleave(groups, dim=2)
    Q_f = Q.cpu().float()
    scores = torch.einsum("bqhd,bkhd->bhqk", Q_f, K_f)
    scores = scores / (dim**0.5)
    if is_causal:
        seq_len = Q_f.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    P = F.softmax(scores, dim=-1)
    O = torch.einsum("bhqk,bkhd->bqhd", P, V_f)
    return O.to(Q.dtype)


def main(
    batch: int = 2,
    heads: int = 4,
    seq_len: int = 256,
    dim: int = 128,
    is_causal: bool = False,
    groups: int = 1,
    variant: str = "ss",
):
    head_kv = heads // groups
    assert heads % groups == 0
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    print(f"=== Blackwell GQA Forward ({variant.upper()}) ===")
    print(
        f"batch={batch}, heads={heads}, head_kv={head_kv}, groups={groups}, "
        f"seq_len={seq_len}, dim={dim}, causal={is_causal}"
    )

    fn = flashattn_ss if variant == "ss" else flashattn_ts
    threads = 128 if variant == "ss" else 256  # ts: 256
    kernel = fn(
        batch, heads, seq_len, dim, is_causal, groups=groups,
        block_M=128, block_N=128, threads=threads,
    )

    Q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(batch, seq_len, head_kv, dim, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(batch, seq_len, head_kv, dim, device="cuda", dtype=torch.bfloat16)

    out = kernel(Q, K, V)
    ref = ref_program(Q, K, V, is_causal, groups).to(out.device)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("Correctness check passed.")

    latency = do_bench(lambda: kernel(Q, K, V), warmup=100)
    print(f"Blackwell GQA fwd ({variant}): {latency:.2f} ms")
    print(f"Blackwell GQA fwd ({variant}): {total_flops / latency * 1e-9:.2f} TFlops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--groups", type=int, default=1, help="GQA: head_kv = heads // groups")
    parser.add_argument(
        "--variant", choices=["ss", "ts"], default="ss",
        help="ss: pipeline (default); ts: 256 threads",
    )
    args = parser.parse_args()
    main(
        args.batch, args.heads, args.seq_len, args.dim,
        args.is_causal, args.groups, args.variant,
    )
