"""Blackwell (SM100) MHA backward, BSHD layout.

Pipeline (default): --variant ss or default.
ts (optional): --variant ts (256 threads, 2 stages).
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import argparse


PASS_CFG = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}


@tilelang.jit(out_idx=[3, 4], pass_configs=PASS_CFG)
def flashattn_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    """Forward to get O and LSE (for backward)."""
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
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N)
            )
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.tcgen05_gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s, acc_s_cast)
                T.tcgen05_gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return main


@tilelang.jit(out_idx=[2], pass_configs=PASS_CFG)
def flashattn_bwd_preprocess(batch, heads, seq_len, dim):
    dtype = T.bfloat16
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 32

    @T.prim_func
    def main(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                T.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return main


def make_dq_layout(dQ):
    return T.Layout(
        dQ.shape,
        lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2],
    )


@tilelang.jit(out_idx=[1], pass_configs=PASS_CFG)
def flashattn_bwd_postprocess(batch, heads, seq_len, dim):
    dtype = T.bfloat16
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 64

    @T.prim_func
    def main(
        dQ: T.Tensor(shape, accum_dtype),
        dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk : (bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk : (bx + 1) * blk, by, :],
            )

    return main


@tilelang.jit(pass_configs=PASS_CFG)
def flashattn_bwd(batch, heads, seq_len, dim, is_causal, block_M, block_N, threads=128, num_stages=2):
    """Blackwell MHA backward. Pipeline default (128, 2); ts = (256, 2)."""
    sm_scale = (1.0 / dim) ** 0.5
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(shape, accum_dtype),
        dK: T.Tensor(shape, dtype),
        dV: T.Tensor(shape, dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_M, dim], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim], dtype)
            dk_shared = T.alloc_shared([block_M, dim], dtype)

            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(K[bz, by * block_M : (by + 1) * block_M, bx, :], K_shared)
            T.copy(V[bz, by * block_M : (by + 1) * block_M, bx, :], V_shared)
            T.clear(dv)
            T.clear(dk)
            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.tcgen05_gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j, qkT[i, j], 0)
                T.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                T.clear(dsT)
                T.tcgen05_gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.tcgen05_gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.tcgen05_gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.tcgen05_gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim):
                    T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, by * block_M : (by + 1) * block_M, bx, :])
            T.copy(dk_shared, dK[bz, by * block_M : (by + 1) * block_M, bx, :])

    return main


def flashattn_bwd_pipeline(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    """Pipeline (default): 128 threads, 2 stages."""
    return flashattn_bwd(batch, heads, seq_len, dim, is_causal, block_M, block_N, threads=128, num_stages=2)


def flashattn_bwd_warp(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    """ts: 256 threads, 2 stages. Use --variant ts to enable."""
    return flashattn_bwd(batch, heads, seq_len, dim, is_causal, block_M, block_N, threads=256, num_stages=2)


def ref_program(Q, K, V, is_causal):
    """CPU reference forward (for validation); backward ref not implemented."""
    dim = Q.size(-1)
    Q_f = Q.cpu().float()
    K_f = K.cpu().float()
    V_f = V.cpu().float()
    scores = torch.einsum("bqhd,bkhd->bhqk", Q_f, K_f)
    scores = scores / (dim**0.5)
    if is_causal:
        seq_len = Q_f.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    P = F.softmax(scores, dim=-1)
    out_ref = torch.einsum("bhqk,bkhd->bqhd", P, V_f)
    return out_ref.to(Q.dtype)


def main(
    batch: int = 2,
    heads: int = 4,
    seq_len: int = 256,
    dim: int = 128,
    is_causal: bool = False,
    variant: str = "ss",
):
    """Run MHA backward kernels (fwd + preprocess + bwd + postprocess)."""
    block_M = 64
    block_N = 64 if dim <= 64 else 32
    use_ts = variant == "ts"
    bwd_fn = flashattn_bwd_warp if use_ts else flashattn_bwd_pipeline

    kernel_fwd = flashattn_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N)
    kernel_prep = flashattn_bwd_preprocess(batch, heads, seq_len, dim)
    kernel_post = flashattn_bwd_postprocess(batch, heads, seq_len, dim)
    kernel_bwd = bwd_fn(batch, heads, seq_len, dim, is_causal, block_M, block_N)

    Q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    dO = torch.randn_like(Q)
    O, lse = kernel_fwd(Q, K, V)
    Delta = kernel_prep(O, dO)
    dQ = torch.zeros(batch, seq_len, heads, dim, device="cuda", dtype=torch.float32)
    dK = torch.empty_like(K, device="cuda")
    dV = torch.empty_like(V, device="cuda")
    kernel_bwd(Q, K, V, dO, lse, Delta, dQ, dK, dV)
    _ = kernel_post(dQ)  # dQ_out in output layout; not compared to ref (no backward ref)
    print("Blackwell MHA bwd ({}): run OK (backward gradients not verified against ref).".format(variant))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument(
        "--variant",
        choices=["ss", "ts"],
        default="ss",
        help="ss: pipeline (default); ts: 256 threads",
    )
    args = parser.parse_args()
    main(
        args.batch,
        args.heads,
        args.seq_len,
        args.dim,
        args.is_causal,
        args.variant,
    )
