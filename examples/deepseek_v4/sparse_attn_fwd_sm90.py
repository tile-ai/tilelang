"""DSV4-style MQA Sparse Attention Kernel (BSHD layout).

- Partial RoPE, instead of extra 64-dim RoPE
- Support for attention sink
- Designed for sm90

NOTE: This impl is simply an illustration and maybe not optimal,
or comparable to handwrite version
"""

# TODO(wt): Support window size

import math
import torch
import tilelang
from tilelang.profiler import do_bench
import tilelang.language as T
import argparse


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def sparse_attn_fwd(
    batch,
    heads,
    seq_len,
    seq_len_kv,
    dim,
    topk,
    sm_scale=None,
    block_N=64,
    num_stages=2,
    threads=256,
    dtype: T.dtype = T.bfloat16,
):
    assert topk % block_N == 0, "topk must be divisible by block_N for now"

    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504  # log2(e)

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, dim]
    idx_shape = [batch, seq_len, topk]
    o_shape = [batch, seq_len, heads, dim]
    accum_dtype = T.float32

    BI = block_N
    NI = T.ceildiv(topk, BI)

    if heads > 64:
        assert heads % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = heads // 64
    else:
        REPLICATE_H = 1

    H_per_block = 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        TopkIndices: T.Tensor(idx_shape, T.int32),
        Output: T.Tensor(o_shape, dtype),
        Sinks: T.Tensor([heads], dtype),
    ):
        with T.Kernel(REPLICATE_H, seq_len, batch, threads=threads) as (h_block, seq_idx, by):
            h_start = h_block * H_per_block

            Q_shared = T.alloc_shared([H_per_block, dim], dtype)
            KV_shared = T.alloc_shared([BI, dim], dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            Sinks_shared = T.alloc_shared([H_per_block], dtype)

            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            acc_o = T.alloc_fragment([H_per_block, dim], accum_dtype)
            acc_o_shared = T.alloc_shared([H_per_block, dim], dtype)
            scores_max = T.alloc_fragment([H_per_block], accum_dtype)
            scores_max_prev = T.alloc_fragment([H_per_block], accum_dtype)
            scores_scale = T.alloc_fragment([H_per_block], accum_dtype)
            scores_sum = T.alloc_fragment([H_per_block], accum_dtype)
            logsum = T.alloc_fragment([H_per_block], accum_dtype)
            mask = T.alloc_fragment([BI], T.bool)

            T.copy(Q[by, seq_idx, h_start : h_start + H_per_block, :], Q_shared)
            T.copy(Sinks[h_start : h_start + H_per_block], Sinks_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -(2**30))  # avoid -inf - inf to cause nan

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                # Valid mask: skip padding indices (-1)
                for bi_i in T.Parallel(BI):
                    idx = TopkIndices[by, seq_idx, i_i * BI + bi_i]
                    mask[bi_i] = idx >= 0

                # Gather KV block using indices
                for bi_i, d_i in T.Parallel(BI, dim):
                    idx = TopkIndices[by, seq_idx, i_i * BI + bi_i]
                    KV_shared[bi_i, d_i] = KV[by, idx, d_i]

                # Initialize scores with mask
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))

                # QK^T GEMM: (H_per_block, dim) @ (dim, BI) -> (H_per_block, BI)
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # Online softmax with exp2
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    scores_max[h_i] = T.max(scores_max[h_i], scores_max_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    scores_scale[h_i] = T.exp2(scores_max_prev[h_i] * scale - scores_max[h_i] * scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * scale - scores_max[h_i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)

                # Rescale acc_o by correction factor
                for h_i, d_i in T.Parallel(H_per_block, dim):
                    acc_o[h_i, d_i] *= scores_scale[h_i]

                # Accumulate: P @ V
                T.copy(acc_s, S_shared)
                T.gemm(
                    S_shared,
                    KV_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # Update logsum
                for h_i in T.Parallel(H_per_block):
                    logsum[h_i] = logsum[h_i] * scores_scale[h_i] + scores_sum[h_i]

            # Attention sink (per-head)
            for h_i in T.Parallel(H_per_block):
                logsum[h_i] += T.exp2(Sinks_shared[h_i] * 1.44269504 - scores_max[h_i] * scale)

            # Normalize
            for h_i, d_i in T.Parallel(H_per_block, dim):
                acc_o[h_i, d_i] /= logsum[h_i]

            # Store output
            T.copy(acc_o, acc_o_shared)
            T.copy(acc_o_shared, Output[by, seq_idx, h_start : h_start + H_per_block, :])

    return main


def torch_sparse_attention(
    q: torch.Tensor,  # [b, s, h, d]
    kv: torch.Tensor,  # [b, skv, d]
    attn_sink: torch.Tensor,  # [h]
    topk_idxs: torch.Tensor,  # [b, s, topk] int32
    softmax_scale: float,
) -> torch.Tensor:
    """Reference: gather KV by indices, compute attention with sink."""
    batch, seq_len, heads, dim = q.shape
    topk = topk_idxs.shape[-1]

    # Expand KV to [b, s, skv, d] and gather by indices -> [b, s, topk, d]
    kv_expanded = kv[:, None, :, :].expand(batch, seq_len, -1, dim)
    idx_expanded = topk_idxs[:, :, :, None].expand(batch, seq_len, topk, dim).long()
    gathered_kv = torch.gather(kv_expanded, 2, idx_expanded)

    # Scores: [b, s, h, topk]
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered_kv.float()) * softmax_scale

    # Concatenate sink and softmax
    sink = attn_sink[None, None, :, None].expand(batch, seq_len, heads, 1)
    attn = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)

    # Output: [b, s, h, d]
    out = torch.einsum("bmht,bmtd->bmhd", attn[:, :, :, :-1], gathered_kv.float())
    return out.to(q.dtype)


def test_correctness(
    BATCH: int = 1,
    H: int = 8,
    N_CTX: int = 512,
    N_KV: int = 1024,
    D_HEAD: int = 128,
    TOPK: int = 256,
    dtype_str: str = "bfloat16",
):
    T_dtype = T.dtype(dtype_str)
    torch_dtype = T_dtype.as_torch()

    torch.manual_seed(42)
    Q = torch.randn(BATCH, N_CTX, H, D_HEAD, dtype=torch_dtype, device="cuda")
    KV = torch.randn(BATCH, N_KV, D_HEAD, dtype=torch_dtype, device="cuda")
    attn_sink = torch.randn(H, dtype=torch_dtype, device="cuda")
    topk_idxs = torch.randint(0, N_KV, (BATCH, N_CTX, TOPK), dtype=torch.int32, device="cuda")

    scale = 1.0 / math.sqrt(D_HEAD)

    kernel = sparse_attn_fwd(BATCH, H, N_CTX, N_KV, D_HEAD, TOPK, dtype=T_dtype)
    print(kernel.get_kernel_source())

    # TileLang forward
    O_tl = kernel(Q, KV, topk_idxs, attn_sink)

    # PyTorch reference
    O_ref = torch_sparse_attention(Q, KV, attn_sink, topk_idxs, scale)

    rtol, atol = 1e-2, 1e-2
    max_err = (O_tl - O_ref).abs().max().item()
    assert torch.allclose(O_tl, O_ref, rtol=rtol, atol=atol), f"O max err: {max_err}"
    print(f"[{dtype_str}] Correctness OK, max error: {max_err:.6f}")


def benchmark_fwd(
    BATCH: int = 1,
    H: int = 8,
    N_CTX: int = 512,
    N_KV: int = 1024,
    D_HEAD: int = 128,
    TOPK: int = 256,
    dtype_str: str = "bfloat16",
):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]

    with torch.no_grad():
        torch.manual_seed(42)
        Q = torch.randn(BATCH, N_CTX, H, D_HEAD, dtype=torch_dtype, device="cuda")
        KV = torch.randn(BATCH, N_KV, D_HEAD, dtype=torch_dtype, device="cuda")
        attn_sink = torch.randn(H, dtype=torch_dtype, device="cuda")
        topk_idxs = torch.randint(0, N_KV, (BATCH, N_CTX, TOPK), dtype=torch.int32, device="cuda")

        kernel = sparse_attn_fwd(BATCH, H, N_CTX, N_KV, D_HEAD, TOPK, dtype=dtype_str)
        latency_tl = do_bench(lambda: kernel(Q, KV, topk_idxs, attn_sink))

        flops = 4.0 * BATCH * N_CTX * H * TOPK * D_HEAD
        print(f"[{dtype_str}] B={BATCH}, S={N_CTX}, SKV={N_KV}, H={H}, D={D_HEAD}, TOPK={TOPK}")
        print(f"  tilelang: {latency_tl:.3f} ms  ({flops / latency_tl * 1e-9:.2f} TFlops)")

        return latency_tl


def main(
    BATCH: int = 1,
    H: int = 8,
    N_CTX: int = 1024,
    N_KV: int = 1024,
    D_HEAD: int = 64,
    TOPK: int = 256,
    dtype_str: str = "bfloat16",
):
    test_correctness(BATCH, H, N_CTX, N_KV, D_HEAD, TOPK, dtype_str)
    benchmark_fwd(BATCH, H, N_CTX, N_KV, D_HEAD, TOPK, dtype_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--n_ctx", type=int, default=4096)
    parser.add_argument("--n_kv", type=int, default=8192)
    parser.add_argument("--d_head", type=int, default=512)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()
    main(args.batch, args.h, args.n_ctx, args.n_kv, args.d_head, args.topk, args.dtype)
