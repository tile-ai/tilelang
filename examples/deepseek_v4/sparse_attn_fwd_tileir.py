"""DeepSeek V4 MQA sparse attention with a TileIR-specific schedule.

For shapes (H=64, D=512, topk=512), B200 measurements favor
H_per_block=64, num_ctas=1, and num_worker_warps=8 with the other defaults.
Use --heads 64 --topk 512 --head-tile 64 --num-ctas 1 --worker-warps 8
with the command-line example; set --seq-len 512 for the DSV4 Flash config
reference point.

Might need to autotune for other shape.

"""

import argparse
import math

import torch

import tilelang
import tilelang.language as T
from tilelang.tileir.checks import check_tileir_available


@tilelang.jit(
    out_idx=[3],
    target="tileir",
    execution_backend="tileir",
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_TILEIR_OPT_LEVEL: 3,
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
    H_per_block=128,
    num_ctas=2,
    num_worker_warps=4,
    occupancy=24,
):
    assert min(batch, heads, seq_len, seq_len_kv, dim, topk, block_N, H_per_block) > 0
    assert topk % block_N == 0, "topk must be divisible by block_N"
    assert all(value & (value - 1) == 0 for value in (dim, block_N, H_per_block)), "Tile dimensions must be powers of two"
    assert dtype in (T.float16, T.bfloat16), "Only float16 and bfloat16 are supported"

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

    assert heads % H_per_block == 0
    REPLICATE_H = heads // H_per_block

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        TopkIndices: T.Tensor(idx_shape, T.int32),
        Output: T.Tensor(o_shape, dtype),
        Sinks: T.Tensor([heads], dtype),
    ):
        with T.Kernel(
            REPLICATE_H, seq_len, batch, threads=threads, num_ctas=num_ctas, num_worker_warps=num_worker_warps, occupancy=occupancy
        ) as (h_block, seq_idx, by):
            h_start = h_block * H_per_block

            Indices_shared = T.alloc_shared([BI], T.int32)
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
                # Stage indices once for the mask and KV gather. Disabling TMA
                # avoids multicast atoms below 128 bytes with multi-CTA clusters.
                T.copy(TopkIndices[by, seq_idx, i_i * BI : (i_i + 1) * BI], Indices_shared, disable_tma=True)
                # Valid mask: skip padding indices (-1)
                for bi_i in T.Parallel(BI):
                    idx = Indices_shared[bi_i]
                    mask[bi_i] = idx >= 0
                    Indices_shared[bi_i] = T.max(idx, 0)

                # Reuse row-wise clamped indices instead of clamping a full
                # BI x dim pointer tile. The mask excludes padding from softmax.
                for bi_i, d_i in T.Parallel(BI, dim):
                    idx = Indices_shared[bi_i]
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


def check_output(output, q, kv, indices, sinks, positions=None):
    """Check complete query rows against an independent float32 reference.

    Small tests check every row; the production benchmark samples three rows
    to avoid materializing a sequence-wide gathered KV tensor.
    """
    assert torch.isfinite(output).all(), "Sparse attention output must be finite"
    if positions is None:
        positions = ((b, s) for b in range(q.shape[0]) for s in range(q.shape[1]))
    for b, s in positions:
        selected_indices = indices[b, s].long()
        selected_kv = kv[b, selected_indices.clamp_min(0)].float()
        scores = q[b, s].float() @ selected_kv.T / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(selected_indices[None, :] < 0, -torch.inf)
        weights = torch.softmax(torch.cat((scores, sinks.float()[:, None]), dim=1), dim=1)[:, :-1]
        expected = (weights @ selected_kv).to(q.dtype)
        torch.testing.assert_close(output[b, s], expected, rtol=1e-2, atol=1e-2)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--seq-len-kv", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--head-tile", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-ctas", type=int, default=2)
    parser.add_argument("--worker-warps", type=int, default=4)
    parser.add_argument("--occupancy", type=int, default=24)
    args = parser.parse_args()
    check_tileir_available()
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    q = torch.randn(args.batch, args.seq_len, args.heads, args.dim, device="cuda", dtype=dtype)
    kv = torch.randn(args.batch, args.seq_len_kv, args.dim, device="cuda", dtype=dtype)
    indices = torch.randint(0, args.seq_len_kv, (args.batch, args.seq_len, args.topk), device="cuda", dtype=torch.int32)
    sinks = torch.randn(args.heads, device="cuda", dtype=dtype)
    kernel = sparse_attn_fwd(
        args.batch,
        args.heads,
        args.seq_len,
        args.seq_len_kv,
        args.dim,
        args.topk,
        dtype=T.dtype(args.dtype),
        H_per_block=args.head_tile,
        block_N=args.block_n,
        num_ctas=args.num_ctas,
        num_worker_warps=args.worker_warps,
        occupancy=args.occupancy,
    )
    output = kernel(q, kv, indices, sinks)
    positions = {(0, 0), (args.batch // 2, args.seq_len // 2), (args.batch - 1, args.seq_len - 1)}
    check_output(output, q, kv, indices, sinks, positions)
    latency = kernel.get_profiler().do_bench(input_tensors=[q, kv, indices, sinks], n_warmup=5, n_repeat=50, backend="cupti")
    print(f"{torch.cuda.get_device_name()}: {latency:.3f} ms (TileIR, sampled reference passed)")


if __name__ == "__main__":
    main()
