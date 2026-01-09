# ruff: noqa
# Using varlen (variable length) format with attention sink

import argparse
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.profiler import do_bench
from typing import Optional
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../flash_attention"))
from varlen_utils import generate_random_padding_mask, generate_qkv


@tilelang.jit(
    out_idx=[7],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_sink(
    batch_size,
    groups,
    UQ,
    UKV,
    heads,
    dim,
    is_causal,
    window_size=None,  # None for full causal attention
    sm_scale=None,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=128,
):
    if window_size is not None:
        assert window_size % block_N == 0, "window_size must be divisible by block_N"

    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [UQ, heads, dim]
    kv_shape = [UKV, head_kv, dim]
    o_shape = [UQ, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(kv_shape, dtype),
        V_unpad: T.Tensor(kv_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        max_seqlen_q: T.int32,
        Sinks: T.Tensor([heads], dtype),
        Output_unpad: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            sinks = T.alloc_fragment([block_M], dtype)

            batch_idx = bz
            head_idx = by
            kv_head_idx = head_idx // groups

            q_start_idx = cu_seqlens_q[batch_idx]
            kv_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = k_end_idx - kv_start_idx

            T.copy(Q_unpad[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, head_idx, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            for i in T.Parallel(block_M):
                sinks[i] = Sinks[head_idx]

            offset = kv_current_seqlen - q_current_seqlen  # always align on the right
            max_visible_k_idx = offset + (bx + 1) * block_M

            # Determine loop range based on causal mask and sliding window
            if is_causal:
                if window_size is not None:
                    # Sliding window + causal: start from window boundary
                    start = T.max(0, (offset + bx * block_M - window_size + 1) // block_N)
                    end = T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_current_seqlen, block_N))
                else:
                    # Full causal attention
                    start = 0
                    end = T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_current_seqlen, block_N))
            else:
                if window_size is not None:
                    start = T.max(0, (offset + bx * block_M - window_size + 1) // block_N)
                    end = T.ceildiv(kv_current_seqlen, block_N)
                else:
                    start = 0
                    end = T.ceildiv(kv_current_seqlen, block_N)

            loop_range = end - start

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                actual_k = k + start
                T.copy(K_unpad[kv_start_idx + actual_k * block_N : kv_start_idx + (actual_k + 1) * block_N, kv_head_idx, :], K_shared)

                # Build mask considering causal, sliding window, and padding
                if is_causal:
                    if window_size is not None:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i + offset
                            k_idx = actual_k * block_N + j
                            # Causal + sliding window mask
                            acc_s[i, j] = T.if_then_else(
                                (q_idx < k_idx)  # causal: can't see future
                                or (q_idx >= k_idx + window_size)  # sliding window: too old
                                or (bx * block_M + i >= q_current_seqlen or actual_k * block_N + j >= kv_current_seqlen),
                                -T.infinity(acc_s.dtype),
                                0,
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                (bx * block_M + i + offset < actual_k * block_N + j)
                                or (bx * block_M + i >= q_current_seqlen or actual_k * block_N + j >= kv_current_seqlen),
                                -T.infinity(acc_s.dtype),
                                0,
                            )
                else:
                    if window_size is not None:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i + offset
                            k_idx = actual_k * block_N + j
                            acc_s[i, j] = T.if_then_else(
                                (q_idx >= k_idx + window_size)  # sliding window: too old
                                or (bx * block_M + i >= q_current_seqlen or actual_k * block_N + j >= kv_current_seqlen),
                                -T.infinity(acc_s.dtype),
                                0,
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                (bx * block_M + i >= q_current_seqlen or actual_k * block_N + j >= kv_current_seqlen),
                                -T.infinity(acc_s.dtype),
                                0,
                            )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                # Check_inf for sliding window attention
                if window_size is not None:
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V_unpad[kv_start_idx + actual_k * block_N : kv_start_idx + (actual_k + 1) * block_N, kv_head_idx, :], V_shared)

                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Attention sink: add sink contribution to logsum
            for i in T.Parallel(block_M):
                logsum[i] += T.exp2(sinks[i] * 1.44269504 - scores_max[i] * scale)

            for i, j in T.Parallel(block_M, dim):
                # When sq > skv, some tokens can see nothing (for causal)
                acc_o[i, j] = 0 if is_causal and bx * block_M + i + offset < 0 else acc_o[i, j] / logsum[i]

            T.copy(acc_o, O_shared)
            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i < q_current_seqlen:
                    Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

    return main


def ref_program(
    q_unpad: torch.Tensor,
    k_unpad: torch.Tensor,
    v_unpad: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sinks: torch.Tensor,
    batch_size: int,
    is_causal: bool,
    sliding_window: Optional[int] = None,
    groups: int = 1,
) -> torch.Tensor:
    """Reference implementation for varlen attention with sinks."""
    # q_unpad: [total_q, heads, dim]
    # k_unpad: [total_kv, head_kv, dim]
    # v_unpad: [total_kv, head_kv, dim]
    total_q, num_heads, head_dim = q_unpad.shape
    _, num_key_value_heads, _ = k_unpad.shape

    sm_scale = 1.0 / head_dim**0.5

    output = torch.zeros_like(q_unpad)

    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()

        q_len = q_end - q_start
        k_len = k_end - k_start

        if q_len == 0:
            continue

        # Extract sequences for this batch
        q_seq = q_unpad[q_start:q_end]  # [q_len, heads, dim]
        k_seq = k_unpad[k_start:k_end]  # [k_len, head_kv, dim]
        v_seq = v_unpad[k_start:k_end]  # [k_len, head_kv, dim]

        # Reshape for GQA
        q_seq = q_seq.view(q_len, num_key_value_heads, groups, head_dim)  # [q_len, head_kv, groups, dim]
        sinks_expanded = sinks.view(num_key_value_heads, groups, 1, 1).float()  # [head_kv, groups, 1, 1]

        k_seq = k_seq.unsqueeze(2)  # [k_len, head_kv, 1, dim]
        v_seq = v_seq.unsqueeze(2)  # [k_len, head_kv, 1, dim]

        # Compute attention
        # q_seq: [q_len, head_kv, groups, dim], k_seq: [k_len, head_kv, 1, dim]
        logits = torch.einsum("qhgd,khgd->hgqk", q_seq.float(), k_seq.float()) * sm_scale

        # Build mask
        start_q = k_len - q_len  # offset for causal alignment
        pos_keys = torch.arange(k_len, device=q_unpad.device)
        pos_queries = torch.arange(q_len, device=q_unpad.device) + start_q

        if is_causal:
            mask = pos_keys[None, :] > pos_queries[:, None]
            mask = mask.float().masked_fill(mask, float("-inf"))
        else:
            mask = torch.zeros(q_len, k_len, device=q_unpad.device)

        if sliding_window is not None:
            too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
            mask.masked_fill_(too_old, float("-inf"))

        logits = logits + mask[None, None, :, :]  # [head_kv, groups, q_len, k_len]

        # Apply sink-adjusted softmax
        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        logits_or_sinks_max = torch.maximum(sinks_expanded, logits_max)
        sinks_exp = torch.exp(sinks_expanded - logits_or_sinks_max)
        unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks_exp
        scores = unnormalized_scores / normalizer

        # Compute output
        out = torch.einsum("hgqk,khgd->qhgd", scores, v_seq.float())
        out = out.reshape(q_len, num_heads, head_dim).to(q_unpad.dtype)

        output[q_start:q_end] = out

    return output


def main(
    batch: int = 1,
    heads: int = 64,
    q_seqlen: int = 2048,
    k_seqlen: int = 2048,
    dim: int = 128,
    groups: int = 16,
    is_causal: bool = True,
    window_size: Optional[int] = None,
):
    assert heads % groups == 0, "heads must be divisible by groups"

    flops_per_matmul = 2.0 * batch * heads * q_seqlen * k_seqlen * dim
    total_flops = 2 * flops_per_matmul

    tilelang.testing.set_random_seed(0)

    if is_causal:
        total_flops *= 0.5

    if window_size is not None:
        print(f"Using sliding window attention with window_size={window_size}")
        flops_per_matmul = 2.0 * batch * heads * min(window_size, k_seqlen // 2) * q_seqlen * dim
        total_flops = 2 * flops_per_matmul

    dtype = torch.float16
    device = torch.device("cuda")

    head_kv = heads // groups
    q = torch.randn(batch, q_seqlen, heads, dim, dtype=dtype, device=device)
    k = torch.randn(batch, k_seqlen, head_kv, dim, dtype=dtype, device=device)
    v = torch.randn(batch, k_seqlen, head_kv, dim, dtype=dtype, device=device)
    sinks = torch.randn(heads, dtype=dtype, device=device)

    query_padding_mask = generate_random_padding_mask(q_seqlen, batch, device, mode="random")
    key_padding_mask = generate_random_padding_mask(k_seqlen, batch, device, mode="random")

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        _,
        _,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    UQ = q_unpad.shape[0]
    UKV = k_unpad.shape[0]

    kernel = flashattn_sink(
        batch, groups, UQ, UKV, heads, dim, is_causal, window_size=window_size, block_M=128, block_N=128, num_stages=2, threads=256
    )

    out_unpad = kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, sinks)
    out = output_pad_fn(out_unpad)

    # Reference implementation
    ref_out_unpad = ref_program(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sinks,
        batch,
        is_causal,
        sliding_window=window_size,
        groups=groups,
    )
    ref_out = output_pad_fn(ref_out_unpad)

    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)

    print("All checks passed.✅")
    latency = do_bench(
        lambda: kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, sinks),
        warmup=500,
    )
    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="query heads")
    parser.add_argument("--groups", type=int, default=16, help="groups")
    parser.add_argument("--q_seqlen", type=int, default=2048, help="query sequence length")
    parser.add_argument("--k_seqlen", type=int, default=2048, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=128, help="head dim")
    parser.add_argument("--is_causal", action="store_true", help="causal attention")
    parser.add_argument("--window_size", type=int, default=None, help="sliding window size (default: None for full attention)")
    args = parser.parse_args()
    main(args.batch, args.heads, args.q_seqlen, args.k_seqlen, args.dim, args.groups, args.is_causal, args.window_size)
