# ruff: noqa
import argparse
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from einops import rearrange, repeat
from tilelang.profiler import do_bench
from varlen_utils import generate_random_padding_mask, generate_qkv


def attention_ref(
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        causal=False,
        window_size=(-1, -1),
        upcast=True,
):
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    dim = q.shape[-1]
    scale = (1.0 / dim)**0.5
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    scores = torch.einsum("bthd,bshd->bhts", q, k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    scores = scores * scale
    attention = torch.softmax(scores, dim=-1).to(v.dtype)

    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


@tilelang.jit(
    out_idx=[6], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def flashattn(batch_size,
              groups,
              UQ,
              UKV,
              heads,
              dim,
              is_causal,
              block_M=64,
              block_N=64,
              num_stages=1,
              threads=128):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [UQ, heads, dim]
    kv_shape = [UKV, head_kv, dim]
    o_shape = [UQ, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
            Q_unpad: T.Tensor(q_shape, dtype),
            K_unpad: T.Tensor(kv_shape, dtype),
            V_unpad: T.Tensor(kv_shape, dtype),
            cu_seqlens_q: T.Tensor([batch_size + 1], "int32"),
            cu_seqlens_k: T.Tensor([batch_size + 1], "int32"),
            max_seqlen_q: T.int32,
            Output_unpad: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(
                T.ceildiv(max_seqlen_q, block_M), heads, batch_size,
                threads=threads) as (bx, by, bz):
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

            batch_idx = bz
            head_idx = by
            kv_head_idx = head_idx // groups

            q_start_idx = cu_seqlens_q[batch_idx]
            k_start_idx = cu_seqlens_k[batch_idx]
            v_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]
            v_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            k_current_seqlen = k_end_idx - k_start_idx
            v_current_seqlen = v_end_idx - v_start_idx

            T.copy(
                Q_unpad[q_start_idx + bx * block_M:q_start_idx + (bx + 1) * block_M, head_idx, :],
                Q_shared)
            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i >= q_current_seqlen:
                    Q_shared[i, d] = 0

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(k_current_seqlen, block_N)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    K_unpad[k_start_idx + k * block_N:k_start_idx + (k + 1) * block_N,
                            kv_head_idx, :], K_shared)
                for i, d in T.Parallel(block_N, dim):
                    if k * block_N + i >= k_current_seqlen:
                        K_shared[i, d] = 0

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else((bx * block_M + i >= k * block_N + j) and
                                                     (bx * block_M + i >= q_current_seqlen or
                                                      k * block_N + j >= k_current_seqlen),
                                                     -T.infinity(acc_s.dtype), 0)
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else((bx * block_M + i >= q_current_seqlen or
                                                      k * block_N + j >= k_current_seqlen),
                                                     -T.infinity(acc_s.dtype), 0)

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)

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

                T.copy(
                    V_unpad[v_start_idx + k * block_N:v_start_idx + (k + 1) * block_N,
                            kv_head_idx, :], V_shared)
                for i, d in T.Parallel(block_N, dim):
                    if k * block_N + i >= v_current_seqlen:
                        V_shared[i, d] = 0

                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)

            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i < q_current_seqlen:
                    Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

    return main


def main(batch: int = 1,
         heads: int = 64,
         q_seqlen: int = 2048,
         k_seqlen: int = 2048,
         dim: int = 128,
         groups: int = 16,
         is_causal: bool = False):
    assert heads % groups == 0, "heads must be divisible by groups"

    flops_per_matmul = 2.0 * batch * heads * q_seqlen * k_seqlen * dim
    total_flops = 2 * flops_per_matmul

    tilelang.testing.set_random_seed(0)

    causal = False
    if causal:
        total_flops *= 0.5

    tilelang.testing.set_random_seed(0)

    dtype = torch.float16
    device = torch.device("cuda")

    head_kv = heads // groups
    q = torch.randn(batch, q_seqlen, heads, dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch, k_seqlen, head_kv, dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch, k_seqlen, head_kv, dim, dtype=dtype, device=device, requires_grad=True)

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
    ) = generate_qkv(
        q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    UQ = q_unpad.shape[0]
    UKV = k_unpad.shape[0]

    kernel = flashattn(
        batch,
        groups,
        UQ,
        UKV,
        heads,
        dim,
        is_causal,
        block_M=64,
        block_N=64,
        num_stages=1,
        threads=128)

    out_unpad = kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
    out = output_pad_fn(out_unpad)

    out_ref, _ = attention_ref(
        q,
        k,
        v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=is_causal,
    )
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    print("All checks passed.✅")
    latency = do_bench(
        lambda: kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q))
    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=64, help='query heads')
    parser.add_argument('--groups', type=int, default=16, help='groups')
    parser.add_argument('--q_seqlen', type=int, default=2048, help='query sequence length')
    parser.add_argument('--k_seqlen', type=int, default=2048, help='key/value sequence length')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--is_causal', action='store_true', help='causal attention')
    args = parser.parse_args()
    main(args.batch, args.heads, args.q_seqlen, args.k_seqlen, args.dim, args.groups,
         args.is_causal)
