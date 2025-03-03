# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ruff: noqa
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
import argparse

import torch
from einops import rearrange, repeat
from flash_attn.bert_padding import pad_input, unpad_input


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths)
    return padding_mask


def generate_qkv(q,
                 k,
                 v,
                 query_padding_mask=None,
                 key_padding_mask=None,
                 kvpacked=False,
                 qkvpacked=False):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(output_unpad, indices_q, batch_size, seqlen_q
                                                      )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device)
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size)

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device)
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k if key_padding_mask is None else rearrange(
            key_padding_mask.sum(-1), "b -> b 1 1 1"))
    sq = (
        seqlen_q if query_padding_mask is None else rearrange(
            query_padding_mask.sum(-1), "b -> b 1 1 1"))
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite window size
        upcast=True,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    scale = (1.0 / dim)**0.5  # log2(e)
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    scores = torch.einsum("bthd,bshd->bhts", q, k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
        # scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0)
    scores = scores * scale
    attention = torch.softmax(scores, dim=-1).to(v.dtype)

    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def flashattn(batch_size, UQ, UKV, heads, dim, is_causal):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    q_shape = [UQ, heads, dim]
    k_shape = [UKV, heads, dim]
    v_shape = [UKV, heads, dim]
    o_shape = [UQ, heads, dim]
    block_M = 64
    block_N = 64
    num_stages = 0
    threads = 32

    dtype = "float16"
    accum_dtype = "float"

    def kernel_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def main(
                Q_unpad: T.Buffer(q_shape, dtype),
                K_unpad: T.Buffer(k_shape, dtype),
                V_unpad: T.Buffer(v_shape, dtype),
                cu_seqlens_q: T.Buffer([batch_size + 1], "int32"),
                cu_seqlens_k: T.Buffer([batch_size + 1], "int32"),
                max_seqlen_q: T.int32,
                Output_unpad: T.Buffer(o_shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(max_seqlen_q, block_M), heads, batch_size,
                    threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype, "shared")
                K_shared = T.alloc_shared([block_N, dim], dtype, "shared")
                V_shared = T.alloc_shared([block_N, dim], dtype, "shared")
                O_shared = T.alloc_shared([block_M, dim], dtype, "shared")
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

                q_start_idx = cu_seqlens_q[batch_idx]
                k_start_idx = cu_seqlens_k[batch_idx]
                v_start_idx = cu_seqlens_k[batch_idx]
                q_end_idx = cu_seqlens_q[batch_idx + 1]
                k_end_idx = cu_seqlens_k[batch_idx + 1]
                v_end_idx = cu_seqlens_k[batch_idx + 1]

                q_current_seqlen = q_end_idx - q_start_idx
                k_current_seqlen = k_end_idx - k_start_idx
                v_current_seqlen = v_end_idx - v_start_idx

                for i, d in T.Parallel(block_M, dim):
                    if bx * block_M + i < q_current_seqlen:
                        Q_shared[i, d] = Q_unpad[q_start_idx + bx * block_M + i, head_idx, d]
                    else:
                        Q_shared[i, d] = 0

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv(k_current_seqlen, block_N)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    # Q * K
                    for i, d in T.Parallel(block_N, dim):
                        if k * block_N + i < k_current_seqlen:
                            K_shared[i, d] = K_unpad[k_start_idx + k * block_N + i, head_idx, d]
                        else:
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

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)

                    # Softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    # To do causal softmax, we need to set the scores_max to 0 if it is -inf
                    # This process is called Check_inf in FlashAttention3 code, and it only need to be done
                    # in the first ceil_div(kBlockM, kBlockN) steps.
                    # for i in T.Parallel(block_M):
                    #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                        # max * log_2(e)) This allows the compiler to use the ffma
                        # instruction instead of fadd and fmul separately.
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    # Rescale
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]

                    # V * softmax(Q * K)
                    for i, d in T.grid(block_N, dim):
                        if k * block_N + i < v_current_seqlen:
                            V_shared[i, d] = V_unpad[v_start_idx + k * block_N + i, head_idx, d]
                        else:
                            V_shared[i, d] = 0

                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)

                for i, d in T.Parallel(block_M, dim):
                    if bx * block_M + i < q_current_seqlen:
                        Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

        return main

    return kernel_func(block_M, block_N, num_stages, threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=2, help='batch size')
    parser.add_argument('--heads', type=int, default=16, help='heads')
    parser.add_argument('--seq_len', type=int, default=256, help='sequence length')
    parser.add_argument('--dim', type=int, default=32, help='dim')

    args = parser.parse_args()
    batch, heads, seq_len, dim = args.batch, args.heads, args.seq_len, args.dim
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul

    tilelang.testing.set_random_seed(0)

    causal = False
    if causal:
        total_flops *= 0.5

    dtype = torch.float16
    device = torch.device("cuda")
    window_size = (-1, -1)

    q = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)
    k = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)
    v = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)

    query_padding_mask = generate_random_padding_mask(seq_len, batch, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seq_len, batch, device, mode="random")
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
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(
        q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    UQ = q_unpad.shape[0]  # unpadded query length
    UK = k_unpad.shape[0]  # unpadded key length
    UKV = k_unpad.shape[0]  # unpadded query key length

    program = flashattn(batch, UQ, UKV, heads, dim, causal)
    kernel = tilelang.compile(program, out_idx=-1, execution_backend="cython")
    print(kernel.get_kernel_source())

    out_unpad = kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
    out = output_pad_fn(out_unpad)

    out_ref, _ = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
    )
    import flash_attn
    fla_out_unpad = flash_attn.flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        causal=causal,
    )
    fla_out = output_pad_fn(fla_out_unpad)
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    print("Assert Equal Passed")
