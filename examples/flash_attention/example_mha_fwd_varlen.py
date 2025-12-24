# ruff: noqa
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
import argparse
from tilelang.profiler import do_bench
from tilelang.autotuner import set_autotune_inputs, autotune

import torch
from einops import rearrange, repeat
from varlen_utils import generate_random_padding_mask, generate_qkv
import itertools


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
    dim = q.shape[-1]
    scale = (1.0 / dim) ** 0.5  # log2(e)
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


def get_configs():
    iter_params = dict(block_M=[64, 128], block_N=[64, 128], num_stages=[0, 1, 2, 3], threads=[128, 256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs())
@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(batch_size, UQ, UKV, heads, dim, is_causal, block_M=64, block_N=64, num_stages=1, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [UQ, heads, dim]
    k_shape = [UKV, heads, dim]
    v_shape = [UKV, heads, dim]
    o_shape = [UQ, heads, dim]

    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(k_shape, dtype),
        V_unpad: T.Tensor(v_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        max_seqlen_q: T.int32,
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

            T.copy(
                Q_unpad[q_start_idx + bx * block_M : q_start_idx + bx * block_M + block_M, head_idx, :], Q_shared
            )  # OOB positions will be handled below

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(k_current_seqlen, block_N)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # Q * K
                T.copy(
                    K_unpad[k_start_idx + k * block_N : k_start_idx + k * block_N + block_N, head_idx, :], K_shared
                )  # OOB positions will be handled below
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i >= k * block_N + j)
                            and (bx * block_M + i >= q_current_seqlen or k * block_N + j >= k_current_seqlen),
                            -T.infinity(acc_s.dtype),
                            0,
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i >= q_current_seqlen or k * block_N + j >= k_current_seqlen), -T.infinity(acc_s.dtype), 0
                        )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Softmax
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
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
                T.copy(
                    V_unpad[v_start_idx + k * block_N : v_start_idx + k * block_N + block_N, head_idx, :], V_shared
                )  # OOB positions' weights are 0

                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared, Output_unpad[q_start_idx + bx * block_M : q_start_idx + bx * block_M + block_M, head_idx, :]
            )  # TMA will handle OOB

    return main


def main(batch: int = 8, heads: int = 64, seq_len: int = 2048, dim: int = 128, tune: bool = False):
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
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    UQ = q_unpad.shape[0]  # unpadded query length
    UK = k_unpad.shape[0]  # unpadded key length
    UKV = k_unpad.shape[0]  # unpadded query key length

    if tune:
        with set_autotune_inputs(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q):
            kernel = flashattn(batch, UQ, UKV, heads, dim, causal)
    else:
        kernel = flashattn(batch, UQ, UKV, heads, dim, causal, block_M=64, block_N=64, num_stages=1, threads=128)
        # NOTE: (128, 128, 2or3, 256) is recommended for Hopper

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
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)

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
    torch.testing.assert_close(out, fla_out, rtol=1e-2, atol=1e-2)

    print("All checks passed.✅")

    # benchmark
    t = do_bench(lambda: kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q))
    print(f"Tilelang time: {t} ms")
    print(f"Tilelang: {total_flops / t * 1e-9} TFlops")
    t = do_bench(
        lambda: flash_attn.flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 0.0, causal=causal
        )
    )
    print(f"FA2 time: {t} ms")
    print(f"FA2: {total_flops / t * 1e-9} TFlops")


def run_regression_perf(batch: int = 8, heads: int = 64, seq_len: int = 2048, dim: int = 128):
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
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    UQ = q_unpad.shape[0]
    UK = k_unpad.shape[0]
    UKV = k_unpad.shape[0]
    kernel = flashattn(batch, UQ, UKV, heads, dim, causal, block_M=128, block_N=128, num_stages=2, threads=256)

    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)

    return do_bench(run_kernel_only, warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="heads")
    parser.add_argument("--seq_len", type=int, default=2048, help="sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--tune", action="store_true", default=False, help="tune the kernel")

    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.tune)
