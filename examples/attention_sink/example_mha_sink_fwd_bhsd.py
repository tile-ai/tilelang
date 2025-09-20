# Adapted from tilelang/examples/flash_attn/example_mha_fwd_bhsd.py
# Add support for sliding window attention and sink tokens

import tilelang
import tilelang.language as T
import torch
from tilelang.autotuner import *
import itertools
import argparse
from functools import partial


def get_configs():
    iter_params = dict(block_M=[64], block_N=[64], num_stages=[0, 1, 2, 3], threads=[128, 256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[3], debug_root_path='/home/wt/tilelang/debug')
def mha_with_sink(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        window_size=None,  # None means full attn
        block_M=64,
        block_N=64,
        num_stages=1,
        threads=128):

    assert window_size is None, "Sliding window attention is not supported yet"
    #TODO: support sliding window attention

    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.macro
    def MMA0(
        K: T.Tensor(kv_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        past_len = seq_kv - seq_q
        T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
        for i, j in T.Parallel(block_M, block_N):
            q_idx = bx * block_M + i + past_len
            k_idx = k * block_N + j
            if window_size is not None:
                acc_s[i, j] = T.if_then_else(q_idx >= k_idx and q_idx - window_size + 1 <= k_idx, 0,
                                             -T.infinity(acc_s.dtype))
            else:
                acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        V_shared: T.SharedBuffer([block_M, dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            scores_max: T.FragmentBuffer([block_M], accum_dtype),
            scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
            scores_sum: T.FragmentBuffer([block_M], accum_dtype),
            logsum: T.FragmentBuffer([block_M], accum_dtype),
            sinks: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], sinks[i])
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
            sinks[i] = scores_max[i] * scale  # Note that sinks also need to be scaled
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
            Sinks: T.Tensor([heads], dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
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
            sinks = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            for i in T.Parallel(block_M):
                sinks[i] = Sinks[by]

            loop_range = T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M, block_N))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum,
                        logsum, sinks)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= (logsum[i] + T.exp2(sinks[i] - scores_max[i]))
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

    return main


# Modified from https://github.com/openai/gpt-oss/blob/main/gpt_oss/triton/attention.py
def attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,  # for BLOCK_M=64, BLOCK_N=64
    sliding_window: int | None = None,
    start_q: torch.LongTensor = 0,
):
    batch_size, num_key_value_heads, num_keys, head_dim = key.shape
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    num_queries = query.shape[2]
    query = query.transpose(1, 2).contiguous().view(batch_size, num_queries, num_key_value_heads,
                                                    -1, head_dim)
    num_key_value_groups = query.shape[3]

    sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)
    sinks = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
    scores = unnormalized_scores / normalizer

    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups,
                            head_dim).bfloat16().transpose(1, 2).contiguous()
    return output


def main(
    batch: int = 1,
    heads: int = 1,
    seq_q: int = 256,
    seq_kv: int = 256,
    dim: int = 64,
    window_size: int | None = None,
    tune: bool = False,
):
    if window_size is not None:
        print('Using sliding window attention.')
        assert window_size <= seq_q
        assert window_size <= seq_kv
        flops_per_matmul = 2.0 * batch * heads * window_size * seq_kv * dim  # rough estimation
    else:
        print('Using full attention.')
        flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim * 0.5
    total_flops = 2 * flops_per_matmul

    if tune:
        kernel = mha_with_sink(batch, heads, seq_q, seq_kv, dim, window_size)
        print(f"Best config: {kernel.config}")
    else:
        kernel = mha_with_sink(
            batch,
            heads,
            seq_q,
            seq_kv,
            dim,
            window_size,
            block_M=64,
            block_N=64,
            num_stages=1,
            threads=128)

    ref = partial(attention_ref, sliding_window=window_size)

    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref, rtol=0.01, atol=0.01)
    print("All checks pass.✅")

    latency = profiler.do_bench(ref, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = profiler.do_bench(warmup=500)
    print("Tile-lang: {:.4f} ms".format(latency))
    print("Tile-lang: {:.4f} TFlops".format(total_flops / latency * 1e-9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--seq_q', type=int, default=8192, help='query sequence length')
    parser.add_argument('--seq_kv', type=int, default=8192, help='key/value sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument(
        '--window_size',
        type=int,
        default=None,
        help='window size(default: None), None means full attn')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()

    main(args.batch, args.heads, args.seq_q, args.seq_kv, args.dim, args.window_size, args.tune)
