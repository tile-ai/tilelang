import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=128, help="batch size")
parser.add_argument("--heads", type=int, default=16, help="heads")
parser.add_argument("--seq_q", type=int, default=1024, help="query sequence length")
parser.add_argument("--seq_kv", type=int, default=1024, help="key/value sequence length")
parser.add_argument("--dim", type=int, default=256, help="dim")
parser.add_argument("--is_causal", action="store_true", help="causal")
parser.add_argument("--tune", action="store_true", help="tune configs")
parser.add_argument("--use_v2", action="store_true")

args = parser.parse_args()

use_v2 = args.use_v2


def get_configs():
    iter_params = dict(block_M=[128], block_N=[128], num_stages=[2], threads=[256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, block_M=64, block_N=64, num_stages=0, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

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
        T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                q_idx = bx * block_M + i + past_len
                k_idx = k * block_N + j
                acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
        else:
            T.clear(acc_s)
        if use_v2:
            T.gemm_v2(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
        else:
            T.gemm_v1(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        V_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
        # T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
        if use_v2:
            T.gemm_v2(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
        else:
            T.gemm_v1(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
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

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return main


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device), seq_kv - seq_q)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output


def main(
    batch: int = 1,
    heads: int = 1,
    seq_q: int = 256,
    seq_kv: int = 256,
    dim: int = 64,
    is_causal: bool = False,
    tune: bool = False,
):
    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    if not tune:
        kernel = flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, block_M=64, block_N=64, num_stages=0, threads=128)
        print(kernel.get_kernel_source())
        ref_program_processed = partial(ref_program, is_causal=is_causal)

        profiler = kernel.get_profiler()
        profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = profiler.do_bench(ref_program_processed, warmup=500)
        print(f"Ref: {latency:.2f} ms")
        print(f"Ref: {total_flops / latency * 1e-9:.2f} TFlops")
        latency = profiler.do_bench(warmup=500)
        print(f"Tile-lang: {latency:.2f} ms")
        print(f"Tile-lang: {total_flops / latency * 1e-9:.2f} TFlops")
    else:
        kernel = flashattn(batch, heads, seq_q, seq_kv, dim, is_causal)
        best_latency = kernel.latency
        best_config = kernel.config
        ref_latency = kernel.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")


if __name__ == "__main__":
    tilelang.disable_cache()
    main(args.batch, args.heads, args.seq_q, args.seq_kv, args.dim, args.is_causal, args.tune)
