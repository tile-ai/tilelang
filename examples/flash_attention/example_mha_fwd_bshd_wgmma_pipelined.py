import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial


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
def flashattn(batch, heads, seq_len, dim, is_causal, block_M=128, block_N=128, num_stages=2, threads=256):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
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

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(
                loop_range,
                num_stages=num_stages,
                order=[-1, 0, 3, 1, -1, 2],
                stage=[-1, 0, 0, 1, -1, 1],
                group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12], [13], [14]],
            ):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
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

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
    return output


def main(
    batch: int = 8,
    heads: int = 32,
    seq_len: int = 4096,
    dim: int = 128,
    is_causal: bool = False,
    tune: bool = False,
):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    if not tune:
        kernel = flashattn(batch, heads, seq_len, dim, is_causal, block_M=128, block_N=128, num_stages=2, threads=256)
        ref_program_processed = partial(ref_program, is_causal=is_causal)
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
        profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = profiler.do_bench(ref_program_processed, warmup=500)
        print("Ref: {:.2f} ms".format(latency))
        print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(warmup=500)
        print("Tile-lang: {:.2f} ms".format(latency))
        print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    else:
        kernel = flashattn(batch, heads, seq_len, dim, is_causal)
        best_latency = kernel.latency
        best_config = kernel.config
        ref_latency = kernel.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")


def run_regression_perf(batch: int = 8, heads: int = 32, seq_len: int = 4096, dim: int = 128, is_causal: bool = False):
    kernel = flashattn(batch, heads, seq_len, dim, is_causal, block_M=128, block_N=128, num_stages=2, threads=256)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    return profiler.do_bench(warmup=500, backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="heads")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.tune)
