# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


def flashattn(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages, threads):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.macro
    def MMA0(
        K: T.Buffer(shape, dtype),
        Q_shared: T.Buffer([block_M, dim], dtype),
        K_shared: T.Buffer([block_N, dim], dtype),
        acc_s: T.Buffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                             -T.infinity(acc_s.dtype))
        else:
            T.clear(acc_s)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
            V: T.Buffer(shape, dtype),
            V_shared: T.Buffer([block_M, dim], dtype),
            acc_s_cast: T.Buffer([block_M, block_N], dtype),
            acc_o: T.Buffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
    ):
        T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
            acc_s: T.Buffer([block_M, block_N], accum_dtype),
            acc_s_cast: T.Buffer([block_M, block_N], dtype),
            scores_max: T.Buffer([block_M], accum_dtype),
            scores_max_prev: T.Buffer([block_M], accum_dtype),
            scores_scale: T.Buffer([block_M], accum_dtype),
            scores_sum: T.Buffer([block_M], accum_dtype),
            logsum: T.Buffer([block_M], accum_dtype),
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
            acc_o: T.Buffer([block_M, dim], accum_dtype),
            scores_scale: T.Buffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.prim_func
    def main(
            Q: T.Buffer(shape, dtype),
            K: T.Buffer(shape, dtype),
            V: T.Buffer(shape, dtype),
            Output: T.Buffer(shape, dtype),
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

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                    (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum,
                        logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

    return main


def run_mha(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages=2, threads=128):
    program = flashattn(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages,
                        threads)

    kernel = tilelang.compile(program, out_idx=[3])
    profiler = kernel.get_profiler()

    def ref_program(Q, K, V):
        import torch
        import torch.nn.functional as F
        dim = Q.size(-1)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
        scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
        if is_causal:
            seq_len = Q.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
        return output

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.05)


def test_mha_causal_dim64():
    run_mha(
        batch=4,
        heads=8,
        seq_len=8192,
        dim=64,
        is_causal=True,
        block_M=64,
        block_N=64,
        num_stages=2,
        threads=128)


def test_mha_no_causal_dim64():
    run_mha(
        batch=4,
        heads=8,
        seq_len=8192,
        dim=64,
        is_causal=False,
        block_M=64,
        block_N=64,
        num_stages=2,
        threads=128)


# def test_mha_causal_dim128():
#     run_mha(
#         batch=4,
#         heads=8,
#         seq_len=8192,
#         dim=128,
#         is_causal=True,
#         block_M=64,
#         block_N=64,
#         num_stages=1,
#         threads=128)

# def test_mha_no_causal_dim128():
#     run_mha(
#         batch=4,
#         heads=8,
#         seq_len=8192,
#         dim=128,
#         is_causal=False,
#         block_M=64,
#         block_N=64,
#         num_stages=1,
#         threads=128)


def test_mha_causal_dim256():
    run_mha(
        batch=4,
        heads=8,
        seq_len=8192,
        dim=256,
        is_causal=True,
        block_M=64,
        block_N=64,
        num_stages=1,
        threads=128)


def test_mha_no_causal_dim256():
    run_mha(
        batch=4,
        heads=8,
        seq_len=8192,
        dim=256,
        is_causal=False,
        block_M=64,
        block_N=64,
        num_stages=1,
        threads=128)


if __name__ == "__main__":
    tilelang.testing.main()
