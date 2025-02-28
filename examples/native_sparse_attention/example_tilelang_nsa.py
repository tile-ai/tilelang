# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ruff: noqa
import torch
from reference import naive_nsa
import tilelang
from tilelang import language as T
import tilelang.testing

tilelang.testing.set_random_seed(0)


def native_sparse_attention(batch,
                            heads,
                            seq_len,
                            dim,
                            is_causal,
                            scale=None,
                            groups=1,
                            selected_blocks=16):
    if scale is None:
        scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    dtype = "float16"
    accum_dtype = "float"
    block_S = 64
    block_T = min(128, tilelang.math.next_power_of_2(dim))

    S = selected_blocks
    NS = S
    G = groups
    BS = block_S
    BK = BV = block_T
    num_stages = 0
    threads = 32

    def kernel_func(block_S, block_T, num_stages, threads):

        @T.prim_func
        def main(
                Q: T.Buffer(q_shape, dtype),
                K: T.Buffer(kv_shape, dtype),
                V: T.Buffer(kv_shape, dtype),
                BlockIndices: T.Buffer(block_indices_shape, block_indices_dtype),
                Output: T.Buffer(q_shape, dtype),
        ):
            with T.Kernel(
                    dim // block_T, seq_len, batch * head_kv, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([G, BK], dtype)
                K_shared = T.alloc_shared([BS, BK], dtype)
                V_shared = T.alloc_shared([BS, BV], dtype)
                O_shared = T.alloc_shared([G, BV], dtype)

                acc_s = T.alloc_fragment([G, BS], accum_dtype)
                acc_s_cast = T.alloc_fragment([G, BS], dtype)
                acc_o = T.alloc_fragment([G, BV], accum_dtype)
                scores_max = T.alloc_fragment([G], accum_dtype)
                scores_max_prev = T.alloc_fragment([G], accum_dtype)
                scores_scale = T.alloc_fragment([G], accum_dtype)
                scores_sum = T.alloc_fragment([G], accum_dtype)
                logsum = T.alloc_fragment([G], accum_dtype)

                i_v, i_t, i_bh = bx, by, bz
                i_b, i_h = i_bh // heads, i_bh % heads

                T.copy(Q[i_b, i_t, i_h * G:(i_h + 1) * G, :], Q_shared)

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for i in T.Pipelined(NS, num_stages=num_stages):
                    i_s = BlockIndices[i_b, i_t, i_h, i]
                    if i_s <= i_t:
                        # Q * K
                        T.copy(K[i_b, i_s * BS:(i_s + 1) * BS, i_h, :], K_shared)

                        if is_causal:
                            for i, j in T.Parallel(G, BS):
                                acc_s[i, j] = T.if_then_else(i_t >= (i_s * BS + j), 0,
                                                             -T.infinity(acc_s.dtype))
                        else:
                            T.clear(acc_s)
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow)

                        # Softmax
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
                        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
                        # in the first ceil_div(kBlockM, kBlockN) steps.
                        # for i in T.Parallel(block_M):
                        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
                        for i in T.Parallel(G):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale -
                                                     scores_max[i] * scale)
                        for i, j in T.Parallel(G, BS):
                            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                            # max * log_2(e)) This allows the compiler to use the ffma
                            # instruction instead of fadd and fmul separately.
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(G):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)

                        # Rescale
                        for i, j in T.Parallel(G, BV):
                            acc_o[i, j] *= scores_scale[i]

                        # V * softmax(Q * K)
                        T.copy(V[i_b, i_s * BS:(i_s + 1) * BS, i_h, :], V_shared)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(G, BV):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[i_b, i_t, i_h * G:(i_h + 1) * G, :])

        return main

    def kernel(block_S, block_T, num_stages, threads):
        return kernel_func(block_S, block_T, num_stages, threads)

    return kernel(block_S, block_T, num_stages, threads)


if __name__ == "__main__":
    B, SEQ_LEN, H, HQ, D, S, block_size, dtype, scale = 1, 64, 4, 64, 32, 16, 64, torch.float16, None

    program = native_sparse_attention(
        batch=B,
        heads=HQ,
        seq_len=SEQ_LEN,
        dim=D,
        is_causal=True,
        scale=scale,
        groups=HQ // H,
        selected_blocks=S,
    )
    kernel = tilelang.compile(program, out_idx=[4])

    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    DO = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device='cuda')

    block_indices = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.long, device='cuda')
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                i_i = torch.randperm(max(1, (t // block_size)))[:S]
                block_indices[b, t, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    block_counts = torch.randint(1, S + 1, (B, SEQ_LEN, H), device='cuda')

    out = kernel(Q, K, V, block_indices.to(torch.int32))

    print(out)

    ref = naive_nsa(
        q=Q,
        k=K,
        v=V,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale)

    print(ref)
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)
