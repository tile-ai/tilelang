# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
#
# Modified to implement FlashAttention-2 forward pass principles.
# Corrected loop implementation using T.while_loop.

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import itertools
import argparse
from functools import partial


# PyTorch 参考实现保持不变
def ref_program(Q, K, V, is_causal, groups=1):
    assert Q.size(
        2) == K.size(2) * groups, f"Q heads {Q.size(2)} K heads {K.size(2)} groups {groups}"
    assert Q.size(
        2) == V.size(2) * groups, f"Q heads {Q.size(2)} V heads {V.size(2)} groups {groups}"
    dim = Q.size(-1)
    K = K.repeat_interleave(groups, dim=2)
    V = V.repeat_interleave(groups, dim=2)
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


def get_v2_configs():
    """Generates configurations for the autotuner, tailored for FA-2 style parallelism."""
    block_M = [64, 128, 256]
    block_N = [32, 64, 128]
    threads = [128, 256, 512]
    num_split_q = [32, 64, 128]
    num_stages = [1, 2, 3]
    enable_rasterization = [True]
    k_pack = [2]

    valid_configs = []

    for m, n, s, t, stages, r, k in itertools.product(block_M, block_N, num_split_q, threads,
                                                      num_stages, enable_rasterization, k_pack):
        valid_configs.append({
            "block_M": m,
            "block_N": n,
            "num_split_q": s,
            "threads": t,
            "num_stages": stages,
            "enable_rasterization": r,
            "k_pack": k
        })
    if not valid_configs:
        valid_configs.append({
            'block_M': 64,
            'block_N': 64,
            'num_split_q': 64,
            'threads': 256,
            'num_stages': 1,
            'enable_rasterization': True,
            'k_pack': 2
        })
    return valid_configs


@tilelang.autotune(configs=get_v2_configs(), cache_input_tensors=True)
@tilelang.jit(out_idx=[3])
def fast_flashattn_v2(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups,
    block_M: int,
    block_N: int,
    num_split_q: int,
    threads: int,
    num_stages: int,
    enable_rasterization: bool,
    k_pack: int,
):
    scale = (1.0 / dim)**0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = "float16"
    accum_dtype = "float"

    v_vec_size = 4

    vec_size = 4 * k_pack

    @T.macro
    def compute_block(
            bz,
            by,
            bx,
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            m_i: T.FragmentBuffer([block_M], accum_dtype),
            l_i: T.FragmentBuffer([block_M], accum_dtype),
    ):
        Q_shared = T.alloc_shared([block_M, dim], dtype)
        K_shared = T.alloc_shared([block_N, dim], dtype)
        V_shared = T.alloc_shared([block_N, dim], dtype)
        P_shared = T.alloc_shared([block_M, block_N], dtype)

        acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
        m_prev = T.alloc_fragment([block_M], accum_dtype)
        scale_factor = T.alloc_fragment([block_M], accum_dtype)

        q_block_offset = bx * block_M
        T.copy(
            Q[bz, q_block_offset:q_block_offset + block_M, by, :],
            Q_shared,
            coalesced_width=vec_size)

        loop_end_k = T.ceildiv(q_block_offset +
                               block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
        for k in T.Pipelined(loop_end_k, num_stages=num_stages):
            kv_idx = k * block_N
            T.copy(
                K[bz, kv_idx:kv_idx + block_N, by // groups, :], K_shared, coalesced_width=vec_size)
            T.copy(
                V[bz, kv_idx:kv_idx + block_N, by // groups, :],
                V_shared,
                coalesced_width=v_vec_size)

            T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, k_pack=k_pack)

            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(q_block_offset + i >= kv_idx + j, acc_s[i, j],
                                                 -T.infinity(acc_s.dtype))

            T.copy(m_i, m_prev)
            T.reduce_max(acc_s, m_i, dim=1, clear=False)

            for i in T.Parallel(block_M):
                sf = T.exp2(m_prev[i] * scale - m_i[i] * scale)
                l_i[i] *= sf
                scale_factor[i] = sf

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scale_factor[i]

            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - m_i[i] * scale)

            row_sum = T.alloc_fragment([block_M], accum_dtype)
            T.reduce_sum(acc_s, row_sum, dim=1)
            for i in T.Parallel(block_M):
                l_i[i] += row_sum[i]

            T.copy(acc_s, P_shared)
            T.sync_threads()

            T.gemm(P_shared, V_shared, acc_o)

    # 修复：将宏移至内核外部，以实现清晰的代码结构。
    @T.macro
    def scale_and_write_back(src_buffer, scale_vector, dest_tensor, bz, by, q_block_offset):
        # 此宏执行融合的缩放和写回操作，这对性能至关重要。
        for i, j in T.Parallel(block_M, dim):
            dest_tensor[bz, q_block_offset + i, by, j] = src_buffer[i, j] * scale_vector[i]

    @T.macro
    def flash_attn_forward_kernel(Q: T.Tensor(q_shape, dtype), K: T.Tensor(kv_shape, dtype),
                                  V: T.Tensor(kv_shape, dtype), Output: T.Tensor(q_shape, dtype)):
        with T.Kernel(num_split_q, batch * heads, threads=threads) as (b_split, byz_combined):
            T.use_swizzle(10, enable=enable_rasterization)

            bz = byz_combined // heads
            by = byz_combined % heads

            num_q_blocks = T.ceildiv(seq_len, block_M)

            bx = T.alloc_var("int32")
            bx[0] = b_split

            with T.While(bx[0] < num_q_blocks):
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                m_i = T.alloc_fragment([block_M], accum_dtype)
                l_i = T.alloc_fragment([block_M], accum_dtype)
                T.fill(acc_o, 0)
                T.fill(m_i, -T.infinity(accum_dtype))
                T.fill(l_i, 0)

                current_bx = bx[0]

                compute_block(bz, by, current_bx, Q, K, V, acc_o, m_i, l_i)

                l_inv = T.alloc_fragment([block_M], accum_dtype)
                for i in T.Parallel(block_M):
                    safe_l = T.if_then_else(l_i[i] > 1e-6, l_i[i], 1.0)
                    l_inv[i] = 1.0 / safe_l

                # 修复：现在对宏的调用对编译器来说更清晰。
                q_block_offset = current_bx * block_M
                scale_and_write_back(acc_o, l_inv, Output, bz, by, q_block_offset)

                bx[0] = current_bx + num_split_q

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        flash_attn_forward_kernel(Q, K, V, Output)

    return main


# main 函数保持不变
def main_v2(batch: int = 1,
            heads: int = 8,
            seq_len: int = 4096,
            dim: int = 128,
            is_causal: bool = False,
            groups: int = 1):

    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    print("Starting autotuning for FlashAttention-V2...")
    kernel = fast_flashattn_v2(batch, heads, seq_len, dim, is_causal, groups=groups)
    print(f"Autotuning finished. Best Configuration: {kernel.config}")

    ref_program_processed = partial(ref_program, is_causal=is_causal, groups=groups)

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    print("Verifying correctness...")
    profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = profiler.do_bench(ref_program_processed, warmup=100)
    print(f"Reference (PyTorch): {latency:.2f} ms | {total_flops / latency * 1e-9:.2f} TFlops")

    latency = profiler.do_bench(warmup=100)
    print(
        f"Fast Flash Attention V2 (Tile-lang): {latency:.2f} ms | {total_flops / latency * 1e-9:.2f} TFlops"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=8, help='heads')
    parser.add_argument('--seq_len', type=int, default=4096, help='sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--is_causal', action='store_true', help='causal')
    parser.add_argument('--groups', type=int, default=1, help='groups')
    args = parser.parse_args()
    main_v2(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.groups)
