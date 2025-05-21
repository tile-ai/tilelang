# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# use default stage 1 template, not the optimal
# schedule, please checkout examples/deepseek_mla
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from einops import rearrange, einsum


def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H, num_split):
    scale = (1.0 / (dim + pe_dim))**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.macro
    def flash_attn(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(batch, heads // min(block_H, kv_group_num), threads=384) as (bx, by):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = by // (kv_group_num // block_H)
            T.use_swizzle(10)
            T.annotate_layout({
                O_shared: tilelang.layout.make_swizzled_layout(O_shared),
            })
            T.create_list_of_mbarrier(128, 128, 256, 128)

            loop_range = T.ceildiv(seqlen_kv, block_N)
            with T.ws(2):
                T.dec_max_nreg(24)
                T.copy(Q[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_shared)
                T.copy(Q_pe[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_pe_shared)
                T.barrier_arrive(barrier_id=3)
                for k in T.serial(loop_range):
                    T.barrier_wait(barrier_id=(k % 1) + 2, parity=(k % 2) ^ 1)
                    T.copy(KV[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :], KV_shared)
                    T.barrier_arrive(k % 1)
                    T.copy(K_pe[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :], K_pe_shared)
                    T.barrier_arrive(k % 1 + 1)
            with T.ws(0, 1):
                T.inc_max_nreg(240)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.barrier_wait(3, 0)
                for k in T.serial(loop_range):
                    T.clear(acc_s)
                    T.barrier_wait(barrier_id=k % 1, parity=(k // 1) % 2)
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol)
                    T.barrier_wait(barrier_id=k % 1 + 1, parity=(k // 1) % 2)
                    T.gemm(
                        Q_pe_shared,
                        K_pe_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    T.copy(acc_s, S_shared)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
                    T.barrier_arrive(barrier_id=k % 1 + 2)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :])

    @T.prim_func
    def main_no_split(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
    ):
        flash_attn(Q, Q_pe, KV, K_pe, Output)

    return main_no_split


def ref_program(q, q_pe, kv, k_pe, glse, Output_partial):
    #     """
    #     Inputs:
    #     - q (Tensor): [batch, heads, dim]
    #     - q_pe (Tensor): [batch, heads, pe_dim]
    #     - kv (Tensor): [batch, seqlen_kv, kv_head_num, dim]
    #     - k_pe (Tensor): [batch, seqlen_kv, kv_head_num, pe_dim]
    #     - glse (Tensor): [batch, heads, num_split]
    #     - Output_partial (Tensor): [batch, heads, num_split, dim]
    #     Outputs:
    #     - output (Tensor): [batch, heads, dim]
    #     """
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    num_head_groups = q.shape[1] // kv.shape[2]
    scale = (dim + pe_dim)**0.5
    q = rearrange(
        q, 'b (h g) d -> b g h d', g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

    q_pe = rearrange(
        q_pe, 'b (h g) d -> b g h d',
        g=num_head_groups)  # [batch_size, num_head_groups, groups, pe_dim]

    kv = rearrange(kv, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

    k_pe = rearrange(k_pe, 'b n h d -> b h n d')  # [batch_size, num_head_groups, groups, pe_dim]

    query = torch.concat([q, q_pe], dim=-1)
    key = torch.concat([kv, k_pe], dim=-1)

    scores = einsum(
        query, key,
        'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

    attention = F.softmax(
        scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

    out = einsum(attention, kv,
                 'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
    out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
    return out


def main():
    batch = 128
    heads = 128
    kv_heads = 1
    kv_ctx = 8192
    dim = 512
    pe_dim = 64

    qk_flops = 2 * batch * heads * kv_ctx * (dim + pe_dim)
    pv_flops = 2 * batch * heads * kv_ctx * dim
    total_flops = qk_flops + pv_flops
    BLOCK_N = 64
    BLOCK_H = 64
    num_split = 1

    program = flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, BLOCK_N, BLOCK_H, num_split)
    kernel = tilelang.compile(program, out_idx=[6])
    print(kernel.get_kernel_source())

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    latency = profiler.do_bench(warmup=500)
    print(f"Latency: {latency} ms")
    print(f"TFlops: {total_flops / latency * 1e-9} TFlops")


if __name__ == "__main__":
    main()
