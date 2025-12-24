import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum
import argparse


@tilelang.jit(out_idx=[6])
def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H, num_split):
    scale = (1.0 / (dim + pe_dim)) ** 0.5 * 1.44269504  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"
    h_dim = dim // 2

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
        with T.Kernel(heads // min(block_H, kv_group_num), batch, threads=256) as (hid, bid):
            # smem_sQ
            Q_shared_l = T.alloc_shared([block_H, h_dim], dtype)
            Q_shared_r = T.alloc_shared([block_H, h_dim], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            Q_pe_local_0 = T.alloc_fragment([block_H, pe_dim], dtype)
            Q_pe_local_1 = T.alloc_fragment([block_H, pe_dim], dtype)

            # smem_sK0
            KV_shared_0_l = T.alloc_shared([block_N, h_dim], dtype)
            KV_shared_0_r = T.alloc_shared([block_N, h_dim], dtype)
            K_pe_shared_0 = T.alloc_shared([block_N, pe_dim], dtype)

            # smem_sK1
            KV_shared_1_l = T.alloc_shared([block_N, h_dim], dtype)
            KV_shared_1_r = T.alloc_shared([block_N, h_dim], dtype)
            K_pe_shared_1 = T.alloc_shared([block_N, pe_dim], dtype)

            # smem_sP0
            SP0_shared = T.alloc_shared([block_H, block_N], dtype)

            # smem_sP1 reuse Q_pe_shared
            SP1_shared = Q_pe_shared

            # smem_sM
            scores_max = T.alloc_shared([block_H], accum_dtype)

            # smem_sScale0
            scores_scale_0 = T.alloc_shared([block_H], accum_dtype)
            # smem_sScale1
            scores_scale_1 = T.alloc_shared([block_H], accum_dtype)

            logsum = T.alloc_shared([block_H], accum_dtype)

            O_shared_l = Q_shared_l
            O_shared_r = Q_shared_r

            acc_s_0 = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_0_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_s_1 = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_1_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o_l = T.alloc_fragment([block_H, h_dim], accum_dtype)
            acc_o_r = T.alloc_fragment([block_H, h_dim], accum_dtype)
            scores_max_0 = T.alloc_fragment([block_H], accum_dtype)
            scores_max_1 = T.alloc_fragment([block_H], accum_dtype)

            scores_max_prev_0 = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev_1 = T.alloc_fragment([block_H], accum_dtype)

            scores_sum_0 = T.alloc_fragment([block_H], accum_dtype)
            scores_sum_1 = T.alloc_fragment([block_H], accum_dtype)
            logsum_0 = T.alloc_fragment([block_H], accum_dtype)
            logsum_1 = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = hid // (kv_group_num // block_H)

            # barriers_Q
            q_shared_ready_barrier = T.alloc_barrier(arrive_count=256)

            # barriers_K0
            kv_shared_0_l_is_ready = T.alloc_barrier(arrive_count=128)
            kv_shared_0_r_is_ready = T.alloc_barrier(arrive_count=128)
            kv_shared_0_pe_is_ready = T.alloc_barrier(arrive_count=128)
            # barriers_K1
            kv_shared_1_l_is_ready = T.alloc_barrier(arrive_count=128)
            kv_shared_1_r_is_ready = T.alloc_barrier(arrive_count=128)
            kv_shared_1_pe_is_ready = T.alloc_barrier(arrive_count=128)

            # redundant barriers
            score_max_0_ready_barrier = T.alloc_barrier(arrive_count=128)
            scale_1_ready_barrier = T.alloc_barrier(arrive_count=128)
            p0_1_1_ready_barrier = T.alloc_barrier(arrive_count=128)
            lse_0_ready_barrier = T.alloc_barrier(arrive_count=128)
            lse_1_ready_barrier = T.alloc_barrier(arrive_count=128)
            s_shared_ready_barrier = T.alloc_barrier(arrive_count=128)

            tx = T.get_thread_binding()

            T.copy(Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :h_dim], Q_shared_l)
            T.copy(Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, h_dim:], Q_shared_r)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.barrier_arrive(q_shared_ready_barrier)
            T.barrier_wait(q_shared_ready_barrier, 0)

            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv, (block_N * 2))

            if tx < 128:
                T.copy(Q_pe_shared, Q_pe_local_0)
                T.fill(acc_o_l, 0)
                T.fill(logsum_0, 0)

                T.copy(KV[bid, block_N : 2 * block_N, cur_kv_head, :h_dim], KV_shared_1_l)
                T.barrier_arrive(kv_shared_1_l_is_ready)

                T.copy(KV[bid, block_N : 2 * block_N, cur_kv_head, h_dim:], KV_shared_1_r)
                T.barrier_arrive(kv_shared_1_r_is_ready)

                T.copy(K_pe[bid, block_N : 2 * block_N, cur_kv_head, :], K_pe_shared_1)
                T.barrier_arrive(kv_shared_1_pe_is_ready)

                for k in T.serial(loop_range):
                    T.barrier_wait(kv_shared_0_l_is_ready, k % 2)
                    T.gemm(Q_shared_l, KV_shared_0_l, acc_s_0, transpose_B=True, clear_accum=True, wg_wait=-1)
                    T.barrier_wait(kv_shared_0_r_is_ready, k % 2)
                    T.gemm(Q_shared_r, KV_shared_0_r, acc_s_0, transpose_B=True, wg_wait=-1)

                    T.barrier_wait(kv_shared_0_pe_is_ready, k % 2)
                    T.gemm(Q_pe_local_0, K_pe_shared_0, acc_s_0, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    # Step 3.
                    T.copy(scores_max, scores_max_0)
                    T.copy(scores_max_0, scores_max_prev_0)
                    T.fill(scores_max_0, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s_0, scores_max_0, dim=1, clear=False)
                    T.copy(scores_max_0, scores_max)

                    # Step 4.
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s_0[i, j] = T.exp2(acc_s_0[i, j] * scale - scores_max[i] * scale)
                    for i in T.Parallel(block_H):
                        scores_scale_0[i] = T.exp2(scores_max_prev_0[i] * scale - scores_max[i] * scale)

                    T.reduce_sum(acc_s_0, scores_sum_0, dim=1)

                    # Step 5.
                    T.copy(acc_s_0, acc_s_0_cast)

                    for i, j in T.Parallel(block_H, h_dim):
                        acc_o_l[i, j] *= scores_scale_0[i]

                    for i in T.Parallel(block_H):
                        logsum_0[i] = logsum_0[i] * scores_scale_0[i] + scores_sum_0[i]

                    # Step 6.
                    T.gemm(acc_s_0_cast, KV_shared_0_l, acc_o_l)
                    T.barrier_arrive(score_max_0_ready_barrier)

                    T.barrier_wait(scale_1_ready_barrier, k % 2)

                    if k < loop_range - 1:
                        T.copy(KV[bid, (2 * k + 2) * block_N : (2 * k + 3) * block_N, cur_kv_head, :h_dim], KV_shared_0_l)
                        T.barrier_arrive(kv_shared_0_l_is_ready)

                    # Step 11.
                    for i, j in T.Parallel(block_H, block_N):
                        SP0_shared[i, j] = acc_s_0[i, j] * scores_scale_1[i]

                    T.barrier_arrive(p0_1_1_ready_barrier)

                    # Step 13.
                    for i, j in T.Parallel(block_H, h_dim):
                        acc_o_l[i, j] *= scores_scale_1[i]
                    for i in T.Parallel(block_H):
                        logsum_0[i] = logsum_0[i] * scores_scale_1[i]
                    T.barrier_wait(s_shared_ready_barrier, k % 2)

                    # Step 14.
                    T.gemm(SP1_shared, KV_shared_1_l, acc_o_l)

                    if k < loop_range - 1:
                        T.copy(KV[bid, (2 * k + 3) * block_N : (2 * k + 4) * block_N, cur_kv_head, :h_dim], KV_shared_1_l)
                        T.barrier_arrive(kv_shared_1_l_is_ready)

                        T.copy(K_pe[bid, (2 * k + 3) * block_N : (2 * k + 4) * block_N, cur_kv_head, :], K_pe_shared_1)
                        T.barrier_arrive(kv_shared_1_pe_is_ready)

                T.copy(logsum_0, logsum)
                T.barrier_arrive(lse_0_ready_barrier)
                T.barrier_wait(lse_1_ready_barrier, 0)
                for i, j in T.Parallel(block_H, h_dim):
                    acc_o_l[i, j] /= logsum[i]
                T.copy(acc_o_l, O_shared_l)
                T.copy(O_shared_l, Output[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :h_dim])

            else:
                T.copy(Q_pe_shared, Q_pe_local_1)
                T.fill(acc_o_r, 0)
                T.fill(logsum_1, 0)

                T.copy(KV[bid, :block_N, cur_kv_head, :h_dim], KV_shared_0_l)
                T.barrier_arrive(kv_shared_0_l_is_ready)
                T.copy(KV[bid, :block_N, cur_kv_head, h_dim:], KV_shared_0_r)
                T.barrier_arrive(kv_shared_0_r_is_ready)
                T.copy(K_pe[bid, :block_N, cur_kv_head, :], K_pe_shared_0)
                T.barrier_arrive(kv_shared_0_pe_is_ready)

                for k in T.serial(loop_range):
                    # Step 2.
                    T.barrier_wait(kv_shared_1_l_is_ready, k % 2)
                    T.gemm(Q_shared_l, KV_shared_1_l, acc_s_1, transpose_B=True, clear_accum=True, wg_wait=-1)

                    T.barrier_wait(kv_shared_1_r_is_ready, k % 2)
                    T.gemm(Q_shared_r, KV_shared_1_r, acc_s_1, transpose_B=True, wg_wait=-1)

                    T.barrier_wait(kv_shared_1_pe_is_ready, k % 2)
                    T.gemm(Q_pe_local_1, K_pe_shared_1, acc_s_1, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    # Step 7.
                    T.barrier_wait(score_max_0_ready_barrier, k % 2)

                    T.copy(scores_max, scores_max_prev_1)
                    T.fill(scores_max_1, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s_1, scores_max_1, dim=1, clear=False)
                    T.copy(scores_max_1, scores_max)

                    for i in T.Parallel(block_H):
                        scores_scale_1[i] = T.exp2(scores_max_prev_1[i] * scale - scores_max[i] * scale)

                    # Step 8.
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s_1[i, j] = T.exp2(acc_s_1[i, j] * scale - scores_max[i] * scale)

                    # Step 9.
                    T.reduce_sum(acc_s_1, scores_sum_1, dim=1)

                    for i, j in T.Parallel(block_H, h_dim):
                        acc_o_r[i, j] = acc_o_r[i, j] * (scores_scale_0[i] * scores_scale_1[i])

                    for i in T.Parallel(block_H):
                        logsum_1[i] = logsum_1[i] * scores_scale_1[i] * scores_scale_0[i] + scores_sum_1[i]

                    T.barrier_arrive(scale_1_ready_barrier)

                    # Step 10. compute O1 with KV_shared_1_rd
                    T.copy(acc_s_1, acc_s_1_cast)
                    T.gemm(acc_s_1_cast, KV_shared_1_r, acc_o_r, wg_wait=-1)
                    T.copy(acc_s_1_cast, SP1_shared)
                    T.barrier_arrive(s_shared_ready_barrier)

                    if k < loop_range - 1:
                        T.copy(KV[bid, (2 * k + 3) * block_N : (2 * k + 4) * block_N, cur_kv_head, h_dim:], KV_shared_1_r)
                        T.barrier_arrive(kv_shared_1_r_is_ready)

                    T.barrier_wait(p0_1_1_ready_barrier, k % 2)
                    # Step 12.
                    T.gemm(SP0_shared, KV_shared_0_r, acc_o_r)

                    if k < loop_range - 1:
                        T.copy(KV[bid, (2 * k + 2) * block_N : (2 * k + 3) * block_N, cur_kv_head, h_dim:], KV_shared_0_r)
                        T.barrier_arrive(kv_shared_0_r_is_ready)

                        T.copy(K_pe[bid, (2 * k + 2) * block_N : (2 * k + 3) * block_N, cur_kv_head, :], K_pe_shared_0)
                        T.barrier_arrive(kv_shared_0_pe_is_ready)

                T.barrier_wait(lse_0_ready_barrier, 0)
                for i in T.Parallel(block_H):
                    logsum[i] += logsum_1[i]
                T.barrier_arrive(lse_1_ready_barrier)
                for i, j in T.Parallel(block_H, h_dim):
                    acc_o_r[i, j] /= logsum[i]
                T.copy(acc_o_r, O_shared_r)
                T.copy(O_shared_r, Output[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, h_dim:])

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
    scale = (dim + pe_dim) ** 0.5
    q = rearrange(q, "b (h g) d -> b g h d", g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

    q_pe = rearrange(q_pe, "b (h g) d -> b g h d", g=num_head_groups)  # [batch_size, num_head_groups, groups, pe_dim]

    kv = rearrange(kv, "b n h d -> b h n d")  # [batch_size, groups, seqlen_kv, dim]

    k_pe = rearrange(k_pe, "b n h d -> b h n d")  # [batch_size, num_head_groups, groups, pe_dim]

    query = torch.concat([q, q_pe], dim=-1)
    key = torch.concat([kv, k_pe], dim=-1)

    scores = einsum(query, key, "b g h d, b h s d -> b g h s")  # [batch_size, num_head_groups, groups, seqlen_kv]

    attention = F.softmax(scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

    out = einsum(attention, kv, "b g h s, b h s d -> b g h d")  # [batch_size, num_head_groups, groups, dim]
    out = rearrange(out, "b g h d -> b (h g) d")  # [batch_size, heads, dim]
    return out


def main(batch=1, heads=64, kv_heads=1, kv_ctx=1024, dim=512, pe_dim=64):
    qk_flops = 2 * batch * heads * kv_ctx * (dim + pe_dim)
    pv_flops = 2 * batch * heads * kv_ctx * dim
    total_flops = qk_flops + pv_flops
    BLOCK_N = 64
    BLOCK_H = 64
    num_split = 1

    kernel = flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, BLOCK_N, BLOCK_H, num_split)
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    latency = profiler.do_bench(warmup=500)
    print(f"Latency: {latency} ms")
    print(f"TFlops: {total_flops / latency * 1e-9} TFlops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=132, help="batch size")
    parser.add_argument("--heads", type=int, default=128, help="q heads number")
    parser.add_argument("--kv_heads", type=int, default=1, help="kv heads number")
    parser.add_argument("--kv_ctx", type=int, default=8192, help="kv context length")
    parser.add_argument("--dim", type=int, default=512, help="head dim")
    parser.add_argument("--pe_dim", type=int, default=64, help="pe head dim")
    args = parser.parse_args()
    batch, heads, kv_heads, kv_ctx, dim, pe_dim = args.batch, args.heads, args.kv_heads, args.kv_ctx, args.dim, args.pe_dim
    main(batch, heads, kv_heads, kv_ctx, dim, pe_dim)
