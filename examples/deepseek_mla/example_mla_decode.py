import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum
import argparse


@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H, num_split, softmax_scale):
    scale = float(softmax_scale * 1.44269504)  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        # flash_attn_split
        with T.Kernel(batch, heads // min(block_H, kv_group_num), num_split, threads=256) as (bid, hid, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = hid // (kv_group_num // block_H)
            T.use_swizzle(10)

            T.copy(Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                kv_start = (seqlen_kv // num_split) * bz + k * block_N
                kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, kv_start:kv_end, cur_kv_head, :], K_pe_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                T.copy(S_shared, acc_s_cast)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, glse[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz, :])

        # combine
        with T.Kernel(heads, batch, threads=128) as (hid, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local = T.max(lse_max_local, glse[bz, hid, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split = glse[bz, hid, k]
                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, hid, k, i]
                lse_local_split = glse[bz, hid, k]
                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local
            for i in T.Parallel(dim):
                Output[bz, hid, i] = o_accum_local[i]

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

            cur_kv_head = hid // (kv_group_num // block_H)

            T.copy(Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_pe_shared)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
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
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :])

    if num_split > 1:
        return main_split
    else:
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


def main(
    batch=1,
    heads=128,
    kv_heads=1,
    kv_ctx=8192,
    dim=512,
    pe_dim=64,
):
    qk_flops = 2 * batch * heads * kv_ctx * (dim + pe_dim)
    pv_flops = 2 * batch * heads * kv_ctx * dim
    total_flops = qk_flops + pv_flops
    BLOCK_N = 64
    BLOCK_H = min(64, heads // kv_heads)
    num_split = 1
    softmax_scale = (dim + pe_dim) ** -0.5

    kernel = flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, BLOCK_N, BLOCK_H, num_split, softmax_scale)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    profiler.assert_allclose(ref_program, rtol=1e-4, atol=1e-4)
    latency = profiler.do_bench(warmup=500)
    print(f"Latency: {latency} ms")
    print(f"TFlops: {total_flops / latency * 1e-9} TFlops")


def run_regression_perf(
    batch=1,
    heads=128,
    kv_heads=1,
    kv_ctx=8192,
    dim=512,
    pe_dim=64,
):
    BLOCK_N = 64
    BLOCK_H = min(64, heads // kv_heads)
    num_split = 1
    softmax_scale = (dim + pe_dim) ** -0.5

    kernel = flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, BLOCK_N, BLOCK_H, num_split, softmax_scale)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    profiler.assert_allclose(ref_program, rtol=1e-4, atol=1e-4)
    return profiler.do_bench(warmup=500, backend="cupti")


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
