import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse
from tilelang.profiler import do_bench
import math


@tilelang.jit(
    out_idx=[8],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def mla_decode_tilelang(batch, h_q, h_kv, max_seqlen_pad, dv, dpe, block_N, block_H, num_split, block_size, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = (dv + dpe) ** -0.5
    scale = float(softmax_scale * 1.44269504)  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = h_q // h_kv
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert h_kv == 1, "h_kv must be 1"
    assert block_size >= block_N and block_size % block_N == 0, "block_size must be larger than block_N and a multiple of block_N"

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, h_q, dv], dtype),
        Q_pe: T.Tensor([batch, h_q, dpe], dtype),
        KV: T.Tensor([batch * max_seqlen_pad, h_kv, dv], dtype),
        K_pe: T.Tensor([batch * max_seqlen_pad, h_kv, dpe], dtype),
        block_table: T.Tensor([batch, max_seqlen_pad // block_size], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        glse: T.Tensor([batch, h_q, num_split], dtype),
        Output_partial: T.Tensor([batch, h_q, num_split, dv], dtype),
        Output: T.Tensor([batch, h_q, dv], dtype),
    ):
        # split kv
        with T.Kernel(batch, h_q // min(block_H, kv_group_num), num_split, threads=256) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dv], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, dpe], dtype)
            KV_shared = T.alloc_shared([block_N, dv], dtype)
            K_pe_shared = T.alloc_shared([block_N, dpe], dtype)
            O_shared = T.alloc_shared([block_H, dv], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dv], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = by // (kv_group_num // block_H)
            T.use_swizzle(10)

            T.copy(Q[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            total_blocks = T.ceildiv(cache_seqlens[bx], block_N)
            blocks_per_split = T.floordiv(total_blocks, num_split)
            remaining_blocks = T.floormod(total_blocks, num_split)
            loop_range = blocks_per_split + T.if_then_else(bz < remaining_blocks, 1, 0)
            start = (blocks_per_split * bz + T.min(bz, remaining_blocks)) * block_N

            for k in T.Pipelined(loop_range, num_stages=2):
                kv_start = block_table[bx, (start + k * block_N) // block_size] * block_size + (k * block_N) % block_size
                T.copy(KV[kv_start : kv_start + block_N, cur_kv_head, :], KV_shared)
                T.copy(K_pe[kv_start : kv_start + block_N, cur_kv_head, :], K_pe_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(start + k * block_N + j >= cache_seqlens[bx], -T.infinity(accum_dtype), acc_s[i, j])
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
                for i, j in T.Parallel(block_H, dv):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_H, dv):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, glse[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, bz])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, bz, :])

        # combine
        with T.Kernel(h_q, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dv], dtype)
            o_accum_local = T.alloc_fragment([dv], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local = T.max(lse_max_local, glse[bz, by, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split = glse[bz, by, k]
                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
            for k in T.serial(num_split):
                for i in T.Parallel(dv):
                    po_local[i] = Output_partial[bz, by, k, i]
                lse_local_split = glse[bz, by, k]
                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                for i in T.Parallel(dv):
                    o_accum_local[i] += po_local[i] * scale_local
            for i in T.Parallel(dv):
                Output[bz, by, i] = o_accum_local[i]

    @T.prim_func
    def main_no_split(
        Q: T.Tensor([batch, h_q, dv], dtype),
        Q_pe: T.Tensor([batch, h_q, dpe], dtype),
        KV: T.Tensor([batch * max_seqlen_pad, h_kv, dv], dtype),
        K_pe: T.Tensor([batch * max_seqlen_pad, h_kv, dpe], dtype),
        block_table: T.Tensor([batch, max_seqlen_pad // block_size], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        glse: T.Tensor([batch, h_q, num_split], dtype),
        Output_partial: T.Tensor([batch, h_q, num_split, dv], dtype),
        Output: T.Tensor([batch, h_q, dv], dtype),
    ):
        with T.Kernel(batch, h_q // min(block_H, kv_group_num), threads=256) as (bx, by):
            Q_shared = T.alloc_shared([block_H, dv], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, dpe], dtype)
            KV_shared = T.alloc_shared([block_N, dv], dtype)
            K_pe_shared = T.alloc_shared([block_N, dpe], dtype)
            O_shared = T.alloc_shared([block_H, dv], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dv], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = by // (kv_group_num // block_H)
            T.use_swizzle(10)

            T.copy(Q[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(cache_seqlens[bx], block_N)
            for kr in T.Pipelined(loop_range, num_stages=2):
                k = loop_range - 1 - kr
                kv_start = block_table[bx, (k * block_N) // block_size] * block_size + (k * block_N) % block_size
                T.copy(KV[kv_start : kv_start + block_N, cur_kv_head, :], KV_shared)
                T.copy(K_pe[kv_start : kv_start + block_N, cur_kv_head, :], K_pe_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                if kr == 0:
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= cache_seqlens[bx], -T.infinity(accum_dtype), acc_s[i, j])
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
                for i, j in T.Parallel(block_H, dv):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_H, dv):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :])

    if num_split > 1:
        return main_split
    else:
        return main_no_split


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


@torch.inference_mode()
def run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    # q: [b, s_q, h_q, d]
    # block_table: [b, max_seqlen_pad // block_size]
    # blocked_k: [b * max_seqlen_pad // block_size, block_size, h_kv, d]
    # cache_seqlens: [b]
    blocked_v = blocked_k[..., :dv]

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=q.device)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=q.device)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q,
                h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out.to(dtype), lse.to(dtype)

    out_torch, _ = ref_mla()
    return out_torch


def run_tilelang_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    assert d > dv, "mla with rope dim should be larger than no rope dim"
    q_nope, q_pe = q[..., :dv].contiguous(), q[..., dv:].contiguous()
    blocked_k_nope, blocked_k_pe = blocked_k[..., :dv].contiguous(), blocked_k[..., dv:].contiguous()

    dpe = d - dv
    num_kv_splits = 1
    BLOCK_N = 64
    BLOCK_H = min(64, h_q // h_kv)
    softmax_scale = d**-0.5

    out_partial = torch.empty(b, h_q, num_kv_splits, dv, dtype=dtype, device=q.device)
    glse = torch.empty(b, h_q, num_kv_splits, dtype=dtype, device=q.device)
    kernel = mla_decode_tilelang(b, h_q, h_kv, max_seqlen_pad, dv, dpe, BLOCK_N, BLOCK_H, num_kv_splits, block_size, softmax_scale)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    def flash_mla_tilelang():
        out = profiler.func(
            q_nope.view(-1, h_q, dv),
            q_pe.view(-1, h_q, dpe),
            blocked_k_nope.view(-1, h_kv, dv),
            blocked_k_pe.view(-1, h_kv, dpe),
            block_table,
            cache_seqlens,
            glse,
            out_partial,
        )
        return out.view([b, s_q, h_q, dv])

    out_flash = flash_mla_tilelang()
    t = do_bench(flash_mla_tilelang)
    out_ref = run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype)
    torch.testing.assert_close(out_flash, out_ref, rtol=0.01, atol=0.01)
    print("All close")
    return out_flash, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--h_q", type=int, default=128, help="q heads number")
    parser.add_argument("--h_kv", type=int, default=1, help="kv heads number")
    parser.add_argument("--cache_seqlen", type=int, default=8192, help="kv cache context length")
    parser.add_argument("--d", type=int, default=576, help="query/key head dim, d = dv + dpe")
    parser.add_argument("--dv", type=int, default=512, help="value head dim")
    args = parser.parse_args()
    b, h_q, h_kv, cache_seqlen, d, dv = args.batch, args.h_q, args.h_kv, args.cache_seqlen, args.d, args.dv

    device = "cuda"
    dtype = torch.float16

    s_q = 1  # for decode, s_q = 1
    block_size = 64
    cache_seqlens = torch.tensor([cache_seqlen + 2 * i for i in range(b)], dtype=torch.int32, device=device)
    dpe = d - dv
    causal = True

    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen / 256) * 256

    total_flops = s_q * total_seqlens * h_q * d * 2

    q = torch.randn(b, s_q, h_q, d, dtype=dtype, device=device)
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32, device=device).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=dtype, device=device)
    out_flash, latency = run_tilelang_mla(
        q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype
    )

    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
