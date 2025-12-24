# ruff: noqa
import torch
import triton
import triton.language as tl
import argparse
from einops import rearrange, einsum
import torch.nn.functional as F
import math
import time
from heuristic import num_splits_heuristic
from tilelang.profiler import do_bench


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [1, 2, 4] for num_stages in [1, 2, 3, 4, 7]],
    key=["BLOCK_H", "BLOCK_N", "BLOCK_D"],
)
@triton.jit
def _split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_seqlens_ptr,
    o_partial_ptr,
    lse_partial_ptr,
    mask_ptr,
    sm_scale,
    num_splits,
    gqa_group_size,
    max_selected_blocks,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_o_b,
    stride_o_h,
    stride_o_split,
    stride_o_d,
    stride_lse_b,
    stride_lse_h,
    stride_lse_split,
    stride_mask_b,
    stride_mask_h,
    stride_mask_s,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx_kv = tl.program_id(1)
    split_idx = tl.program_id(2)

    head_idx_q = head_idx_kv * gqa_group_size
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    cache_seqlens = tl.load(cache_seqlens_ptr + batch_idx)
    num_blocks = max_selected_blocks
    blocks_per_split = tl.floor(num_blocks / num_splits).to(tl.int32)
    remaining_blocks = num_blocks % num_splits
    if split_idx < remaining_blocks:
        loop_range = blocks_per_split + 1
    else:
        loop_range = blocks_per_split

    q_ptr += batch_idx * stride_q_b + head_idx_q * stride_q_h
    k_cache_ptr += batch_idx * stride_k_b + head_idx_kv * stride_k_h + offs_n[None, :] * stride_k_s + offs_d[:, None] * stride_k_d
    v_cache_ptr += batch_idx * stride_v_b + head_idx_kv * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
    mask_ptr += batch_idx * stride_mask_b + head_idx_kv * stride_mask_h

    q = tl.load(q_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d, mask=offs_h[:, None] < gqa_group_size)
    start = blocks_per_split * split_idx + tl.minimum(split_idx, remaining_blocks)
    for i in range(loop_range):
        block_idx = tl.load(mask_ptr + (start + i) * stride_mask_s)
        if block_idx >= 0:
            start_n = block_idx * BLOCK_N
            k_ptr = k_cache_ptr + start_n * stride_k_s
            v_ptr = v_cache_ptr + start_n * stride_v_s

            k = tl.load(k_ptr, mask=start_n + offs_n[None, :] < cache_seqlens, other=0.0)
            v = tl.load(v_ptr, mask=start_n + offs_n[:, None] < cache_seqlens, other=0.0)

            qk = tl.dot(q, k)
            qk = qk * sm_scale
            qk = tl.where(start_n + offs_n[None, :] < cache_seqlens, qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            p = p.to(v.type.element_ty)
            acc += tl.dot(p, v)
            m_i = m_ij

    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(o_partial_ptr.dtype.element_ty)

    lse_partial_ptr += batch_idx * stride_lse_b + (head_idx_q + offs_h) * stride_lse_h + split_idx * stride_lse_split
    tl.store(lse_partial_ptr, m_i, mask=offs_h < gqa_group_size)

    o_partial_ptr += (
        batch_idx * stride_o_b + (head_idx_q + offs_h[:, None]) * stride_o_h + split_idx * stride_o_split + offs_d[None, :] * stride_o_d
    )
    tl.store(o_partial_ptr, acc, mask=offs_h[:, None] < gqa_group_size)


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [1, 2, 4] for num_stages in [1, 2, 3, 4, 7]],
    key=["BLOCK_D"],
)
@triton.jit
def _merge_kernel(
    o_partial_ptr,
    lse_partial_ptr,
    o_ptr,
    lse_partial_stride_b,
    lse_partial_stride_h,
    lse_partial_stride_split,
    o_partial_stride_b,
    o_partial_stride_h,
    o_partial_stride_split,
    o_partial_stride_d,
    o_stride_b,
    o_stride_h,
    o_stride_d,
    BLOCK_D: tl.constexpr,
    num_splits: tl.constexpr,
    num_splits_pow2: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_splits = tl.arange(0, num_splits_pow2)
    offs_d = tl.arange(0, BLOCK_D)

    lse_offsets = lse_partial_ptr + batch_idx * lse_partial_stride_b + head_idx * lse_partial_stride_h
    lse = tl.load(lse_offsets + offs_splits * lse_partial_stride_split, mask=offs_splits < num_splits, other=float("-inf"))

    lse_max = tl.max(lse)

    o_offsets = o_partial_ptr + batch_idx * o_partial_stride_b + head_idx * o_partial_stride_h
    o_partial = tl.load(
        o_offsets + offs_splits[:, None] * o_partial_stride_split + offs_d[None, :] * o_partial_stride_d,
        mask=offs_splits[:, None] < num_splits,
    )
    sumexp_normalized_splitk = tl.exp(lse - lse_max)
    sumexp_normalized = tl.sum(sumexp_normalized_splitk, axis=0)
    numerator_normalized = tl.sum(o_partial * sumexp_normalized_splitk[:, None], axis=0)
    acc = numerator_normalized / sumexp_normalized
    acc = acc.to(o_ptr.dtype.element_ty)
    o_ptr += batch_idx * o_stride_b + head_idx * o_stride_h
    tl.store(o_ptr + offs_d * o_stride_d, acc)


def block_sparse_flash_decode_gqa_indice_triton(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    max_cache_seqlen,
    max_selected_blocks,
    block_indices,
    block_size,
    sm_scale=None,
):
    batch, heads, dim = q.shape

    if sm_scale is None:
        sm_scale = 1 / math.sqrt(dim)

    _, max_cache_seqlen_cache, heads_kv, dim_v = v_cache.shape
    assert max_cache_seqlen == max_cache_seqlen_cache, "max_cache_seqlen mismatch"
    group_size = heads // heads_kv

    block_H = 16

    num_m_blocks = 1 * (heads // heads_kv + block_H - 1) // block_H
    num_n_blocks = max_selected_blocks

    size_one_kv_head = max_selected_blocks * block_size * (dim + dim_v) * 2  # kv_seqlen * (dim + dim_v) * 2
    total_mblocks = batch * heads_kv * num_m_blocks
    num_sm = 64
    # num_sm = self.num_sm
    num_splits = num_splits_heuristic(
        total_mblocks, num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, is_causal_or_local=True, max_splits=128
    )

    # print("num_splits:", num_splits, "num_blocks:", num_n_blocks)

    num_splits_pow2 = triton.next_power_of_2(num_splits)

    o_partial = torch.empty((batch, heads, num_splits, dim_v), device=q.device, dtype=q.dtype)
    lse_partial = torch.empty((batch, heads, num_splits), device=q.device, dtype=torch.float32)

    BLOCK_D = dim
    BLOCK_H = group_size if group_size > 16 else 16
    grid = (batch, heads_kv, num_splits)
    _split_kernel[grid](
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        o_partial,
        lse_partial,
        block_indices,
        sm_scale,
        num_splits,
        group_size,
        max_selected_blocks,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_indices.stride(0),
        block_indices.stride(1),
        block_indices.stride(2),
        BLOCK_H=BLOCK_H,
        BLOCK_N=block_size,
        BLOCK_D=BLOCK_D,
    )

    output = torch.zeros((batch, heads, dim_v), device=q.device, dtype=q.dtype)
    grid = (batch, heads)
    _merge_kernel[grid](
        o_partial,
        lse_partial,
        output,
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_D=dim_v,
        num_splits=num_splits,
        num_splits_pow2=num_splits_pow2,
    )

    return output


def ref_program_torch(query, key, value, block_indices, cache_seqlens, max_cache_seqlen, num_blocks, block_size):
    batch, heads, dim = query.shape
    heads_kv = key.shape[2]
    dim_v = value.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5
    key = rearrange(key, "b n h d -> b h n d")  # [batch_size, heads_kv, seqlen_kv, dim]
    value = rearrange(value, "b n h d -> b h n d")  # [batch_size, heads_kv, seqlen_kv, dim]

    query = rearrange(query, "b (h g) d -> b g h d", g=num_head_groups)  # [batch_size, num_head_groups, heads_kv, dim]

    scores = einsum(query, key, "b g h d, b h s d -> b g h s")  # [batch_size, num_head_groups, heads_kv, seqlen_kv]

    sparse_mask = torch.zeros_like(scores)
    # Assign mask values based on block_indices
    for b in range(batch):
        for h in range(heads_kv):
            valid_indices = block_indices[b, h]  # Extract indices for this batch and head
            for idx in valid_indices:
                if idx >= 0:
                    sparse_mask[b, :, h, idx * block_size : (idx + 1) * block_size] = 1
    scores = scores.masked_fill(sparse_mask == 0, float("-inf"))

    range_len = torch.arange(scores.shape[-1], device="cuda").unsqueeze(0)
    cache_seqlens_expanded = cache_seqlens.unsqueeze(1)
    pad_mask = range_len >= cache_seqlens_expanded
    pad_mask = pad_mask[:, None, None, :]
    scores = scores.masked_fill(pad_mask, float("-inf"))
    attention = F.softmax(scores / scale, dim=-1)  # [batch_size, num_head_groups, heads_kv, seqlen_kv]

    out = einsum(attention, value, "b g h s, b h s d -> b g h d")  # [batch_size, num_head_groups, heads_kv, dim]
    out = rearrange(out, "b g h d -> b (h g) d")  # [batch_size, heads, dim]
    return out


def ref_program_fa(query, key, value, cache_seqlens):
    # latency reference
    # from flash_attn_interface import flash_attn_with_kvcache # fa3
    from flash_attn import flash_attn_with_kvcache  # fa2

    query = query.unsqueeze(1)
    output = flash_attn_with_kvcache(query, key, value, cache_seqlens=cache_seqlens)
    output = output.squeeze(1)
    return output


def main(batch=64, heads=32, heads_kv=8, max_cache_seqlen=8192, dim=128, dim_v=128, sparse_ratio=0.8, block_size=32):
    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = batch, heads, heads_kv, max_cache_seqlen, dim, dim_v
    sparse_ratio = sparse_ratio
    block_size = block_size
    qk_flops = 2 * batch * heads * max_cache_seqlen * dim
    pv_flops = 2 * batch * heads * max_cache_seqlen * dim_v
    total_flops = qk_flops + pv_flops

    max_selected_blocks = int(math.ceil(max_cache_seqlen * (1 - sparse_ratio) / block_size))
    print("max_selected_blocks: ", max_selected_blocks)
    dtype = torch.float16
    block_H = 64

    Q = torch.randn((batch, heads, dim), dtype=dtype, device="cuda")
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device="cuda")
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device="cuda")
    cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32, device="cuda")
    # cache_seqlens = torch.full((batch,), max_cache_seqlen, dtype=torch.int32, device='cuda')
    # Ensure at least one element equals cache_seqlen
    random_index = torch.randint(0, batch, (1,), device="cuda").item()  # Select a random index
    cache_seqlens[random_index] = max_cache_seqlen  # Assign cache_seqlen to ensure at least one occurrence

    print("cache_seqlens: ", cache_seqlens)

    max_valid_num_blocks = torch.ceil(cache_seqlens / block_size).int()
    print("max_valid_num_blocks: ", max_valid_num_blocks)
    # Initialize block_indices with -1 (for padding blocks)
    block_indices = torch.full((batch, heads_kv, max_selected_blocks), -1, dtype=torch.int32, device="cuda")

    # Assign valid indices while ensuring no duplicates within each batch-group
    for b in range(batch):
        max_valid_block = max_valid_num_blocks[b].item()  # Max valid blocks for this batch
        if max_valid_block > 0:  # Ensure there's at least one valid block
            for h in range(heads_kv):
                valid_indices = torch.randperm(max_valid_block, device="cuda", dtype=torch.int32)[:max_selected_blocks]
                block_indices[b, h, : len(valid_indices)] = valid_indices

    # Sort indices within each batch-group for consistency
    block_indices, _ = block_indices.sort(dim=-1, descending=True)
    # print("block_indices: ", block_indices)
    actual_num_blocks = torch.sum(block_indices != -1, dim=-1).to(torch.int32)[:, 0]
    print("actual_num_blocks: ", actual_num_blocks)
    # print(block_indices.shape, actual_num_blocks.shape)

    max_num_blocks = torch.max(max_valid_num_blocks).item()
    print("max_num_blocks: ", max_num_blocks)

    ref = ref_program_torch(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen, max_num_blocks, block_size)

    triton_out = block_sparse_flash_decode_gqa_indice_triton(
        Q,
        K,
        V,
        cache_seqlens,
        max_cache_seqlen,
        max_selected_blocks,
        block_indices,
        block_size,
    )

    print("max difference: ", torch.max(torch.abs(ref - triton_out)))
    assert torch.allclose(ref, triton_out, atol=1e-2), "Output mismatch between Triton and reference implementation"
    print("Passed the ref test!")

    # Measure performance
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        block_sparse_flash_decode_gqa_indice_triton(
            Q,
            K,
            V,
            cache_seqlens,
            max_cache_seqlen,
            max_selected_blocks,
            block_indices,
            block_size,
        )
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    avg_time = elapsed_time / 1000
    avg_flops = total_flops / avg_time
    print(f"Average time: {avg_time:.6f} seconds")

    # Measure performance of reference implementation
    import flash_attn  # noqa: F401

    start = time.time()
    for _ in range(1000):
        ref_program_fa(Q, K, V, cache_seqlens)
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time_ref = end - start
    avg_time_ref = elapsed_time_ref / 1000
    avg_flops_ref = total_flops / avg_time_ref
    print(f"Average time of ref: {avg_time_ref:.6f} seconds")

    print(f"Speedup: {avg_time_ref / avg_time:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="heads")
    parser.add_argument("--heads_kv", type=int, default=8, help="heads_kv")
    parser.add_argument("--max_cache_seqlen", type=int, default=8192, help="kvcache sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--dim_v", type=int, default=128, help="dim_v")
    parser.add_argument("--sparse_ratio", type=float, default=0.8, help="sparse ratio")
    parser.add_argument("--block_size", type=int, default=32, help="block_size")
    args = parser.parse_args()
    main(args.batch, args.heads, args.heads_kv, args.max_cache_seqlen, args.dim, args.dim_v, args.sparse_ratio, args.block_size)
