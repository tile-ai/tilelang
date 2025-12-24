# ruff: noqa
import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum
import argparse
import time
import math
from tilelang.profiler import do_bench

from heuristic import num_splits_heuristic


def flashattn(batch, heads, heads_kv, dim, dim_v):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // heads_kv

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def kernel_func(
        block_N, block_H, page_block_size, num_split, num_stages, threads, num_pages, max_num_blocks_per_seq, max_selected_blocks
    ):
        shape_q = [batch, heads, dim]
        shape_k = [num_pages, page_block_size, heads_kv, dim]
        shape_v = [num_pages, page_block_size, heads_kv, dim_v]
        shape_indices = [batch, heads_kv, max_selected_blocks]
        shape_block_table = [batch, max_num_blocks_per_seq]
        shape_o = [batch, heads, dim_v]
        part_shape = [batch, heads, num_split, dim_v]
        valid_block_H = min(block_H, kv_group_num)
        assert block_N <= page_block_size and page_block_size % block_N == 0
        block_ratio = page_block_size // block_N

        @T.prim_func
        def main(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            block_indices: T.Tensor(shape_indices, T.int32),
            cache_seqlens: T.Tensor([batch], T.int32),
            block_table: T.Tensor(shape_block_table, T.int32),
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            # flash_attn_split
            with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim_v], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim_v], accum_dtype)

                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)
                has_valid_block = T.alloc_var("bool")

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                num_blocks = max_selected_blocks
                blocks_per_split = T.floordiv(num_blocks, num_split)
                remaining_blocks = T.floormod(num_blocks, num_split)
                loop_range = blocks_per_split + T.if_then_else(sid < remaining_blocks, 1, 0)
                start = blocks_per_split * sid + T.min(sid, remaining_blocks)
                has_valid_block = False
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    logical_block_idx = block_indices[bid, cur_kv_head, start + k]
                    if logical_block_idx >= 0:
                        has_valid_block = True
                        block_table_idx = T.floordiv(logical_block_idx, block_ratio)
                        block_tile_idx = T.floormod(logical_block_idx, block_ratio)
                        physical_block_idx = block_table[bid, block_table_idx]
                        T.copy(K[physical_block_idx, block_tile_idx * block_N : (block_tile_idx + 1) * block_N, cur_kv_head, :], K_shared)
                        T.clear(acc_s)
                        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        if k == 0:  # assume block_indices is sorted in reverse order, otherwise, remove this if condition
                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i, j] = T.if_then_else(
                                    logical_block_idx * block_N + j >= cache_seqlens[bid], -T.infinity(accum_dtype), acc_s[i, j]
                                )
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_H):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_H):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Parallel(block_H, dim_v):
                            acc_o[i, j] *= scores_scale[i]
                        T.copy(V[physical_block_idx, block_tile_idx * block_N : (block_tile_idx + 1) * block_N, cur_kv_head, :], V_shared)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                if has_valid_block:
                    for i, j in T.Parallel(block_H, dim_v):
                        acc_o[i, j] /= logsum[i]

                    for i in T.Parallel(block_H):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                for i in T.Parallel(block_H):
                    if i < valid_block_H:
                        glse[bid, hid * valid_block_H + i, sid] = logsum[i]

                for i, j in T.Parallel(block_H, dim_v):
                    if i < valid_block_H:
                        Output_partial[bid, hid * valid_block_H + i, sid, j] = acc_o[i, j]

            # combine
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim_v], accum_dtype)
                o_accum_local = T.alloc_fragment([dim_v], accum_dtype)
                lse_local_split = T.alloc_var(accum_dtype)
                lse_logsum_local = T.alloc_var(accum_dtype)
                lse_max_local = T.alloc_var(accum_dtype)
                scale_local = T.alloc_var(accum_dtype)
                max_split = T.alloc_var(T.int32)

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                lse_max_local = -T.infinity(accum_dtype)
                for k in T.serial(num_split):
                    lse_local_split = glse[bz, by, k]
                    if lse_local_split != 0:
                        max_split = k
                        lse_max_local = T.max(lse_max_local, glse[bz, by, k])

                for k in T.Pipelined(num_split, num_stages=1):
                    if k <= max_split:
                        lse_local_split = glse[bz, by, k]
                        lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
                lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
                for k in T.serial(num_split):
                    if k <= max_split:
                        for i in T.Parallel(dim_v):
                            po_local[i] = Output_partial[bz, by, k, i]
                        lse_local_split = glse[bz, by, k]
                        scale_local = T.exp2(lse_local_split - lse_logsum_local)
                        for i in T.Parallel(dim_v):
                            o_accum_local[i] += po_local[i] * scale_local
                for i in T.Parallel(dim_v):
                    Output[bz, by, i] = o_accum_local[i]

        return main

    return kernel_func


class SparseFlashAttn(torch.nn.Module):
    def __init__(self, batch, heads, heads_kv, dim, dim_v, page_block_size, block_N, num_pages):
        super(SparseFlashAttn, self).__init__()
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.dim_v = dim_v
        self.block_N = block_N
        self.page_block_size = page_block_size
        self.num_pages = num_pages
        self.block_H = 64

        self.kernel = flashattn(batch, heads, heads_kv, dim, dim_v)(
            block_N=block_N,
            block_H=self.block_H,
            page_block_size=page_block_size,
            num_split=T.dynamic("num_split"),
            num_stages=2,
            threads=128,
            num_pages=num_pages,
            max_num_blocks_per_seq=T.dynamic("max_num_blocks_per_seq"),
            max_selected_blocks=T.dynamic("max_selected_blocks"),
        )

        props = torch.cuda.get_device_properties(torch.device("cuda:0"))
        self.num_sm = props.multi_processor_count

    def forward(self, query, key, value, block_indices, cache_seqlens, block_table):
        batch = self.batch
        heads = self.heads
        heads_kv = self.heads_kv
        dim_v = self.dim_v
        dim = self.dim
        block_size = self.block_N
        max_selected_blocks = block_indices.shape[-1]

        # Compute static scheduling parameters
        num_m_blocks = 1 * (heads // heads_kv + self.block_H - 1) // self.block_H
        num_n_blocks = max_selected_blocks
        size_one_kv_head = max_selected_blocks * block_size * (dim + dim_v) * 2
        total_mblocks = batch * heads_kv * num_m_blocks

        num_sm = self.num_sm

        num_split = num_splits_heuristic(
            total_mblocks, num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, is_causal_or_local=True, max_splits=128
        )

        glse = torch.empty((batch, heads, num_split), dtype=torch.float32, device="cuda")
        output_partial = torch.empty((batch, heads, num_split, dim_v), dtype=torch.float32, device="cuda")

        output = self.kernel(
            query,
            key,
            value,
            block_indices,
            cache_seqlens,
            block_table,
            glse,
            output_partial,
        )
        return output


def ref_program_torch_paged(query, key_cache, value_cache, block_indices, cache_seqlens, block_table, page_block_size, block_size):
    """
    Paged version of sparse attention reference implementation.

    Args:
        query: [batch, heads, dim]
        key_cache: [num_pages, page_block_size, heads_kv, dim]
        value_cache: [num_pages, page_block_size, heads_kv, dim]
        block_indices: [batch, heads_kv, max_selected_blocks] - logical block indices
        cache_seqlens: [batch] - actual sequence lengths
        block_table: [batch, max_num_blocks_per_seq] - maps logical to physical blocks
        page_block_size: size of each page block
        block_size: size of attention blocks (block_N)
    """
    batch, heads, dim = query.shape
    heads_kv = key_cache.shape[2]
    dim_v = value_cache.shape[3]
    num_head_groups = heads // heads_kv
    scale = dim**0.5

    # Reconstruct the full key and value tensors from paged cache
    max_cache_seqlen = max(cache_seqlens).item()
    key_full = torch.zeros((batch, heads_kv, max_cache_seqlen, dim), dtype=key_cache.dtype, device=key_cache.device)
    value_full = torch.zeros((batch, heads_kv, max_cache_seqlen, dim_v), dtype=value_cache.dtype, device=value_cache.device)

    # Reconstruct full tensors from paged cache using block_table
    for b in range(batch):
        seq_len = cache_seqlens[b].item()
        num_blocks_needed = int(math.ceil(seq_len / page_block_size))

        for block_idx in range(num_blocks_needed):
            physical_block_idx = block_table[b, block_idx].item()

            # Calculate the range of tokens for this block
            start_token = block_idx * page_block_size
            end_token = min(start_token + page_block_size, seq_len)
            actual_block_size = end_token - start_token

            # Copy from paged cache to full tensors
            key_full[b, :, start_token:end_token, :] = key_cache[physical_block_idx, :actual_block_size, :, :].transpose(0, 1)
            value_full[b, :, start_token:end_token, :] = value_cache[physical_block_idx, :actual_block_size, :, :].transpose(0, 1)

    # Reshape query for grouped attention
    query = rearrange(query, "b (h g) d -> b g h d", g=num_head_groups)  # [batch_size, num_head_groups, heads_kv, dim]

    # Compute attention scores
    scores = einsum(query, key_full, "b g h d, b h s d -> b g h s")  # [batch_size, num_head_groups, heads_kv, seqlen_kv]

    # Create sparse mask based on block_indices
    sparse_mask = torch.zeros_like(scores)

    # Apply sparse mask based on selected blocks
    for b in range(batch):
        for h in range(heads_kv):
            valid_indices = block_indices[b, h]  # Extract indices for this batch and head
            for idx in valid_indices:
                if idx >= 0:  # Valid block index
                    start_pos = idx * block_size
                    end_pos = min(start_pos + block_size, max_cache_seqlen)
                    sparse_mask[b, :, h, start_pos:end_pos] = 1

    # Apply sparse mask
    scores = scores.masked_fill(sparse_mask == 0, float("-inf"))

    # Apply causal mask based on actual sequence lengths
    range_len = torch.arange(scores.shape[-1], device=scores.device).unsqueeze(0)
    cache_seqlens_expanded = cache_seqlens.unsqueeze(1)
    pad_mask = range_len >= cache_seqlens_expanded
    pad_mask = pad_mask[:, None, None, :]
    scores = scores.masked_fill(pad_mask, float("-inf"))

    # Compute attention weights
    attention = F.softmax(scores / scale, dim=-1)

    # Apply attention to values
    out = einsum(attention, value_full, "b g h s, b h s d -> b g h d")  # [batch_size, num_head_groups, heads_kv, dim]

    # Reshape output back to original format
    out = rearrange(out, "b g h d -> b (h g) d")  # [batch_size, heads, dim]

    return out


def ref_program_fa(query, kcache, vcache, cache_seqlens, block_table):
    # latency reference
    # from flash_attn_interface import flash_attn_with_kvcache # fa3
    from flash_attn import flash_attn_with_kvcache  # fa2

    query = query.unsqueeze(1)
    output = flash_attn_with_kvcache(query, kcache, vcache, cache_seqlens=cache_seqlens, block_table=block_table)
    output = output.squeeze(1)
    return output


def main(args):
    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = (
        args.batch,
        args.heads,
        args.heads_kv,
        args.max_cache_seqlen,
        args.dim,
        args.dim_v,
    )
    sparse_ratio = args.sparse_ratio
    block_N = args.block_N
    page_block_size = args.page_block_size
    num_blocks = args.num_pages  # Use num_pages from args

    # For dense case verification, set sparse_ratio to 0 to select all blocks
    max_selected_blocks = int(math.ceil(max_cache_seqlen / block_N))
    print("max_selected_blocks: ", max_selected_blocks)
    dtype = torch.float16

    # Generate random inputs
    Q = torch.randn((batch, heads, dim), dtype=dtype, device="cuda")
    cache_seqlens = torch.randint(max_cache_seqlen // 2, max_cache_seqlen + 1, (batch,), dtype=torch.int32, device="cuda")
    print("cache_seqlens: ", cache_seqlens)

    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device="cuda")
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device="cuda")

    # Create paged KV cache
    K_cache = torch.zeros((num_blocks, page_block_size, heads_kv, dim), dtype=dtype, device="cuda")
    V_cache = torch.zeros((num_blocks, page_block_size, heads_kv, dim_v), dtype=dtype, device="cuda")

    # Create block table and block indices for dense case (all blocks selected)
    max_num_blocks_per_seq = int(math.ceil(max_cache_seqlen / page_block_size))
    print("max_num_blocks_per_seq: ", max_num_blocks_per_seq)
    block_table = torch.zeros((batch, max_num_blocks_per_seq), dtype=torch.int32, device="cuda")
    block_indices = torch.zeros((batch, heads_kv, max_selected_blocks), dtype=torch.int32, device="cuda")

    # Fill block table and block indices and cache

    # Create a pool of available physical blocks
    total_blocks_needed = sum(int(math.ceil(cache_seqlens[seq_idx].item() / page_block_size)) for seq_idx in range(batch))
    available_blocks = list(range(total_blocks_needed))
    import random

    random.seed(42)  # For reproducibility
    random.shuffle(available_blocks)

    # Fill block table with random physical block indices
    block_assignment = {}  # Map (seq_idx, block_idx) -> physical_block_idx
    block_idx_counter = 0

    for seq_idx in range(batch):
        seq_len = cache_seqlens[seq_idx].item()
        num_blocks_needed = int(math.ceil(seq_len / page_block_size))

        # Assign random physical blocks for each sequence
        for block_idx in range(num_blocks_needed):
            physical_block_idx = available_blocks[block_idx_counter]
            block_table[seq_idx, block_idx] = physical_block_idx
            block_assignment[(seq_idx, block_idx)] = physical_block_idx
            block_idx_counter += 1

    print(f"Block table: {block_table}")

    # Fill K_cache and V_cache with data from original K and V tensors using random block assignment
    for seq_idx in range(batch):
        seq_len = cache_seqlens[seq_idx].item()
        num_blocks_needed = int(math.ceil(seq_len / page_block_size))

        for block_idx in range(num_blocks_needed):
            physical_block_idx = block_assignment[(seq_idx, block_idx)]

            # Calculate the range of tokens for this block
            start_token = block_idx * page_block_size
            end_token = min(start_token + page_block_size, seq_len)
            actual_block_size = end_token - start_token

            # Copy K and V data to the paged cache
            K_cache[physical_block_idx, :actual_block_size, :, :] = K[seq_idx, start_token:end_token, :, :]
            V_cache[physical_block_idx, :actual_block_size, :, :] = V[seq_idx, start_token:end_token, :, :]

    # Fill block_indices for sparse attention
    # For dense case (verification), we select all blocks in reverse order
    # For sparse case, we select a subset of blocks based on sparse_ratio
    for seq_idx in range(batch):
        seq_len = cache_seqlens[seq_idx].item()
        num_tile = int(math.ceil(seq_len / block_N))

        if sparse_ratio == 0.0:
            # Dense case: select all blocks in reverse order
            selected_blocks = min(num_tile, max_selected_blocks)
            for head_idx in range(heads_kv):
                for i in range(selected_blocks):
                    # Select blocks in reverse order (most recent first)
                    block_indices[seq_idx, head_idx, i] = num_tile - 1 - i
                # Fill remaining slots with -1 (invalid)
                for i in range(selected_blocks, max_selected_blocks):
                    block_indices[seq_idx, head_idx, i] = -1
        else:
            # Fill block_indices for all KV heads
            num_selected = int(num_tile * (1.0 - sparse_ratio))
            num_selected = max(1, min(num_selected, max_selected_blocks))
            all_blocks = list(range(num_tile))
            for head_idx in range(heads_kv):
                selected_blocks = []
                # Always include the most recent blocks
                recent_blocks = 1
                selected_blocks.append(num_tile - 1)

                # Randomly select some earlier blocks
                if num_selected > recent_blocks:
                    remaining_blocks = [b for b in all_blocks if b not in selected_blocks]
                    if remaining_blocks:
                        import random

                        random.seed(42)  # For reproducibility
                        additional_blocks = random.sample(remaining_blocks, min(num_selected - recent_blocks, len(remaining_blocks)))
                        selected_blocks.extend(additional_blocks)

                # Sort selected blocks in reverse order (most recent first)
                selected_blocks.sort(reverse=True)

                for i in range(len(selected_blocks)):
                    block_indices[seq_idx, head_idx, i] = selected_blocks[i]
                # Fill remaining slots with -1 (invalid)
                for i in range(len(selected_blocks), max_selected_blocks):
                    block_indices[seq_idx, head_idx, i] = -1

    # Initialize sparse attention module
    sparse_attn = SparseFlashAttn(batch, heads, heads_kv, dim, dim_v, page_block_size, block_N, num_blocks)
    output_sparse = sparse_attn.forward(Q, K_cache, V_cache, block_indices, cache_seqlens, block_table)

    import flash_attn  # noqa: F401

    output_ref_torch = ref_program_torch_paged(Q, K_cache, V_cache, block_indices, cache_seqlens, block_table, page_block_size, block_N)

    output_ref_fa = ref_program_fa(Q, K_cache, V_cache, cache_seqlens, block_table)
    # Check correctness
    if sparse_ratio == 0.0:
        max_diff = torch.max(torch.abs(output_sparse - output_ref_fa)).item()
        mean_diff = torch.mean(torch.abs(output_sparse - output_ref_fa)).item()
        assert torch.allclose(output_ref_fa, output_ref_torch, atol=1e-2), "Reference outputs do not match!"
    else:
        max_diff = torch.max(torch.abs(output_sparse - output_ref_torch)).item()
        mean_diff = torch.mean(torch.abs(output_sparse - output_ref_torch)).item()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-2:
        print("✓ Verification PASSED: Results match within tolerance")
    else:
        print("✗ Verification FAILED: Results differ significantly")

    # Performance measurement
    for _ in range(10):  # Warm-up
        sparse_attn.forward(Q, K_cache, V_cache, block_indices, cache_seqlens, block_table)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):  # Run multiple times for averaging
        sparse_attn.forward(Q, K_cache, V_cache, block_indices, cache_seqlens, block_table)
    torch.cuda.synchronize()
    end_time = time.time()

    kernel_time = (end_time - start_time) / 100 * 1000  # Convert to ms
    print(f"Kernel execution time: {kernel_time:.2f} ms")

    # FA performance measurement
    for _ in range(10):  # Warm-up
        ref_program_fa(Q, K_cache, V_cache, cache_seqlens, block_table)

    torch.cuda.synchronize()
    start_time_fa = time.time()
    for _ in range(100):  # Run multiple times for averaging
        ref_program_fa(Q, K_cache, V_cache, cache_seqlens, block_table)
    torch.cuda.synchronize()
    end_time_fa = time.time()
    kernel_time_fa = (end_time_fa - start_time_fa) / 100 * 1000  # Convert to ms
    print(f"FA kernel execution time: {kernel_time_fa:.2f} ms")

    print(f"Speedup: {kernel_time_fa / kernel_time:.2f}x")


def run_regression_perf(args):
    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = (
        args.batch,
        args.heads,
        args.heads_kv,
        args.max_cache_seqlen,
        args.dim,
        args.dim_v,
    )
    sparse_ratio = args.sparse_ratio
    block_N = args.block_N
    page_block_size = args.page_block_size
    num_blocks = args.num_pages
    max_selected_blocks = int(math.ceil(max_cache_seqlen / block_N))
    dtype = torch.float16
    Q = torch.randn((batch, heads, dim), dtype=dtype, device="cuda")
    cache_seqlens = torch.randint(max_cache_seqlen // 2, max_cache_seqlen + 1, (batch,), dtype=torch.int32, device="cuda")
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device="cuda")
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device="cuda")
    K_cache = torch.zeros((num_blocks, page_block_size, heads_kv, dim), dtype=dtype, device="cuda")
    V_cache = torch.zeros((num_blocks, page_block_size, heads_kv, dim_v), dtype=dtype, device="cuda")
    max_num_blocks_per_seq = int(math.ceil(max_cache_seqlen / page_block_size))
    block_table = torch.zeros((batch, max_num_blocks_per_seq), dtype=torch.int32, device="cuda")
    block_indices = torch.zeros((batch, heads_kv, max_selected_blocks), dtype=torch.int32, device="cuda")
    total_blocks_needed = sum(int(math.ceil(cache_seqlens[seq_idx].item() / page_block_size)) for seq_idx in range(batch))
    available_blocks = list(range(total_blocks_needed))
    import random

    random.seed(42)
    random.shuffle(available_blocks)
    block_assignment = {}
    block_idx_counter = 0
    for seq_idx in range(batch):
        seq_len = cache_seqlens[seq_idx].item()
        num_blocks_needed = int(math.ceil(seq_len / page_block_size))
        for block_idx in range(num_blocks_needed):
            physical_block_idx = available_blocks[block_idx_counter]
            block_table[seq_idx, block_idx] = physical_block_idx
            block_assignment[(seq_idx, block_idx)] = physical_block_idx
            block_idx_counter += 1
    for seq_idx in range(batch):
        seq_len = cache_seqlens[seq_idx].item()
        num_blocks_needed = int(math.ceil(seq_len / page_block_size))
        for block_idx in range(num_blocks_needed):
            physical_block_idx = block_assignment[(seq_idx, block_idx)]
            start_token = block_idx * page_block_size
            end_token = min(start_token + page_block_size, seq_len)
            actual_block_size = end_token - start_token
            K_cache[physical_block_idx, :actual_block_size, :, :] = K[seq_idx, start_token:end_token, :, :]
            V_cache[physical_block_idx, :actual_block_size, :, :] = V[seq_idx, start_token:end_token, :, :]
    for seq_idx in range(batch):
        seq_len = cache_seqlens[seq_idx].item()
        num_tile = int(math.ceil(seq_len / block_N))
        if sparse_ratio == 0.0:
            selected_blocks = min(num_tile, max_selected_blocks)
            for head_idx in range(heads_kv):
                for i in range(selected_blocks):
                    block_indices[seq_idx, head_idx, i] = num_tile - 1 - i
                for i in range(selected_blocks, max_selected_blocks):
                    block_indices[seq_idx, head_idx, i] = -1
        else:
            num_selected = int(num_tile * (1.0 - sparse_ratio))
            num_selected = max(1, min(num_selected, max_selected_blocks))
            all_blocks = list(range(num_tile))
            for head_idx in range(heads_kv):
                selected_blocks = []
                recent_blocks = 1
                selected_blocks.append(num_tile - 1)
                if num_selected > recent_blocks:
                    remaining_blocks = [b for b in all_blocks if b not in selected_blocks]
                    if remaining_blocks:
                        import random

                        random.seed(42)
                        additional_blocks = random.sample(remaining_blocks, min(num_selected - recent_blocks, len(remaining_blocks)))
                        selected_blocks.extend(additional_blocks)

                selected_blocks.sort(reverse=True)

                for i in range(len(selected_blocks)):
                    block_indices[seq_idx, head_idx, i] = selected_blocks[i]
                for i in range(len(selected_blocks), max_selected_blocks):
                    block_indices[seq_idx, head_idx, i] = -1

    sparse_attn = SparseFlashAttn(batch, heads, heads_kv, dim, dim_v, page_block_size, block_N, num_blocks)
    kernel = sparse_attn.kernel
    batch = sparse_attn.batch
    heads = sparse_attn.heads
    heads_kv = sparse_attn.heads_kv
    dim_v = sparse_attn.dim_v
    dim = sparse_attn.dim
    block_size = sparse_attn.block_N
    max_selected_blocks = block_indices.shape[-1]

    num_m_blocks = 1 * (heads // heads_kv + sparse_attn.block_H - 1) // sparse_attn.block_H
    num_n_blocks = max_selected_blocks
    size_one_kv_head = max_selected_blocks * block_size * (dim + dim_v) * 2
    total_mblocks = batch * heads_kv * num_m_blocks

    num_sm = sparse_attn.num_sm

    num_split = num_splits_heuristic(
        total_mblocks, num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, is_causal_or_local=True, max_splits=128
    )

    glse = torch.empty((batch, heads, num_split), dtype=torch.float32, device="cuda")
    output_partial = torch.empty((batch, heads, num_split, dim_v), dtype=torch.float32, device="cuda")

    def run_kernel_only():
        kernel(
            Q,
            K_cache,
            V_cache,
            block_indices,
            cache_seqlens,
            block_table,
            glse,
            output_partial,
        )

    return do_bench(run_kernel_only, warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="heads")
    parser.add_argument("--heads_kv", type=int, default=8, help="heads_kv")
    parser.add_argument("--max_cache_seqlen", type=int, default=8192, help="kvcache sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--dim_v", type=int, default=128, help="dim_v")
    parser.add_argument("--sparse_ratio", type=float, default=0.0, help="sparse ratio")
    parser.add_argument("--block_N", type=int, default=64, help="block_N")
    parser.add_argument("--page_block_size", type=int, default=256, help="block size of pages")
    parser.add_argument("--num_pages", type=int, default=1024, help="total number of pages")
    args = parser.parse_args()
    main(args)
