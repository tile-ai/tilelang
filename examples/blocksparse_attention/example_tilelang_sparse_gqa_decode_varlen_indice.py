import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from einops import rearrange, einsum
import argparse
import time
import math
from heuristic import num_splits_heuristic
from tilelang.profiler import do_bench


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
    def kernel_func(block_N, block_H, num_split, num_stages, threads, max_cache_seqlen, max_selected_blocks):
        shape_q = [batch, heads, dim]
        shape_k = [batch, max_cache_seqlen, heads_kv, dim]
        shape_v = [batch, max_cache_seqlen, heads_kv, dim_v]
        shape_indices = [batch, heads_kv, max_selected_blocks]
        shape_o = [batch, heads, dim_v]
        part_shape = [batch, heads, num_split, dim_v]
        valid_block_H = min(block_H, kv_group_num)

        @T.prim_func
        def main(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            block_indices: T.Tensor(shape_indices, T.int32),
            cache_seqlens: T.Tensor([batch], T.int32),
            # actual_num_blocks: T.Tensor([batch], T.int32),
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            # flash_attn_split(Q, K, V, block_indices, cache_seqlens, actual_num_blocks, glse, Output_partial)
            # flash_attn_split
            with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim_v], dtype)
                # O_shared = T.alloc_shared([valid_block_H, dim_v], dtype)
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
                    i_s = block_indices[bid, cur_kv_head, start + k]
                    if i_s >= 0:
                        has_valid_block = True
                        T.copy(K[bid, i_s * block_N : (i_s + 1) * block_N, cur_kv_head, :], K_shared)
                        T.clear(acc_s)
                        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        if k == 0:  # assume block_indices is sorted in reverse order, otherwise, remove this if condition
                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i, j] = T.if_then_else(i_s * block_N + j >= cache_seqlens[bid], -T.infinity(accum_dtype), acc_s[i, j])
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
                        T.copy(V[bid, i_s * block_N : (i_s + 1) * block_N, cur_kv_head, :], V_shared)
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
    def __init__(self, batch, heads, heads_kv, dim, dim_v, block_size):
        super(SparseFlashAttn, self).__init__()
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.dim_v = dim_v
        self.block_size = block_size

        self.block_H = 64

        self.kernel = flashattn(batch, heads, heads_kv, dim, dim_v)(
            block_N=block_size,
            block_H=self.block_H,
            num_split=T.dynamic("num_split"),
            num_stages=2,
            threads=128,
            max_cache_seqlen=T.dynamic("max_cache_seqlen"),
            max_selected_blocks=T.dynamic("max_selected_blocks"),
        )

        props = torch.cuda.get_device_properties(torch.device("cuda:0"))
        self.num_sm = props.multi_processor_count

    def forward(self, query, key, value, block_indices, cache_seqlens):
        batch = self.batch
        heads = self.heads
        heads_kv = self.heads_kv
        dim_v = self.dim_v
        dim = self.dim
        block_size = self.block_size
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

        output = self.kernel(query, key, value, block_indices, cache_seqlens, glse, output_partial)
        return output


def sparse_gqa_decode_varlen_indice(query, key, value, block_indices, cache_seqlens, max_cache_seqlen, block_size):
    """
    Args:
        query: [batch, heads, dim]
        key: [batch, max_cache_seqlen, heads_kv, dim]
        value: [batch, max_cache_seqlen, heads_kv, dim_v]
        block_indices: [batch, heads_kv, max_selected_blocks], indices of selected blocks, -1 for padding
        cache_seqlens: [batch], sequence lengths of the kvcache
        max_cache_seqlen: maximum sequence length of kvcache
        block_size: block size
    Returns:
        output: [batch, heads, dim_v]

    """

    batch, heads, dim = query.shape
    heads_kv = key.shape[2]
    dim_v = value.shape[-1]
    max_selected_blocks = block_indices.shape[-1]
    block_H = 64

    actual_num_blocks = torch.sum(block_indices != -1, dim=-1).to(torch.int32)
    actual_num_blocks = actual_num_blocks[
        :, 0
    ]  # [batch],  number of valid blocks, assume all groups in the same batch have the same number of blocks

    # get num_split
    num_m_blocks = 1 * (heads // heads_kv + block_H - 1) // block_H
    num_n_blocks = max_selected_blocks  # (kv_seqlen  + block_size - 1 ) // block_size
    # num_n_blocks = torch.sum(actual_num_blocks, dim=-1).item() * heads_kv # total number of blocks

    size_one_kv_head = max_selected_blocks * block_size * (dim + dim_v) * 2  # kv_seqlen * (dim + dim_v) * 2
    total_mblocks = batch * heads_kv * num_m_blocks
    num_sm = 132
    num_split = num_splits_heuristic(
        total_mblocks, num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, is_causal_or_local=True, max_splits=128
    )

    glse = torch.empty((batch, heads, num_split), dtype=torch.float32, device="cuda")
    Output_partial = torch.empty((batch, heads, num_split, dim_v), dtype=torch.float32, device="cuda")
    kernel = flashattn(batch, heads, heads_kv, dim, dim_v)(
        block_N=block_size,
        block_H=block_H,
        num_split=T.dynamic("num_split"),
        num_stages=2,
        threads=128,
        max_cache_seqlen=T.dynamic("max_cache_seqlen"),
        max_selected_blocks=T.dynamic("max_selected_blocks"),
    )

    output = kernel(query, key, value, block_indices, cache_seqlens, glse, Output_partial)
    return output


def ref_program_torch(query, key, value, block_indices, cache_seqlens, max_cache_seqlen, num_blocks, block_size):
    batch, heads, dim = query.shape
    heads_kv = key.shape[2]
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


def ref_program_fa(query, key, value, block_indices, cache_seqlens, max_cache_seqlen, num_blocks, block_size):
    # latency reference
    # from flash_attn_interface import flash_attn_with_kvcache # fa3
    from flash_attn import flash_attn_with_kvcache  # fa2

    query = query.unsqueeze(1)
    output = flash_attn_with_kvcache(query, key, value, cache_seqlens=cache_seqlens)
    output = output.squeeze(1)
    return output


def debug(name, expect, actual, atol=1e-3, rtol=1e-3):
    all_close = torch.allclose(expect, actual, atol=atol, rtol=rtol)
    print(name + "  all_close={}".format(all_close))
    if not all_close:
        diff = (expect - actual).abs()
        print("all_close={}, max={}, min={}, mean={}".format(all_close, diff.max().item(), diff.min().item(), diff.mean().item()))
        max_indices = torch.nonzero(diff == diff.max().item())
        first_index = tuple(max_indices[0].tolist())
        print(f"Index: {first_index}, expect: {expect[first_index]}, actual: {actual[first_index]}")


def main(batch=8, heads=32, heads_kv=8, max_cache_seqlen=8192, dim=128, dim_v=128, sparse_ratio=0.8, block_size=32):
    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = batch, heads, heads_kv, max_cache_seqlen, dim, dim_v
    sparse_ratio = sparse_ratio
    block_size = block_size
    max_selected_blocks = int(math.ceil(max_cache_seqlen * (1 - sparse_ratio) / block_size))
    print("max_selected_blocks: ", max_selected_blocks)
    dtype = torch.float16

    Q = torch.randn((batch, heads, dim), dtype=dtype, device="cuda")
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device="cuda")
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device="cuda")
    cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32, device="cuda")
    # cache_seqlens = torch.full((batch,), max_cache_seqlen, dtype=torch.int32, device='cuda')
    # # Ensure at least one element equals cache_seqlen
    # random_index = torch.randint(0, batch, (1,), device='cuda').item()  # Select a random index
    # # cache_seqlens[random_index] = max_cache_seqlen  # Assign cache_seqlen to ensure at least one occurrence

    print("cache_seqlens: ", cache_seqlens)

    max_valid_num_blocks = torch.ceil(cache_seqlens / block_size).int()
    print("max_valid_num_blocks: ", max_valid_num_blocks)
    # Initialize block_indices with -1 (for padding blocks)
    block_indices = torch.full((batch, heads_kv, max_selected_blocks), -1, dtype=torch.int32, device="cuda")
    # max_num_blocks = int((max_cache_seqlen + block_size - 1)/ block_size)
    # block_indices = torch.full((batch, heads_kv, max_num_blocks), -1, dtype=torch.int32, device='cuda')

    # Assign valid indices while ensuring no duplicates within each batch-group
    for b in range(batch):
        max_valid_block = max_valid_num_blocks[b].item()  # Max valid blocks for this batch
        if max_valid_block > 0:  # Ensure there's at least one valid block
            for h in range(heads_kv):
                valid_indices = torch.randperm(max_valid_block, device="cuda", dtype=torch.int32)[:max_selected_blocks]
                # valid_indices = torch.randperm(max_valid_block, device='cuda', dtype=torch.int32)[:max_num_blocks]
                block_indices[b, h, : len(valid_indices)] = valid_indices

    # Sort indices within each batch-group for consistency
    block_indices, _ = block_indices.sort(dim=-1, descending=True)
    # print("block_indices: ", block_indices)
    actual_num_blocks = torch.sum(block_indices != -1, dim=-1).to(torch.int32)[:, 0]
    print("actual_num_blocks: ", actual_num_blocks)
    # print(block_indices.shape, actual_num_blocks.shape)

    max_num_blocks = torch.max(max_valid_num_blocks).item()
    print("max_num_blocks: ", max_num_blocks)

    # parity reference
    ref = ref_program_torch(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen, max_num_blocks, block_size)

    sparse_kernel = SparseFlashAttn(batch, heads, heads_kv, dim, dim_v, block_size)
    out = sparse_kernel(Q, K, V, block_indices, cache_seqlens)
    debug("output", ref, out, atol=1e-3, rtol=1e-3)

    import flash_attn  # noqa: F401

    ## latency reference
    for _ in range(10):
        ref = ref_program_fa(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen, max_num_blocks, block_size)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        ref = ref_program_fa(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen, max_num_blocks, block_size)
    torch.cuda.synchronize()
    print("dense time: ", (time.time() - start) / 100 * 1000)

    for _ in range(10):
        # out = sparse_gqa_decode_varlen_indice(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen, block_size)
        out = sparse_kernel(Q, K, V, block_indices, cache_seqlens)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        # out = sparse_gqa_decode_varlen_indice(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen, block_size)
        out = sparse_kernel(Q, K, V, block_indices, cache_seqlens)
    torch.cuda.synchronize()
    print("sparse time: ", (time.time() - start) / 100 * 1000)


def run_regression_perf(batch=8, heads=32, heads_kv=8, max_cache_seqlen=8192, dim=128, dim_v=128, sparse_ratio=0.8, block_size=32):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = batch, heads, heads_kv, max_cache_seqlen, dim, dim_v
    sparse_ratio = sparse_ratio
    block_size = block_size
    max_selected_blocks = int(math.ceil(max_cache_seqlen * (1 - sparse_ratio) / block_size))
    dtype = torch.float16
    Q = torch.randn((batch, heads, dim), dtype=dtype, device="cuda")
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device="cuda")
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device="cuda")
    cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32, device="cuda")
    max_valid_num_blocks = torch.ceil(cache_seqlens / block_size).int()
    block_indices = torch.full((batch, heads_kv, max_selected_blocks), -1, dtype=torch.int32, device="cuda")

    for b in range(batch):
        max_valid_block = max_valid_num_blocks[b].item()
        if max_valid_block > 0:
            for h in range(heads_kv):
                valid_indices = torch.randperm(max_valid_block, device="cuda", dtype=torch.int32)[:max_selected_blocks]
                block_indices[b, h, : len(valid_indices)] = valid_indices

    block_indices, _ = block_indices.sort(dim=-1, descending=True)
    sparse_kernel = SparseFlashAttn(batch, heads, heads_kv, dim, dim_v, block_size)
    batch = sparse_kernel.batch
    heads = sparse_kernel.heads
    heads_kv = sparse_kernel.heads_kv
    dim_v = sparse_kernel.dim_v
    dim = sparse_kernel.dim
    block_size = sparse_kernel.block_size
    max_selected_blocks = block_indices.shape[-1]

    num_m_blocks = 1 * (heads // heads_kv + sparse_kernel.block_H - 1) // sparse_kernel.block_H
    num_n_blocks = max_selected_blocks
    size_one_kv_head = max_selected_blocks * block_size * (dim + dim_v) * 2
    total_mblocks = batch * heads_kv * num_m_blocks
    num_sm = sparse_kernel.num_sm

    num_split = num_splits_heuristic(
        total_mblocks, num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, is_causal_or_local=True, max_splits=128
    )

    glse = torch.empty((batch, heads, num_split), dtype=torch.float32, device="cuda")
    output_partial = torch.empty((batch, heads, num_split, dim_v), dtype=torch.float32, device="cuda")
    kernel = sparse_kernel.kernel

    def run_kernel_only():
        kernel(Q, K, V, block_indices, cache_seqlens, glse, output_partial)

    return do_bench(run_kernel_only, warmup=100, rep=1000, backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="heads")
    parser.add_argument("--heads_kv", type=int, default=8, help="heads_kv")
    parser.add_argument("--max_cache_seqlen", type=int, default=8192, help="kvcache sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--dim_v", type=int, default=128, help="dim_v")
    parser.add_argument("--sparse_ratio", type=float, default=0.8, help="sparse ratio")
    parser.add_argument("--block_size", type=int, default=32, help="block_size")
    args = parser.parse_args()
    main(args.batch, args.heads, args.heads_kv, args.max_cache_seqlen, args.dim, args.dim_v, args.sparse_ratio, args.block_size)
