"""GLM-5.3 paged FP8 MQA logits over the compressed k-pool cache.

The GLM sparse-attention indexer compares 32 FP8 query heads with one FP8
pooled key per history group. Per-head query scales are folded into ``weights``
by the caller. This kernel applies the cached per-pool K scale, ReLU-gates each
head score, and reduces the weighted head scores to one FP32 logit per pool.
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T

from examples.kpool.example_glm53_kpool_compress import GLM53_HEAD_DIM
from tilelang.language.fp8 import determine_fp8_type, determine_torch_fp8_type


GLM53_INDEX_HEADS = 32


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def glm53_kpool_fp8_mqa_logits_kernel(
    page_size: int = 64,
    num_heads: int = GLM53_INDEX_HEADS,
    head_dim: int = GLM53_HEAD_DIM,
    block_n: int = 32,
    threads: int = 256,
    num_stages: int = 2,
    fp8_dtype: str | None = None,
):
    """Build the paged pooled-history logits kernel."""
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if num_heads != GLM53_INDEX_HEADS:
        raise ValueError(f"GLM-5.3 indexer requires num_heads={GLM53_INDEX_HEADS}, got {num_heads}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 indexer requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")
    if block_n <= 0:
        raise ValueError(f"block_n must be positive, got {block_n}")
    if threads <= 0:
        raise ValueError(f"threads must be positive, got {threads}")
    if fp8_dtype is None:
        fp8_dtype = determine_fp8_type()

    num_rows = T.dynamic("num_rows")
    num_blocks = T.dynamic("num_blocks")
    num_page_rows = T.dynamic("num_page_rows")
    max_pages = T.dynamic("max_pages")
    max_num_pools = T.dynamic("max_num_pools")

    @T.prim_func
    def main(
        query: T.Tensor((num_rows, num_heads, head_dim), fp8_dtype),
        k_cache: T.Tensor((num_blocks, page_size, head_dim), fp8_dtype),
        scale_cache: T.Tensor((num_blocks, page_size), T.float32),
        weights: T.Tensor((num_rows, num_heads), T.float32),
        pool_page_table: T.Tensor((num_page_rows, max_pages), T.int32),
        page_table_rows: T.Tensor((num_rows,), T.int32),
        pool_starts: T.Tensor((num_rows,), T.int32),
        pool_ends: T.Tensor((num_rows,), T.int32),
        logits: T.Tensor((num_rows, max_num_pools), T.float32),
    ) -> None:
        with T.Kernel(num_rows, threads=threads) as row:
            query_shared = T.alloc_shared((num_heads, head_dim), fp8_dtype)
            key_shared = T.alloc_shared((block_n, head_dim), fp8_dtype)
            scores = T.alloc_fragment((block_n, num_heads), T.float32)
            key_scales = T.alloc_fragment((block_n,), T.float32)
            row_logits = T.alloc_fragment((block_n,), T.float32)
            head_weights = T.alloc_fragment((num_heads,), T.float32)

            T.copy(query[row, 0, 0], query_shared)
            T.copy(weights[row, 0], head_weights)

            page_table_row = page_table_rows[row]
            pool_start = pool_starts[row]
            pool_end = pool_ends[row]

            for tile in T.Pipelined(T.ceildiv(max_num_pools, block_n), num_stages=num_stages):
                for pool_lane, dim in T.Parallel(block_n, head_dim):
                    logical_pool = tile * block_n + pool_lane
                    if logical_pool >= pool_start and logical_pool < pool_end:
                        page = logical_pool // page_size
                        page_offset = logical_pool % page_size
                        if page_table_row >= 0 and page_table_row < num_page_rows and page >= 0 and page < max_pages:
                            physical_block = pool_page_table[page_table_row, page]
                            if physical_block >= 0 and physical_block < num_blocks:
                                key_shared[pool_lane, dim] = k_cache[physical_block, page_offset, dim]
                            else:
                                key_shared[pool_lane, dim] = T.cast(0, fp8_dtype)
                        else:
                            key_shared[pool_lane, dim] = T.cast(0, fp8_dtype)
                    else:
                        key_shared[pool_lane, dim] = T.cast(0, fp8_dtype)

                for pool_lane in T.Parallel(block_n):
                    logical_pool = tile * block_n + pool_lane
                    if logical_pool >= pool_start and logical_pool < pool_end:
                        page = logical_pool // page_size
                        page_offset = logical_pool % page_size
                        if page_table_row >= 0 and page_table_row < num_page_rows and page >= 0 and page < max_pages:
                            physical_block = pool_page_table[page_table_row, page]
                            if physical_block >= 0 and physical_block < num_blocks:
                                key_scales[pool_lane] = scale_cache[physical_block, page_offset]
                            else:
                                key_scales[pool_lane] = 0.0
                        else:
                            key_scales[pool_lane] = 0.0
                    else:
                        key_scales[pool_lane] = 0.0

                T.gemm(
                    key_shared,
                    query_shared,
                    scores,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for pool_lane, head in T.Parallel(block_n, num_heads):
                    scores[pool_lane, head] = T.max(scores[pool_lane, head] * key_scales[pool_lane], 0.0) * head_weights[head]

                T.reduce_sum(scores, row_logits, dim=1, clear=True)

                for pool_lane in T.Parallel(block_n):
                    logical_pool = tile * block_n + pool_lane
                    if logical_pool < max_num_pools:
                        if logical_pool >= pool_start and logical_pool < pool_end:
                            page = logical_pool // page_size
                            if page_table_row >= 0 and page_table_row < num_page_rows and page >= 0 and page < max_pages:
                                physical_block = pool_page_table[page_table_row, page]
                                if physical_block >= 0 and physical_block < num_blocks:
                                    logits[row, logical_pool] = row_logits[pool_lane]
                                else:
                                    logits[row, logical_pool] = -T.infinity(T.float32)
                            else:
                                logits[row, logical_pool] = -T.infinity(T.float32)
                        else:
                            logits[row, logical_pool] = -T.infinity(T.float32)

    return main


def _validate_inputs(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    weights: torch.Tensor,
    pool_page_table: torch.Tensor,
    page_table_rows: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    max_num_pools: int,
    num_heads: int,
    head_dim: int,
) -> str:
    if num_heads != GLM53_INDEX_HEADS:
        raise ValueError(f"GLM-5.3 indexer requires num_heads={GLM53_INDEX_HEADS}, got {num_heads}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 indexer requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")
    if max_num_pools < 0:
        raise ValueError(f"max_num_pools must be nonnegative, got {max_num_pools}")
    if query.ndim != 3:
        raise ValueError(f"query must have rank 3, got shape {tuple(query.shape)}")
    if tuple(query.shape[1:]) != (num_heads, head_dim):
        raise ValueError(f"query must have shape [num_rows, {num_heads}, {head_dim}], got {tuple(query.shape)}")
    if k_cache.ndim != 3 or k_cache.shape[2] != head_dim:
        raise ValueError(f"k_cache must have shape [num_blocks, page_size, {head_dim}], got {tuple(k_cache.shape)}")
    if tuple(scale_cache.shape) != tuple(k_cache.shape[:2]):
        raise ValueError(
            "scale_cache must match the first two k_cache dimensions; "
            f"got k_cache={tuple(k_cache.shape)}, scale_cache={tuple(scale_cache.shape)}"
        )
    num_rows = query.shape[0]
    if tuple(weights.shape) != (num_rows, num_heads):
        raise ValueError(f"weights must have shape {(num_rows, num_heads)}, got {tuple(weights.shape)}")
    if pool_page_table.ndim != 2 or pool_page_table.shape[0] <= 0 or pool_page_table.shape[1] <= 0:
        raise ValueError(f"pool_page_table must be a nonempty rank-2 tensor, got {tuple(pool_page_table.shape)}")
    for name, tensor in (
        ("page_table_rows", page_table_rows),
        ("pool_starts", pool_starts),
        ("pool_ends", pool_ends),
    ):
        if tuple(tensor.shape) != (num_rows,):
            raise ValueError(f"{name} must have shape {(num_rows,)}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be torch.int32, got {tensor.dtype}")

    if weights.dtype != torch.float32:
        raise TypeError(f"weights must be torch.float32, got {weights.dtype}")
    if scale_cache.dtype != torch.float32:
        raise TypeError(f"scale_cache must be torch.float32, got {scale_cache.dtype}")
    if pool_page_table.dtype != torch.int32:
        raise TypeError(f"pool_page_table must be torch.int32, got {pool_page_table.dtype}")

    tensors = (
        query,
        k_cache,
        scale_cache,
        weights,
        pool_page_table,
        page_table_rows,
        pool_starts,
        pool_ends,
    )
    if not query.is_cuda:
        raise ValueError("all tensors must be on a CUDA or ROCm device")
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous")

    fp8_dtype = determine_torch_fp8_type(device=query.device)
    if query.dtype != fp8_dtype:
        raise TypeError(f"query must use the platform FP8 dtype {fp8_dtype}, got {query.dtype}")
    if k_cache.dtype != fp8_dtype:
        raise TypeError(f"k_cache must use the platform FP8 dtype {fp8_dtype}, got {k_cache.dtype}")

    if num_rows == 0:
        return determine_fp8_type(device=query.device)

    min_start = int(pool_starts.min().item())
    max_end = int(pool_ends.max().item())
    if min_start < 0 or torch.any(pool_starts > pool_ends).item():
        raise ValueError("pool ranges must satisfy 0 <= pool_starts <= pool_ends")
    if max_end > max_num_pools:
        raise ValueError(f"pool_ends must not exceed max_num_pools={max_num_pools}, got {max_end}")
    if max_end > pool_page_table.shape[1] * k_cache.shape[1]:
        raise ValueError("pool ranges exceed pool_page_table capacity")

    min_row = int(page_table_rows.min().item())
    max_row = int(page_table_rows.max().item())
    if min_row < 0 or max_row >= pool_page_table.shape[0]:
        raise ValueError(f"page_table_rows must be in [0, {pool_page_table.shape[0]}), got min={min_row}, max={max_row}")

    for row in range(num_rows):
        start = int(pool_starts[row].item())
        end = int(pool_ends[row].item())
        if start == end:
            continue
        table_row = int(page_table_rows[row].item())
        first_page = start // k_cache.shape[1]
        last_page = (end - 1) // k_cache.shape[1]
        blocks = pool_page_table[table_row, first_page : last_page + 1]
        if torch.any(blocks < 0).item() or torch.any(blocks >= k_cache.shape[0]).item():
            raise ValueError(f"active pool_page_table entries for row {row} must name valid cache blocks")

    return determine_fp8_type(device=query.device)


def glm53_kpool_fp8_mqa_logits(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    weights: torch.Tensor,
    pool_page_table: torch.Tensor,
    page_table_rows: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    *,
    max_num_pools: int | None = None,
    num_heads: int = GLM53_INDEX_HEADS,
    head_dim: int = GLM53_HEAD_DIM,
    block_n: int = 32,
    threads: int = 256,
    num_stages: int = 2,
) -> torch.Tensor:
    """Compute bounded FP32 pool logits from caller-owned paged FP8 caches."""
    if max_num_pools is None:
        max_num_pools = int(pool_ends.max().item()) if query.shape[0] else 0
    fp8_dtype = _validate_inputs(
        query,
        k_cache,
        scale_cache,
        weights,
        pool_page_table,
        page_table_rows,
        pool_starts,
        pool_ends,
        max_num_pools,
        num_heads,
        head_dim,
    )
    logits = torch.full(
        (query.shape[0], max_num_pools),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    if query.shape[0] == 0 or max_num_pools == 0:
        return logits

    kernel = glm53_kpool_fp8_mqa_logits_kernel(
        page_size=k_cache.shape[1],
        num_heads=num_heads,
        head_dim=head_dim,
        block_n=block_n,
        threads=threads,
        num_stages=num_stages,
        fp8_dtype=fp8_dtype,
    )
    kernel(
        query,
        k_cache,
        scale_cache,
        weights,
        pool_page_table,
        page_table_rows,
        pool_starts,
        pool_ends,
        logits,
    )
    return logits


def glm53_kpool_fp8_mqa_logits_reference(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    weights: torch.Tensor,
    pool_page_table: torch.Tensor,
    page_table_rows: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    *,
    max_num_pools: int | None = None,
) -> torch.Tensor:
    """PyTorch reference for the paged pooled-history scoring contract."""
    if max_num_pools is None:
        max_num_pools = int(pool_ends.max().item()) if query.shape[0] else 0
    output = torch.full(
        (query.shape[0], max_num_pools),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    page_size = k_cache.shape[1]
    for row in range(query.shape[0]):
        start = int(pool_starts[row].item())
        end = int(pool_ends[row].item())
        if start == end:
            continue
        logical = torch.arange(start, end, dtype=torch.long, device=query.device)
        table_row = int(page_table_rows[row].item())
        blocks = pool_page_table[table_row, logical // page_size].long()
        offsets = logical % page_size
        keys = k_cache[blocks, offsets].float()
        scales = scale_cache[blocks, offsets].float()
        per_head = torch.einsum("nd,hd->nh", keys, query[row].float())
        output[row, start:end] = (per_head.mul(scales[:, None]).clamp_min(0.0) * weights[row][None, :]).sum(dim=1)
    return output
