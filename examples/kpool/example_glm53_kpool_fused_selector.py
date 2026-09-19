"""Fused GLM-5.3 paged k-pool scoring, Top-K, and token transformation.

The production indexer stores each compressed page as contiguous FP8 key bytes
followed by contiguous FP32 scale bytes.  This kernel consumes that interleaved
``uint8`` layout, computes the weighted pooled-history logits, folds the first
radix histogram into the scoring pass, selects the highest-scoring pools, and
emits token indices in one launch.

Callers may provide both scratch and output tensors.  That keeps their storage
stable across CUDA/HIP graph capture and avoids exposing the intermediate
logits as a pipeline result.
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T

from examples.deepseek_v32.topk_selector import convert_to_uint32
from examples.kpool.example_glm53_kpool_compress import GLM53_HEAD_DIM, GLM53_POOL_SIZE
from examples.kpool.example_glm53_kpool_fp8_mqa_logits import (
    GLM53_INDEX_HEADS,
    glm53_kpool_fp8_mqa_logits_reference,
)
from examples.kpool.example_glm53_kpool_topk_transform import (
    GLM53_INDEX_TOPK,
    glm53_kpool_topk_transform_reference,
)
from tilelang.language.fp8 import determine_fp8_type, determine_torch_fp8_type


_IDENTITY = 0
_PAGED = 1
_RAGGED = 2
_RADIX = 1 << 8


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def glm53_kpool_fused_selector_kernel(
    page_size: int = 64,
    token_topk: int = GLM53_INDEX_TOPK,
    pool_size: int = GLM53_POOL_SIZE,
    num_heads: int = GLM53_INDEX_HEADS,
    head_dim: int = GLM53_HEAD_DIM,
    transform_mode: int = _IDENTITY,
    block_n: int = 32,
    threads: int = 256,
    num_stages: int = 2,
    fp8_dtype: str | None = None,
):
    """Build the one-launch paged scorer and token selector."""
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if pool_size <= 1:
        raise ValueError(f"pool_size must be greater than one, got {pool_size}")
    if token_topk <= 0 or token_topk % pool_size != 0:
        raise ValueError(f"token_topk must be a positive multiple of pool_size={pool_size}")
    if num_heads != GLM53_INDEX_HEADS:
        raise ValueError(f"GLM-5.3 indexer requires num_heads={GLM53_INDEX_HEADS}, got {num_heads}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 indexer requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")
    if transform_mode not in (_IDENTITY, _PAGED, _RAGGED):
        raise ValueError(f"unsupported transform_mode={transform_mode}")
    if block_n <= 0:
        raise ValueError(f"block_n must be positive, got {block_n}")
    if threads != _RADIX:
        raise ValueError(f"the fused radix selector requires threads={_RADIX}, got {threads}")
    if fp8_dtype is None:
        fp8_dtype = determine_fp8_type()

    group_topk = token_topk // pool_size
    out_cols = token_topk + pool_size - 1
    scale_offset = page_size * head_dim

    num_rows = T.dynamic("num_rows")
    num_blocks = T.dynamic("num_blocks")
    num_pool_page_rows = T.dynamic("num_pool_page_rows")
    max_pool_pages = T.dynamic("max_pool_pages")
    max_num_pools = T.dynamic("max_num_pools")
    num_token_page_rows = T.dynamic("num_token_page_rows")
    max_tokens = T.dynamic("max_tokens")

    @T.prim_func
    def main(
        query: T.Tensor((num_rows, num_heads, head_dim), fp8_dtype),
        cache_u8: T.Tensor((num_blocks, page_size * (head_dim + 4)), T.uint8),
        weights: T.Tensor((num_rows, num_heads), T.float32),
        pool_page_table: T.Tensor((num_pool_page_rows, max_pool_pages), T.int32),
        pool_page_table_rows: T.Tensor((num_rows,), T.int32),
        pool_starts: T.Tensor((num_rows,), T.int32),
        pool_ends: T.Tensor((num_rows,), T.int32),
        seq_lens: T.Tensor((num_rows,), T.int32),
        token_page_table: T.Tensor((num_token_page_rows, max_tokens), T.int32),
        token_page_table_rows: T.Tensor((num_rows,), T.int32),
        topk_offsets: T.Tensor((num_rows,), T.int32),
        logits_workspace: T.Tensor((num_rows, max_num_pools), T.float32),
        token_indices: T.Tensor((num_rows, out_cols), T.int32),
    ) -> None:
        with T.Kernel(num_rows, threads=threads) as row:
            tx = T.get_thread_binding(0)

            query_shared = T.alloc_shared((num_heads, head_dim), fp8_dtype)
            key_bytes_shared = T.alloc_shared((block_n, head_dim), T.uint8)
            key_shared = T.view(key_bytes_shared, (block_n, head_dim), fp8_dtype)
            scale_bytes_shared = T.alloc_shared((block_n, 4), T.uint8)
            key_scales = T.view(scale_bytes_shared, (block_n,), T.float32)
            scores = T.alloc_fragment((block_n, num_heads), T.float32)
            row_logits = T.alloc_fragment((block_n,), T.float32)
            head_weights = T.alloc_fragment((num_heads,), T.float32)

            # Keep the suffix-scan sentinel histogram[tx + 1] in bounds when
            # tx == _RADIX - 1 while retaining whole-stride parallel fills.
            histogram = T.alloc_shared((_RADIX * 2,), T.int32)
            threshold_bin = T.alloc_shared((1,), T.int32)
            output_counts = T.alloc_shared((2,), T.int32)
            selected_pools = T.alloc_shared((group_topk,), T.int32)

            remaining = T.alloc_var(T.int32)
            prefix = T.alloc_var(T.uint32)
            prefix_mask = T.alloc_var(T.uint32)
            key_bits = T.alloc_var(T.uint32)
            bin_id = T.alloc_var(T.int32)
            pos = T.alloc_var(T.int32)

            pool_start = pool_starts[row]
            pool_end = pool_ends[row]
            pool_count = pool_end - pool_start

            if pool_count > group_topk:
                T.copy(query[row, 0, 0], query_shared)
                T.copy(weights[row, 0], head_weights)
                T.fill(histogram, 0)
                if tx == 0:
                    threshold_bin[0] = 0
                T.sync_threads()

                # Score every active pool.  Building the most-significant-byte
                # histogram here removes the first full logits read performed
                # by a separate radix Top-K launch.
                first_tile = pool_start // block_n
                for tile_step in T.Pipelined(
                    T.ceildiv(pool_end, block_n) - first_tile,
                    num_stages=num_stages,
                ):
                    tile = first_tile + tile_step
                    # Keep the predicated byte gather scalar. HIP codegen
                    # requires vector byte broadcasts to use literal values,
                    # while layout lowering hoists these zero fills.
                    for pool_lane, dim in T.Parallel(
                        block_n,
                        head_dim,
                        coalesced_width=T.int32(1),
                    ):
                        logical_pool = tile * block_n + pool_lane
                        if logical_pool >= pool_start and logical_pool < pool_end:
                            page = logical_pool // page_size
                            page_offset = logical_pool % page_size
                            if (
                                pool_page_table_rows[row] >= 0
                                and pool_page_table_rows[row] < num_pool_page_rows
                                and page >= 0
                                and page < max_pool_pages
                            ):
                                physical_block = pool_page_table[pool_page_table_rows[row], page]
                                if physical_block >= 0 and physical_block < num_blocks:
                                    key_bytes_shared[pool_lane, dim] = cache_u8[physical_block, page_offset * head_dim + dim]
                                else:
                                    key_bytes_shared[pool_lane, dim] = 0
                            else:
                                key_bytes_shared[pool_lane, dim] = 0
                        else:
                            key_bytes_shared[pool_lane, dim] = 0

                    for pool_lane, byte in T.Parallel(block_n, 4):
                        logical_pool = tile * block_n + pool_lane
                        if logical_pool >= pool_start and logical_pool < pool_end:
                            page = logical_pool // page_size
                            page_offset = logical_pool % page_size
                            if (
                                pool_page_table_rows[row] >= 0
                                and pool_page_table_rows[row] < num_pool_page_rows
                                and page >= 0
                                and page < max_pool_pages
                            ):
                                physical_block = pool_page_table[pool_page_table_rows[row], page]
                                if physical_block >= 0 and physical_block < num_blocks:
                                    scale_bytes_shared[pool_lane, byte] = cache_u8[
                                        physical_block,
                                        scale_offset + page_offset * 4 + byte,
                                    ]
                                else:
                                    scale_bytes_shared[pool_lane, byte] = 0
                            else:
                                scale_bytes_shared[pool_lane, byte] = 0
                        else:
                            scale_bytes_shared[pool_lane, byte] = 0

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
                        if logical_pool >= 0 and logical_pool >= pool_start and logical_pool < pool_end and logical_pool < max_num_pools:
                            logits_workspace[row, logical_pool] = row_logits[pool_lane]
                            key_bits = convert_to_uint32(row_logits[pool_lane])
                            bin_id = T.cast((key_bits >> 24) & 0xFF, T.int32)
                            T.atomic_add(histogram[bin_id], 1)

                T.sync_threads()
                remaining = group_topk
                prefix = T.cast(0, T.uint32)
                prefix_mask = T.cast(0, T.uint32)
                scan_width = threads * 4
                first_scan = pool_start // scan_width
                scan_count = T.ceildiv(pool_end, scan_width) - first_scan

                # Select the cutoff one byte at a time.  Later passes rescan
                # caller-owned scratch instead of storing a bounded candidate
                # bucket, so arbitrarily large equal-score groups stay safe.
                for radix_round in T.serial(4):
                    shift = 24 - radix_round * 8
                    if radix_round > 0:
                        T.fill(histogram, 0)
                        if tx == 0:
                            threshold_bin[0] = 0
                        T.sync_threads()
                        for scan_step in T.serial(scan_count):
                            input_base = (first_scan + scan_step) * scan_width + tx * 4
                            for item in T.serial(4):
                                input_idx = input_base + item
                                if input_idx >= pool_start and input_idx < pool_end and input_idx < max_num_pools:
                                    key_bits = convert_to_uint32(logits_workspace[row, input_idx])
                                    if (key_bits & prefix_mask) == prefix:
                                        bin_id = T.cast((key_bits >> shift) & 0xFF, T.int32)
                                        T.atomic_add(histogram[bin_id], 1)
                        T.sync_threads()

                    if tx < _RADIX:
                        for scan_step in T.serial(8):
                            offset = 1 << scan_step
                            T.sync_threads(3, _RADIX)
                            if tx < _RADIX - offset:
                                bin_id = histogram[tx] + histogram[tx + offset]
                            T.sync_threads(3, _RADIX)
                            if tx < _RADIX - offset:
                                histogram[tx] = bin_id

                        T.sync_threads(3, _RADIX)
                        if histogram[tx] >= remaining and histogram[tx + 1] < remaining:
                            threshold_bin[0] = tx
                    T.sync_threads()

                    remaining = remaining - histogram[threshold_bin[0] + 1]
                    prefix = prefix | (T.cast(threshold_bin[0], T.uint32) << shift)
                    prefix_mask = prefix_mask | (T.cast(0xFF, T.uint32) << shift)
                    T.sync_threads()

                for fill_tile in T.serial(T.ceildiv(group_topk, threads)):
                    fill_idx = fill_tile * threads + tx
                    if fill_idx < group_topk:
                        selected_pools[fill_idx] = -1
                if tx < 2:
                    output_counts[tx] = 0
                T.sync_threads()

                for scan_step in T.serial(scan_count):
                    input_base = (first_scan + scan_step) * scan_width + tx * 4
                    for item in T.serial(4):
                        input_idx = input_base + item
                        if input_idx >= pool_start and input_idx < pool_end and input_idx < max_num_pools:
                            key_bits = convert_to_uint32(logits_workspace[row, input_idx])
                            if key_bits > prefix:
                                pos = T.atomic_add(output_counts[0], 1, return_prev=True)
                                if pos < group_topk - remaining:
                                    selected_pools[pos] = input_idx
                            elif key_bits == prefix:
                                pos = T.atomic_add(output_counts[1], 1, return_prev=True)
                                if pos < remaining:
                                    selected_pools[group_topk - remaining + pos] = input_idx
                T.sync_threads()

            selected_count = T.min(pool_count, group_topk)
            history_count = selected_count * pool_size
            tail_start = pool_end * pool_size
            tail_count = seq_lens[row] - tail_start

            for output_tile in T.serial(T.ceildiv(out_cols, threads)):
                col = output_tile * threads + tx
                if col < out_cols:
                    raw_token = T.alloc_var(T.int32, init=-1)
                    if col < history_count:
                        group_rank = col // pool_size
                        slot = col % pool_size
                        group_id = T.if_then_else(
                            pool_count <= group_topk,
                            pool_start + group_rank,
                            selected_pools[group_rank],
                        )
                        raw_token = group_id * pool_size + slot
                    elif col < history_count + tail_count:
                        raw_token = tail_start + col - history_count

                    if raw_token >= 0:
                        if transform_mode == _PAGED:
                            page_table_row = token_page_table_rows[row]
                            if page_table_row >= 0 and page_table_row < num_token_page_rows and raw_token < max_tokens:
                                token_indices[row, col] = token_page_table[page_table_row, raw_token]
                            else:
                                token_indices[row, col] = -1
                        elif transform_mode == _RAGGED:
                            token_indices[row, col] = raw_token + topk_offsets[row]
                        else:
                            token_indices[row, col] = raw_token
                    else:
                        token_indices[row, col] = -1

    return main


def pack_glm53_kpool_cache(
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
) -> torch.Tensor:
    """Pack separate FP8 keys and FP32 scales into the production byte layout."""
    if k_cache.ndim != 3 or k_cache.shape[2] != GLM53_HEAD_DIM:
        raise ValueError(f"k_cache must have shape [num_blocks, page_size, {GLM53_HEAD_DIM}], got {tuple(k_cache.shape)}")
    if tuple(scale_cache.shape) != tuple(k_cache.shape[:2]):
        raise ValueError(
            "scale_cache must match the first two k_cache dimensions; "
            f"got k_cache={tuple(k_cache.shape)}, scale_cache={tuple(scale_cache.shape)}"
        )
    if k_cache.device != scale_cache.device:
        raise ValueError("k_cache and scale_cache must be on the same device")
    if not k_cache.is_contiguous() or not scale_cache.is_contiguous():
        raise ValueError("k_cache and scale_cache must be contiguous")
    fp8_dtype = determine_torch_fp8_type(device=k_cache.device)
    if k_cache.dtype != fp8_dtype:
        raise TypeError(f"k_cache must use the platform FP8 dtype {fp8_dtype}, got {k_cache.dtype}")
    if scale_cache.dtype != torch.float32:
        raise TypeError(f"scale_cache must be torch.float32, got {scale_cache.dtype}")

    num_blocks, page_size, head_dim = k_cache.shape
    key_bytes = page_size * head_dim
    packed = torch.empty(
        (num_blocks, key_bytes + page_size * 4),
        dtype=torch.uint8,
        device=k_cache.device,
    )
    packed[:, :key_bytes].copy_(k_cache.view(torch.uint8).reshape(num_blocks, key_bytes))
    packed[:, key_bytes:].copy_(scale_cache.view(torch.uint8).reshape(num_blocks, page_size * 4))
    return packed


def _unpack_glm53_kpool_cache(
    cache_u8: torch.Tensor,
    page_size: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack the byte layout for the PyTorch reference only."""
    key_bytes = page_size * head_dim
    k_cache = cache_u8[:, :key_bytes].contiguous().view(fp8_dtype).reshape(cache_u8.shape[0], page_size, head_dim)
    scale_cache = cache_u8[:, key_bytes:].contiguous().view(torch.float32).reshape(cache_u8.shape[0], page_size)
    return k_cache, scale_cache


def _validate_inputs(
    query: torch.Tensor,
    cache_u8: torch.Tensor,
    weights: torch.Tensor,
    pool_page_table: torch.Tensor,
    pool_page_table_rows: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    logits_workspace: torch.Tensor,
    token_indices: torch.Tensor,
    token_topk: int,
    pool_size: int,
    token_page_table: torch.Tensor | None,
    token_page_table_rows: torch.Tensor | None,
    topk_offsets: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
    validate: bool = True,
) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
    if num_heads != GLM53_INDEX_HEADS:
        raise ValueError(f"GLM-5.3 indexer requires num_heads={GLM53_INDEX_HEADS}, got {num_heads}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 indexer requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")
    if pool_size <= 1:
        raise ValueError(f"pool_size must be greater than one, got {pool_size}")
    if token_topk <= 0 or token_topk % pool_size != 0:
        raise ValueError(f"token_topk must be a positive multiple of pool_size={pool_size}")
    if query.ndim != 3 or tuple(query.shape[1:]) != (num_heads, head_dim):
        raise ValueError(f"query must have shape [num_rows, {num_heads}, {head_dim}], got {tuple(query.shape)}")
    if cache_u8.ndim != 2 or cache_u8.shape[0] <= 0:
        raise ValueError(f"cache_u8 must be a nonempty rank-2 interleaved cache, got {tuple(cache_u8.shape)}")
    if cache_u8.dtype != torch.uint8:
        raise TypeError(f"cache_u8 must be torch.uint8, got {cache_u8.dtype}")
    if cache_u8.shape[1] % (head_dim + 4) != 0:
        raise ValueError(f"cache_u8 width must equal page_size * ({head_dim} + 4), got {cache_u8.shape[1]}")
    page_size = cache_u8.shape[1] // (head_dim + 4)
    if page_size <= 0:
        raise ValueError("the inferred cache page size must be positive")

    rows = query.shape[0]
    group_topk = token_topk // pool_size
    out_cols = token_topk + pool_size - 1
    if tuple(weights.shape) != (rows, num_heads) or weights.dtype != torch.float32:
        raise ValueError(f"weights must be float32 with shape {(rows, num_heads)}")
    if pool_page_table.ndim != 2 or pool_page_table.shape[0] <= 0 or pool_page_table.shape[1] <= 0:
        raise ValueError(f"pool_page_table must be a nonempty rank-2 tensor, got {tuple(pool_page_table.shape)}")
    if pool_page_table.dtype != torch.int32:
        raise TypeError(f"pool_page_table must be torch.int32, got {pool_page_table.dtype}")
    for name, tensor in (
        ("pool_page_table_rows", pool_page_table_rows),
        ("pool_starts", pool_starts),
        ("pool_ends", pool_ends),
        ("seq_lens", seq_lens),
    ):
        if tuple(tensor.shape) != (rows,) or tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32 with shape {(rows,)}")

    max_num_pools = logits_workspace.shape[1] if logits_workspace.ndim == 2 else -1
    if tuple(logits_workspace.shape) != (rows, max_num_pools) or max_num_pools <= 0:
        raise ValueError(f"logits_workspace must have shape [num_rows, positive_max_num_pools], got {tuple(logits_workspace.shape)}")
    if logits_workspace.dtype != torch.float32:
        raise TypeError(f"logits_workspace must be torch.float32, got {logits_workspace.dtype}")
    if tuple(token_indices.shape) != (rows, out_cols) or token_indices.dtype != torch.int32:
        raise ValueError(f"token_indices must be int32 with shape {(rows, out_cols)}")

    tensors = (
        query,
        cache_u8,
        weights,
        pool_page_table,
        pool_page_table_rows,
        pool_starts,
        pool_ends,
        seq_lens,
        logits_workspace,
        token_indices,
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

    if validate and rows:
        min_start = int(pool_starts.min().item())
        max_end = int(pool_ends.max().item())
        if min_start < 0 or torch.any(pool_starts > pool_ends).item():
            raise ValueError("pool ranges must satisfy 0 <= pool_starts <= pool_ends")
        if max_end > max_num_pools:
            raise ValueError(f"pool_ends must not exceed logits workspace width {max_num_pools}, got {max_end}")
        if max_end > pool_page_table.shape[1] * page_size:
            raise ValueError("pool ranges exceed pool_page_table capacity")
        min_row = int(pool_page_table_rows.min().item())
        max_row = int(pool_page_table_rows.max().item())
        if min_row < 0 or max_row >= pool_page_table.shape[0]:
            raise ValueError(f"pool_page_table_rows must be in [0, {pool_page_table.shape[0]})")
        for row in range(rows):
            start = int(pool_starts[row].item())
            end = int(pool_ends[row].item())
            if start == end:
                continue
            table_row = int(pool_page_table_rows[row].item())
            first_page = start // page_size
            last_page = (end - 1) // page_size
            blocks = pool_page_table[table_row, first_page : last_page + 1]
            if torch.any(blocks < 0).item() or torch.any(blocks >= cache_u8.shape[0]).item():
                raise ValueError(f"active pool_page_table entries for row {row} must name valid cache blocks")
        tail_counts = seq_lens - pool_ends * pool_size
        if torch.any(tail_counts < 0).item() or torch.any(tail_counts >= pool_size).item():
            raise ValueError("seq_lens must add between zero and pool_size-1 tail tokens")

    if token_page_table is not None and topk_offsets is not None:
        raise ValueError("token_page_table and topk_offsets are mutually exclusive")
    transform_mode = _IDENTITY
    if token_page_table is not None:
        transform_mode = _PAGED
        if (
            token_page_table.ndim != 2
            or token_page_table.shape[0] <= 0
            or token_page_table.shape[1] <= 0
            or token_page_table.dtype != torch.int32
        ):
            raise ValueError("token_page_table must be a nonempty rank-2 int32 tensor")
        if token_page_table_rows is None:
            if token_page_table.shape[0] != rows:
                raise ValueError("token_page_table_rows is required when token_page_table rows differ from query rows")
            token_page_table_rows = torch.arange(rows, dtype=torch.int32, device=query.device)
        if tuple(token_page_table_rows.shape) != (rows,) or token_page_table_rows.dtype != torch.int32:
            raise ValueError(f"token_page_table_rows must be int32 with shape {(rows,)}")
        if validate and rows:
            min_row = int(token_page_table_rows.min().item())
            max_row = int(token_page_table_rows.max().item())
            if min_row < 0 or max_row >= token_page_table.shape[0]:
                raise ValueError(f"token_page_table_rows must be in [0, {token_page_table.shape[0]})")
            max_seq_len = int(seq_lens.max().item())
            if max_seq_len > token_page_table.shape[1]:
                raise ValueError("token_page_table is shorter than an active sequence")
            for row in range(rows):
                table_row = int(token_page_table_rows[row].item())
                seq_len = int(seq_lens[row].item())
                if seq_len and torch.any(token_page_table[table_row, :seq_len] < 0).item():
                    raise ValueError(f"active token_page_table entries for row {row} must be nonnegative")
    else:
        token_page_table = torch.zeros((1, 1), dtype=torch.int32, device=query.device)
        token_page_table_rows = torch.zeros((rows,), dtype=torch.int32, device=query.device)

    if topk_offsets is not None:
        transform_mode = _RAGGED
        if tuple(topk_offsets.shape) != (rows,) or topk_offsets.dtype != torch.int32:
            raise ValueError(f"topk_offsets must be int32 with shape {(rows,)}")
    else:
        topk_offsets = torch.zeros((rows,), dtype=torch.int32, device=query.device)

    extra_tensors = (token_page_table, token_page_table_rows, topk_offsets)
    if any(tensor.device != query.device for tensor in extra_tensors):
        raise ValueError("transform tensors must be on the same device as query")
    if any(not tensor.is_contiguous() for tensor in extra_tensors):
        raise ValueError("transform tensors must be contiguous")

    return (
        page_size,
        group_topk,
        token_page_table,
        token_page_table_rows,
        topk_offsets,
        transform_mode,
        determine_fp8_type(device=query.device),
    )


def glm53_kpool_fused_select(
    query: torch.Tensor,
    cache_u8: torch.Tensor,
    weights: torch.Tensor,
    pool_page_table: torch.Tensor,
    pool_page_table_rows: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    token_topk: int = GLM53_INDEX_TOPK,
    pool_size: int = GLM53_POOL_SIZE,
    token_page_table: torch.Tensor | None = None,
    token_page_table_rows: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
    logits_workspace: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    num_heads: int = GLM53_INDEX_HEADS,
    head_dim: int = GLM53_HEAD_DIM,
    block_n: int = 32,
    threads: int = 256,
    num_stages: int = 2,
    validate: bool = True,
) -> torch.Tensor:
    """Run the graph-safe fused selector over an interleaved paged cache.

    Before CUDA/HIP graph capture, validate the inputs once outside capture.
    During capture, pass persistent framework-owned ``logits_workspace`` and
    ``out`` tensors and set ``validate=False`` to skip device-to-host metadata
    checks. Shape, dtype, device, and contiguity checks remain enabled.
    """
    rows = query.shape[0] if query.ndim else 0
    if logits_workspace is None:
        max_num_pools = int(pool_ends.max().item()) if rows else 0
        logits_workspace = torch.empty(
            (rows, max(1, max_num_pools)),
            dtype=torch.float32,
            device=query.device,
        )
    if out is None:
        out = torch.empty(
            (rows, token_topk + pool_size - 1),
            dtype=torch.int32,
            device=query.device,
        )

    (
        page_size,
        _,
        token_page_table,
        token_page_table_rows,
        topk_offsets,
        transform_mode,
        fp8_dtype,
    ) = _validate_inputs(
        query,
        cache_u8,
        weights,
        pool_page_table,
        pool_page_table_rows,
        pool_starts,
        pool_ends,
        seq_lens,
        logits_workspace,
        out,
        token_topk,
        pool_size,
        token_page_table,
        token_page_table_rows,
        topk_offsets,
        num_heads,
        head_dim,
        validate=validate,
    )
    if rows == 0:
        return out

    kernel = glm53_kpool_fused_selector_kernel(
        page_size=page_size,
        token_topk=token_topk,
        pool_size=pool_size,
        num_heads=num_heads,
        head_dim=head_dim,
        transform_mode=transform_mode,
        block_n=block_n,
        threads=threads,
        num_stages=num_stages,
        fp8_dtype=fp8_dtype,
    )
    kernel(
        query,
        cache_u8,
        weights,
        pool_page_table,
        pool_page_table_rows,
        pool_starts,
        pool_ends,
        seq_lens,
        token_page_table,
        token_page_table_rows,
        topk_offsets,
        logits_workspace,
        out,
    )
    return out


def glm53_kpool_fused_select_reference(
    query: torch.Tensor,
    cache_u8: torch.Tensor,
    weights: torch.Tensor,
    pool_page_table: torch.Tensor,
    pool_page_table_rows: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    token_topk: int = GLM53_INDEX_TOPK,
    pool_size: int = GLM53_POOL_SIZE,
    token_page_table: torch.Tensor | None = None,
    token_page_table_rows: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose the established PyTorch references for the fused contract."""
    fp8_dtype = determine_torch_fp8_type(device=query.device)
    if cache_u8.ndim != 2 or cache_u8.shape[1] % (query.shape[-1] + 4) != 0:
        raise ValueError("cache_u8 does not contain a valid interleaved k-pool layout")
    page_size = cache_u8.shape[1] // (query.shape[-1] + 4)
    k_cache, scale_cache = _unpack_glm53_kpool_cache(cache_u8, page_size, query.shape[-1], fp8_dtype)
    max_num_pools = max(1, int(pool_ends.max().item()) if query.shape[0] else 0)
    logits = glm53_kpool_fp8_mqa_logits_reference(
        query,
        k_cache,
        scale_cache,
        weights,
        pool_page_table,
        pool_page_table_rows,
        pool_starts,
        pool_ends,
        max_num_pools=max_num_pools,
    )
    return glm53_kpool_topk_transform_reference(
        logits,
        pool_starts,
        pool_ends,
        seq_lens,
        token_topk=token_topk,
        pool_size=pool_size,
        token_page_table=token_page_table,
        page_table_rows=token_page_table_rows,
        topk_offsets=topk_offsets,
    )
