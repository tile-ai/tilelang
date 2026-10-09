"""GLM-5.3 pool-level Top-K and token-index transformation.

The public checkpoint selects 512 four-token pools for a 2,048-token history
budget, expands each selected pool to its four logical tokens, and appends the
incomplete zero-to-three-token tail. The resulting indices may remain logical,
be translated through a per-request token table, or receive a ragged offset.
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T

from examples.deepseek_v32.topk_selector import tl_topk
from examples.kpool.example_glm53_kpool_compress import GLM53_POOL_SIZE


GLM53_INDEX_TOPK = 2048
_IDENTITY = 0
_PAGED = 1
_RAGGED = 2


@tilelang.jit
def glm53_kpool_transform_kernel(
    token_topk: int = GLM53_INDEX_TOPK,
    pool_size: int = GLM53_POOL_SIZE,
    transform_mode: int = _IDENTITY,
    threads: int = 256,
):
    """Build the selected-pool expansion and tail-append kernel."""
    if pool_size <= 1:
        raise ValueError(f"pool_size must be greater than one, got {pool_size}")
    if token_topk <= 0 or token_topk % pool_size != 0:
        raise ValueError(f"token_topk must be a positive multiple of pool_size={pool_size}")
    if transform_mode not in (_IDENTITY, _PAGED, _RAGGED):
        raise ValueError(f"unsupported transform_mode={transform_mode}")

    group_topk = token_topk // pool_size
    out_cols = token_topk + pool_size - 1
    rows = T.dynamic("rows")
    page_rows = T.dynamic("page_rows")
    max_tokens = T.dynamic("max_tokens")

    @T.prim_func
    def main(
        selected_pools: T.Tensor((rows, group_topk), T.int32),
        pool_starts: T.Tensor((rows,), T.int32),
        pool_ends: T.Tensor((rows,), T.int32),
        seq_lens: T.Tensor((rows,), T.int32),
        token_page_table: T.Tensor((page_rows, max_tokens), T.int32),
        page_table_rows: T.Tensor((rows,), T.int32),
        topk_offsets: T.Tensor((rows,), T.int32),
        token_indices: T.Tensor((rows, out_cols), T.int32),
    ) -> None:
        with T.Kernel(rows, T.ceildiv(out_cols, threads), threads=threads) as (row, tile):
            lane = T.get_thread_binding(0)
            col = tile * threads + lane
            pool_start = pool_starts[row]
            pool_end = pool_ends[row]
            pool_count = pool_end - pool_start
            selected_count = T.min(pool_count, group_topk)
            history_count = selected_count * pool_size
            tail_start = pool_end * pool_size
            tail_count = seq_lens[row] - tail_start

            if col < out_cols:
                raw_token = T.alloc_var(T.int32, init=-1)
                if col < history_count:
                    group_rank = col // pool_size
                    slot = col % pool_size
                    group_id = T.if_then_else(
                        pool_count <= group_topk,
                        pool_start + group_rank,
                        selected_pools[row, group_rank],
                    )
                    raw_token = group_id * pool_size + slot
                elif col < history_count + tail_count:
                    raw_token = tail_start + col - history_count

                if raw_token >= 0:
                    if transform_mode == _PAGED:
                        page_table_row = page_table_rows[row]
                        if page_table_row >= 0 and page_table_row < page_rows and raw_token < max_tokens:
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


def _validate_inputs(
    logits: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    token_topk: int,
    pool_size: int,
    token_page_table: torch.Tensor | None,
    page_table_rows: torch.Tensor | None,
    topk_offsets: torch.Tensor | None,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if logits.ndim != 2:
        raise ValueError(f"logits must have rank 2, got shape {tuple(logits.shape)}")
    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be torch.float32, got {logits.dtype}")
    if not logits.is_cuda:
        raise ValueError("all tensors must be on a CUDA or ROCm device")
    if pool_size <= 1:
        raise ValueError(f"pool_size must be greater than one, got {pool_size}")
    if token_topk <= 0 or token_topk % pool_size != 0:
        raise ValueError(f"token_topk must be a positive multiple of pool_size={pool_size}")

    rows = logits.shape[0]
    group_topk = token_topk // pool_size
    for name, tensor in (
        ("pool_starts", pool_starts),
        ("pool_ends", pool_ends),
        ("seq_lens", seq_lens),
    ):
        if tuple(tensor.shape) != (rows,):
            raise ValueError(f"{name} must have shape {(rows,)}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be torch.int32, got {tensor.dtype}")
        if tensor.device != logits.device:
            raise ValueError(f"{name} must be on the same device as logits")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not logits.is_contiguous():
        raise ValueError("logits must be contiguous")

    if token_page_table is not None and topk_offsets is not None:
        raise ValueError("token_page_table and topk_offsets are mutually exclusive")
    if rows:
        min_start = int(pool_starts.min().item())
        max_end = int(pool_ends.max().item())
        if min_start < 0 or torch.any(pool_starts > pool_ends).item():
            raise ValueError("pool ranges must satisfy 0 <= pool_starts <= pool_ends")
        if max_end > logits.shape[1]:
            raise ValueError(f"pool_ends must not exceed logits width {logits.shape[1]}")
        full_pool_tokens = pool_ends * pool_size
        tail_counts = seq_lens - full_pool_tokens
        if torch.any(tail_counts < 0).item() or torch.any(tail_counts >= pool_size).item():
            raise ValueError("seq_lens must add between zero and pool_size-1 tail tokens")

    transform_mode = _IDENTITY
    if token_page_table is not None:
        transform_mode = _PAGED
        if token_page_table.ndim != 2 or token_page_table.shape[0] <= 0 or token_page_table.shape[1] <= 0:
            raise ValueError(f"token_page_table must be a nonempty rank-2 tensor, got {tuple(token_page_table.shape)}")
        if token_page_table.dtype != torch.int32:
            raise TypeError(f"token_page_table must be torch.int32, got {token_page_table.dtype}")
        if token_page_table.device != logits.device or not token_page_table.is_contiguous():
            raise ValueError("token_page_table must be contiguous and on the same device as logits")
        if page_table_rows is None:
            if token_page_table.shape[0] != rows:
                raise ValueError("page_table_rows is required when token_page_table rows differ from logits rows")
            page_table_rows = torch.arange(rows, dtype=torch.int32, device=logits.device)
        if tuple(page_table_rows.shape) != (rows,) or page_table_rows.dtype != torch.int32:
            raise ValueError(f"page_table_rows must be int32 with shape {(rows,)}")
        if page_table_rows.device != logits.device or not page_table_rows.is_contiguous():
            raise ValueError("page_table_rows must be contiguous and on the same device as logits")
        if rows:
            min_row = int(page_table_rows.min().item())
            max_row = int(page_table_rows.max().item())
            if min_row < 0 or max_row >= token_page_table.shape[0]:
                raise ValueError(f"page_table_rows must be in [0, {token_page_table.shape[0]})")
            max_seq_len = int(seq_lens.max().item())
            if max_seq_len > token_page_table.shape[1]:
                raise ValueError("token_page_table is shorter than an active sequence")
            for row in range(rows):
                table_row = int(page_table_rows[row].item())
                seq_len = int(seq_lens[row].item())
                if seq_len and torch.any(token_page_table[table_row, :seq_len] < 0).item():
                    raise ValueError(f"active token_page_table entries for row {row} must be nonnegative")
    else:
        token_page_table = torch.zeros((1, 1), dtype=torch.int32, device=logits.device)
        page_table_rows = torch.zeros((rows,), dtype=torch.int32, device=logits.device)

    if topk_offsets is not None:
        transform_mode = _RAGGED
        if tuple(topk_offsets.shape) != (rows,) or topk_offsets.dtype != torch.int32:
            raise ValueError(f"topk_offsets must be int32 with shape {(rows,)}")
        if topk_offsets.device != logits.device or not topk_offsets.is_contiguous():
            raise ValueError("topk_offsets must be contiguous and on the same device as logits")
    else:
        topk_offsets = torch.zeros((rows,), dtype=torch.int32, device=logits.device)

    return group_topk, token_page_table, page_table_rows, topk_offsets, transform_mode


def glm53_kpool_topk_transform(
    logits: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    token_topk: int = GLM53_INDEX_TOPK,
    pool_size: int = GLM53_POOL_SIZE,
    token_page_table: torch.Tensor | None = None,
    page_table_rows: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select pools, expand them to tokens, append the tail, and transform."""
    group_topk, token_page_table, page_table_rows, topk_offsets, transform_mode = _validate_inputs(
        logits,
        pool_starts,
        pool_ends,
        seq_lens,
        token_topk,
        pool_size,
        token_page_table,
        page_table_rows,
        topk_offsets,
    )
    out_cols = token_topk + pool_size - 1
    output = torch.full((logits.shape[0], out_cols), -1, dtype=torch.int32, device=logits.device)
    if logits.shape[0] == 0:
        return output

    topk_request = min(group_topk, logits.shape[1])
    if topk_request:
        selected_pools = tl_topk(logits, pool_starts, pool_ends, topk_request)
    else:
        selected_pools = torch.empty((logits.shape[0], 0), dtype=torch.int32, device=logits.device)
    if topk_request < group_topk:
        padded_pools = torch.zeros(
            (logits.shape[0], group_topk),
            dtype=torch.int32,
            device=logits.device,
        )
        padded_pools[:, :topk_request] = selected_pools
        selected_pools = padded_pools
    kernel = glm53_kpool_transform_kernel(
        token_topk=token_topk,
        pool_size=pool_size,
        transform_mode=transform_mode,
    )
    kernel(
        selected_pools,
        pool_starts,
        pool_ends,
        seq_lens,
        token_page_table,
        page_table_rows,
        topk_offsets,
        output,
    )
    return output


def glm53_kpool_topk_transform_reference(
    logits: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    token_topk: int = GLM53_INDEX_TOPK,
    pool_size: int = GLM53_POOL_SIZE,
    token_page_table: torch.Tensor | None = None,
    page_table_rows: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch reference for pool selection and token transformation."""
    group_topk = token_topk // pool_size
    out_cols = token_topk + pool_size - 1
    output = torch.full((logits.shape[0], out_cols), -1, dtype=torch.int32, device=logits.device)
    for row in range(logits.shape[0]):
        start = int(pool_starts[row].item())
        end = int(pool_ends[row].item())
        count = end - start
        if count <= group_topk:
            groups = torch.arange(start, end, dtype=torch.int64, device=logits.device)
        else:
            groups = torch.topk(logits[row, start:end], group_topk, sorted=False).indices.to(torch.int64) + start
        history = (groups[:, None] * pool_size + torch.arange(pool_size, dtype=torch.int64, device=logits.device)[None, :]).flatten()
        tail = torch.arange(end * pool_size, int(seq_lens[row].item()), dtype=torch.int64, device=logits.device)
        raw = torch.cat((history, tail))
        if token_page_table is not None:
            table_row = row if page_table_rows is None else int(page_table_rows[row].item())
            transformed = token_page_table[table_row, raw]
        elif topk_offsets is not None:
            transformed = raw.to(torch.int32) + topk_offsets[row]
        else:
            transformed = raw.to(torch.int32)
        output[row, : transformed.numel()] = transformed
    return output
