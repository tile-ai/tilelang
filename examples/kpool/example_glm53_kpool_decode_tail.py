"""GLM-5.3 k-pool decode-tail maintenance and pool-close writes.

The sparse-attention indexer keeps one rolling BF16 tail per request. Prefill
seeds the last ``pool_size`` tokens into that tail. Decode then processes each
request's tokens in position order, updates the tail, and emits one compressed
FP8 cache entry whenever a four-token pool closes.
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T

from examples.kpool.example_glm53_kpool_compress import (
    GLM53_HEAD_DIM,
    GLM53_POOL_SIZE,
    _HADAMARD128_SCALE,
    _fast_log2_ceil,
    _fast_pow2,
    glm53_kpool_reference,
)
from tilelang.language.fp8 import determine_fp8_type, determine_torch_fp8_type


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def glm53_kpool_seed_tail_kernel(
    pool_size: int = GLM53_POOL_SIZE,
    head_dim: int = GLM53_HEAD_DIM,
):
    """Build the prefill kernel that seeds each request's rolling tail.

    ``tail_slot_mapping`` is a flat physical slot with the form
    ``tail_block * pool_size + position % pool_size``. ``cu_seqlens`` provides
    explicit request boundaries so one program can seed only that request's
    final rolling window.
    """
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 k-pool requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")

    num_tokens = T.dynamic("num_tokens")
    num_requests = T.dynamic("num_requests")
    num_tail_blocks = T.dynamic("num_tail_blocks")

    @T.prim_func
    def main(
        key: T.Tensor((num_tokens, head_dim), T.bfloat16),
        slot_score: T.Tensor((num_tokens, head_dim), T.bfloat16),
        tail_slot_mapping: T.Tensor((num_tokens,), T.int32),
        cu_seqlens: T.Tensor((num_requests + 1,), T.int32),
        tail_cache: T.Tensor((num_tail_blocks, 2, pool_size, head_dim), T.bfloat16),
    ) -> None:
        with T.Kernel(num_requests, threads=head_dim) as request_id:
            dim = T.get_thread_binding(0)
            request_start = cu_seqlens[request_id]
            request_end = cu_seqlens[request_id + 1]
            seed_start = T.max(request_start, request_end - pool_size)

            for seed_offset in T.serial(pool_size):
                token_id = seed_start + seed_offset
                if token_id < request_end and token_id < num_tokens:
                    tail_slot = tail_slot_mapping[token_id]
                    tail_block = tail_slot // pool_size
                    if tail_slot >= 0 and tail_slot < num_tail_blocks * pool_size:
                        tail_offset = tail_slot % pool_size
                        tail_cache[tail_block, 0, tail_offset, dim] = key[token_id, dim]
                        tail_cache[tail_block, 1, tail_offset, dim] = slot_score[token_id, dim]

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def glm53_kpool_decode_tail_kernel(
    next_n: int,
    pool_size: int = GLM53_POOL_SIZE,
    page_size: int = 64,
    head_dim: int = GLM53_HEAD_DIM,
    fp8_dtype: str | None = None,
    fp8_max: float | None = None,
    round_scale: bool = True,
):
    """Build the ordered decode-tail update and pool-close writer kernel."""
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 k-pool requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if fp8_dtype is None:
        fp8_dtype = determine_fp8_type()
    if fp8_max is None:
        fp8_max = float(torch.finfo(getattr(torch, fp8_dtype)).max)

    num_requests = T.dynamic("num_requests")
    num_tail_blocks = T.dynamic("num_tail_blocks")
    num_cache_blocks = T.dynamic("num_cache_blocks")

    @T.prim_func
    def main(
        tail_cache: T.Tensor((num_tail_blocks, 2, pool_size, head_dim), T.bfloat16),
        tail_slot_mapping: T.Tensor((num_requests, next_n), T.int32),
        key: T.Tensor((num_requests, next_n, head_dim), T.bfloat16),
        slot_score: T.Tensor((num_requests, next_n, head_dim), T.bfloat16),
        ape: T.Tensor((pool_size, head_dim), T.float32),
        cache_loc: T.Tensor((num_requests, next_n), T.int32),
        positions: T.Tensor((num_requests, next_n), T.int32),
        k_cache: T.Tensor((num_cache_blocks, page_size, head_dim), fp8_dtype),
        scale_cache: T.Tensor((num_cache_blocks, page_size), T.float32),
    ) -> None:
        with T.Kernel(num_requests, threads=head_dim) as request_id:
            dim = T.get_thread_binding(0)
            exchange = T.alloc_shared((head_dim,), T.float32)

            # Tokens for one request are deliberately processed serially. A
            # speculative token can fill a tail slot that a later token in the
            # same launch must read when it closes the pool.
            for token_offset in T.serial(next_n):
                position = positions[request_id, token_offset]
                tail_slot = tail_slot_mapping[request_id, token_offset]
                tail_block = tail_slot // pool_size
                tail_offset = position % pool_size

                if position >= 0 and tail_slot >= 0 and tail_block >= 0 and tail_block < num_tail_blocks:
                    current_k = T.alloc_var(
                        T.float32,
                        init=T.cast(key[request_id, token_offset, dim], T.float32),
                    )
                    current_score = T.alloc_var(
                        T.float32,
                        init=T.cast(slot_score[request_id, token_offset, dim], T.float32),
                    )
                    physical_loc = cache_loc[request_id, token_offset]

                    if tail_offset == pool_size - 1 and physical_loc >= 0:
                        max_score = T.alloc_var(T.float32, init=-T.infinity(T.float32))
                        denom = T.alloc_var(T.float32, init=0.0)
                        pooled = T.alloc_var(T.float32, init=0.0)

                        for pool_slot in T.serial(pool_size):
                            historical_score = T.cast(tail_cache[tail_block, 1, pool_slot, dim], T.float32)
                            score = (
                                T.if_then_else(
                                    pool_slot == tail_offset,
                                    current_score,
                                    historical_score,
                                )
                                + ape[pool_slot, dim]
                            )
                            max_score = T.max(max_score, score)

                        for pool_slot in T.serial(pool_size):
                            historical_score = T.cast(tail_cache[tail_block, 1, pool_slot, dim], T.float32)
                            score = (
                                T.if_then_else(
                                    pool_slot == tail_offset,
                                    current_score,
                                    historical_score,
                                )
                                + ape[pool_slot, dim]
                            )
                            probability = T.exp(score - max_score)
                            denom = denom + probability
                            historical_k = T.cast(tail_cache[tail_block, 0, pool_slot, dim], T.float32)
                            pool_k = T.if_then_else(
                                pool_slot == tail_offset,
                                current_k,
                                historical_k,
                            )
                            pooled = pooled + pool_k * probability

                        pooled = T.cast(T.cast(pooled / denom, T.bfloat16), T.float32)
                        # A prior pool-closing iteration may still have threads
                        # reading exchange[0] for its absmax. Synchronize before
                        # any lane reuses shared memory for the next closure.
                        T.sync_threads()
                        exchange[dim] = pooled
                        T.sync_threads()

                        for stage in T.serial(7):
                            stride = 1 << stage
                            own = T.alloc_var(T.float32, init=exchange[dim])
                            peer = T.alloc_var(T.float32, init=exchange[dim ^ stride])
                            T.sync_threads()
                            exchange[dim] = T.if_then_else(
                                (dim & stride) == 0,
                                own + peer,
                                peer - own,
                            )
                            T.sync_threads()

                        rotated = T.alloc_var(
                            T.float32,
                            init=T.cast(
                                T.cast(exchange[dim] * _HADAMARD128_SCALE, T.bfloat16),
                                T.float32,
                            ),
                        )
                        exchange[dim] = T.abs(rotated)
                        T.sync_threads()
                        for stage in T.serial(7):
                            stride = 64 >> stage
                            if dim < stride:
                                exchange[dim] = T.max(exchange[dim], exchange[dim + stride])
                            T.sync_threads()

                        absmax = T.max(exchange[0], T.cast(1e-4, T.float32))
                        if round_scale:
                            scale = _fast_pow2(_fast_log2_ceil(absmax / fp8_max))
                        else:
                            scale = absmax / fp8_max
                        quantized = T.max(T.min(rotated / scale, fp8_max), -fp8_max)

                        cache_block = physical_loc // page_size
                        page_offset = physical_loc % page_size
                        if cache_block >= 0 and cache_block < num_cache_blocks:
                            k_cache[cache_block, page_offset, dim] = T.cast(quantized, fp8_dtype)
                            if dim == 0:
                                scale_cache[cache_block, page_offset] = scale

                    # Completion reads the prior ring before the current token
                    # is stashed. Non-closing tokens simply advance the ring.
                    tail_cache[tail_block, 0, tail_offset, dim] = key[request_id, token_offset, dim]
                    tail_cache[tail_block, 1, tail_offset, dim] = slot_score[request_id, token_offset, dim]

    return main


def _require_device_tensors(tensors: dict[str, torch.Tensor], device: torch.device) -> None:
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same device as key")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _validate_tail_cache(tail_cache: torch.Tensor, pool_size: int, head_dim: int) -> None:
    if tail_cache.ndim != 4 or tuple(tail_cache.shape[1:]) != (2, pool_size, head_dim):
        raise ValueError(f"tail_cache must have shape [num_tail_blocks, 2, pool_size, head_dim], got {tuple(tail_cache.shape)}")
    if tail_cache.shape[0] <= 0:
        raise ValueError("tail_cache must contain at least one block")
    if tail_cache.dtype != torch.bfloat16:
        raise TypeError(f"tail_cache must be torch.bfloat16, got {tail_cache.dtype}")


def _validate_seed_inputs(
    tail_cache: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pool_size: int,
    head_dim: int,
) -> None:
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 k-pool requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")
    _validate_tail_cache(tail_cache, pool_size, head_dim)
    if key.ndim != 2 or key.shape[1] != head_dim:
        raise ValueError(f"key must have shape [num_tokens, {head_dim}], got {tuple(key.shape)}")
    if tuple(slot_score.shape) != tuple(key.shape):
        raise ValueError(f"slot_score must have shape {tuple(key.shape)}, got {tuple(slot_score.shape)}")
    if tuple(tail_slot_mapping.shape) != (key.shape[0],):
        raise ValueError(f"tail_slot_mapping must have shape {(key.shape[0],)}, got {tuple(tail_slot_mapping.shape)}")
    if key.dtype != torch.bfloat16:
        raise TypeError(f"key must be torch.bfloat16, got {key.dtype}")
    if slot_score.dtype != torch.bfloat16:
        raise TypeError(f"slot_score must be torch.bfloat16, got {slot_score.dtype}")
    if tail_slot_mapping.dtype != torch.int32:
        raise TypeError(f"tail_slot_mapping must be torch.int32, got {tail_slot_mapping.dtype}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must have rank 1, got shape {tuple(cu_seqlens.shape)}")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"cu_seqlens must be torch.int32, got {cu_seqlens.dtype}")
    if not key.is_cuda:
        raise ValueError("all tensors must be on a CUDA or ROCm device")
    _require_device_tensors(
        {
            "key": key,
            "slot_score": slot_score,
            "tail_slot_mapping": tail_slot_mapping,
            "cu_seqlens": cu_seqlens,
            "tail_cache": tail_cache,
        },
        key.device,
    )

    boundaries = cu_seqlens.cpu().tolist()
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != key.shape[0]:
        raise ValueError("cu_seqlens must start at 0 and end at num_tokens")
    if any(start > end for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")

    tail_slots = tail_cache.shape[0] * pool_size
    request_blocks = []
    for request_start, request_end in zip(boundaries, boundaries[1:]):
        request_slots = tail_slot_mapping[request_start:request_end]
        if request_slots.numel() == 0:
            continue
        if torch.any(request_slots < 0).item() or int(request_slots.max().item()) >= tail_slots:
            raise ValueError(f"request tail slots must be in [0, {tail_slots})")
        blocks = request_slots // pool_size
        if torch.any(blocks != blocks[0]).item():
            raise ValueError("all prefill tokens for one request must use the same tail block")
        if int(request_slots[0].item()) % pool_size != 0:
            raise ValueError("the first prefill tail slot must have phase zero")
        if request_slots.numel() > 1:
            phase_step = (request_slots[1:] - request_slots[:-1]).remainder(pool_size)
            if torch.any(phase_step != 1).item():
                raise ValueError("prefill tail slots must advance by one modulo pool_size")
        request_blocks.append(int(blocks[0].item()))
    if len(set(request_blocks)) != len(request_blocks):
        raise ValueError("active prefill requests must use distinct tail blocks")


def glm53_kpool_seed_tail_cache(
    tail_cache: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    pool_size: int = GLM53_POOL_SIZE,
    head_dim: int = GLM53_HEAD_DIM,
) -> None:
    """Seed each request's rolling tail from a flattened prefill batch."""
    _validate_seed_inputs(
        tail_cache,
        key,
        slot_score,
        tail_slot_mapping,
        cu_seqlens,
        pool_size,
        head_dim,
    )
    if key.shape[0] == 0 or cu_seqlens.numel() <= 1:
        return
    kernel = glm53_kpool_seed_tail_kernel(pool_size=pool_size, head_dim=head_dim)
    kernel(key, slot_score, tail_slot_mapping, cu_seqlens, tail_cache)


def _validate_decode_inputs(
    tail_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    cache_loc: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    pool_size: int,
    head_dim: int,
) -> tuple[str, float]:
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 k-pool requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")
    _validate_tail_cache(tail_cache, pool_size, head_dim)
    if key.ndim != 3 or key.shape[2] != head_dim:
        raise ValueError(f"key must have shape [num_requests, next_n, {head_dim}], got {tuple(key.shape)}")
    expected_metadata_shape = tuple(key.shape[:2])
    if tuple(slot_score.shape) != tuple(key.shape):
        raise ValueError(f"slot_score must have shape {tuple(key.shape)}, got {tuple(slot_score.shape)}")
    if tuple(ape.shape) != (pool_size, head_dim):
        raise ValueError(f"ape must have shape {(pool_size, head_dim)}, got {tuple(ape.shape)}")
    for name, tensor in {
        "tail_slot_mapping": tail_slot_mapping,
        "cache_loc": cache_loc,
        "positions": positions,
    }.items():
        if tuple(tensor.shape) != expected_metadata_shape:
            raise ValueError(f"{name} must have shape {expected_metadata_shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be torch.int32, got {tensor.dtype}")

    if key.dtype != torch.bfloat16:
        raise TypeError(f"key must be torch.bfloat16, got {key.dtype}")
    if slot_score.dtype != torch.bfloat16:
        raise TypeError(f"slot_score must be torch.bfloat16, got {slot_score.dtype}")
    if ape.dtype != torch.float32:
        raise TypeError(f"ape must be torch.float32, got {ape.dtype}")
    if scale_cache.dtype != torch.float32:
        raise TypeError(f"scale_cache must be torch.float32, got {scale_cache.dtype}")
    if scale_cache.ndim != 2:
        raise ValueError(f"scale_cache must have rank 2, got shape {tuple(scale_cache.shape)}")
    if k_cache.ndim != 3 or tuple(k_cache.shape[1:]) != (scale_cache.shape[1], head_dim):
        raise ValueError(
            "k_cache must have shape [num_blocks, page_size, head_dim] and agree with scale_cache; "
            f"got k_cache={tuple(k_cache.shape)}, scale_cache={tuple(scale_cache.shape)}"
        )
    if scale_cache.shape[0] != k_cache.shape[0]:
        raise ValueError(
            "scale_cache must have shape [num_blocks, page_size] and agree with k_cache; "
            f"got k_cache={tuple(k_cache.shape)}, scale_cache={tuple(scale_cache.shape)}"
        )
    if k_cache.shape[0] <= 0 or k_cache.shape[1] <= 0:
        raise ValueError(f"cache dimensions must be positive, got {tuple(k_cache.shape)}")
    if not key.is_cuda:
        raise ValueError("all tensors must be on a CUDA or ROCm device")

    _require_device_tensors(
        {
            "tail_cache": tail_cache,
            "tail_slot_mapping": tail_slot_mapping,
            "key": key,
            "slot_score": slot_score,
            "ape": ape,
            "cache_loc": cache_loc,
            "positions": positions,
            "k_cache": k_cache,
            "scale_cache": scale_cache,
        },
        key.device,
    )

    fp8_dtype_name = determine_fp8_type(device=key.device)
    expected_fp8_dtype = determine_torch_fp8_type(device=key.device)
    if k_cache.dtype != expected_fp8_dtype:
        raise TypeError(f"k_cache must use the platform FP8 dtype {expected_fp8_dtype}, got {k_cache.dtype}")

    valid = positions >= 0
    if torch.any(valid != (tail_slot_mapping >= 0)).item():
        raise ValueError("positions and tail_slot_mapping must use matching negative padding")
    if torch.any((~valid) & (cache_loc >= 0)).item():
        raise ValueError("padded tokens must use a negative cache_loc")
    if key.shape[1] > 1:
        if torch.any(valid[:, 1:] & ~valid[:, :-1]).item():
            raise ValueError("valid decode tokens must form a prefix in each request row")
        consecutive = positions[:, 1:] == positions[:, :-1] + 1
        if torch.any(valid[:, 1:] & ~consecutive).item():
            raise ValueError("valid positions must be consecutive within each request")

    active_tail_slots = tail_slot_mapping[valid]
    if active_tail_slots.numel() != 0:
        tail_slots = tail_cache.shape[0] * pool_size
        if int(active_tail_slots.max().item()) >= tail_slots:
            raise ValueError(f"active tail slots must be in [0, {tail_slots})")
        if torch.any(active_tail_slots % pool_size != positions[valid] % pool_size).item():
            raise ValueError("tail slot phase must equal position modulo pool_size")

        tail_blocks = torch.where(valid, tail_slot_mapping // pool_size, -1)
        active_rows = valid[:, 0]
        row_blocks = tail_blocks[:, 0]
        if torch.any(valid & (tail_blocks != row_blocks[:, None])).item():
            raise ValueError("all tokens for one request must use the same tail block")
        active_row_blocks = row_blocks[active_rows]
        if torch.unique(active_row_blocks).numel() != active_row_blocks.numel():
            raise ValueError("active requests must use distinct tail blocks")

    completion = valid & (positions % pool_size == pool_size - 1)
    if torch.any(completion != (cache_loc >= 0)).item():
        raise ValueError("cache_loc must be nonnegative exactly when a valid token closes a pool")
    active_cache_locs = cache_loc[completion]
    if active_cache_locs.numel() != 0:
        cache_slots = k_cache.shape[0] * k_cache.shape[1]
        if int(active_cache_locs.max().item()) >= cache_slots:
            raise ValueError(f"active cache locations must be in [0, {cache_slots})")
        if torch.unique(active_cache_locs).numel() != active_cache_locs.numel():
            raise ValueError("active cache locations must be unique to avoid concurrent writes")

    return fp8_dtype_name, float(torch.finfo(expected_fp8_dtype).max)


def glm53_kpool_decode_update_and_maybe_write_cache(
    tail_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    cache_loc: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    *,
    pool_size: int = GLM53_POOL_SIZE,
    head_dim: int = GLM53_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    """Update decode tails in order and write each newly completed pool."""
    fp8_dtype, fp8_max = _validate_decode_inputs(
        tail_cache,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        cache_loc,
        positions,
        k_cache,
        scale_cache,
        pool_size,
        head_dim,
    )
    if key.shape[0] == 0 or key.shape[1] == 0:
        return
    kernel = glm53_kpool_decode_tail_kernel(
        next_n=key.shape[1],
        pool_size=pool_size,
        page_size=k_cache.shape[1],
        head_dim=head_dim,
        fp8_dtype=fp8_dtype,
        fp8_max=fp8_max,
        round_scale=round_scale,
    )
    kernel(
        tail_cache,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        cache_loc,
        positions,
        k_cache,
        scale_cache,
    )


def glm53_kpool_seed_tail_reference(
    tail_cache: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    pool_size: int = GLM53_POOL_SIZE,
) -> None:
    """PyTorch reference for prefill tail seeding; mutates ``tail_cache``."""
    boundaries = cu_seqlens.cpu().tolist()
    for request_start, request_end in zip(boundaries, boundaries[1:]):
        seed_start = max(request_start, request_end - pool_size)
        for token_id in range(seed_start, request_end):
            tail_slot = int(tail_slot_mapping[token_id].item())
            tail_block = tail_slot // pool_size
            tail_offset = tail_slot % pool_size
            tail_cache[tail_block, 0, tail_offset] = key[token_id]
            tail_cache[tail_block, 1, tail_offset] = slot_score[token_id]


def glm53_kpool_decode_tail_reference(
    tail_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    cache_loc: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    *,
    pool_size: int = GLM53_POOL_SIZE,
    round_scale: bool = True,
) -> None:
    """Sequential PyTorch reference; mutates the tail and compressed caches."""
    num_requests, next_n = key.shape[:2]
    flat_k_cache = k_cache.view(-1, GLM53_HEAD_DIM)
    flat_scale_cache = scale_cache.view(-1)
    for request_id in range(num_requests):
        for token_offset in range(next_n):
            position = int(positions[request_id, token_offset].item())
            tail_slot = int(tail_slot_mapping[request_id, token_offset].item())
            if position < 0 or tail_slot < 0:
                continue
            tail_block = tail_slot // pool_size
            tail_offset = position % pool_size
            if tail_offset == pool_size - 1:
                pool_k = tail_cache[tail_block, 0].clone()
                pool_score = tail_cache[tail_block, 1].clone()
                pool_k[tail_offset] = key[request_id, token_offset]
                pool_score[tail_offset] = slot_score[request_id, token_offset]
                compressed_k, compressed_scale = glm53_kpool_reference(
                    pool_k.unsqueeze(0),
                    pool_score.unsqueeze(0),
                    ape,
                    round_scale=round_scale,
                )
                physical_loc = int(cache_loc[request_id, token_offset].item())
                flat_k_cache[physical_loc] = compressed_k[0]
                flat_scale_cache[physical_loc] = compressed_scale[0]
            tail_cache[tail_block, 0, tail_offset] = key[request_id, token_offset]
            tail_cache[tail_block, 1, tail_offset] = slot_score[request_id, token_offset]
