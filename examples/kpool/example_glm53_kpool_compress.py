"""GLM-5.3 k-pool compression and safe paged-cache writes.

This example implements the fused operation used by the GLM-5.3-Flash sparse
attention indexer: per-dimension softmax pooling, Hadamard-128 rotation,
per-vector FP8 quantization, and an indexed cache write.
"""

from __future__ import annotations

import math

import torch
import tilelang
import tilelang.language as T
from tilelang.language.fp8 import determine_fp8_type, determine_torch_fp8_type


GLM53_HEAD_DIM = 128
GLM53_POOL_SIZE = 16
_HADAMARD128_SCALE = 1.0 / math.sqrt(GLM53_HEAD_DIM)


def _fast_log2_ceil(x):
    """Compute ceil(log2(x)) through the IEEE-754 float32 representation."""
    bits_x = T.reinterpret(x, "uint32")
    exp_x = (bits_x >> 23) & 0xFF
    mantissa = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(mantissa != 0, 1, 0))


def _fast_pow2(x):
    """Compute 2**x for an integer exponent through its float32 bits."""
    return T.reinterpret((x + 127) << 23, "float32")


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def glm53_kpool_compress_kernel(
    pool_size: int = GLM53_POOL_SIZE,
    page_size: int = 64,
    head_dim: int = GLM53_HEAD_DIM,
    fp8_dtype: str | None = None,
    fp8_max: float | None = None,
    round_scale: bool = True,
):
    """Build the fused compressor/cache-writer kernel.

    One 128-thread program processes one pool. Each thread owns one head
    dimension during the softmax and pooling phases. The Hadamard transform
    and absmax reduction exchange those values through shared memory.
    """
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

    num_pools = T.dynamic("num_pools")
    num_blocks = T.dynamic("num_blocks")

    @T.prim_func
    def main(
        slot_k: T.Tensor((num_pools, pool_size, head_dim), T.bfloat16),
        slot_score: T.Tensor((num_pools, pool_size, head_dim), T.bfloat16),
        ape: T.Tensor((pool_size, head_dim), T.float32),
        loc: T.Tensor((num_pools,), T.int64),
        write_mask: T.Tensor((num_pools,), T.bool),
        k_cache: T.Tensor((num_blocks, page_size, head_dim), fp8_dtype),
        scale_cache: T.Tensor((num_blocks, page_size), T.float32),
    ) -> None:
        with T.Kernel(num_pools, threads=head_dim) as pool_id:
            dim = T.get_thread_binding(0)
            exchange = T.alloc_shared((head_dim,), T.float32)

            # The branch is uniform across the block. Masked rows never load a
            # location and never enter the barrier-containing body.
            if write_mask[pool_id]:
                max_score = T.alloc_var(T.float32, init=-T.infinity(T.float32))
                denom = T.alloc_var(T.float32, init=0.0)
                pooled = T.alloc_var(T.float32, init=0.0)

                for slot in T.serial(pool_size):
                    score = T.cast(slot_score[pool_id, slot, dim], T.float32) + ape[slot, dim]
                    max_score = T.max(max_score, score)

                for slot in T.serial(pool_size):
                    score = T.cast(slot_score[pool_id, slot, dim], T.float32) + ape[slot, dim]
                    probability = T.exp(score - max_score)
                    denom = denom + probability
                    pooled = pooled + T.cast(slot_k[pool_id, slot, dim], T.float32) * probability

                # Match the production path's BF16 boundary before Hadamard.
                pooled = T.cast(T.cast(pooled / denom, T.bfloat16), T.float32)

                exchange[dim] = pooled
                T.sync_threads()

                # Normalized Sylvester Hadamard-128. A barrier between the
                # loads and stores prevents an in-place read/write race.
                for stage in T.serial(7):
                    stride = 1 << stage
                    own = T.alloc_var(T.float32, init=exchange[dim])
                    peer = T.alloc_var(T.float32, init=exchange[dim ^ stride])
                    T.sync_threads()
                    exchange[dim] = T.if_then_else((dim & stride) == 0, own + peer, peer - own)
                    T.sync_threads()

                rotated = T.alloc_var(
                    T.float32,
                    init=T.cast(T.cast(exchange[dim] * _HADAMARD128_SCALE, T.bfloat16), T.float32),
                )

                # Reuse shared memory for the block-wide per-vector absmax.
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

                physical_loc = loc[pool_id]
                block_id = physical_loc // page_size
                page_offset = physical_loc % page_size
                # Host validation is authoritative; retain a device-side block
                # guard so malformed metadata still cannot escape the cache.
                if block_id >= 0 and block_id < num_blocks:
                    k_cache[block_id, page_offset, dim] = T.cast(quantized, fp8_dtype)
                    if dim == 0:
                        scale_cache[block_id, page_offset] = scale

    return main


def _validate_kpool_inputs(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    pool_size: int,
    head_dim: int,
) -> tuple[torch.Tensor, str, float]:
    """Validate the public contract and return launch-specific FP8 metadata."""
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 k-pool requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")
    if slot_k.ndim != 3:
        raise ValueError(f"slot_k must have rank 3, got shape {tuple(slot_k.shape)}")
    expected_slot_shape = (slot_k.shape[0], pool_size, head_dim)
    if tuple(slot_k.shape) != expected_slot_shape:
        raise ValueError(f"slot_k must have shape {expected_slot_shape}, got {tuple(slot_k.shape)}")
    if tuple(slot_score.shape) != expected_slot_shape:
        raise ValueError(f"slot_score must have shape {expected_slot_shape}, got {tuple(slot_score.shape)}")
    if tuple(ape.shape) != (pool_size, head_dim):
        raise ValueError(f"ape must have shape {(pool_size, head_dim)}, got {tuple(ape.shape)}")
    if tuple(loc.shape) != (slot_k.shape[0],):
        raise ValueError(f"loc must have shape {(slot_k.shape[0],)}, got {tuple(loc.shape)}")

    if slot_k.dtype != torch.bfloat16:
        raise TypeError(f"slot_k must be torch.bfloat16, got {slot_k.dtype}")
    if slot_score.dtype != torch.bfloat16:
        raise TypeError(f"slot_score must be torch.bfloat16, got {slot_score.dtype}")
    if ape.dtype != torch.float32:
        raise TypeError(f"ape must be torch.float32, got {ape.dtype}")
    if loc.dtype != torch.int64:
        raise TypeError(f"loc must be torch.int64, got {loc.dtype}")
    if scale_cache.dtype != torch.float32:
        raise TypeError(f"scale_cache must be torch.float32, got {scale_cache.dtype}")

    if k_cache.ndim != 3:
        raise ValueError(f"k_cache must have rank 3, got shape {tuple(k_cache.shape)}")
    if scale_cache.ndim != 2:
        raise ValueError(f"scale_cache must have rank 2, got shape {tuple(scale_cache.shape)}")
    if tuple(k_cache.shape[1:]) != (scale_cache.shape[1], head_dim):
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

    tensors = (slot_k, slot_score, ape, loc, k_cache, scale_cache)
    if not slot_k.is_cuda:
        raise ValueError("all tensors must be on a CUDA or ROCm device")
    if any(t.device != slot_k.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")

    if write_mask is None:
        write_mask = torch.ones((slot_k.shape[0],), dtype=torch.bool, device=slot_k.device)
    else:
        if tuple(write_mask.shape) != (slot_k.shape[0],):
            raise ValueError(f"write_mask must have shape {(slot_k.shape[0],)}, got {tuple(write_mask.shape)}")
        if write_mask.dtype != torch.bool:
            raise TypeError(f"write_mask must be torch.bool, got {write_mask.dtype}")
        if write_mask.device != slot_k.device:
            raise ValueError("write_mask must be on the same device as slot_k")
        if not write_mask.is_contiguous():
            raise ValueError("write_mask must be contiguous")

    fp8_dtype_name = determine_fp8_type(device=slot_k.device)
    expected_fp8_dtype = determine_torch_fp8_type(device=slot_k.device)
    if k_cache.dtype != expected_fp8_dtype:
        raise TypeError(f"k_cache must use the platform FP8 dtype {expected_fp8_dtype}, got {k_cache.dtype}")

    active_locs = loc[write_mask]
    if active_locs.numel() != 0:
        cache_slots = k_cache.shape[0] * k_cache.shape[1]
        min_loc = int(active_locs.min().item())
        max_loc = int(active_locs.max().item())
        if min_loc < 0 or max_loc >= cache_slots:
            raise ValueError(f"active loc values must be in [0, {cache_slots}), got min={min_loc}, max={max_loc}")
        if torch.unique(active_locs).numel() != active_locs.numel():
            raise ValueError("active loc values must be unique to avoid concurrent cache writes")

    return write_mask, fp8_dtype_name, float(torch.finfo(expected_fp8_dtype).max)


def glm53_kpool_compress_and_write_cache(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    *,
    write_mask: torch.Tensor | None = None,
    pool_size: int = GLM53_POOL_SIZE,
    head_dim: int = GLM53_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    """Compress pools and write caller-owned, separately typed caches in place.

    ``loc`` is a flat physical slot in page-major order. Masked rows may carry
    any sentinel location; only active locations are range- and uniqueness-
    checked. Validation intentionally runs on the host before kernel launch.
    """
    write_mask, fp8_dtype, fp8_max = _validate_kpool_inputs(
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        k_cache,
        scale_cache,
        pool_size,
        head_dim,
    )
    if slot_k.shape[0] == 0:
        return

    kernel = glm53_kpool_compress_kernel(
        pool_size=pool_size,
        page_size=k_cache.shape[1],
        head_dim=head_dim,
        fp8_dtype=fp8_dtype,
        fp8_max=fp8_max,
        round_scale=round_scale,
    )
    kernel(slot_k, slot_score, ape, loc, write_mask, k_cache, scale_cache)


def hadamard128_reference(x: torch.Tensor) -> torch.Tensor:
    """Matrix-free normalized Sylvester Hadamard-128 reference."""
    if x.shape[-1] != GLM53_HEAD_DIM:
        raise ValueError(f"expected last dimension {GLM53_HEAD_DIM}, got {x.shape[-1]}")
    result = x.float().clone()
    half = 1
    while half < GLM53_HEAD_DIM:
        pairs = result.reshape(*result.shape[:-1], -1, 2, half)
        left = pairs[..., 0, :].clone()
        right = pairs[..., 1, :].clone()
        pairs[..., 0, :] = left + right
        pairs[..., 1, :] = left - right
        half *= 2
    return result * _HADAMARD128_SCALE


def glm53_kpool_reference(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    *,
    round_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference returning compressed FP8 values and FP32 scales."""
    scores = slot_score.float() + ape.unsqueeze(0)
    max_score = scores.max(dim=1).values
    denom = torch.zeros_like(max_score)
    pooled = torch.zeros_like(max_score)
    for slot in range(slot_k.shape[1]):
        probability = torch.exp(scores[:, slot] - max_score)
        denom += probability
        pooled += slot_k[:, slot].float() * probability
    pooled = (pooled / denom).to(torch.bfloat16).float()
    rotated = hadamard128_reference(pooled).to(torch.bfloat16).float()

    fp8_dtype = determine_torch_fp8_type(device=slot_k.device)
    fp8_max = float(torch.finfo(fp8_dtype).max)
    absmax = rotated.abs().amax(dim=1).clamp(min=1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax / fp8_max)))
    else:
        scale = absmax / fp8_max
    quantized = (rotated / scale[:, None]).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return quantized, scale
