"""MXFP8 (FP8 data + UE8M0 scales) quantization utilities for SM120.

The scale storage is byte-identical to the MXFP4 layout: one UE8M0 byte per
``scale_block_k = 32`` consecutive K elements, packed into the SM120
BlockScaledBasicChunk K-major word order (one uint32 word covers K=128).
Only the data side differs - FP8 elements are stored one per byte, so the
global A/B footprint is exactly ``numel`` bytes (the compression invariant
tests pin this).
"""

from __future__ import annotations

from .mxfp4 import (
    _MXFP4_SCALE_BLOCK_K as _MXFP8_SCALE_BLOCK_K,
    _SCALE_BYTES_PER_WORD,
    decode_ue8m0_scale_bytes,
    encode_ue8m0_scale_bytes,
    pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes,
)
from .nvfp4 import _check_block_shape, _import_torch

_FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}


def quantize_bf16_to_mxfp8_blockscaled(
    x,
    *,
    dtype: str = "e4m3",
    block_rows: int = 128,
    block_words: int = 2,
    scale_block_k: int = 32,
    return_scale_bytes: bool = False,
):
    """Quantize a BF16 activation tensor to SM120 MXFP8 blockscaled storage.

    Every ``scale_block_k = 32`` consecutive K elements share one UE8M0
    (power-of-two, ceil-rounded) scale chosen so the scaled block fits the
    target FP8 range. Returns ``(fp8_data, scale_source)`` — a
    ``torch.float8_*[rows, K]`` tensor and the packed
    ``torch.uint32[rows_pad, K/128]`` K-major scale source — plus the
    semantic ``[rows, K/32]`` scale bytes when ``return_scale_bytes`` is set.

    Special values follow the MXFP4 quantizer: NaN poisons its block's amax
    (scale byte ``0x00``, block written as zeros); Inf saturates the block
    scale to ``0xFE`` and the Inf element to the FP8 max.
    """

    torch = _import_torch()
    _check_block_shape(block_rows, block_words)
    if scale_block_k != _MXFP8_SCALE_BLOCK_K:
        raise ValueError(f"SM120 MXFP8 scale_block_k must be 32, got {scale_block_k}")
    if dtype not in _FP8_MAX:
        raise ValueError(f"dtype must be one of {sorted(_FP8_MAX)}, got {dtype!r}")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(f"x must be a 2D tensor, got shape {tuple(x.shape)}")
    if not x.dtype.is_floating_point:
        raise TypeError(f"x must be a floating-point tensor, got {x.dtype}")

    rows, cols = x.shape
    scale_cols_per_tile = block_words * _SCALE_BYTES_PER_WORD
    if rows % block_rows != 0 or cols % (scale_block_k * scale_cols_per_tile) != 0:
        raise ValueError(
            f"SM120 MXFP8 quantization requires rows multiple of {block_rows} and K multiple of "
            f"{scale_block_k * scale_cols_per_tile}, got {tuple(x.shape)}"
        )

    x_f32 = x.contiguous().to(torch.float32)
    blocks = x_f32.reshape(rows, cols // scale_block_k, scale_block_k)
    amax = blocks.abs().amax(dim=2)
    scale_bytes = encode_ue8m0_scale_bytes(amax / _FP8_MAX[dtype], rounding="ceil")
    scale_values = decode_ue8m0_scale_bytes(scale_bytes)
    scaled_blocks = torch.where(amax[..., None] > 0, blocks / scale_values[..., None], torch.zeros_like(blocks))
    torch_fp8 = torch.float8_e4m3fn if dtype == "e4m3" else torch.float8_e5m2
    fp8_data = scaled_blocks.reshape(rows, cols).to(torch_fp8)
    packed_scales = pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(scale_bytes, block_rows=block_rows, block_words=block_words)
    if return_scale_bytes:
        return fp8_data, packed_scales, scale_bytes
    return fp8_data, packed_scales
