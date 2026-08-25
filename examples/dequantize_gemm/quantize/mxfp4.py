"""MXFP4 (E2M1 data + UE8M0 scales) quantization and scale layout utilities.

The SM120 ``blockscaled_chunk_kmajor`` scale layout is shared with NVFP4: the
physical word order is the same 128-row BlockScaledBasicChunk stack, only the
per-byte coverage changes (32 K elements per UE8M0 byte, so one uint32 word
covers K=128 and a ``block_K`` stage holds ``block_K // 128`` words).
"""

from __future__ import annotations

from .nvfp4 import (
    _check_block_shape,
    _import_torch,
    _pack_scale_bytes_to_words,
    _FP4_E2M1_MAX,
    encode_fp4_e2m1_values,
    pack_fp4_e2m1_codes,
    swizzle_blockscaled_chunk_kmajor_scale_words,
)

_MXFP4_SCALE_BLOCK_K = 32
_SCALE_BYTES_PER_WORD = 4


def decode_ue8m0_scale_bytes(scale_bytes):
    """Decode UE8M0 scale bytes (biased power-of-two exponents) to float32."""

    torch = _import_torch()
    if not isinstance(scale_bytes, torch.Tensor):
        raise TypeError(f"scale_bytes must be a torch.Tensor, got {type(scale_bytes)!r}")
    if scale_bytes.dtype != torch.uint8:
        raise TypeError(f"scale_bytes must have dtype torch.uint8, got {scale_bytes.dtype}")

    u = scale_bytes.to(torch.int32)
    value = torch.pow(torch.tensor(2.0, device=scale_bytes.device), (u - 127).to(torch.float32))
    return torch.where(u == 0xFF, torch.full_like(value, float("nan")), value)


def encode_ue8m0_scale_bytes(values, *, rounding: str = "ceil"):
    """Encode non-negative floats as UE8M0 bytes (power-of-two scales).

    ``rounding="ceil"`` rounds up to the next power of two so values divided
    by the scale stay within the FP4 range; zero maps to byte ``0x00``.
    Values above the largest representable scale (``2**127``) saturate to
    byte ``0xFE`` (``0xFF`` is the UE8M0 NaN encoding and is never produced).
    """

    torch = _import_torch()
    if rounding != "ceil":
        raise ValueError(f"rounding must be 'ceil', got {rounding!r}")
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"values must be a torch.Tensor, got {type(values)!r}")

    x = torch.nan_to_num(values.to(torch.float32), nan=0.0, posinf=2.0**127, neginf=0.0).clamp(min=0.0, max=2.0**127).contiguous()
    bits = x.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    code = (exponent + (mantissa != 0).to(torch.int32)).clamp(1, 254)
    code = torch.where(x == 0, torch.zeros_like(code), code)
    return code.to(torch.uint8)


def pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(scale_bytes, block_rows: int = 128, block_words: int = 2):
    """Pack UE8M0 scale bytes into SM120 BlockScaledBasicChunk K-major words.

    ``scale_bytes`` carries semantic row-major UE8M0 bytes with shape
    ``[rows, K / 32]``. The result is
    ``torch.uint32[ceil(rows / 128) * 128, K // 128]`` in the same physical
    word order as the NVFP4 packer; ``block_words`` is the per-``block_K``
    word count (``block_K // 128``).
    """

    torch = _import_torch()
    _check_block_shape(block_rows, block_words)
    if not isinstance(scale_bytes, torch.Tensor):
        raise TypeError(f"scale_bytes must be a torch.Tensor, got {type(scale_bytes)!r}")
    if scale_bytes.dtype != torch.uint8:
        raise TypeError(f"scale_bytes must have dtype torch.uint8, got {scale_bytes.dtype}")
    if scale_bytes.ndim != 2:
        raise ValueError(f"scale_bytes must be a 2D tensor, got shape {tuple(scale_bytes.shape)}")

    scale_cols = scale_bytes.shape[1]
    scale_cols_per_tile = block_words * _SCALE_BYTES_PER_WORD
    if scale_cols % scale_cols_per_tile != 0:
        raise ValueError(
            f"blockscaled_chunk_kmajor UE8M0 scale bytes require K/32 columns multiple of "
            f"{scale_cols_per_tile}, got {tuple(scale_bytes.shape)}"
        )

    words = _pack_scale_bytes_to_words(scale_bytes)
    return swizzle_blockscaled_chunk_kmajor_scale_words(words, block_rows, block_words)


def quantize_bf16_to_mxfp4_blockscaled(
    x,
    *,
    block_rows: int = 128,
    block_words: int = 2,
    scale_block_k: int = 32,
    return_scale_bytes: bool = False,
):
    """Quantize a BF16 activation tensor to SM120 MXFP4 blockscaled storage.

    Every ``scale_block_k = 32`` consecutive K elements share one UE8M0
    (power-of-two, ceil-rounded) scale. Returns ``(packed_fp4,
    scale_source)`` — ``torch.int8[rows, K/2]`` FP4 data and the packed
    ``torch.uint32[rows_pad, K/128]`` K-major scale source — plus the
    semantic ``[rows, K/32]`` scale bytes when ``return_scale_bytes`` is set.
    """

    torch = _import_torch()
    _check_block_shape(block_rows, block_words)
    if scale_block_k != _MXFP4_SCALE_BLOCK_K:
        raise ValueError(f"SM120 MXFP4 scale_block_k must be 32, got {scale_block_k}")
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
            f"SM120 MXFP4 quantization requires rows multiple of {block_rows} and K multiple of "
            f"{scale_block_k * scale_cols_per_tile}, got {tuple(x.shape)}"
        )

    x_f32 = x.contiguous().to(torch.float32)
    blocks = x_f32.reshape(rows, cols // scale_block_k, scale_block_k)
    amax = blocks.abs().amax(dim=2)
    scale_bytes = encode_ue8m0_scale_bytes(amax / _FP4_E2M1_MAX, rounding="ceil")
    scale_values = decode_ue8m0_scale_bytes(scale_bytes)
    scaled_blocks = torch.where(amax[..., None] > 0, blocks / scale_values[..., None], torch.zeros_like(blocks))
    fp4_codes = encode_fp4_e2m1_values(scaled_blocks.reshape(rows, cols))
    packed_fp4 = pack_fp4_e2m1_codes(fp4_codes)
    packed_scales = pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(scale_bytes, block_rows=block_rows, block_words=block_words)
    if return_scale_bytes:
        return packed_fp4, packed_scales, scale_bytes
    return packed_fp4, packed_scales
