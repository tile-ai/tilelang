"""MXFP6 (E2M3/E3M2 data + UE8M0 scales) quantization utilities for SM120.

Scale storage is byte-identical to the MXFP4/MXFP8 layout (one UE8M0 byte
per 32 K elements, BlockScaledBasicChunk K-major words). Data storage:
torch has no fp6 dtype, so packed fp6 lives in uint8 blobs - 4 elements per
3 bytes as an LSB-first 6-bit stream, matching the bit order of the
16U6_ALIGN16B smem form and the ``b6x16_p32`` ldmatrix (hardware-pinned in
the sub-byte ldmatrix tests). Global footprint is exactly ``numel * 3 / 4``
bytes (0.75 B/elem - the compression invariant).
"""

from __future__ import annotations

from .mxfp4 import (
    _MXFP4_SCALE_BLOCK_K as _MXFP6_SCALE_BLOCK_K,
    _SCALE_BYTES_PER_WORD,
    decode_ue8m0_scale_bytes,
    encode_ue8m0_scale_bytes,
    pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes,
)
from .nvfp4 import _check_block_shape, _import_torch

# e2m3: 1 sign, 2 exponent (bias 1), 3 mantissa -> max 1.875 * 2^2 = 7.5
# e3m2: 1 sign, 3 exponent (bias 3), 2 mantissa -> max 1.75 * 2^4 = 28
_FP6_MAX = {"e2m3": 7.5, "e3m2": 28.0}
_FP6_SPECS = {"e2m3": (2, 3, 1), "e3m2": (3, 2, 3)}  # (exp_bits, man_bits, bias)


def _fp6_code_values(dtype: str):
    """Decode table: value of each of the 64 fp6 codes (sign at bit 5)."""
    torch = _import_torch()
    exp_bits, man_bits, bias = _FP6_SPECS[dtype]
    values = []
    for code in range(64):
        sign = -1.0 if (code >> 5) & 1 else 1.0
        exp = (code >> man_bits) & ((1 << exp_bits) - 1)
        man = code & ((1 << man_bits) - 1)
        if exp == 0:  # subnormal: man * 2^(1 - bias - man_bits)
            value = man * 2.0 ** (1 - bias - man_bits)
        else:  # normal; fp6 has no Inf/NaN encodings (fn types)
            value = (1.0 + man * 2.0**-man_bits) * 2.0 ** (exp - bias)
        values.append(sign * value)
    return torch.tensor(values, dtype=torch.float32)


def encode_fp6_values(values, dtype: str):
    """Round-to-nearest fp6 codes for a float tensor (saturating).

    Ties resolve toward the even mantissa code, matching IEEE
    round-to-nearest-even on the code lattice.
    """
    torch = _import_torch()
    if dtype not in _FP6_SPECS:
        raise ValueError(f"dtype must be one of {sorted(_FP6_SPECS)}, got {dtype!r}")
    lut = _fp6_code_values(dtype).to(values.device)
    magnitude_codes = torch.arange(32, device=values.device)
    magnitudes = lut[:32]  # codes 0..31 are the non-negative values, ascending
    x = torch.nan_to_num(values.to(torch.float32), nan=0.0)
    sign = x < 0
    mag = x.abs().clamp(max=_FP6_MAX[dtype])
    # Midpoints separate nearest-code regions; on an exact midpoint the two
    # bucketize flavors disagree by one, and we pick the even code (RN-even).
    midpoints = (magnitudes[:-1] + magnitudes[1:]) / 2
    idx_low = torch.bucketize(mag, midpoints, right=False)
    idx_high = torch.bucketize(mag, midpoints, right=True)
    tie = idx_low != idx_high
    idx = torch.where(tie & (idx_low % 2 == 0), idx_low, idx_high)
    codes = magnitude_codes[idx]
    return (codes | (sign.to(torch.int64) << 5)).to(torch.uint8)


def decode_fp6_values(codes, dtype: str):
    torch = _import_torch()
    lut = _fp6_code_values(dtype).to(codes.device)
    return lut[codes.to(torch.long)]


def pack_fp6_codes(codes):
    """Pack 6-bit codes as an LSB-first bitstream: 4 elements -> 3 bytes.

    The bit order matches the 16U6_ALIGN16B payload consumed by the
    ``b6x16_p32`` ldmatrix (element i occupies bits [6i, 6i+6) of the
    stream), so packed groups can be staged into smem byte-for-byte.
    """
    torch = _import_torch()
    if codes.dtype != torch.uint8:
        raise TypeError(f"codes must be torch.uint8, got {codes.dtype}")
    rows, cols = codes.shape
    if cols % 4 != 0:
        raise ValueError(f"fp6 packing requires K multiple of 4, got {tuple(codes.shape)}")
    c = codes.reshape(rows, cols // 4, 4).to(torch.int32)
    b0 = c[:, :, 0] | ((c[:, :, 1] & 0x3) << 6)
    b1 = (c[:, :, 1] >> 2) | ((c[:, :, 2] & 0xF) << 4)
    b2 = (c[:, :, 2] >> 4) | (c[:, :, 3] << 2)
    return torch.stack([b0, b1, b2], dim=2).reshape(rows, cols * 3 // 4).to(torch.uint8)


def unpack_fp6_bytes(packed, cols: int):
    """Inverse of ``pack_fp6_codes`` (for round-trip tests and references)."""
    torch = _import_torch()
    rows = packed.shape[0]
    b = packed.reshape(rows, cols // 4, 3).to(torch.int32)
    c0 = b[:, :, 0] & 0x3F
    c1 = ((b[:, :, 0] >> 6) | (b[:, :, 1] << 2)) & 0x3F
    c2 = ((b[:, :, 1] >> 4) | (b[:, :, 2] << 4)) & 0x3F
    c3 = (b[:, :, 2] >> 2) & 0x3F
    return torch.stack([c0, c1, c2, c3], dim=2).reshape(rows, cols).to(torch.uint8)


def quantize_bf16_to_mxfp6_blockscaled(
    x,
    *,
    dtype: str = "e3m2",
    block_rows: int = 128,
    block_words: int = 2,
    scale_block_k: int = 32,
    return_scale_bytes: bool = False,
):
    """Quantize a BF16 tensor to SM120 MXFP6 blockscaled storage.

    Returns ``(packed_fp6, scale_source)`` - a ``torch.uint8[rows, K*3/4]``
    LSB-first fp6 blob and the packed ``torch.uint32[rows_pad, K/128]``
    K-major scale source - plus the semantic ``[rows, K/32]`` scale bytes
    when ``return_scale_bytes`` is set. NaN/Inf behavior matches the
    MXFP4/MXFP8 quantizers (NaN zeroes its block via scale 0x00; Inf
    saturates the block scale to 0xFE).
    """
    torch = _import_torch()
    _check_block_shape(block_rows, block_words)
    if scale_block_k != _MXFP6_SCALE_BLOCK_K:
        raise ValueError(f"SM120 MXFP6 scale_block_k must be 32, got {scale_block_k}")
    if dtype not in _FP6_MAX:
        raise ValueError(f"dtype must be one of {sorted(_FP6_MAX)}, got {dtype!r}")
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
            f"SM120 MXFP6 quantization requires rows multiple of {block_rows} and K multiple of "
            f"{scale_block_k * scale_cols_per_tile}, got {tuple(x.shape)}"
        )

    x_f32 = x.contiguous().to(torch.float32)
    blocks = x_f32.reshape(rows, cols // scale_block_k, scale_block_k)
    amax = blocks.abs().amax(dim=2)
    scale_bytes = encode_ue8m0_scale_bytes(amax / _FP6_MAX[dtype], rounding="ceil")
    scale_values = decode_ue8m0_scale_bytes(scale_bytes)
    scaled_blocks = torch.where(amax[..., None] > 0, blocks / scale_values[..., None], torch.zeros_like(blocks))
    fp6_codes = encode_fp6_values(scaled_blocks.reshape(rows, cols), dtype)
    packed_fp6 = pack_fp6_codes(fp6_codes)
    packed_scales = pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(scale_bytes, block_rows=block_rows, block_words=block_words)
    if return_scale_bytes:
        return packed_fp6, packed_scales, scale_bytes
    return packed_fp6, packed_scales
