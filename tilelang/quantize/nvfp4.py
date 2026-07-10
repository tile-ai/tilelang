"""NVFP4 quantization and scale layout utilities."""

from __future__ import annotations

import tilelang
import tilelang.language as T


_BLOCKSCALED_CHUNK_ROWS = 128
_BLOCKSCALED_CHUNK_WORDS = 4
_SCALE_BYTES_PER_WORD = 4
_NVFP4_SCALE_BLOCK_K = 16
_SCALE_LAYOUT_BLOCKSCALED_CHUNK_KMAJOR = "blockscaled_chunk_kmajor"
_SCALE_LAYOUT_ID_BLOCKSCALED_CHUNK_KMAJOR = 0
_FP4_E2M1_MAX = 6.0
_UE4M3_MIN_SUBNORMAL = 2.0**-9
_UE4M3_MIN_NORMAL = 2.0**-6
_UE4M3_MAX = 1.75 * (2.0**8)
_FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_TILELANG_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
    tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
}


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch installed
        raise ImportError("NVFP4 scale layout utilities require PyTorch tensors") from exc
    return torch


def _check_block_shape(block_rows: int, block_words: int) -> None:
    if block_rows != _BLOCKSCALED_CHUNK_ROWS or block_words != _BLOCKSCALED_CHUNK_WORDS:
        raise ValueError(
            "SM120 BlockScaledBasicChunk K-major scale packing currently supports "
            f"block_rows={_BLOCKSCALED_CHUNK_ROWS} and block_words={_BLOCKSCALED_CHUNK_WORDS}, "
            f"got block_rows={block_rows}, block_words={block_words}"
        )


def _scale_layout_id(scale_layout: str) -> int:
    if scale_layout == _SCALE_LAYOUT_BLOCKSCALED_CHUNK_KMAJOR:
        return _SCALE_LAYOUT_ID_BLOCKSCALED_CHUNK_KMAJOR
    raise ValueError(f"Unsupported SM120 NVFP4 scale_layout={scale_layout!r}; expected {_SCALE_LAYOUT_BLOCKSCALED_CHUNK_KMAJOR!r}")


def decode_ue4m3_scale_bytes(scale_bytes):
    """Decode unsigned E4M3 scale bytes to float32 values."""

    torch = _import_torch()
    if not isinstance(scale_bytes, torch.Tensor):
        raise TypeError(f"scale_bytes must be a torch.Tensor, got {type(scale_bytes)!r}")
    if scale_bytes.dtype != torch.uint8:
        raise TypeError(f"scale_bytes must have dtype torch.uint8, got {scale_bytes.dtype}")

    u = scale_bytes.to(torch.int32) & 0x7F
    exponent = (u >> 3) & 0x0F
    mantissa = u & 0x07
    normal = (1.0 + mantissa.to(torch.float32) / 8.0) * torch.pow(2.0, exponent.to(torch.float32) - 7.0)
    subnormal = (mantissa.to(torch.float32) / 8.0) * torch.pow(torch.tensor(2.0, device=scale_bytes.device), -6.0)
    value = torch.where(exponent == 0, subnormal, normal)
    return torch.where(u == 0x7F, torch.full_like(value, float("nan")), value)


def encode_ue4m3_scale_bytes(values, *, rounding: str = "nearest", min_positive: bool = False):
    """Encode non-negative float values as unsigned E4M3 scale bytes."""

    torch = _import_torch()
    if rounding not in ("nearest", "ceil"):
        raise ValueError(f"rounding must be 'nearest' or 'ceil', got {rounding!r}")
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"values must be a torch.Tensor, got {type(values)!r}")

    x = torch.nan_to_num(values.to(torch.float32), nan=0.0, posinf=_UE4M3_MAX, neginf=0.0).clamp(min=0.0)
    x = torch.minimum(x, torch.tensor(_UE4M3_MAX, device=x.device, dtype=torch.float32))
    positive = x > 0

    if rounding == "ceil":
        sub_mantissa = torch.ceil(x / _UE4M3_MIN_SUBNORMAL).to(torch.int32)
    else:
        sub_mantissa = torch.floor(x / _UE4M3_MIN_SUBNORMAL + 0.5).to(torch.int32)
    sub_code = sub_mantissa.clamp(0, 7).to(torch.int32)
    sub_overflow_code = torch.full_like(sub_code, 0x08)
    sub_code = torch.where(sub_mantissa > 7, sub_overflow_code, sub_code)

    safe_x = torch.clamp(x, min=_UE4M3_MIN_NORMAL)
    exponent_unbiased = torch.floor(torch.log2(safe_x)).clamp(-6, 8)
    scale = torch.pow(2.0, exponent_unbiased)
    mantissa_real = (safe_x / scale - 1.0) * 8.0
    if rounding == "ceil":
        mantissa = torch.ceil(mantissa_real).to(torch.int32)
    else:
        mantissa = torch.floor(mantissa_real + 0.5).to(torch.int32)
    exponent = (exponent_unbiased.to(torch.int32) + 7).clamp(1, 15)
    carry = mantissa >= 8
    exponent = torch.where(carry, exponent + 1, exponent).clamp(1, 15)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa).clamp(0, 7)
    normal_code = ((exponent << 3) | mantissa).clamp(0, 0x7E).to(torch.int32)

    code = torch.where(x < _UE4M3_MIN_NORMAL, sub_code, normal_code)
    if min_positive:
        code = torch.where(positive & (code == 0), torch.ones_like(code), code)
    return code.to(torch.uint8)


def encode_fp4_e2m1_values(values):
    """Encode float values to raw FP4 E2M1 nibble codes."""

    torch = _import_torch()
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"values must be a torch.Tensor, got {type(values)!r}")

    x = torch.nan_to_num(values.to(torch.float32), nan=0.0, posinf=_FP4_E2M1_MAX, neginf=-_FP4_E2M1_MAX)
    abs_x = x.abs()
    mag = torch.zeros_like(abs_x, dtype=torch.int32)
    mag = torch.where(abs_x >= 0.25, torch.ones_like(mag), mag)
    mag = torch.where(abs_x >= 0.75, torch.full_like(mag, 2), mag)
    mag = torch.where(abs_x >= 1.25, torch.full_like(mag, 3), mag)
    mag = torch.where(abs_x >= 1.75, torch.full_like(mag, 4), mag)
    mag = torch.where(abs_x >= 2.5, torch.full_like(mag, 5), mag)
    mag = torch.where(abs_x >= 3.5, torch.full_like(mag, 6), mag)
    mag = torch.where(abs_x >= 5.0, torch.full_like(mag, 7), mag)
    sign = ((x < 0) & (mag != 0)).to(torch.int32) << 3
    return (mag | sign).to(torch.uint8)


def pack_fp4_e2m1_codes(codes):
    """Pack raw FP4 E2M1 nibble codes into int8 storage accepted by TileLang."""

    torch = _import_torch()
    if not isinstance(codes, torch.Tensor):
        raise TypeError(f"codes must be a torch.Tensor, got {type(codes)!r}")
    if codes.dtype != torch.uint8:
        raise TypeError(f"codes must have dtype torch.uint8, got {codes.dtype}")
    if codes.ndim != 2:
        raise ValueError(f"codes must be a 2D tensor, got shape {tuple(codes.shape)}")
    if codes.shape[1] % 2 != 0:
        raise ValueError(f"FP4 packing requires an even K dimension, got {tuple(codes.shape)}")

    low = codes[:, 0::2] & 0x0F
    high = (codes[:, 1::2] & 0x0F) << 4
    return (low | high).contiguous().view(torch.int8)


def decode_packed_fp4_e2m1(packed, cols: int | None = None):
    """Decode packed FP4 E2M1 int8/uint8 storage to float32 values."""

    torch = _import_torch()
    if not isinstance(packed, torch.Tensor):
        raise TypeError(f"packed must be a torch.Tensor, got {type(packed)!r}")
    if packed.ndim != 2:
        raise ValueError(f"packed must be a 2D tensor, got shape {tuple(packed.shape)}")
    if cols is None:
        cols = packed.shape[1] * 2
    if cols != packed.shape[1] * 2:
        raise ValueError(f"cols must equal packed.shape[1] * 2, got cols={cols}, packed shape={tuple(packed.shape)}")

    u = packed.contiguous().view(torch.uint8)
    lut = torch.tensor(_FP4_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    out = torch.empty((packed.shape[0], cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(u & 0x0F).long()]
    out[:, 1::2] = lut[((u >> 4) & 0x0F).long()]
    return out


def blockscaled_chunk_kmajor_word_offset(row: int, k64_word: int, block_rows: int = 128, block_words: int = 4) -> tuple[int, int]:
    """Return tile-local uint32 coordinates for CUTLASS BlockScaledBasicChunk K-major.

    ``row`` is a tile-local M/N coordinate in ``[0, 128)``. ``k64_word`` is the
    tile-local K64 word in ``[0, 4)``. The returned coordinates index the
    compressed uint32 source tile whose bytes hold the four adjacent K/16 scale
    groups. Flattening these coordinates with a row-major stride of four words
    matches the CUDA helper ``sm120_blockscaled_chunk_kmajor_sf_word``.
    """

    _check_block_shape(block_rows, block_words)
    if row < 0 or row >= block_rows:
        raise ValueError(f"row must be in [0, {block_rows}), got {row}")
    if k64_word < 0 or k64_word >= block_words:
        raise ValueError(f"k64_word must be in [0, {block_words}), got {k64_word}")
    return k64_word * 32 + (row % 32), row // 32


def swizzle_blockscaled_chunk_kmajor_scale_words(words, block_rows: int = 128, block_words: int = 4):
    """Convert semantic row-major scale words to SM120 BlockScaledBasicChunk K-major.

    Parameters
    ----------
    words:
        ``torch.uint32`` tensor with shape ``[rows, K // 64]``. Each word packs
        four already-encoded UE4M3 scale bytes for consecutive K/16 groups.
    block_rows:
        The source-layout tile rows. SM120 dense NVFP4 uses ``128``.
    block_words:
        The source-layout tile K64 words. SM120 dense NVFP4 uses ``4``.

    Returns
    -------
    torch.Tensor
        ``torch.uint32`` tensor physically ordered for the CUTLASS
        ``BlockScaledBasicChunk`` K-major source layout. Rows that are not a
        multiple of ``block_rows`` are zero-padded to the next multiple
        (matching the ceil convention CUTLASS/cuBLAS/CuTeDSL use for the
        blocked SF tensor), so the result has
        ``ceil(rows / block_rows) * block_rows`` rows.
    """

    torch = _import_torch()
    _check_block_shape(block_rows, block_words)
    if not isinstance(words, torch.Tensor):
        raise TypeError(f"words must be a torch.Tensor, got {type(words)!r}")
    if words.dtype != torch.uint32:
        raise TypeError(f"words must have dtype torch.uint32, got {words.dtype}")
    if words.ndim != 2:
        raise ValueError(f"words must be a 2D tensor, got shape {tuple(words.shape)}")

    rows, cols = words.shape
    if cols % block_words != 0:
        raise ValueError(f"blockscaled_chunk_kmajor scale storage requires K/64 words multiple of {block_words}, got {tuple(words.shape)}")
    if rows % block_rows != 0:
        padded_rows = (rows + block_rows - 1) // block_rows * block_rows
        padded = torch.zeros((padded_rows, cols), dtype=words.dtype, device=words.device)
        padded[:rows] = words
        words = padded
        rows = padded_rows

    row_blocks = rows // block_rows
    src = words.contiguous().reshape(row_blocks, 4, 32, cols)
    return src.permute(0, 3, 2, 1).contiguous().reshape(rows, cols)


def unswizzle_blockscaled_chunk_kmajor_scale_words(words, block_rows: int = 128, block_words: int = 4):
    """Convert BlockScaledBasicChunk K-major source words back to semantic row-major words.

    This is intended for reference checks and debugging. GEMM kernels should
    consume the source-layout tensor produced by the quantizer directly.
    """

    torch = _import_torch()
    _check_block_shape(block_rows, block_words)
    if not isinstance(words, torch.Tensor):
        raise TypeError(f"words must be a torch.Tensor, got {type(words)!r}")
    if words.dtype != torch.uint32:
        raise TypeError(f"words must have dtype torch.uint32, got {words.dtype}")
    if words.ndim != 2:
        raise ValueError(f"words must be a 2D tensor, got shape {tuple(words.shape)}")

    rows, cols = words.shape
    if rows % block_rows != 0 or cols % block_words != 0:
        raise ValueError(
            f"blockscaled_chunk_kmajor scale storage requires rows multiple of {block_rows} "
            f"and K/64 words multiple of {block_words}, got {tuple(words.shape)}"
        )

    row_blocks = rows // block_rows
    src = words.contiguous().reshape(row_blocks, cols, 32, 4)
    return src.permute(0, 3, 2, 1).contiguous().reshape(rows, cols)


def _pack_scale_bytes_to_words(scale_bytes):
    torch = _import_torch()
    scale_i64 = scale_bytes.to(torch.int64).reshape(
        scale_bytes.shape[0], scale_bytes.shape[1] // _SCALE_BYTES_PER_WORD, _SCALE_BYTES_PER_WORD
    )
    words = scale_i64[:, :, 0]
    words = words | (scale_i64[:, :, 1] << 8)
    words = words | (scale_i64[:, :, 2] << 16)
    words = words | (scale_i64[:, :, 3] << 24)
    return words.to(torch.uint32)


def pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes, block_rows: int = 128, block_words: int = 4):
    """Pack UE4M3 scale bytes into SM120 BlockScaledBasicChunk K-major words.

    This is a layout packer, not a numeric quantizer. ``scale_bytes`` must
    already contain UE4M3-encoded scale bytes with semantic row-major shape
    ``[rows, K / 16]``. The result is the compressed source tensor expected by
    the SM120 NVFP4 blockscaled GEMM path:
    ``torch.uint32[ceil(rows / 128) * 128, K // 64]`` — rows are zero-padded
    to full 128-row tiles, matching the CUTLASS/cuBLAS/CuTeDSL ceil
    convention for the blocked SF tensor.

    The same function applies to SFA and SFB. For SFA, ``rows`` is logical M; for
    SFB, ``rows`` is logical N.

    The packed words are byte-identical to the CUTLASS/CuTeDSL canonical
    blocked SF layout (``blockscaled_utils.tile_atom_to_shape_SF`` with
    ``sf_vec_size=16``), so the same buffer can feed a CuTeDSL NVFP4
    blockscaled GEMM: view it with ``.view(torch.uint8)`` for the byte tensor,
    or reinterpret an existing CuTeDSL SF byte buffer with
    ``.view(torch.uint32).reshape(rows, k // 64)`` — both are zero-copy.
    """

    torch = _import_torch()
    _check_block_shape(block_rows, block_words)
    if not isinstance(scale_bytes, torch.Tensor):
        raise TypeError(f"scale_bytes must be a torch.Tensor, got {type(scale_bytes)!r}")
    if scale_bytes.dtype != torch.uint8:
        raise TypeError(f"scale_bytes must have dtype torch.uint8, got {scale_bytes.dtype}")
    if scale_bytes.ndim != 2:
        raise ValueError(f"scale_bytes must be a 2D tensor, got shape {tuple(scale_bytes.shape)}")

    rows, scale_cols = scale_bytes.shape
    scale_cols_per_tile = block_words * _SCALE_BYTES_PER_WORD
    if scale_cols % scale_cols_per_tile != 0:
        raise ValueError(
            f"blockscaled_chunk_kmajor scale bytes require K/16 columns multiple of {scale_cols_per_tile}, got {tuple(scale_bytes.shape)}"
        )

    words = _pack_scale_bytes_to_words(scale_bytes)
    return swizzle_blockscaled_chunk_kmajor_scale_words(words, block_rows, block_words)


def pack_nvfp4_scale_bytes(
    scale_bytes,
    *,
    scale_layout: str = _SCALE_LAYOUT_BLOCKSCALED_CHUNK_KMAJOR,
    block_rows: int = 128,
    block_words: int = 4,
):
    """Pack semantic UE4M3 scale bytes for a supported SM120 NVFP4 scale layout."""

    _scale_layout_id(scale_layout)
    return pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes, block_rows=block_rows, block_words=block_words)


def _tl_encode_ue4m3_scale_byte_ceil(x):
    x = T.alloc_var(T.float32, init=T.max(T.min(x, _UE4M3_MAX), 0.0))
    sub_mantissa = T.cast(T.ceil(x[0] * 512.0), T.int32)
    sub_code = T.min(T.max(sub_mantissa, 0), 7)
    sub_code = T.if_then_else(sub_mantissa > 7, 0x08, sub_code)

    bits = T.reinterpret(x[0], T.uint32)
    exponent_unbiased = T.cast((bits >> 23) & 0xFF, T.int32) - 127
    exponent_unbiased = T.min(T.max(exponent_unbiased, -6), 8)
    mantissa_bits = T.cast(bits & 0x7FFFFF, T.int32)
    mantissa = (mantissa_bits >> 20) + T.if_then_else((mantissa_bits & 0xFFFFF) != 0, 1, 0)
    biased_exponent = exponent_unbiased + 7
    carry = mantissa >= 8
    biased_exponent = T.if_then_else(carry, biased_exponent + 1, biased_exponent)
    biased_exponent = T.min(T.max(biased_exponent, 1), 15)
    mantissa = T.if_then_else(carry, 0, mantissa)
    mantissa = T.min(T.max(mantissa, 0), 7)
    normal_code = (biased_exponent << 3) | mantissa

    code = T.if_then_else(x[0] < _UE4M3_MIN_NORMAL, sub_code, normal_code)
    return T.if_then_else(x[0] <= 0.0, 0, code)


def _tl_decode_ue4m3_scale_byte(code):
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07
    mantissa_f32 = T.cast(mantissa, T.float32)
    normal_bits = T.cast(((exponent + 120) << 23) | (mantissa << 20), T.uint32)
    normal = T.reinterpret(normal_bits, T.float32)
    subnormal = mantissa_f32 * 0.125 * _UE4M3_MIN_NORMAL
    return T.if_then_else(exponent == 0, subnormal, normal)


def _tl_ue4m3_scale_byte_inverse(code):
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07

    normal_factor = T.if_then_else(
        mantissa == 0,
        1.0,
        T.if_then_else(
            mantissa == 1,
            0.8888888955116272,
            T.if_then_else(
                mantissa == 2,
                0.800000011920929,
                T.if_then_else(
                    mantissa == 3,
                    0.7272727489471436,
                    T.if_then_else(
                        mantissa == 4,
                        0.6666666865348816,
                        T.if_then_else(
                            mantissa == 5,
                            0.6153846383094788,
                            T.if_then_else(mantissa == 6, 0.5714285969734192, 0.5333333611488342),
                        ),
                    ),
                ),
            ),
        ),
    )
    normal_bits = T.cast((134 - exponent) << 23, T.uint32)
    normal_inv = T.reinterpret(normal_bits, T.float32) * normal_factor

    subnormal_inv = T.if_then_else(
        mantissa == 0,
        0.0,
        T.if_then_else(
            mantissa == 1,
            512.0,
            T.if_then_else(
                mantissa == 2,
                256.0,
                T.if_then_else(
                    mantissa == 3,
                    170.6666717529297,
                    T.if_then_else(
                        mantissa == 4,
                        128.0,
                        T.if_then_else(
                            mantissa == 5,
                            102.4000015258789,
                            T.if_then_else(mantissa == 6, 85.33333587646484, 73.14286041259766),
                        ),
                    ),
                ),
            ),
        ),
    )
    return T.if_then_else(exponent == 0, subnormal_inv, normal_inv)


def _tl_encode_fp4_e2m1_code(x):
    return T.reinterpret(T.cast(x, T.float4_e2m1fn), T.uint8)


def _tl_scale_source_coords(logical_row, k64_word, _scale_layout_id: int, scale_cols):
    row_in_chunk = logical_row % _BLOCKSCALED_CHUNK_ROWS
    flat_word = (
        (logical_row // _BLOCKSCALED_CHUNK_ROWS) * _BLOCKSCALED_CHUNK_ROWS * scale_cols
        + k64_word * _BLOCKSCALED_CHUNK_ROWS
        + (row_in_chunk % 32) * 4
        + (row_in_chunk // 32)
    )
    physical_row = flat_word // scale_cols
    physical_word = flat_word % scale_cols
    return physical_row, physical_word


@tilelang.jit(pass_configs=_TILELANG_PASS_CONFIGS)
def _tilelang_nvfp4_blockscaled_quantize_kernel_tiled(
    x,
    rows_per_cta: int = 64,
    tma_store: bool = True,
    num_stages: int = 2,
    num_threads: int = 512,
    scale_layout_id: int = _SCALE_LAYOUT_ID_BLOCKSCALED_CHUNK_KMAJOR,
):
    """TileLang BF16 -> packed NVFP4 over a ``rows_per_cta x 256`` CTA tile.

    One kernel covers the tuned tile variants: TMA input staging, warp-strided
    row ownership (``rows_per_cta // num_warps`` rows per warp), and either a
    TMA store through shared memory (``tma_store=True``) or direct global
    stores. ``rows_per_cta`` in {16, 32, 64} reproduces the previous
    per-variant kernels exactly.
    """

    M = T.dynamic("M")
    K = T.const("K")
    rows_per_warp = rows_per_cta // (num_threads // 32)
    in_dtype = T.bfloat16
    compute_dtype = T.float32

    x: T.Tensor[(M, K), in_dtype]

    quant = T.empty((M, K // 2), T.int8)
    scale_source = T.empty((M, K // 64), T.uint32)

    with T.Kernel(M // rows_per_cta, K // 256, threads=num_threads) as (pid_m, pid_k256):
        lane = T.get_lane_idx()
        warp = T.get_warp_idx_sync()
        x_shared = T.alloc_shared((rows_per_cta, 256), in_dtype)
        if tma_store:
            quant_shared = T.alloc_shared((rows_per_cta, 128), T.int8)
        scale_codes = T.alloc_shared((rows_per_cta, 16), T.int32)

        T.copy(x[pid_m * rows_per_cta, pid_k256 * 256], x_shared, prefer_instruction="tma")
        T.sync_threads()

        for row_iter in T.serial(rows_per_warp):
            row = warp * rows_per_warp + row_iter
            for k64_inner in T.serial(_BLOCKSCALED_CHUNK_WORDS):
                elem_col = k64_inner * 64 + lane * 2
                byte_col = k64_inner * 32 + lane
                group = k64_inner * 4 + (lane // 8)
                group_lane = lane % 8
                value0 = T.cast(x_shared[row, elem_col], compute_dtype)
                value1 = T.cast(x_shared[row, elem_col + 1], compute_dtype)
                amax = T.alloc_var(compute_dtype, init=T.max(T.abs(value0), T.abs(value1)))
                amax = T.max(amax, T.shfl_xor(amax, 4, width=8))
                amax = T.max(amax, T.shfl_xor(amax, 2, width=8))
                amax = T.max(amax, T.shfl_xor(amax, 1, width=8))

                scale_code = T.alloc_var(T.int32, init=0)
                scale_inv = T.alloc_var(compute_dtype, init=0.0)
                if group_lane == 0:
                    scale_code = _tl_encode_ue4m3_scale_byte_ceil(amax / _FP4_E2M1_MAX)
                    scale_inv = _tl_ue4m3_scale_byte_inverse(scale_code)
                    scale_codes[row, group] = scale_code
                scale_inv = T.shfl_sync(scale_inv, 0, width=8)
                fp4_code0 = _tl_encode_fp4_e2m1_code(value0 * scale_inv)
                fp4_code1 = _tl_encode_fp4_e2m1_code(value1 * scale_inv)
                packed_byte = fp4_code0 | (fp4_code1 << 4)
                if tma_store:
                    quant_shared[row, byte_col] = T.cast(packed_byte, T.int8)
                else:
                    quant[pid_m * rows_per_cta + row, pid_k256 * 128 + byte_col] = T.cast(packed_byte, T.int8)

        T.sync_threads()
        for row_iter in T.serial(rows_per_warp):
            row = warp * rows_per_warp + row_iter
            if lane < _BLOCKSCALED_CHUNK_WORDS:
                k64_inner = lane
                scale_word = (
                    T.cast(scale_codes[row, k64_inner * 4], T.uint32)
                    | (T.cast(scale_codes[row, k64_inner * 4 + 1], T.uint32) << 8)
                    | (T.cast(scale_codes[row, k64_inner * 4 + 2], T.uint32) << 16)
                    | (T.cast(scale_codes[row, k64_inner * 4 + 3], T.uint32) << 24)
                )
                logical_row = pid_m * rows_per_cta + row
                k64_word = pid_k256 * _BLOCKSCALED_CHUNK_WORDS + k64_inner
                physical_row, physical_word = _tl_scale_source_coords(logical_row, k64_word, scale_layout_id, K // 64)
                scale_source[physical_row, physical_word] = scale_word

        if tma_store:
            T.sync_threads()
            T.tma_copy(quant_shared, quant[pid_m * rows_per_cta, pid_k256 * 128])
            T.tma_store_wait()

    return quant, scale_source


def tilelang_quantize_bf16_to_nvfp4_blockscaled(
    x,
    *,
    rows_per_cta: int = 32,
    num_stages: int = 4,
    num_threads: int = 512,
    scale_layout: str = _SCALE_LAYOUT_BLOCKSCALED_CHUNK_KMAJOR,
):
    """Quantize a CUDA BF16 activation tensor with a TileLang kernel.

    Parameters
    ----------
    x:
        CUDA ``torch.bfloat16`` tensor with shape ``[rows, K]``. ``rows`` must
        be a multiple of ``128`` and ``K`` a multiple of ``256``.
    rows_per_cta:
        CTA tile rows: ``16``, ``32`` or ``64`` (each CTA covers a
        ``rows_per_cta x 256`` tile).
    num_stages:
        Pipeline stage count for the TMA input staging.
    num_threads:
        CTA thread count; ``512`` (default) or ``1024`` for
        ``rows_per_cta`` in ``(32, 64)``.
    scale_layout:
        Scale-source memory layout. ``"blockscaled_chunk_kmajor"`` is the
        CUTLASS ``BlockScaledBasicChunk`` K-major source contract used by the
        TileLang SM120 NVFP4 GEMM path.

    Returns
    -------
    tuple
        ``(packed_fp4, scale_source)``. ``packed_fp4`` is ``torch.int8[rows, K/2]``.
        ``scale_source`` is ``torch.uint32[rows, K/64]`` in CUTLASS
        ``BlockScaledBasicChunk`` K-major source order.
        The same contract applies to SFA and SFB; ``rows`` is interpreted as M
        for A and N for B.
    """

    torch = _import_torch()
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(f"x must be a 2D tensor, got shape {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("TileLang NVFP4 quantization requires a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"TileLang NVFP4 quantization requires torch.bfloat16 input, got {x.dtype}")
    scale_layout_id = _scale_layout_id(scale_layout)
    if rows_per_cta not in (16, 32, 64):
        raise ValueError(f"rows_per_cta must be 16, 32, or 64, got {rows_per_cta}")
    if num_threads % 32 != 0:
        raise ValueError(f"num_threads must be a multiple of warp size 32, got {num_threads}")
    if rows_per_cta == 16 and num_threads != 512:
        raise ValueError("rows_per_cta=16 uses the one-warp-per-row schedule and requires num_threads=512")
    if rows_per_cta in (32, 64) and num_threads not in (512, 1024):
        raise ValueError("rows_per_cta=32/64 currently supports num_threads=512 or 1024")

    rows, cols = x.shape
    k_tile = _NVFP4_SCALE_BLOCK_K * _SCALE_BYTES_PER_WORD * _BLOCKSCALED_CHUNK_WORDS
    if rows % _BLOCKSCALED_CHUNK_ROWS != 0 or cols % k_tile != 0:
        raise ValueError(
            f"SM120 NVFP4 quantization requires rows multiple of {_BLOCKSCALED_CHUNK_ROWS} and K multiple of {k_tile}, got {tuple(x.shape)}"
        )

    quant, scale_source = _tilelang_nvfp4_blockscaled_quantize_kernel_tiled(
        x.contiguous(),
        rows_per_cta=rows_per_cta,
        tma_store=(rows_per_cta != 32),
        num_stages=num_stages,
        num_threads=num_threads,
        scale_layout_id=scale_layout_id,
    )
    return quant.contiguous().view(torch.int8), scale_source


def tilelang_quantize_nvfp4_blockscaled(*args, **kwargs):
    """Alias for ``tilelang_quantize_bf16_to_nvfp4_blockscaled``."""

    return tilelang_quantize_bf16_to_nvfp4_blockscaled(*args, **kwargs)


def quantize_bf16_to_nvfp4_blockscaled(
    x,
    *,
    block_rows: int = 128,
    block_words: int = 4,
    scale_block_k: int = 16,
    scale_layout: str = _SCALE_LAYOUT_BLOCKSCALED_CHUNK_KMAJOR,
    return_scale_bytes: bool = False,
):
    """Quantize a BF16 activation tensor to SM120 NVFP4 blockscaled storage.

    Parameters
    ----------
    x:
        2D BF16 activation tensor with shape ``[rows, K]``. FP16/FP32 tensors are
        accepted for tests and converted internally, but the intended runtime
        contract is BF16 activation input. The SM120 source contract requires
        ``rows`` to be a multiple of ``128`` and ``K`` to be a multiple of
        ``256`` for the promoted ``blockscaled_chunk_kmajor`` scale layout.
    scale_block_k:
        Number of consecutive K elements sharing one UE4M3 scale byte. SM120
        dense NVFP4 uses ``16``.
    return_scale_bytes:
        When true, also return semantic row-major UE4M3 scale bytes with shape
        ``[rows, K / 16]`` for reference checks.
    scale_layout:
        Scale-source memory layout for the returned ``scale_source``.

    Returns
    -------
    tuple
        ``(packed_fp4, scale_source)`` or
        ``(packed_fp4, scale_source, scale_bytes)``. ``packed_fp4`` is
        ``torch.int8[rows, K / 2]`` with two FP4 E2M1 values per byte.
        ``scale_source`` is ``torch.uint32[rows, K // 64]`` in ``scale_layout`` order.
    """

    torch = _import_torch()
    _check_block_shape(block_rows, block_words)
    _scale_layout_id(scale_layout)
    if scale_block_k != _NVFP4_SCALE_BLOCK_K:
        raise ValueError(f"SM120 NVFP4 scale_block_k must be 16, got {scale_block_k}")
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
            f"SM120 NVFP4 quantization requires rows multiple of {block_rows} and K multiple of "
            f"{scale_block_k * scale_cols_per_tile}, got {tuple(x.shape)}"
        )

    x_f32 = x.contiguous().to(torch.float32)
    blocks = x_f32.reshape(rows, cols // scale_block_k, scale_block_k)
    amax = blocks.abs().amax(dim=2)
    target_scale = amax / _FP4_E2M1_MAX
    scale_bytes = encode_ue4m3_scale_bytes(target_scale, rounding="ceil", min_positive=True)
    scale_bytes = torch.where(amax == 0, torch.zeros_like(scale_bytes), scale_bytes)
    scale_values = decode_ue4m3_scale_bytes(scale_bytes)
    scaled_blocks = torch.where(scale_values[..., None] > 0, blocks / scale_values[..., None], torch.zeros_like(blocks))
    fp4_codes = encode_fp4_e2m1_values(scaled_blocks.reshape(rows, cols))
    packed_fp4 = pack_fp4_e2m1_codes(fp4_codes)
    packed_scales = pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes, block_rows=block_rows, block_words=block_words)
    if return_scale_bytes:
        return packed_fp4, packed_scales, scale_bytes
    return packed_fp4, packed_scales


def quantize_nvfp4_blockscaled(*args, **kwargs):
    """Backward-compatible alias for ``quantize_bf16_to_nvfp4_blockscaled``."""

    return quantize_bf16_to_nvfp4_blockscaled(*args, **kwargs)
