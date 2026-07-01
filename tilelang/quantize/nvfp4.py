"""NVFP4 quantization and scale layout utilities."""

from __future__ import annotations

import tilelang
import tilelang.language as T


_BLOCKSCALED_CHUNK_ROWS = 128
_BLOCKSCALED_CHUNK_WORDS = 4
_SCALE_BYTES_PER_WORD = 4
_NVFP4_SCALE_BLOCK_K = 16
_FP4_E2M1_MAX = 6.0
_UE4M3_MIN_SUBNORMAL = 2.0**-9
_UE4M3_MIN_NORMAL = 2.0**-6
_UE4M3_MAX = 1.875 * (2.0**8)
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


def decode_ue4m3_scale_bytes(scale_bytes):
    """Decode unsigned E4M3 scale bytes to float32 values."""

    torch = _import_torch()
    if not isinstance(scale_bytes, torch.Tensor):
        raise TypeError(f"scale_bytes must be a torch.Tensor, got {type(scale_bytes)!r}")
    if scale_bytes.dtype != torch.uint8:
        raise TypeError(f"scale_bytes must have dtype torch.uint8, got {scale_bytes.dtype}")

    u = scale_bytes.to(torch.int32)
    exponent = (u >> 3) & 0x0F
    mantissa = u & 0x07
    normal = (1.0 + mantissa.to(torch.float32) / 8.0) * torch.pow(2.0, exponent.to(torch.float32) - 7.0)
    subnormal = (mantissa.to(torch.float32) / 8.0) * torch.pow(torch.tensor(2.0, device=scale_bytes.device), -6.0)
    return torch.where(exponent == 0, subnormal, normal)


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
    normal_code = ((exponent << 3) | mantissa).clamp(0, 0x7F).to(torch.int32)

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


def _sm120_sfa_row_in_lane(lane: int) -> int:
    return 8 * (lane & 1) + (lane >> 2)


def _sm120_sfb_col_in_lane(lane: int) -> int:
    return lane >> 2


def _sm120_sfa_selector_source_lane(lane: int, scale_a_thread_id: int) -> int:
    """Return the source lane selected by OMMA.SF's A-side thread selector."""

    return (lane & ~3) | ((scale_a_thread_id & 1) << 1) | (lane & 1)


def _sm120_sfb_selector_source_lane(lane: int, scale_b_thread_id: int) -> int:
    """Return the source lane selected by OMMA.SF's B-side thread selector."""

    return (lane & ~3) | (scale_b_thread_id & 3)


def _sm120_compact_sfa_owner_row(lane: int, warp_m: int, reg_group: int) -> int:
    qlane = lane & 3
    return warp_m * 64 + reg_group * 32 + (qlane >> 1) * 16 + _sm120_sfa_row_in_lane(lane)


def _sm120_compact_sfb_owner_row(lane: int, warp_n: int, reg_group: int) -> int:
    qlane = lane & 3
    return warp_n * 64 + reg_group * 32 + qlane * 8 + _sm120_sfb_col_in_lane(lane)


def _sm120_compact_sfa_effective_row(lane: int, warp_m: int, mma_i: int) -> int:
    source_lane = _sm120_sfa_selector_source_lane(lane, mma_i & 1)
    return _sm120_compact_sfa_owner_row(source_lane, warp_m, mma_i >> 1)


def _sm120_compact_sfb_effective_row(lane: int, warp_n: int, mma_j: int, half: int) -> int:
    source_lane = _sm120_sfb_selector_source_lane(lane, (mma_j & 1) * 2 + half)
    return _sm120_compact_sfb_owner_row(source_lane, warp_n, mma_j >> 1)


def _sm120_compact_scale_issue_contract(lane: int, warp_m: int, warp_n: int) -> list[dict[str, int | str]]:
    """Return the compact-selector OMMA.SF scale contract for one lane.

    This is a pure specification helper for tests and future lowering work. It
    models which lane supplies the SFA/SFB scale register selected by each
    ``mma.sync.m16n8k64.kind::mxf4nvf4.block_scale`` issue.
    """

    issues: list[dict[str, int | str]] = []
    for mma_i in range(4):
        sa_reg_group = mma_i >> 1
        sa_tid = mma_i & 1
        sa_source_lane = _sm120_sfa_selector_source_lane(lane, sa_tid)
        for mma_j in range(4):
            sb_reg_group = mma_j >> 1
            for half in range(2):
                sb_tid = (mma_j & 1) * 2 + half
                sb_source_lane = _sm120_sfb_selector_source_lane(lane, sb_tid)
                issues.append(
                    {
                        "mma_i": mma_i,
                        "mma_j": mma_j,
                        "half": half,
                        "sa_reg": f"sa{sa_reg_group}",
                        "sa_tid": sa_tid,
                        "sa_source_lane": sa_source_lane,
                        "sfa_row": _sm120_compact_sfa_owner_row(sa_source_lane, warp_m, sa_reg_group),
                        "sb_reg": f"sb{sb_reg_group}",
                        "sb_tid": sb_tid,
                        "sb_source_lane": sb_source_lane,
                        "sfb_row": _sm120_compact_sfb_owner_row(sb_source_lane, warp_n, sb_reg_group),
                    }
                )
    return issues


def _sm120_compact_scale_copy_view_contract(lane: int, warp_m: int, warp_n: int, k64_word: int) -> list[dict[str, object]]:
    """Return per-issue scale rows plus BlockScaledBasicChunk word addresses."""

    issues: list[dict[str, object]] = []
    for issue in _sm120_compact_scale_issue_contract(lane, warp_m, warp_n):
        sfa_word_coord = blockscaled_chunk_kmajor_word_offset(int(issue["sfa_row"]), k64_word)
        sfb_word_coord = blockscaled_chunk_kmajor_word_offset(int(issue["sfb_row"]), k64_word)
        issues.append(
            {
                **issue,
                "k64_word": k64_word,
                "sfa_word_coord": sfa_word_coord,
                "sfa_word_offset": sfa_word_coord[0] * _BLOCKSCALED_CHUNK_WORDS + sfa_word_coord[1],
                "sfb_word_coord": sfb_word_coord,
                "sfb_word_offset": sfb_word_coord[0] * _BLOCKSCALED_CHUNK_WORDS + sfb_word_coord[1],
            }
        )
    return issues


def _sm120_compact_scale_producer_contract(warp_m: int, warp_n: int) -> list[dict[str, object]]:
    """Return the source-lane register assignment required by current OMMA.SF.

    This is the inverse view of ``_sm120_compact_scale_issue_contract``. It says
    which producer lane supplies the scale register consumed by each issue. A
    future compact scale-copy lowering must preserve these semantic rows while
    changing how the producer lanes load/package the registers.
    """

    rows: list[dict[str, object]] = []
    for mma_i in range(4):
        sa_tid = mma_i & 1
        sa_reg_group = mma_i >> 1
        for producer_lane in range(32):
            consumers = [lane for lane in range(32) if _sm120_sfa_selector_source_lane(lane, sa_tid) == producer_lane]
            if not consumers:
                continue
            rows.append(
                {
                    "kind": "SFA",
                    "mma_i": mma_i,
                    "sa_tid": sa_tid,
                    "sa_reg": f"sa{sa_reg_group}",
                    "producer_lane": producer_lane,
                    "consumers": tuple(consumers),
                    "row": _sm120_compact_sfa_owner_row(producer_lane, warp_m, sa_reg_group),
                }
            )

    for mma_j in range(4):
        sb_reg_group = mma_j >> 1
        for half in range(2):
            sb_tid = (mma_j & 1) * 2 + half
            for producer_lane in range(32):
                consumers = [lane for lane in range(32) if _sm120_sfb_selector_source_lane(lane, sb_tid) == producer_lane]
                if not consumers:
                    continue
                rows.append(
                    {
                        "kind": "SFB",
                        "mma_j": mma_j,
                        "half": half,
                        "sb_tid": sb_tid,
                        "sb_reg": f"sb{sb_reg_group}",
                        "producer_lane": producer_lane,
                        "consumers": tuple(consumers),
                        "row": _sm120_compact_sfb_owner_row(producer_lane, warp_n, sb_reg_group),
                    }
                )
    return rows


def _sm120_compact_package_consumer_byte_offsets(lane: int, warp_m: int, warp_n: int, k64_word: int) -> dict[str, tuple[int, ...]]:
    """Return the current package's per-consumer byte offsets for one K64 word."""

    issues = _sm120_compact_scale_copy_view_contract(lane, warp_m, warp_n, k64_word)
    sfa_offsets = tuple(int(issue["sfa_word_offset"]) * 4 for issue in issues[0::8])
    sfb_offsets = tuple(int(issue["sfb_word_offset"]) * 4 for issue in issues[:8])
    return {"SFA": sfa_offsets, "SFB": sfb_offsets}


def _sm120_cutlass_issue_site_contract(site: int) -> dict[str, int]:
    """Return CUTLASS/CuTe SM120 fulltile issue-site coordinates.

    The traversal covers four K64 atoms. Each K64 atom issues eight N8 atoms,
    and each N8 atom visits four M atoms. Odd N8 atoms reverse the M order.
    """

    if site < 0 or site >= 128:
        raise ValueError(f"site must be in [0, 128), got {site}")
    k_block = site // 32
    site_in_k = site % 32
    n8_atom = site_in_k // 4
    m_serp = site_in_k % 4
    m_atom = m_serp if n8_atom % 2 == 0 else 3 - m_serp
    n16_col = n8_atom // 2
    n8_half = n8_atom % 2
    return {
        "site": site,
        "k_block": k_block,
        "m_atom": m_atom,
        "n8_atom": n8_atom,
        "n16_col": n16_col,
        "n8_half": n8_half,
        "A_slot0": 128 * m_atom + 32 * k_block,
        "B_slot0": 64 * n8_half + 128 * n16_col + 16 * k_block,
        "SFA_slot0": 4 * m_atom + 16 * k_block,
        "SFB_slot0": 4 * n16_col + 16 * n8_half + 32 * k_block,
        "C_slot0": 4 * m_atom + 32 * n16_col + 16 * n8_half,
    }


def _sm120_tilelang_current_issue_site_contract(site: int) -> dict[str, int]:
    """Return the current package helper's issue coordinates for comparison."""

    if site < 0 or site >= 128:
        raise ValueError(f"site must be in [0, 128), got {site}")
    k_block = site // 32
    site_in_k = site % 32
    mma_i = site_in_k // 8
    n_site = site_in_k % 8
    mma_j = n_site // 2
    half = n_site & 1
    return {
        "site": site,
        "k_block": k_block,
        "mma_i": mma_i,
        "mma_j": mma_j,
        "half": half,
        "sa_reg_group": mma_i >> 1,
        "sa_tid": mma_i & 1,
        "sb_reg_group": mma_j >> 1,
        "sb_tid": (mma_j & 1) * 2 + half,
    }


def swizzle_blockscaled_chunk_kmajor_scale_words(words, block_rows: int = 128, block_words: int = 4):
    """Convert semantic row-major scale words to SM120 BlockScaledBasicChunk K-major.

    Parameters
    ----------
    words:
        ``torch.uint32`` tensor with shape ``[rows, K / 64]``. Each word packs
        four already-encoded UE4M3 scale bytes for consecutive K/16 groups.
    block_rows:
        The source-layout tile rows. SM120 dense NVFP4 uses ``128``.
    block_words:
        The source-layout tile K64 words. SM120 dense NVFP4 uses ``4``.

    Returns
    -------
    torch.Tensor
        ``torch.uint32`` tensor with the same shape, physically ordered for the
        CUTLASS ``BlockScaledBasicChunk`` K-major source layout.
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
    k64_groups = cols // block_words
    src = words.contiguous().reshape(row_blocks, 4, 32, k64_groups, block_words)
    return src.permute(0, 4, 2, 3, 1).contiguous().reshape(rows, cols)


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
    k64_groups = cols // block_words
    src = words.contiguous().reshape(row_blocks, block_words, 32, k64_groups, 4)
    return src.permute(0, 4, 2, 3, 1).contiguous().reshape(rows, cols)


def pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes, block_rows: int = 128, block_words: int = 4):
    """Pack UE4M3 scale bytes into SM120 BlockScaledBasicChunk K-major words.

    This is a layout packer, not a numeric quantizer. ``scale_bytes`` must
    already contain UE4M3-encoded scale bytes with semantic row-major shape
    ``[rows, K / 16]``. The result is the compressed source tensor expected by
    the SM120 NVFP4 blockscaled GEMM path: ``torch.uint32[rows, K / 64]``.

    The same function applies to SFA and SFB. For SFA, ``rows`` is logical M; for
    SFB, ``rows`` is logical N.
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
    if rows % block_rows != 0 or scale_cols % scale_cols_per_tile != 0:
        raise ValueError(
            f"blockscaled_chunk_kmajor scale bytes require rows multiple of {block_rows} "
            f"and K/16 columns multiple of {scale_cols_per_tile}, got {tuple(scale_bytes.shape)}"
        )

    scale_i64 = scale_bytes.to(torch.int64).reshape(rows, scale_cols // _SCALE_BYTES_PER_WORD, _SCALE_BYTES_PER_WORD)
    words = scale_i64[:, :, 0]
    words = words | (scale_i64[:, :, 1] << 8)
    words = words | (scale_i64[:, :, 2] << 16)
    words = words | (scale_i64[:, :, 3] << 24)
    return swizzle_blockscaled_chunk_kmajor_scale_words(words.to(torch.uint32), block_rows, block_words)


def _tl_encode_ue4m3_scale_byte_ceil(x):
    x = T.max(T.min(x, _UE4M3_MAX), 0.0)
    sub_mantissa = T.cast(T.ceil(x * 512.0), T.int32)
    sub_code = T.min(T.max(sub_mantissa, 0), 7)
    sub_code = T.if_then_else(sub_mantissa > 7, 0x08, sub_code)

    safe_x = T.max(x, _UE4M3_MIN_NORMAL)
    exponent_unbiased = T.cast(T.floor(T.log2(safe_x)), T.int32)
    exponent_unbiased = T.min(T.max(exponent_unbiased, -6), 8)
    scale = T.exp2(T.cast(exponent_unbiased, T.float32))
    mantissa = T.cast(T.ceil((safe_x / scale - 1.0) * 8.0), T.int32)
    biased_exponent = exponent_unbiased + 7
    carry = mantissa >= 8
    biased_exponent = T.if_then_else(carry, biased_exponent + 1, biased_exponent)
    biased_exponent = T.min(T.max(biased_exponent, 1), 15)
    mantissa = T.if_then_else(carry, 0, mantissa)
    mantissa = T.min(T.max(mantissa, 0), 7)
    normal_code = (biased_exponent << 3) | mantissa

    code = T.if_then_else(x < _UE4M3_MIN_NORMAL, sub_code, normal_code)
    return T.if_then_else(x <= 0.0, 0, code)


def _tl_decode_ue4m3_scale_byte(code):
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07
    mantissa_f32 = T.cast(mantissa, T.float32)
    normal = (1.0 + mantissa_f32 * 0.125) * T.exp2(T.cast(exponent - 7, T.float32))
    subnormal = mantissa_f32 * 0.125 * _UE4M3_MIN_NORMAL
    return T.if_then_else(exponent == 0, subnormal, normal)


@tilelang.jit(pass_configs=_TILELANG_PASS_CONFIGS)
def _tilelang_nvfp4_blockscaled_quantize_kernel(x, rows_per_cta: int = 16):
    """TileLang BF16 -> packed NVFP4 + BlockScaledBasicChunk K-major scales."""

    M = T.dynamic("M")
    K = T.const("K")
    in_dtype = T.bfloat16
    fp4_dtype = T.float4_e2m1fn
    compute_dtype = T.float32

    x: T.Tensor[(M, K), in_dtype]

    quant = T.empty((M, K), fp4_dtype)
    scale_source = T.empty((M, K // 64), T.uint32)

    with T.Kernel(T.ceildiv(M, rows_per_cta), K // 64, threads=128) as (pid_m, pid_k64):
        x_local = T.alloc_fragment((rows_per_cta, 64), in_dtype)
        amax0 = T.alloc_fragment((rows_per_cta,), compute_dtype)
        amax1 = T.alloc_fragment((rows_per_cta,), compute_dtype)
        amax2 = T.alloc_fragment((rows_per_cta,), compute_dtype)
        amax3 = T.alloc_fragment((rows_per_cta,), compute_dtype)
        scale_value0 = T.alloc_fragment((rows_per_cta,), compute_dtype)
        scale_value1 = T.alloc_fragment((rows_per_cta,), compute_dtype)
        scale_value2 = T.alloc_fragment((rows_per_cta,), compute_dtype)
        scale_value3 = T.alloc_fragment((rows_per_cta,), compute_dtype)
        scale_code0 = T.alloc_fragment((rows_per_cta,), T.int32)
        scale_code1 = T.alloc_fragment((rows_per_cta,), T.int32)
        scale_code2 = T.alloc_fragment((rows_per_cta,), T.int32)
        scale_code3 = T.alloc_fragment((rows_per_cta,), T.int32)
        scale_word = T.alloc_fragment((rows_per_cta,), T.uint32)
        quant_local = T.alloc_fragment((rows_per_cta, 64), fp4_dtype)

        for i, j in T.Parallel(rows_per_cta, 64):
            x_local[i, j] = x[pid_m * rows_per_cta + i, pid_k64 * 64 + j]

        for i in T.Parallel(rows_per_cta):
            amax0[i] = 0.0
            for j in T.serial(16):
                amax0[i] = T.max(amax0[i], T.abs(T.cast(x_local[i, j], compute_dtype)))
            scale_code0[i] = _tl_encode_ue4m3_scale_byte_ceil(amax0[i] / _FP4_E2M1_MAX)
            scale_value0[i] = _tl_decode_ue4m3_scale_byte(scale_code0[i])

        for i in T.Parallel(rows_per_cta):
            amax1[i] = 0.0
            for j in T.serial(16):
                amax1[i] = T.max(amax1[i], T.abs(T.cast(x_local[i, 16 + j], compute_dtype)))
            scale_code1[i] = _tl_encode_ue4m3_scale_byte_ceil(amax1[i] / _FP4_E2M1_MAX)
            scale_value1[i] = _tl_decode_ue4m3_scale_byte(scale_code1[i])

        for i in T.Parallel(rows_per_cta):
            amax2[i] = 0.0
            for j in T.serial(16):
                amax2[i] = T.max(amax2[i], T.abs(T.cast(x_local[i, 32 + j], compute_dtype)))
            scale_code2[i] = _tl_encode_ue4m3_scale_byte_ceil(amax2[i] / _FP4_E2M1_MAX)
            scale_value2[i] = _tl_decode_ue4m3_scale_byte(scale_code2[i])

        for i in T.Parallel(rows_per_cta):
            amax3[i] = 0.0
            for j in T.serial(16):
                amax3[i] = T.max(amax3[i], T.abs(T.cast(x_local[i, 48 + j], compute_dtype)))
            scale_code3[i] = _tl_encode_ue4m3_scale_byte_ceil(amax3[i] / _FP4_E2M1_MAX)
            scale_value3[i] = _tl_decode_ue4m3_scale_byte(scale_code3[i])

        for i, j in T.Parallel(rows_per_cta, 16):
            value0 = T.cast(x_local[i, j], compute_dtype)
            scaled0 = T.if_then_else(scale_value0[i] > 0.0, value0 / scale_value0[i], 0.0)
            quant_local[i, j] = T.cast(T.clamp(scaled0, -_FP4_E2M1_MAX, _FP4_E2M1_MAX), fp4_dtype)

        for i, j in T.Parallel(rows_per_cta, 16):
            value1 = T.cast(x_local[i, 16 + j], compute_dtype)
            scaled1 = T.if_then_else(scale_value1[i] > 0.0, value1 / scale_value1[i], 0.0)
            quant_local[i, 16 + j] = T.cast(T.clamp(scaled1, -_FP4_E2M1_MAX, _FP4_E2M1_MAX), fp4_dtype)

        for i, j in T.Parallel(rows_per_cta, 16):
            value2 = T.cast(x_local[i, 32 + j], compute_dtype)
            scaled2 = T.if_then_else(scale_value2[i] > 0.0, value2 / scale_value2[i], 0.0)
            quant_local[i, 32 + j] = T.cast(T.clamp(scaled2, -_FP4_E2M1_MAX, _FP4_E2M1_MAX), fp4_dtype)

        for i, j in T.Parallel(rows_per_cta, 16):
            value3 = T.cast(x_local[i, 48 + j], compute_dtype)
            scaled3 = T.if_then_else(scale_value3[i] > 0.0, value3 / scale_value3[i], 0.0)
            quant_local[i, 48 + j] = T.cast(T.clamp(scaled3, -_FP4_E2M1_MAX, _FP4_E2M1_MAX), fp4_dtype)

        for i, j in T.Parallel(rows_per_cta, 64):
            quant[pid_m * rows_per_cta + i, pid_k64 * 64 + j] = quant_local[i, j]

        for i in T.Parallel(rows_per_cta):
            scale_word[i] = (
                T.cast(scale_code0[i], T.uint32)
                | (T.cast(scale_code1[i], T.uint32) << 8)
                | (T.cast(scale_code2[i], T.uint32) << 16)
                | (T.cast(scale_code3[i], T.uint32) << 24)
            )
            logical_row = pid_m * rows_per_cta + i
            row_in_chunk = logical_row % _BLOCKSCALED_CHUNK_ROWS
            physical_row = (
                (logical_row // _BLOCKSCALED_CHUNK_ROWS) * _BLOCKSCALED_CHUNK_ROWS
                + (pid_k64 % _BLOCKSCALED_CHUNK_WORDS) * 32
                + (row_in_chunk % 32)
            )
            physical_word = (pid_k64 // _BLOCKSCALED_CHUNK_WORDS) * _BLOCKSCALED_CHUNK_WORDS + (row_in_chunk // 32)
            scale_source[physical_row, physical_word] = scale_word[i]

    return quant, scale_source


def tilelang_quantize_bf16_to_nvfp4_blockscaled(x, *, rows_per_cta: int = 16):
    """Quantize a CUDA BF16 activation tensor with a TileLang kernel.

    Parameters
    ----------
    x:
        CUDA ``torch.bfloat16`` tensor with shape ``[rows, K]``. ``rows`` must
        be a multiple of ``128`` and ``K`` a multiple of ``256``.

    Returns
    -------
    tuple
        ``(packed_fp4, scale_source)``. ``packed_fp4`` is ``torch.int8[rows, K/2]``.
        ``scale_source`` is ``torch.uint32[rows, K/64]`` in CUTLASS
        ``BlockScaledBasicChunk`` K-major source order. The same contract
        applies to SFA and SFB; ``rows`` is interpreted as M for A and N for B.
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
    if rows_per_cta <= 0 or _BLOCKSCALED_CHUNK_ROWS % rows_per_cta != 0:
        raise ValueError(f"rows_per_cta must divide {_BLOCKSCALED_CHUNK_ROWS}, got {rows_per_cta}")

    rows, cols = x.shape
    k_tile = _NVFP4_SCALE_BLOCK_K * _SCALE_BYTES_PER_WORD * _BLOCKSCALED_CHUNK_WORDS
    if rows % _BLOCKSCALED_CHUNK_ROWS != 0 or cols % k_tile != 0:
        raise ValueError(
            f"SM120 NVFP4 quantization requires rows multiple of {_BLOCKSCALED_CHUNK_ROWS} and K multiple of {k_tile}, got {tuple(x.shape)}"
        )

    quant, scale_source = _tilelang_nvfp4_blockscaled_quantize_kernel(x.contiguous(), rows_per_cta=rows_per_cta)
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

    Returns
    -------
    tuple
        ``(packed_fp4, scale_source)`` or
        ``(packed_fp4, scale_source, scale_bytes)``. ``packed_fp4`` is
        ``torch.int8[rows, K / 2]`` with two FP4 E2M1 values per byte.
        ``scale_source`` is ``torch.uint32[rows, K / 64]`` in CUTLASS
        ``BlockScaledBasicChunk`` K-major source order.
    """

    torch = _import_torch()
    _check_block_shape(block_rows, block_words)
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
