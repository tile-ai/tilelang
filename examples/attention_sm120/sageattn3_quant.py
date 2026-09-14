"""SageAttention3-style NVFP4 quantization contract and golden reference (SM120).

Canonical (logical-order) representation used by every exporter and by the
golden reference:

``q_codes``  uint8 ``[B, H, N, D]``      e2m1 codes (bit 3 = sign), Q smoothed by the
                                          128-row block mean and quantized per 16 head-dim elems
``q_sf``     uint8 ``[B, H, N, D // 16]`` ue4m3 scale bytes
``k_codes``  uint8 ``[B, H, N, D]``      keys in ORIGINAL order, K smoothed by the seq mean
``k_sf``     uint8 ``[B, H, N, D // 16]``
``vt_codes`` uint8 ``[B, H, D, N]``      V transposed; scale groups run along the key axis
``vt_sf``    uint8 ``[B, H, D, N // 16]``
``delta_s``  fp32  ``[B, H, N // 128, N]`` = q_mean_block @ K_smoothed^T (the smoothing-Q term)

Two exporters map the canonical form to the input formats of (a) the original
``sageattn3_blackwell`` CUDA kernel (nibble packing, 64-token blocked SF swizzle,
K rows permuted within 32-groups) and (b) the TileLang kernel in this directory
(nibble packing, row-major uint32 scale words, same K permutation).

The quantization semantics replicate ``fp4_quantization_4d.cu`` and
``softmax_fused.h`` / ``mainloop_tma_ws.h`` of thu-ml/SageAttention:
scale = e4m3_rn(amax / 6) (round-tripped through e4m3 before the inverse for
Q/K/V, but NOT round-tripped for the in-kernel P quantization), e2m1 by
``cvt.rn.satfinite`` (round-to-nearest-even, saturate to 6).
"""

from __future__ import annotations

import math

import torch

FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0
LOG2E = 1.4426950408889634
# Two-level P scaling constants (softmax_fused.h): P2 = P * 448 * 6 folded into exp2.
LOG2_P1_SCALE = math.log2(1.0 / (FP8_E4M3_MAX * FP4_MAX))  # -11.392317422778762
LOG2_FP4_SCALE = math.log2(1.0 / FP4_MAX)  # -2.584962500721156

# Key permutation applied to K rows inside every 32-key group (fp4_quantization_4d.cu:157-168).
# Permuted position r holds original key PERM32[r].
PERM32 = torch.tensor([(r // 8) * 2 + ((r % 8) // 2) * 8 + (r % 2) for r in range(32)], dtype=torch.long)

_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


# --------------------------------------------------------------------------------------
# Element codecs
# --------------------------------------------------------------------------------------
def e4m3_rn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 -> (ue4m3 byte, exact fp32 value) with round-to-nearest-even and satfinite."""
    x32 = x.to(torch.float32).clamp(min=0.0, max=FP8_E4M3_MAX)
    f8 = x32.to(torch.float8_e4m3fn)
    return f8.view(torch.uint8), f8.to(torch.float32)


def e2m1_rn_codes(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> e2m1 code (0..15) emulating ``cvt.rn.satfinite.e2m1x2.f32``."""
    x32 = x.to(torch.float32)
    a = x32.abs().clamp(max=FP4_MAX)
    a = torch.nan_to_num(a, nan=0.0)
    mids = _E2M1_MIDPOINTS.to(x32.device)
    m_gt = (a.unsqueeze(-1) > mids).sum(-1)
    m_ge = (a.unsqueeze(-1) >= mids).sum(-1)
    tie = m_gt != m_ge
    m = torch.where(tie, torch.where(m_gt % 2 == 0, m_gt, m_gt + 1), m_gt)
    sign = torch.signbit(x32).to(torch.long)
    return (m + 8 * sign).to(torch.uint8)


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    c = codes.to(torch.long)
    vals = _E2M1_VALUES.to(codes.device)[c & 7]
    return torch.where((c & 8) != 0, -vals, vals)


def ue4m3_decode(sf_bytes: torch.Tensor) -> torch.Tensor:
    return sf_bytes.view(torch.float8_e4m3fn).to(torch.float32)


# --------------------------------------------------------------------------------------
# Row quantization (Q/K/V semantics of fp4_quantization_4d.cu)
# --------------------------------------------------------------------------------------
def quantize_rows_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last axis in groups of 16: returns (codes uint8 [..., K], sf uint8 [..., K//16])."""
    x32 = x.to(torch.float32)
    *lead, k = x32.shape
    assert k % 16 == 0
    blocks = x32.reshape(*lead, k // 16, 16)
    amax = blocks.abs().amax(dim=-1)
    sf_bytes, sf_val = e4m3_rn(amax / FP4_MAX)
    inv = torch.where(sf_val > 0, 1.0 / sf_val, torch.zeros_like(sf_val))
    y = blocks * inv.unsqueeze(-1)
    codes = e2m1_rn_codes(y).reshape(*lead, k)
    return codes, sf_bytes


def dequantize_rows_nvfp4(codes: torch.Tensor, sf_bytes: torch.Tensor) -> torch.Tensor:
    *lead, k = codes.shape
    vals = e2m1_decode(codes).reshape(*lead, k // 16, 16)
    return (vals * ue4m3_decode(sf_bytes).unsqueeze(-1)).reshape(*lead, k)


# --------------------------------------------------------------------------------------
# Preprocessing (api.py preprocess_qkv) and canonical quantization
# --------------------------------------------------------------------------------------
def pad_seq_to_128(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[2]
    pad = (128 - n % 128) % 128
    if pad == 0:
        return x.contiguous()
    return torch.nn.functional.pad(x, (0, 0, 0, pad), value=0).contiguous()


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Smoothing K (seq mean) and Q (128-row block mean) + delta_s, in the original's dtypes.

    Returns (q_smoothed bf16/fp16, k_smoothed, v, q_mean [B,H,N//128,D], delta_s fp32).
    The seq-mean and the delta_s matmul are done with the same torch ops as the original.
    """
    k = k - k.mean(dim=-2, keepdim=True)
    q, k, v = (pad_seq_to_128(t) for t in (q, k, v))
    b, h, n, d = q.shape
    q32 = q.to(torch.float32).reshape(b, h, n // 128, 128, d)
    qm32 = q32.mean(dim=3)
    q_sm = (q32 - qm32.unsqueeze(3)).reshape(b, h, n, d).to(q.dtype)
    qm = qm32.to(q.dtype)
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q_sm, k, v, qm, delta_s


def quantize_canonical(q_sm: torch.Tensor, k_sm: torch.Tensor, v: torch.Tensor) -> dict:
    """Canonical fp4 codes + scale bytes for smoothed Q, smoothed K (original key order) and V^T."""
    q_codes, q_sf = quantize_rows_nvfp4(q_sm)
    k_codes, k_sf = quantize_rows_nvfp4(k_sm)
    vt_codes, vt_sf = quantize_rows_nvfp4(v.transpose(-2, -1).contiguous())
    return dict(q_codes=q_codes, q_sf=q_sf, k_codes=k_codes, k_sf=k_sf, vt_codes=vt_codes, vt_sf=vt_sf)


# --------------------------------------------------------------------------------------
# Layout exporters
# --------------------------------------------------------------------------------------
def permute_keys(x: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """Apply the 32-group key permutation along ``dim`` (position r <- original PERM32[r])."""
    x = x.movedim(dim, -1)
    *lead, n = x.shape
    assert n % 32 == 0
    x = x.reshape(*lead, n // 32, 32)[..., PERM32.to(x.device)].reshape(*lead, n)
    return x.movedim(-1, dim).contiguous()


def unpermute_keys(x: torch.Tensor, dim: int = -2) -> torch.Tensor:
    inv = torch.empty_like(PERM32)
    inv[PERM32] = torch.arange(32)
    x = x.movedim(dim, -1)
    *lead, n = x.shape
    x = x.reshape(*lead, n // 32, 32)[..., inv.to(x.device)].reshape(*lead, n)
    return x.movedim(-1, dim).contiguous()


def pack_codes_to_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] e2m1 codes -> [..., K//2] bytes, even element in the low nibble."""
    c = codes.to(torch.uint8)
    return (c[..., 0::2] & 0xF) | ((c[..., 1::2] & 0xF) << 4)


def unpack_nibbles_to_codes(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.uint8)
    out = torch.empty(*p.shape[:-1], p.shape[-1] * 2, dtype=torch.uint8, device=p.device)
    out[..., 0::2] = p & 0xF
    out[..., 1::2] = p >> 4
    return out


def _sage_sf_offsets(rows: int, groups: int, device) -> torch.Tensor:
    """Byte offset of (row t, group j) inside the original's blocked SF layout.

    fp4_quantization_4d.cu: base = (t // 64) * 64 * groups;
    offset = (j // 4) * 256 + (j % 4) + ((t % 64) // 16) * 4 + (t % 16) * 16.
    """
    t = torch.arange(rows, device=device).unsqueeze(1)
    j = torch.arange(groups, device=device).unsqueeze(0)
    return (t // 64) * 64 * groups + (j // 4) * 256 + (j % 4) + ((t % 64) // 16) * 4 + (t % 16) * 16


def sf_to_sage_layout(sf_bytes: torch.Tensor) -> torch.Tensor:
    """[..., R, G] logical scale bytes -> same shape, physically in the original's swizzle."""
    *lead, rows, groups = sf_bytes.shape
    assert rows % 64 == 0 and groups % 4 == 0
    flat = sf_bytes.reshape(*lead, rows * groups)
    out = torch.empty_like(flat)
    idx = _sage_sf_offsets(rows, groups, sf_bytes.device).reshape(-1)
    out[..., idx] = flat
    return out.reshape(*lead, rows, groups)


def sf_from_sage_layout(sf_phys: torch.Tensor) -> torch.Tensor:
    *lead, rows, groups = sf_phys.shape
    idx = _sage_sf_offsets(rows, groups, sf_phys.device).reshape(-1)
    return sf_phys.reshape(*lead, rows * groups)[..., idx].reshape(*lead, rows, groups)


def sf_to_rowmajor_words(sf_bytes: torch.Tensor) -> torch.Tensor:
    """[..., R, G] scale bytes -> [..., R, G//4] uint32 words, 4 consecutive groups per word (LSB first)."""
    *lead, rows, groups = sf_bytes.shape
    assert groups % 4 == 0
    b = sf_bytes.to(torch.int64).reshape(*lead, rows, groups // 4, 4)
    words = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)
    return words.to(torch.uint32).contiguous()


def export_for_sage(c: dict) -> dict:
    """Inputs of ``fp4attn_cuda.fwd``: packed q/k/v (uint8) and swizzled float8 scales."""
    k_codes_p = permute_keys(c["k_codes"], dim=-2)
    k_sf_p = permute_keys(c["k_sf"], dim=-2)
    return dict(
        q=pack_codes_to_nibbles(c["q_codes"]),
        k=pack_codes_to_nibbles(k_codes_p),
        v=pack_codes_to_nibbles(c["vt_codes"]),
        sfq=sf_to_sage_layout(c["q_sf"]).view(torch.float8_e4m3fn),
        sfk=sf_to_sage_layout(k_sf_p).view(torch.float8_e4m3fn),
        sfv=sf_to_sage_layout(c["vt_sf"]).view(torch.float8_e4m3fn),
    )


def export_for_tilelang(c: dict) -> dict:
    """Inputs of the TileLang kernel: packed nibbles + row-major uint32 scale words; K permuted."""
    k_codes_p = permute_keys(c["k_codes"], dim=-2)
    k_sf_p = permute_keys(c["k_sf"], dim=-2)
    return dict(
        q=pack_codes_to_nibbles(c["q_codes"]),
        k=pack_codes_to_nibbles(k_codes_p),
        vt=pack_codes_to_nibbles(c["vt_codes"]),
        sfq=sf_to_rowmajor_words(c["q_sf"]),
        sfk=sf_to_rowmajor_words(k_sf_p),
        sfv=sf_to_rowmajor_words(c["vt_sf"]),
    )


def canonical_from_sage(q, k, v, sfq, sfk, sfv) -> dict:
    """Inverse of ``export_for_sage`` (accepts the original quant kernels' outputs)."""
    return dict(
        q_codes=unpack_nibbles_to_codes(q),
        q_sf=sf_from_sage_layout(sfq.view(torch.uint8)),
        k_codes=unpermute_keys(unpack_nibbles_to_codes(k), dim=-2),
        k_sf=unpermute_keys(sf_from_sage_layout(sfk.view(torch.uint8)), dim=-2),
        vt_codes=unpack_nibbles_to_codes(v),
        vt_sf=sf_from_sage_layout(sfv.view(torch.uint8)),
    )


# --------------------------------------------------------------------------------------
# Golden reference (mainloop_tma_ws.h + softmax_fused.h semantics, fp64 accumulation)
# --------------------------------------------------------------------------------------
def golden_attention(
    c: dict,
    delta_s: torch.Tensor,
    softmax_scale: float,
    *,
    block_n: int = 128,
    round_trip_p_scale: bool = False,
) -> torch.Tensor:
    """FP4 attention with the original's in-kernel P quantization, on dequantized canonical inputs.

    ``round_trip_p_scale=False`` reproduces the original (P divided by the *unrounded* fp32
    per-16 scale while the MMA multiplies by the e4m3-rounded scale); ``True`` divides by the
    rounded scale (the numerically consistent variant).
    """
    q = dequantize_rows_nvfp4(c["q_codes"], c["q_sf"]).to(torch.float64)
    k = dequantize_rows_nvfp4(c["k_codes"], c["k_sf"]).to(torch.float64)
    vt = dequantize_rows_nvfp4(c["vt_codes"], c["vt_sf"]).to(torch.float64)
    b, h, n, d = q.shape
    sl2 = softmax_scale * LOG2E
    s_full = torch.matmul(q, k.transpose(-2, -1)).to(torch.float32)  # [B,H,N,N]
    ds = delta_s.to(torch.float32).repeat_interleave(128, dim=2)[:, :, :n, :]
    s_full = s_full + ds
    m_i = torch.full((b, h, n), -float("inf"), dtype=torch.float32, device=q.device)
    l_i = torch.zeros((b, h, n), dtype=torch.float32, device=q.device)
    acc = torch.zeros((b, h, n, d), dtype=torch.float64, device=q.device)
    for j0 in range(0, n, block_n):
        s = s_full[:, :, :, j0 : j0 + block_n]
        m_new = torch.maximum(m_i, s.amax(dim=-1))
        rescale = torch.exp2((m_i - m_new) * sl2)
        max_scaled = m_new * sl2 + LOG2_P1_SCALE
        p2 = torch.exp2(s * sl2 - max_scaled.unsqueeze(-1))
        s16 = s.reshape(b, h, n, block_n // 16, 16).amax(dim=-1)
        absmax = torch.exp2(s16 * sl2 - max_scaled.unsqueeze(-1) + LOG2_FP4_SCALE)
        sfp_bytes, sfp_val = e4m3_rn(absmax)
        div = sfp_val if round_trip_p_scale else absmax
        inv = torch.where(div > 0, 1.0 / div, torch.zeros_like(div))
        p_hat = e2m1_rn_codes(p2.reshape(b, h, n, block_n // 16, 16) * inv.unsqueeze(-1))
        p_deq = (e2m1_decode(p_hat) * sfp_val.unsqueeze(-1)).reshape(b, h, n, block_n).to(torch.float64)
        acc = acc * rescale.unsqueeze(-1).to(torch.float64) + torch.matmul(p_deq, vt[:, :, :, j0 : j0 + block_n].transpose(-2, -1))
        l_i = l_i * rescale + p2.sum(dim=-1)
        m_i = m_new
    return (acc / l_i.unsqueeze(-1).to(torch.float64)).to(torch.float32)


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """Unquantized fp64 attention on the ORIGINAL (unsmoothed, unpadded) q/k/v."""
    q64, k64, v64 = (t.to(torch.float64) for t in (q, k, v))
    p = torch.softmax(torch.matmul(q64, k64.transpose(-2, -1)) * softmax_scale, dim=-1)
    return torch.matmul(p, v64).to(torch.float32)


# --------------------------------------------------------------------------------------
# Metrics (paper definitions)
# --------------------------------------------------------------------------------------
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.to(torch.float64).flatten(), b.to(torch.float64).flatten()
    return float((a64 @ b64) / (a64.norm() * b64.norm()))


def rel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    return float((a64 - b64).abs().sum() / b64.abs().sum())


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.to(torch.float64) - b.to(torch.float64)) ** 2).mean().sqrt())


def alignment_ratio(o_test: torch.Tensor, o_sage: torch.Tensor, o_ref: torch.Tensor) -> float:
    """D16 criterion: ||O_test - O_sage|| / ||O_sage - O_ref|| (must be << 1)."""
    num = (o_test.to(torch.float64) - o_sage.to(torch.float64)).norm()
    den = (o_sage.to(torch.float64) - o_ref.to(torch.float64)).norm()
    return float(num / den)
