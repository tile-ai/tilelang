"""Tests for the SM120 W4A16 NVFP4 GEMM example (examples/gemm_sm120/sm120_nvfp4_w4a16_gemm.py) and its stream-K /
tuned-table extension (maint/gemm/gemm_sm120/sm120_nvfp4_w4a16_streamk.py).

The layout, validation and configuration tests run on the CPU: they emulate the kernel's register dequantization bit
for bit and place every result where the PTX ``mma.m16n8k16`` B-fragment layout says the lane feeds it. The GPU tests
need an SM120 GPU (compute capability 12.0), the only GPU the kernels are tested and tuned on; the sweeps marked
``slow`` cover every tuned-table entry and many more shapes.
"""

import pytest
import torch

import tilelang.testing
from examples.dequantize_gemm.quantize.nvfp4 import (
    decode_packed_fp4_e2m1,
    decode_ue4m3_scale_bytes,
    quantize_bf16_to_nvfp4_blockscaled,
)
from examples.gemm_sm120 import sm120_nvfp4_w4a16_gemm as example
from maint.gemm.gemm_sm120 import sm120_nvfp4_w4a16_streamk as tuned

_TOL = 2.5e-3  # relative Frobenius error vs the FP32 reference (BF16 output rounding is ~1.6e-3)
_ROW_TOL = 1e-2  # worst per-row relative error
_MODULES = [pytest.param(example, id="example"), pytest.param(tuned, id="tuned")]


def _bf16_value(bits):
    bits = bits & 0xFFFF
    return torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16).view(torch.bfloat16).float()


def _emulate_b_fragments(weight):
    """(N, K) FP32 weights as the kernel's B fragments hold them (fp4 * scale * 2^-7), rebuilt from the prepared
    layout by emulating nvfp4_w4a16_dequant_chunk for every lane and placing register r of word w = 2 * st + j at the
    PTX m16n8k16 B-fragment position: b0, b1 at (k = 2 * (lane % 4) + {0, 1}, n = lane / 4), b2, b3 at k + 8; the two
    n8 halves of an n16 tile are registers (0, 1) and (2, 3)."""
    Wq = weight.Wq.cpu().to(torch.int64) & 0xFFFFFFFF
    Sq = weight.Sq.cpu().to(torch.int64) & 0xFFFFFFFF
    NB, KB = Wq.shape[:2]
    lane = torch.arange(32)
    q = Wq.view(NB, KB, 32, 4)
    s = Sq.view(NB, KB, 8, 2)[:, :, lane // 4, :]  # each quad of lanes loads the same 8 scale bytes
    mask = 0x81C081C0
    out = torch.full((NB, 32, KB, 32), float("nan"))
    for st in range(2):
        sw = s[..., st]
        code = [(sw >> (8 * i)) & 0xFF for i in range(4)]
        for j in range(2):
            x = q[..., 2 * st + j]
            regs = (
                x & mask,
                (x << 3) & mask,
                (x << 6) & mask,
                ((x << 1) & 0x80008000) | ((x >> 3) & 0x01800180) | ((x >> 7) & 0x00400040),
            )
            scale_lo = _bf16_value(0x7000 | (code[j] << 4))  # bytes 0, 2 (j = 0) and 1, 3 (j = 1)
            scale_hi = _bf16_value(0x7000 | (code[j + 2] << 4))
            for r in range(4):
                scale = scale_lo if r < 2 else scale_hi
                for h in range(2):
                    value = _bf16_value(regs[r] >> (16 * h)) * scale
                    n = 16 * j + lane // 4 + 8 * (r // 2)
                    k = 16 * st + 2 * (lane % 4) + h + 8 * (r % 2)
                    out[:, n, :, k] = value.permute(2, 0, 1)
    return out.reshape(NB * 32, KB * 32)


def _all_finite_e4m3_scales(N, K):
    codes = torch.tensor([c for c in range(256) if c & 0x7F != 0x7F], dtype=torch.uint8)
    return codes.repeat(N * K // 16 // codes.numel() + 1)[: N * K // 16].view(N, K // 16).view(torch.float8_e4m3fn)


def test_prepare_layout_matches_mma_b_fragments():
    N, K = 128, 512
    gen = torch.Generator().manual_seed(0)
    packed = torch.randint(-128, 128, (N, K // 2), dtype=torch.int8, generator=gen)
    scale = _all_finite_e4m3_scales(N, K)  # every finite E4M3 value, including zero, negatives and subnormals
    weight = example.prepare_nvfp4_weight(packed, scale, 0.5)
    expected = decode_packed_fp4_e2m1(packed) * scale.float().repeat_interleave(16, dim=1) * 2.0**-7
    assert torch.equal(_emulate_b_fragments(weight), expected)
    assert torch.equal(weight.alpha, torch.full((N,), 0.5 * 128.0))
    assert weight.Wq.numel() * 4 == N * K // 2 and weight.Sq.numel() * 4 == N * K // 16


def test_prepare_accepts_ue4m3_bytes_and_column_alpha():
    N, K = 256, 256
    w = torch.randn(N, K) * 3
    packed, _, scale_bytes = quantize_bf16_to_nvfp4_blockscaled(w.to(torch.bfloat16), return_scale_bytes=True)
    alpha = torch.rand(N) + 0.5
    weight = example.prepare_nvfp4_weight(packed, scale_bytes, alpha)
    expected = decode_packed_fp4_e2m1(packed) * decode_ue4m3_scale_bytes(scale_bytes).repeat_interleave(16, dim=1) * 2.0**-7
    assert torch.equal(_emulate_b_fragments(weight), expected)
    assert torch.equal(weight.alpha, alpha * 128.0)


def test_prepare_rejects_bad_inputs():
    packed = torch.zeros(128, 128, dtype=torch.int8)
    scale = torch.zeros(128, 16, dtype=torch.uint8)
    with pytest.raises(ValueError):
        example.prepare_nvfp4_weight(torch.zeros(96, 128, dtype=torch.int8), torch.zeros(96, 16, dtype=torch.uint8))
    with pytest.raises(ValueError):
        example.prepare_nvfp4_weight(packed, torch.zeros(128, 8, dtype=torch.uint8))
    with pytest.raises(ValueError):
        example.prepare_nvfp4_weight(packed, torch.full((128, 16), 0x7F, dtype=torch.uint8))  # NaN scales
    with pytest.raises(ValueError):
        example.prepare_nvfp4_weight(packed, scale, torch.ones(3))  # alpha neither scalar nor (N,)
    # Only E4M3 scales: float32 512.0 is above the E4M3 range, and other dtypes are not checkpoint formats.
    for bad in (torch.full((128, 16), 512.0), torch.ones(128, 16, dtype=torch.bfloat16), torch.ones(128, 16, dtype=torch.int32)):
        with pytest.raises(TypeError):
            example.prepare_nvfp4_weight(packed, bad)
    with pytest.raises(TypeError):
        example.prepare_nvfp4_weight(packed.to(torch.int32), scale)


def test_linear_rejects_bad_inputs():
    N, K = 128, 256
    weight = example.prepare_nvfp4_weight(torch.zeros(N, K // 2, dtype=torch.uint8), torch.zeros(N, K // 16, dtype=torch.uint8))
    x = torch.zeros(4, K, dtype=torch.bfloat16)
    for module in (example, tuned):
        with pytest.raises(ValueError):
            module.nvfp4_w4a16_linear(x.float(), weight)
        with pytest.raises(ValueError):
            module.nvfp4_w4a16_linear(torch.zeros(4, K + 256, dtype=torch.bfloat16), weight)
        with pytest.raises(ValueError):
            module.nvfp4_w4a16_linear(x, weight, out=torch.empty(4, N + 128, dtype=torch.bfloat16))
        with pytest.raises(ValueError):
            module.nvfp4_w4a16_linear(x, weight, out=torch.empty(N, 4, dtype=torch.bfloat16).t())  # not contiguous
        with pytest.raises(TypeError):
            module.nvfp4_w4a16_linear(x, (weight.Wq, weight.Sq))


def _frag_ok(N, K, cfg):
    return N % (32 * cfg.n_warps) == 0 and K % cfg.block_K == 0 and (K // cfg.block_K) % cfg.split_k == 0


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize("num_sms", [188, 170])
def test_configs_cover_all_rows(module, num_sms):
    shapes = [(128, 256), (640, 2304), (768, 1280), (4096, 4096), (11008, 4096)] + list(tuned._TUNED_CONFIGS[188])
    for N, K in shapes:
        for M in range(1, example.MAX_M + 1):
            cfg = module.get_config(N, K, M, num_sms=num_sms)
            if cfg == example.FRAG:
                for r0 in range(0, M, example.FRAG_MAX_M):
                    frag = example.frag_config(N, K, min(example.FRAG_MAX_M, M - r0), num_sms)
                    assert _frag_ok(N, K, frag), (N, K, M, frag)
                continue
            if isinstance(cfg, example.FragConfig):
                assert _frag_ok(N, K, cfg), (N, K, M, cfg)
                continue
            block_N = cfg.warps_n * 32 * cfg.nb
            assert N % block_N == 0 and K % 64 == 0, (N, K, M, cfg)
            if isinstance(cfg, example.TileConfig):
                assert (K // cfg.block_K) % cfg.split_k == 0, (N, K, M, cfg)
            smem = tuned.smem_bytes(cfg)
            assert smem <= 99 * 1024 and smem * cfg.min_blocks <= 100 * 1024, (N, K, M, cfg)


def test_tuned_table_is_per_gpu():
    for N, K in tuned._TUNED_CONFIGS[188]:
        for M in (1, 100, 300, 1024):
            assert tuned.get_config(N, K, M, num_sms=170) == tuned._heuristic_config(N, K, M, 170)


# ---------------------------------------------------------------------------------------------------------------
# GPU tests.
def _layer(N, K, seed=0, global_scale=448.0):
    """An NVFP4 checkpoint-style weight: a BF16 weight times the global scale, quantized with alpha = 1/global scale."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(N, K, device="cuda", generator=gen) * 0.02 * global_scale
    packed, _, scale_bytes = quantize_bf16_to_nvfp4_blockscaled(w.to(torch.bfloat16), return_scale_bytes=True)
    alpha = 1.0 / global_scale
    w_ref = decode_packed_fp4_e2m1(packed) * decode_ue4m3_scale_bytes(scale_bytes).repeat_interleave(16, dim=1) * alpha
    return example.prepare_nvfp4_weight(packed, scale_bytes, alpha), w_ref


def _check(y, x, w_ref=None, ref=None):
    if ref is None:
        ref = x.float() @ w_ref.T
    assert y.dtype == torch.bfloat16 and y.shape == ref.shape
    assert bool(torch.isfinite(y).all())
    d = (y.float() - ref).abs()
    err = d.norm() / ref.norm()
    row_err = (d.norm(dim=1) / ref.norm(dim=1)).max()
    assert err <= _TOL and row_err <= _ROW_TOL, (x.shape[0], err.item(), row_err.item())
    # elementwise: one BF16 rounding of the exact result plus FP32 accumulation-order noise
    row_rms = ref.pow(2).mean(dim=1, keepdim=True).sqrt()
    bad = int((d > 2.0**-8 * ref.abs() + 1e-3 * row_rms).sum())
    assert bad == 0, (x.shape[0], bad)


def _sweep(module, N, K, ms, seed=0):
    weight, w_ref = _layer(N, K, seed)
    x_all = torch.randn(max(ms), K, device="cuda", dtype=torch.bfloat16)
    ref_all = x_all.float() @ w_ref.T
    for M in ms:
        x = x_all[:M]
        y = module.nvfp4_w4a16_linear(x, weight)
        _check(y, x, ref=ref_all[:M])
        assert torch.equal(y, module.nvfp4_w4a16_linear(x, weight)), f"{N}x{K} M={M}: not deterministic"


# Shapes: a 27B-class model's layers (tuned on 188-SM GPUs: fragment, tile and stream-K kernels) and untuned shapes
# (heuristics), including K that is not a multiple of 1024, N that is not a multiple of 256 and the smallest shape.
_FAST_SHAPES = [(5120, 6144), (4096, 4096), (640, 2304), (128, 256)]
_FAST_MS = [1, 17, 64, 200, 1024]
_SLOW_SHAPES = [(5120, 6144), (16384, 5120), (4096, 4096), (768, 1280), (640, 2304), (128, 256), (4096, 11008)]
_SLOW_MS = [1, 2, 4, 8, 16, 17, 32, 33, 64, 65, 97, 128, 200, 256, 300, 513, 1024]


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize("N,K", _FAST_SHAPES)
def test_nvfp4_w4a16_gemm(module, N, K):
    _sweep(module, N, K, _FAST_MS)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.slow
@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize("N,K", _SLOW_SHAPES)
def test_nvfp4_w4a16_gemm_sweep(module, N, K):
    _sweep(module, N, K, _SLOW_MS)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.slow
@pytest.mark.parametrize("N,K", list(tuned._TUNED_CONFIGS[188]))
def test_nvfp4_w4a16_gemm_every_tuned_entry(N, K):
    """Every entry of a tuned table at the start, middle and end of its M range, through that exact config."""
    table = tuned._TUNED_CONFIGS[188][(N, K)]
    weight, w_ref = _layer(N, K, seed=4)
    x_all = torch.randn(example.MAX_M, K, device="cuda", dtype=torch.bfloat16)
    ref_all = x_all.float() @ w_ref.T
    lo = 1
    for m_max, cfg in table:
        for M in sorted({lo, (lo + m_max) // 2, m_max}):
            x = x_all[:M]
            _check(tuned.nvfp4_w4a16_linear(x, weight, config=cfg), x, ref=ref_all[:M])
        lo = m_max + 1


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize(
    "config,ms",
    [
        (example.FragConfig(16, 4, 64, 4, 4), [1, 16, 40]),  # 40 rows: three 16-row passes
        (example.FragConfig(48, 4, 64, 2, 3), [40, 48]),  # 40 rows: the tile kernel with the same tile
        (example.FRAG, [1, 64, 100]),
        (example.TileConfig(1, 32, 4, 1, 64, 2, 4), [1, 16, 100, 333]),
        (example.TileConfig(2, 64, 4, 2, 64, 1, 3), [1, 16, 100, 333]),
        (example.TileConfig(1, 64, 4, 1, 64, 4, 3, min_blocks=2), [1, 100, 333]),
        (tuned.StreamKConfig(1, 64, 4, 1, 3, 1, 1, False), [1, 16, 100, 333]),
        (tuned.StreamKConfig(2, 64, 4, 2, 3, 1, 1, True), [1, 16, 100, 333]),
        (tuned.StreamKConfig(1, 80, 4, 1, 3, 2, 2, False), [1, 16, 100, 333]),
    ],
)
def test_nvfp4_w4a16_gemm_explicit_config(config, ms):
    N, K = 4096, 4096
    weight, w_ref = _layer(N, K, seed=1)
    for M in ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        _check(tuned.nvfp4_w4a16_linear(x, weight, config=config), x, w_ref)
        if not isinstance(config, tuned.StreamKConfig):
            _check(example.nvfp4_w4a16_linear(x, weight, config=config), x, w_ref)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("module", _MODULES)
def test_nvfp4_w4a16_gemm_zero_negative_scales_chunking_and_out(module):
    N, K = 1024, 2048
    gen = torch.Generator(device="cuda").manual_seed(2)
    packed = torch.randint(-128, 128, (N, K // 2), dtype=torch.int8, device="cuda", generator=gen)
    scale = (torch.rand(N, K // 16, device="cuda", generator=gen) * 4 - 1).to(torch.float8_e4m3fn)  # includes negatives
    scale[:, ::7] = 0
    alpha = torch.rand(N, device="cuda", generator=gen) + 0.5
    weight = example.prepare_nvfp4_weight(packed, scale, alpha)
    w_ref = decode_packed_fp4_e2m1(packed) * scale.float().repeat_interleave(16, dim=1) * alpha[:, None]
    for M in [3, 48, 1100]:  # 1100 > MAX_M runs as two chunks
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        _check(module.nvfp4_w4a16_linear(x, weight), x, w_ref)
        out = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        assert module.nvfp4_w4a16_linear(x, weight, out=out) is out
        _check(out, x, w_ref)
    # an x whose storage is not 16-byte aligned is copied, not read misaligned
    base = torch.randn(1 + 8 * K, device="cuda", dtype=torch.bfloat16)
    x = base[1:].view(8, K)
    _check(module.nvfp4_w4a16_linear(x, weight), x, w_ref)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize(
    "module,M",
    [(tuned, 8), (tuned, 33), (tuned, 128), (tuned, 1024), (example, 33), (example, 200), (example, 1024)],
    ids=["tuned-8", "tuned-33", "tuned-128", "tuned-1024", "example-33", "example-200", "example-1024"],
)
def test_nvfp4_w4a16_gemm_cuda_graph(module, M):
    N, K = 5120, 6144  # tuned: fragment (8), tile with wrapped padding rows (33), stream-K (128, 1024)
    weight, w_ref = _layer(N, K, seed=3)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    module.nvfp4_w4a16_linear(x, weight)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        y = module.nvfp4_w4a16_linear(x, weight)
    for _ in range(2):
        x.copy_(torch.randn(M, K, device="cuda", dtype=torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        _check(y, x, w_ref)
        assert torch.equal(y, module.nvfp4_w4a16_linear(x, weight))


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_nvfp4_w4a16_gemm_first_call_in_shared_pool_capture():
    """The first call on a device inside a capture that shares a memory pool with an earlier graph: the split-K
    tickets must not come from (and alias) the pool."""
    N, K, M = 5120, 6144, 128  # stream-K with split tiles
    weight, w_ref = _layer(N, K, seed=5)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    tuned.nvfp4_w4a16_linear(x, weight)  # compile only
    torch.cuda.synchronize()
    example._WORKSPACES.clear()  # as in a fresh process
    weight, w_ref = _layer(N, K, seed=6)  # prepare creates the workspace outside any capture
    pool = torch.cuda.graph_pool_handle()
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, pool=pool, capture_error_mode="thread_local"):
        junk = torch.full((1 << 16,), 3, dtype=torch.int32, device="cuda")
        junk.add_(1)
    del junk  # its block goes back to the pool
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, pool=pool, capture_error_mode="thread_local"):
        y = tuned.nvfp4_w4a16_linear(x, weight)
    _check(tuned.nvfp4_w4a16_linear(x, weight), x, w_ref)  # eager, before any replay
    g2.replay()
    g1.replay()
    torch.cuda.synchronize()
    _check(y, x, w_ref)
    _check(tuned.nvfp4_w4a16_linear(x, weight), x, w_ref)


if __name__ == "__main__":
    tilelang.testing.main()
