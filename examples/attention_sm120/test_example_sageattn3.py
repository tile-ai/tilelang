"""Acceptance tests for the SM120 SageAttention3-style NVFP4 attention example (S3 v1)."""

import pytest

import tilelang
import tilelang.testing

torch = pytest.importorskip("torch")

from examples.attention_sm120 import sageattn3_quant as sq  # noqa: E402
from examples.attention_sm120.sm120_sageattn3_fwd import prepare_inputs, sm120_sageattn3_fwd  # noqa: E402
from examples.attention_sm120 import sm120_sageattn3_fwd_ws as ws  # noqa: E402


def _run(b, h, n, d=128):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    canon, kargs, delta_s = prepare_inputs(q, k, v)
    kernel = sm120_sageattn3_fwd(b, h, n, d)
    o = kernel(*kargs).float()
    return q, k, v, canon, kargs, delta_s, kernel, o


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("n", [256, 1024, 4096])
def test_sageattn3_fwd_matches_golden(n):
    """A1: the kernel reproduces the documented quantization semantics (golden) almost exactly."""
    q, k, v, canon, _, delta_s, _, o = _run(1, 2, n)
    o_ref = sq.reference_attention(q, k, v, 128**-0.5)
    o_gold = sq.golden_attention(canon, delta_s, 128**-0.5)
    ratio = sq.alignment_ratio(o, o_gold, o_ref)
    assert ratio < 0.02, ratio  # measured 0.009
    assert sq.cos_sim(o, o_ref) > 0.975


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sageattn3_fwd_matches_original_kernel():
    """A2: closer to the original CUDA kernel than the golden is (residual = fast-math ulp noise)."""
    fp4attn_cuda = pytest.importorskip("fp4attn_cuda")
    n = 1024
    q, k, v, canon, _, delta_s, _, o = _run(1, 2, n)
    o_ref = sq.reference_attention(q, k, v, 128**-0.5)
    s = sq.export_for_sage(canon)
    o_sage, _ = fp4attn_cuda.fwd(s["q"], s["k"], s["v"], s["sfq"], s["sfk"], s["sfv"], delta_s, n, None, 128**-0.5, False, True, True)
    ratio = sq.alignment_ratio(o, o_sage.float(), o_ref)
    assert ratio < 0.25, ratio  # measured 0.12-0.18 (golden vs original: 0.16-0.18)
    assert abs(sq.cos_sim(o, o_ref) - sq.cos_sim(o_sage.float(), o_ref)) < 1e-3


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sageattn3_fwd_codegen_contract():
    """A3: the intended instructions are engaged (no silent fallback)."""
    _, _, _, _, _, _, kernel, _ = _run(1, 1, 256)
    src = kernel.get_kernel_source()
    assert "tl::tma_load(" in src or "tl::cp_async_gs<" in src  # packed fp4 K/V^T staging (TMA under WS, cp.async in the simple form)
    assert "SM120MmaBlockScaledKind::kMxf4nvf4, 4, tl::SM120MmaScaleType::kUE4M3" in src  # m16n8k64 4X ue4m3
    assert "tl_cvt_e2m1x2_rn(" in src  # cvt.rn.satfinite.e2m1x2 P quantization
    assert "AllReduce<tl::MaxOp, 2, 1" in src  # per-16-key max = one xor-1 shuffle (K permutation contract)
    assert "= -CUDART_INF_F" in src  # running row max initialised
    assert "tl_sts_u8(" in src and "tl_syncwarp()" in src  # warp-private P-scale store, no block barrier before PV
    assert "software_pipeline" not in src  # manual schedule fully lowered


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sageattn3_fwd_perf_smoke():
    """A4: runs and reports TOPS (no threshold)."""
    from tilelang.profiler import do_bench

    _, _, _, _, kargs, _, kernel, _ = _run(4, 32, 2048)
    ms = do_bench(lambda: kernel(*kargs), warmup=10, rep=30, backend="cudagraph", return_mode="median")
    tops = 4.0 * 4 * 32 * 2048 * 2048 * 128 / (ms * 1e-3) / 1e12
    print(f"\nsageattn3 v1 kernel-only B=4 H=32 N=2048: {ms:.4f} ms, {tops:.1f} TOPS")
    assert tops > 0


# ---------------------------------------------------------------------------
# Persistent warp-specialized kernel (sm120_sageattn3_fwd_ws.py)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("n", [256, 512, 1024, 4096])
def test_sageattn3_ws_matches_base(n):
    """The persistent warp-specialized kernel is bit-identical to the plain one.

    256 and 512 stay in the matrix on purpose: sequences shorter than the pipeline depth are
    where prologue/epilogue range bugs show up.
    """
    torch.manual_seed(0)
    b, h, d = 1, 2, 128
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    _, kargs, _ = prepare_inputs(q, k, v)
    base = sm120_sageattn3_fwd(b, h, n, d)(*kargs)
    out = ws.sm120_sageattn3_fwd(b, h, n, d)(*kargs)
    assert torch.equal(base, out)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sageattn3_ws_codegen_contract():
    """The producer/consumer shape actually materialized (no silent fallback to the plain form)."""
    torch.manual_seed(0)
    b, h, n, d = 1, 1, 256, 128
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    _, kargs, _ = prepare_inputs(q, k, v)
    src = ws.sm120_sageattn3_fwd(b, h, n, d).get_kernel_source()
    assert "tl::tma_load(" in src  # Q/K/V/delta_s ride TMA, which only the producer path selects
    assert "cp_async_gs" not in src  # ... so nothing is left on the cp.async path
    assert "tl::warpgroup_reg_dealloc<" in src and "tl::warpgroup_reg_alloc<" in src
    assert "__launch_bounds__(384, 1)" in src  # the second argument is what makes setmaxnreg bind
    assert "SM120MmaBlockScaledKind::kMxf4nvf4, 4, tl::SM120MmaScaleType::kUE4M3" in src
    assert "tl_cvt_e2m1x2_rn(" in src
    assert "tl_sts_u8(" in src and "tl_syncwarp()" in src


def test_sageattn3_ws_register_split_leaves_slack():
    """setmaxnreg.inc blocks forever if the two roles ask for the whole register file.

    A CPU-only guard on the defaults: 32/240 is exactly 65536 and hangs the kernel, so the
    product must stay strictly under it. Regression for a deadlock that costs hours to find.
    """
    import inspect

    sig = inspect.signature(ws.build_sm120_sageattn3_fwd)
    threads = sig.parameters["threads"].default
    producer_regs = sig.parameters["producer_regs"].default
    consumer_regs = sig.parameters["consumer_regs"].default
    consumer_threads = 256
    producer_threads = threads - consumer_threads
    total = producer_threads * producer_regs + consumer_threads * consumer_regs
    assert total <= 65536 - 1024, (total, producer_regs, consumer_regs)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sageattn3_ws_perf_smoke():
    """Runs and reports TOPS (no threshold)."""
    from tilelang.profiler import do_bench

    torch.manual_seed(0)
    b, h, n, d = 4, 32, 2048, 128
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    _, kargs, _ = prepare_inputs(q, k, v)
    kernel = ws.sm120_sageattn3_fwd(b, h, n, d)
    ms = do_bench(lambda: kernel(*kargs), warmup=10, rep=30, backend="cudagraph", return_mode="median")
    tops = 4.0 * b * h * n * n * d / (ms * 1e-3) / 1e12
    print(f"\nsageattn3 persistent-ws kernel-only B={b} H={h} N={n}: {ms:.4f} ms, {tops:.1f} TOPS")
    assert tops > 0


if __name__ == "__main__":
    tilelang.testing.main()
