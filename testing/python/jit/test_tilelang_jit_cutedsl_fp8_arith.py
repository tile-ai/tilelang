"""CuTeDSL FP8 (e4m3 / e5m2) elementwise arithmetic.

No PTX instruction computes on FP8 registers before sm_100a, so the CuTeDSL
backend widens to f16x2 in inline PTX. These tests pin the results to the
correctly rounded FP8 values for every operand pair of both formats.
"""

import numpy as np
import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

DTYPES = {"float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}
OPS = ["add", "sub", "mul", "div", "max", "min"]


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


def _binary_kernel(dtype, op, n, vec):
    def apply(a, b):
        return {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "div": lambda: a / b,
            "max": lambda: T.max(a, b),
            "min": lambda: T.min(a, b),
        }[op]()

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, 1024), threads=1024 // vec) as bx:
            for i in T.Parallel(1024 // vec):
                for v in T.vectorized(vec):
                    idx = bx * 1024 + i * vec + v
                    if idx < n:
                        C[idx] = apply(A[idx], B[idx])

    return main


def _clamp_kernel(dtype, n, vec):
    lo, hi = T.cast(-2.0, dtype), T.cast(3.0, dtype)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, 1024), threads=1024 // vec) as bx:
            for i in T.Parallel(1024 // vec):
                for v in T.vectorized(vec):
                    idx = bx * 1024 + i * vec + v
                    if idx < n:
                        C[idx] = T.clamp(A[idx], lo, hi)

    return main


def _quantize(values, dtype):
    """Round an exact float64 result to FP8 the way ``cvt.rn.satfinite`` does."""
    finite = torch.finfo(dtype).max
    nan = torch.isnan(values)
    out = values.clamp(-finite, finite).to(dtype)
    return torch.where(nan, torch.full_like(values, float("nan")).to(dtype), out)


def _all_pairs(dtype):
    values = torch.arange(256, dtype=torch.uint8).view(DTYPES[dtype]).cuda()
    a, b = torch.meshgrid(values, values, indexing="ij")
    return a.reshape(-1).contiguous(), b.reshape(-1).contiguous()


def _assert_bitwise_equal(got, ref):
    got_bits, ref_bits = got.view(torch.uint8), ref.view(torch.uint8)
    nan = torch.isnan(got.float()) & torch.isnan(ref.float())
    bad = (got_bits != ref_bits) & ~nan
    if bad.any():
        idx = bad.nonzero()[:8].flatten()
        raise AssertionError(f"FP8 result mismatch at {idx.tolist()}: got {got_bits[idx].tolist()} expected {ref_bits[idx].tolist()}")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize("vec", [1, 2, 4, 8])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_cutedsl_fp8_binary_exhaustive(dtype, op, vec):
    """Every operand pair of the format must give the correctly rounded FP8 result."""
    _require_cutedsl()
    a, b = _all_pairs(dtype)
    n = a.numel()
    out = torch.empty(n, dtype=DTYPES[dtype], device="cuda")
    tilelang.compile(_binary_kernel(dtype, op, n, vec), target="cutedsl")(a, b, out)

    exact = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "max": torch.maximum,
        "min": torch.minimum,
    }[op](a.double(), b.double())
    _assert_bitwise_equal(out, _quantize(exact, DTYPES[dtype]))


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize("vec", [1, 2, 4])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_cutedsl_fp8_clamp_exhaustive(dtype, vec):
    _require_cutedsl()
    a, _ = _all_pairs(dtype)
    n = a.numel()
    out = torch.empty(n, dtype=DTYPES[dtype], device="cuda")
    tilelang.compile(_clamp_kernel(dtype, n, vec), target="cutedsl")(a, out)
    _assert_bitwise_equal(out, _quantize(a.double().clamp(-2.0, 3.0), DTYPES[dtype]))


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_cutedsl_fp8_nan_propagates(dtype):
    """max/min/clamp propagate NaN like ``arith.maximumf``, and +0 beats -0 in max."""
    _require_cutedsl()
    nan = float("nan")
    pairs = np.array([[nan, 1.0], [1.0, nan], [nan, nan], [0.0, -0.0], [-0.0, -0.0], [1.0, 2.0]], dtype=np.float32)
    a = torch.tensor(pairs[:, 0], device="cuda").to(DTYPES[dtype])
    b = torch.tensor(pairs[:, 1], device="cuda").to(DTYPES[dtype])
    n = a.numel()
    for op in ("max", "min"):
        out = torch.empty(n, dtype=DTYPES[dtype], device="cuda")
        tilelang.compile(_binary_kernel(dtype, op, n, 1), target="cutedsl")(a, b, out)
        got = out.float()
        assert torch.isnan(got[:3]).all(), f"{op} dropped NaN: {got.tolist()}"
        reference = (torch.maximum if op == "max" else torch.minimum)(a.float(), b.float())
        torch.testing.assert_close(got[3:6], reference[3:6], atol=0, rtol=0)
        if op == "max":
            assert out.view(torch.uint8)[3].item() == 0x00, "max(+0, -0) must be +0"
        else:
            assert out.view(torch.uint8)[3].item() == 0x80, "min(+0, -0) must be -0"

    nan_in_clamp = torch.tensor([nan, 1.0, 1.0], device="cuda").to(DTYPES[dtype])
    out = torch.empty(nan_in_clamp.numel(), dtype=DTYPES[dtype], device="cuda")
    tilelang.compile(_clamp_kernel(dtype, nan_in_clamp.numel(), 1), target="cutedsl")(nan_in_clamp, out)
    assert torch.isnan(out.float()[0]), "clamp dropped a NaN input"
