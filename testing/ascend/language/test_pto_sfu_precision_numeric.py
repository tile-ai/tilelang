"""Numerical validation of PTO SFU precision wrappers on the NPU.

Covers the software wrappers selected by the per-op ``precision=`` kwarg
(the per-op annotation path wired up in codegen_pto.cc):

- ``tl.vdiv_precise_f32``     (vdiv  precision="exact",     0 ulp)
- ``tl.vexp_1ulp_ftz_false``  (vexp  precision="ftz_false", 1 ulp, subnormal outputs)
- ``tl.vln_1ulp_ftz_false``   (vln   precision="ftz_false", 1 ulp, subnormal inputs)
- ``tl.vsqrt_0ulp_ftz_false`` (vsqrt precision="ftz_false", 0 ulp, subnormal inputs)

References are computed on CPU: exp/log in float64 then rounded to float32
(1 double-rounding, compared within 1 ulp), division/sqrt directly in
float32 (IEEE correctly rounded, compared bitwise).
"""

import struct

import numpy as np
import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing

VEC = 64


@tilelang.jit(out_idx=[1], target="pto")
def sfu_vexp_ftz_false(N: int):
    @T.prim_func
    def main(X: T.Tensor((N,), T.float32), Y: T.Tensor((N,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), T.float32)
            y_ub = T.alloc_shared((N,), T.float32)
            T.copy(X, x_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                for i in range(N // VEC):
                    x = T.simd.vld(x_ub[i * VEC])
                    y = T.simd.vexp(x, full, precision="ftz_false")
                    T.simd.vsts(y_ub[i * VEC], y, full)
            T.copy(y_ub, Y)

    return main


@tilelang.jit(out_idx=[1], target="pto")
def sfu_vln_ftz_false(N: int):
    @T.prim_func
    def main(X: T.Tensor((N,), T.float32), Y: T.Tensor((N,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), T.float32)
            y_ub = T.alloc_shared((N,), T.float32)
            T.copy(X, x_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                for i in range(N // VEC):
                    x = T.simd.vld(x_ub[i * VEC])
                    y = T.simd.vln(x, full, precision="ftz_false")
                    T.simd.vsts(y_ub[i * VEC], y, full)
            T.copy(y_ub, Y)

    return main


@tilelang.jit(out_idx=[1], target="pto")
def sfu_vsqrt_ftz_false(N: int):
    @T.prim_func
    def main(X: T.Tensor((N,), T.float32), Y: T.Tensor((N,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), T.float32)
            y_ub = T.alloc_shared((N,), T.float32)
            T.copy(X, x_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                for i in range(N // VEC):
                    x = T.simd.vld(x_ub[i * VEC])
                    y = T.simd.vsqrt(x, full, precision="ftz_false")
                    T.simd.vsts(y_ub[i * VEC], y, full)
            T.copy(y_ub, Y)

    return main


@tilelang.jit(out_idx=[2], target="pto")
def sfu_vdiv_exact(N: int):
    @T.prim_func
    def main(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
        Y: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), T.float32)
            b_ub = T.alloc_shared((N,), T.float32)
            y_ub = T.alloc_shared((N,), T.float32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                for i in range(N // VEC):
                    a = T.simd.vld(a_ub[i * VEC])
                    b = T.simd.vld(b_ub[i * VEC])
                    y = T.simd.vdiv(a, b, full, precision="exact")
                    T.simd.vsts(y_ub[i * VEC], y, full)
            T.copy(y_ub, Y)

    return main


def _bits_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _ulp_distance(got: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    gb = got.view(torch.int32).to(torch.int64)
    rb = ref.view(torch.int32).to(torch.int64)
    same_sign = (gb < 0) == (rb < 0)
    dist = (gb - rb).abs()
    # Sign flips are only legitimate between equal-valued zeros.
    both_zero = (got == 0) & (ref == 0) & (gb == rb)
    return torch.where(same_sign | both_zero, dist, torch.full_like(dist, 1 << 30))


def assert_close_ulp(got, ref, max_ulp, context):
    got = got.detach().cpu().to(torch.float32)
    ref = ref.detach().cpu().to(torch.float32)
    assert got.shape == ref.shape
    nan_pair = torch.isnan(got) & torch.isnan(ref)
    inf_equal = torch.isinf(got) & (got == ref)
    finite_ok = torch.isfinite(got) & torch.isfinite(ref) & (_ulp_distance(got, ref) <= max_ulp)
    ok = nan_pair | inf_equal | finite_ok
    if not bool(ok.all()):
        bad = (~ok).nonzero().flatten()
        for i in bad[:8].tolist():
            print(
                f"{context}[{i}]: got={got[i].item()!r} ({got[i].view(torch.int32).item():#010x})"
                f" ref={ref[i].item()!r} ({ref[i].view(torch.int32).item():#010x})"
            )
    assert bool(ok.all()), f"{context}: {int((~ok).sum())}/{got.numel()} mismatches (max_ulp={max_ulp})"


def _pad64(values):
    values = list(values)
    assert len(values) > 0
    while len(values) % VEC != 0:
        values.append(values[len(values) % len(values)])
    return np.asarray(values, dtype=np.float32)


def _run_kernel(kernel, *inputs):
    outputs = kernel(*inputs)
    torch.npu.synchronize()
    return outputs


def _make_inputs(values, device="npu"):
    return torch.tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32, device=device)


@pytest.mark.pto
def test_pto_vexp_ftz_false_numeric():
    # Normal outputs, subnormal outputs (x in (-104, -87.34)), and underflow
    # to +0 (e^x < 2^-150), where the wrapper's (e^(x/2))^2 path must agree
    # with the correctly-rounded reference.
    values = (
        list(np.linspace(-80.0, 10.0, 64))
        + list(np.linspace(-104.0, -87.5, 64))
        + [-104.5, -120.0, -149.0, -174.0, -175.0, -200.0, -240.0, 0.0]
    )
    x = _make_inputs(_pad64(values))
    y = _run_kernel(sfu_vexp_ftz_false(x.numel()), x)
    ref = np.exp(x.detach().cpu().numpy().astype(np.float64)).astype(np.float32)
    assert_close_ulp(y, torch.from_numpy(ref), max_ulp=1, context="vexp ftz_false")


@pytest.mark.pto
def test_pto_vln_ftz_false_numeric():
    values = (
        # Subnormal inputs: min/largest subnormal, and interior patterns.
        [_bits_to_f32(b) for b in (0x00000001, 0x00001234, 0x00400000, 0x007FFFFF)]
        + [2.0**-127, 2.0**-140, 2.0**-149]
        # Normal inputs spanning the f32 exponent range.
        + list(np.geomspace(1e-30, 1e30, 48))
        + [1.0, 2.0, float(np.e), 0.5, 0.0, -1.0, -0.0]
    )
    x = _make_inputs(_pad64(values))
    y = _run_kernel(sfu_vln_ftz_false(x.numel()), x)
    with np.errstate(all="ignore"):
        ref = np.log(x.detach().cpu().numpy().astype(np.float64)).astype(np.float32)
    assert_close_ulp(y, torch.from_numpy(ref), max_ulp=1, context="vln ftz_false")


@pytest.mark.pto
def test_pto_vsqrt_ftz_false_numeric():
    values = (
        # Subnormal inputs, including the largest subnormal 0x007fffff that
        # the C++ reference implementation specifically guards.
        [_bits_to_f32(b) for b in (0x00000001, 0x00001234, 0x00400000, 0x007FFFFF)]
        + [2.0**-127, 2.0**-140]
        # (0, 1) region scaled up by the FAST_INVERSE path.
        + list(np.geomspace(2.0**-126, 1.0, 32))
        + list(np.geomspace(1.0, 1e30, 24))
        # +-0 and +inf pass through bitwise.
        + [0.0, -0.0, float("inf"), 1.0, 4.0, 0.25, 0.5, 1e-10, 2.0**-50, 2.0**-100]
    )
    x = _make_inputs(_pad64(values))
    y = _run_kernel(sfu_vsqrt_ftz_false(x.numel()), x)
    ref = np.sqrt(x.detach().cpu().numpy()).astype(np.float32)
    assert_close_ulp(y, torch.from_numpy(ref), max_ulp=0, context="vsqrt ftz_false")


@pytest.mark.pto
def test_pto_vdiv_exact_numeric():
    rng = np.random.default_rng(0)
    n = 3 * VEC
    a_vals = rng.standard_normal(n).astype(np.float32) * np.float32(10.0 ** rng.uniform(-20, 20, n).astype(np.float32))
    b_vals = rng.standard_normal(n).astype(np.float32) * np.float32(10.0 ** rng.uniform(-20, 20, n).astype(np.float32))
    # Special values: +/-0, inf, NaN operands exercise the wrapper's
    # special-value passthrough.
    a_vals[0:8] = [1.0, -1.0, 0.0, 0.0, 2.0, float("inf"), float("inf"), float("nan")]
    b_vals[0:8] = [0.0, 0.0, 0.0, 2.0, float("inf"), 2.0, float("inf"), 1.0]
    a = _make_inputs(a_vals)
    b = _make_inputs(b_vals)
    y = _run_kernel(sfu_vdiv_exact(a.numel()), a, b)
    with np.errstate(all="ignore"):
        ref = a.detach().cpu().numpy() / b.detach().cpu().numpy()
    assert_close_ulp(y, torch.from_numpy(ref), max_ulp=0, context="vdiv exact")


if __name__ == "__main__":
    tilelang.testing.main()
