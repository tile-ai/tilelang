"""End-to-end NPU tests for the Parallel->VMI path.

These kernels contain ``T.Parallel`` inside ``T.SimdVF(lanes=...)`` and run
through the real ``tilelang.compile(target="pto")`` pipeline, which routes
them to ``VerifyParallelToPTO`` + ``VectorizeParallelToPTO`` instead of
``AscendSimdVFLowerParallel``. Auto-schedule stays enabled: the non-autosched
pipeline has no InsertSync (MTE sync), so numeric results would be garbage.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import tilelang
import tilelang.ascend.language as T

from tilelang import tvm
from tilelang.engine.lower import lower


def _require_npu_runtime():
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu is unavailable in the current environment")

    try:
        _ = torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
    except Exception as err:
        pytest.skip(f"NPU runtime is unavailable in the current environment: {err}")


def _vl_padded(n, lanes):
    """Extent padded to whole VL chunks. Tail vloads read the full VL from
    their chunk base (loads are unmasked by ISA design; the reserved top 8K
    of UB keeps the extra read-only lanes fault-free), and the padded input
    makes this test's readable storage explicit. The Parallel extent and
    the compact output stay at ``n``."""
    return (n + lanes - 1) // lanes * lanes


def _vecadd(n, lanes):
    """C = A * (A + B) over a 1D Parallel, multi-chunk when n > lanes."""

    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32"), C: T.Buffer((n,), "float32")):
        with T.Kernel(1):
            t1 = T.alloc_shared((n,), "float32")
            t2 = T.alloc_shared((n,), "float32")
            t3 = T.alloc_shared((n,), "float32")
            T.copy(A, t1)
            T.copy(B, t2)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    t3[i] = t1[i] * (t1[i] + t2[i])
            T.copy(t3, C)

    return main


def _i32_index(n, lanes):
    """C[i] = A[i] + i: int32 loop index feeds the value stream, exercising the
    integer VCI signless bridge (vinterpret_cast) on the new path."""

    @T.prim_func
    def main(A: T.Buffer((n,), "int32"), C: T.Buffer((n,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int32")
            c = T.alloc_shared((n,), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = a[i] + i
            T.copy(c, C)

    return main


def _row_kernel(m, n, lanes):
    """2D j-continuous: c[i, j] = a[i, j] + i + j over T.Parallel(m, n)."""

    @T.prim_func
    def main(A: T.Buffer((m, n), "int32"), C: T.Buffer((m, n), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((m, n), "int32")
            c = T.alloc_shared((m, n), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(m, n):
                    c[i, j] = a[i, j] + i + j
            T.copy(c, C)

    return main


def _column_kernel(m, n, lanes):
    """2D i-continuous: buffers are (n, m) and accessed [j, i]."""

    @T.prim_func
    def main(A: T.Buffer((n, m), "int32"), C: T.Buffer((n, m), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((n, m), "int32")
            c = T.alloc_shared((n, m), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(m, n):
                    c[j, i] = a[j, i] + i + j
            T.copy(c, C)

    return main


def _cast_f16_muladd_1d(n, lanes):
    """1D widening cast: c[i] = f32(a[i]) * (f32(a[i]) + f32(b[i]))."""

    @T.prim_func
    def main(A: T.Buffer((n,), "float16"), B: T.Buffer((n,), "float16"), C: T.Buffer((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float16")
            b = T.alloc_shared((n,), "float16")
            c = T.alloc_shared((n,), "float32")
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.Cast("float32", a[i]) * (T.Cast("float32", a[i]) + T.Cast("float32", b[i]))
            T.copy(c, C)

    return main


def _cast_bf16_add_1d(n, lanes):
    """1D bf16 -> f32: c[i] = f32(a[i]) + f32(a[i])."""

    @T.prim_func
    def main(A: T.Buffer((n,), "bfloat16"), C: T.Buffer((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "bfloat16")
            c = T.alloc_shared((n,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.Cast("float32", a[i]) + T.Cast("float32", a[i])
            T.copy(c, C)

    return main


def _cast_i16_square_1d(n, lanes):
    """1D int16 -> int32 widening: c[i] = i32(a[i]) * i32(a[i])."""

    @T.prim_func
    def main(A: T.Buffer((n,), "int16"), C: T.Buffer((n,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int16")
            c = T.alloc_shared((n,), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.Cast("int32", a[i]) * T.Cast("int32", a[i])
            T.copy(c, C)

    return main


def _cast_f16_add_2d(m, n, lanes):
    """2D j-continuous widening cast: c[i, j] = f32(a[i, j]) + f32(b[i, j])."""

    @T.prim_func
    def main(A: T.Buffer((m, n), "float16"), B: T.Buffer((m, n), "float16"), C: T.Buffer((m, n), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((m, n), "float16")
            b = T.alloc_shared((m, n), "float16")
            c = T.alloc_shared((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(m, n):
                    c[i, j] = T.Cast("float32", a[i, j]) + T.Cast("float32", b[i, j])
            T.copy(c, C)

    return main


def _fp8_cast_1d(n, lanes):
    """Cast to fp8: c[i] = f8(a[i]). Outside the supported dtype set."""

    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), C: T.Buffer((n,), "float8_e4m3")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            c = T.alloc_shared((n,), "float8_e4m3")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.Cast("float8_e4m3", a[i])
            T.copy(c, C)

    return main


def _i8_add_1d(n, lanes):
    """8-bit control: c[i] = a[i] + b[i] over int8, a whitelisted category."""

    @T.prim_func
    def main(A: T.Buffer((n,), "int8"), B: T.Buffer((n,), "int8"), C: T.Buffer((n,), "int8")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int8")
            b = T.alloc_shared((n,), "int8")
            c = T.alloc_shared((n,), "int8")
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = a[i] + b[i]
            T.copy(c, C)

    return main


def _new_path_signatures(source):
    """Signatures proving the kernel took the new Parallel->VMI path."""
    return {
        "vload": source.count("pto.vmi.vload("),
        "vstore": source.count("pto.vmi.vstore("),
        "create_mask": source.count("pto.vmi.create_mask("),
        "vlds": source.count(".vlds("),
        "set_flag": source.count("set_flag"),
        "wait_flag": source.count("wait_flag"),
    }


def _assert_new_path(source, *, expect_vinterpret_cast=False, expect_scalar_cast=False, expect_vcvt_to=None):
    """The compiled PTODSL must come from VectorizeParallelToPTO: VMI ops with
    full mask, MTE sync from auto-schedule, and no legacy simd vlds."""
    sig = _new_path_signatures(source)
    assert sig["vload"] > 0 and sig["vstore"] > 0, f"no VMI memory ops in source:\n{source}"
    assert sig["create_mask"] > 0, f"no create_mask in source:\n{source}"
    assert sig["vlds"] == 0, f"legacy simd vlds leaked into source:\n{source}"
    assert sig["set_flag"] > 0 and sig["wait_flag"] > 0, (
        f"no MTE sync (set_flag/wait_flag) in source; auto-schedule must stay enabled for numeric correctness:\n{source}"
    )
    if expect_vinterpret_cast:
        assert "pto.vmi.vinterpret_cast(" in source, f"integer VCI bridge (vinterpret_cast) missing:\n{source}"
    if expect_scalar_cast:
        # PR269 emitted `scalar.cast(...)` here; the current PTOAS PTODSL surface
        # exposes the same cast as `pto.cast(...)`, so the bridge marker is the
        # cast nested directly in the reinterpreted VMI source.
        bridged = "vinterpret_cast(pto.vmi.vci(pto.cast(" in source or "vinterpret_cast(pto.vmi.vbrc(pto.cast(" in source
        assert bridged, f"dynamic integer bridge (pto.cast inside vinterpret_cast) missing:\n{source}"
    if expect_vcvt_to is not None:
        assert "pto.vmi.vcvt(" in source, f"element-wise conversion (vcvt) missing:\n{source}"
        assert f"to_dtype=pto.{expect_vcvt_to}" in source, f"vcvt target {expect_vcvt_to} missing:\n{source}"


@pytest.fixture()
def npu():
    _require_npu_runtime()
    return torch.device("npu")


def _run_and_check(func, inputs, expected, source_checks=None, atol=None):
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    source = kernel.get_kernel_source()
    if source_checks is not None:
        source_checks(source)
    out = kernel(*inputs)
    torch.npu.synchronize()
    got = out.cpu()
    if atol is None:
        assert torch.equal(got, expected), f"numeric mismatch: max_diff={torch.max(torch.abs((got.float() - expected.float()))).item()}"
    else:
        max_diff = torch.max(torch.abs(got.float() - expected.float())).item()
        assert torch.allclose(got.float(), expected.float(), rtol=0, atol=atol), f"numeric mismatch: max_diff={max_diff}"


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_vecadd_1d_multichunk(npu, lanes):
    """1D float32, n=512: multi-chunk (q loop) across all three lane widths."""
    n = 512
    a = torch.randn(n)
    b = torch.randn(n)
    _run_and_check(
        _vecadd(n, lanes),
        (a.to(npu), b.to(npu)),
        a * (a + b),
        source_checks=lambda s: _assert_new_path(s),
    )


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [128, 256])
def test_vecadd_1d_singlechunk(npu, lanes):
    """1D float32, n=lanes: single chunk, VCI base folds to a constant."""
    n = lanes
    a = torch.randn(n)
    b = torch.randn(n)
    _run_and_check(
        _vecadd(n, lanes),
        (a.to(npu), b.to(npu)),
        a * (a + b),
        source_checks=lambda s: _assert_new_path(s),
    )


@pytest.mark.pto
def test_i32_index_multichunk(npu):
    """1D int32 index in the value stream, multi-chunk: dynamic base bridge."""
    n = 512
    a = (torch.randn(n) * 1000).to(torch.int32)
    idx = torch.arange(n, dtype=torch.int32)
    _run_and_check(
        _i32_index(n, 128),
        (a.to(npu),),
        a + idx,
        source_checks=lambda s: _assert_new_path(s, expect_vinterpret_cast=True, expect_scalar_cast=True),
    )


@pytest.mark.pto
def test_i32_index_singlechunk(npu):
    """1D int32 index, single chunk: constant base goes through the bridge too."""
    n = 128
    a = (torch.randn(n) * 1000).to(torch.int32)
    idx = torch.arange(n, dtype=torch.int32)
    _run_and_check(
        _i32_index(n, 128),
        (a.to(npu),),
        a + idx,
        source_checks=lambda s: _assert_new_path(s, expect_vinterpret_cast=True),
    )


@pytest.mark.pto
def test_2d_row_continuous(npu):
    """2D j-continuous unit over T.Parallel(4, 256)."""
    m, n = 4, 256
    a = (torch.randn(m, n) * 100).to(torch.int32)
    i = torch.arange(m, dtype=torch.int32).unsqueeze(1)
    j = torch.arange(n, dtype=torch.int32).unsqueeze(0)
    _run_and_check(
        _row_kernel(m, n, 128),
        (a.to(npu),),
        a + i + j,
        source_checks=lambda s: _assert_new_path(s, expect_vinterpret_cast=True, expect_scalar_cast=True),
    )


@pytest.mark.pto
def test_2d_column_continuous(npu):
    """2D i-continuous unit over T.Parallel(256, 4)."""
    m, n = 256, 4
    a = (torch.randn(n, m) * 100).to(torch.int32)
    i = torch.arange(m, dtype=torch.int32).unsqueeze(0)
    j = torch.arange(n, dtype=torch.int32).unsqueeze(1)
    _run_and_check(
        _column_kernel(m, n, 128),
        (a.to(npu),),
        a + j + i,
        source_checks=lambda s: _assert_new_path(s, expect_vinterpret_cast=True, expect_scalar_cast=True),
    )


@pytest.mark.pto
def test_cast_f16_to_f32_1d(npu):
    """1D f16 inputs widened with vcvt before the element-wise mul-add."""
    n = 256
    a = torch.randn(n).half()
    b = torch.randn(n).half()
    _run_and_check(
        _cast_f16_muladd_1d(n, 128),
        (a.to(npu), b.to(npu)),
        a.float() * (a.float() + b.float()),
        source_checks=lambda s: _assert_new_path(s, expect_vcvt_to="f32"),
        atol=1e-3,
    )


@pytest.mark.pto
def test_cast_bf16_to_f32_1d(npu):
    """1D bf16 widened to f32; bf16 does not satisfy ``is_float()`` here, so the
    widening rule must whitelist it explicitly."""
    n = 256
    a = torch.randn(n).bfloat16()
    _run_and_check(
        _cast_bf16_add_1d(n, 128),
        (a.to(npu),),
        a.float() + a.float(),
        source_checks=lambda s: _assert_new_path(s, expect_vcvt_to="f32"),
        atol=1e-2,
    )


@pytest.mark.pto
def test_cast_i16_to_i32_1d(npu):
    """1D int16 -> int32 widening: exact integer arithmetic after vcvt."""
    n = 256
    a = torch.randint(-30000, 30000, (n,), dtype=torch.int16)
    _run_and_check(
        _cast_i16_square_1d(n, 128),
        (a.to(npu),),
        a.int() * a.int(),
        source_checks=lambda s: _assert_new_path(s, expect_vcvt_to="si32"),
    )


@pytest.mark.pto
def test_cast_f16_to_f32_2d(npu):
    """2D j-continuous f16 -> f32 widening over T.Parallel(4, 256)."""
    m, n = 4, 256
    a = torch.randn(m, n).half()
    b = torch.randn(m, n).half()
    _run_and_check(
        _cast_f16_add_2d(m, n, 128),
        (a.to(npu), b.to(npu)),
        a.float() + b.float(),
        source_checks=lambda s: _assert_new_path(s, expect_vcvt_to="f32"),
        atol=1e-3,
    )


@pytest.mark.pto
def test_fp8_element_type_rejected():
    """fp8 is not supported, and the reason is the element
    *category*, not the bit width: fp8 carries its own dtype code, so
    ``is_float()`` is false for it and ``IsSupportedElementDType``'s category
    whitelist (float / bfloat16 / int / uint / bool) does not admit it. Verify
    rejects the buffer at the dtype check, before any Cast lowering is planned;
    the same holds when an fp8 buffer is a load source rather than a cast
    target. Supporting fp8 needs its rounding and saturation semantics plus
    backend support validated first — a whitelist entry alone is not enough.
    See ``test_int8_control_compiles`` for the same bit width inside the
    whitelisted categories."""
    with pytest.raises(Exception) as exc:
        tilelang.compile(_fp8_cast_1d(128, 128), target="pto")
    message = str(exc.value)
    assert "[VerifyParallelToPTO] dtype:" in message, message
    assert "float8_e4m3" in message, message


@pytest.mark.pto
def test_int8_control_compiles():
    """8-bit control: int8 sits in a whitelisted category, so the same width
    lowers through the new path — which is what makes the fp8 rejection above a
    category question rather than a sub-byte/width one."""
    tilelang.compile(_i8_add_1d(128, 128), target="pto")


@pytest.mark.pto
def test_lane_varying_f16_widening_numeric(npu):
    """Reading the half value per lane makes it a vector conversion, and the
    widening is exact."""
    n = 256

    @T.prim_func
    def main(A: T.Buffer((n,), "float16"), C: T.Buffer((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float16")
            c = T.alloc_shared((n,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(n):
                    c[i] = T.cast(a[i], "float32")
            T.copy(c, C)

    x = (torch.arange(n, dtype=torch.float32) - 96.0) / 8.0
    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    assert "pto.vmi.vcvt(" in kernel.get_kernel_source()
    out = kernel(x.to(torch.float16).to(npu))
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), x.to(torch.float16).float())


@pytest.mark.pto
def test_lane_uniform_int_cast_numeric(npu):
    """Integer scalar cast: int32 -> int16 with a lane-uniform source puts
    int16(a[0]) in every lane."""
    n = 256

    @T.prim_func
    def main(A: T.Buffer((n,), "int32"), C: T.Buffer((n,), "int16")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int32")
            c = T.alloc_shared((n,), "int16")
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(n):
                    c[i] = T.cast(a[0], "int16")
            T.copy(c, C)

    x = torch.arange(n, dtype=torch.int32) - 96
    out = tilelang.compile(main, target="pto", out_idx=-1)(x.to(npu))
    torch.npu.synchronize()
    expected = torch.full((n,), int(x[0].item()), dtype=torch.int16)
    assert torch.equal(out.cpu(), expected), (out.cpu()[:4], expected[:4])


@pytest.mark.pto
def _sub_1d(n, lanes):
    p = _vl_padded(n, lanes)

    @T.prim_func
    def main(A: T.Tensor((p,), "float32"), B: T.Tensor((p,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((p,), "float32")
            b_ub = T.alloc_shared((p,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] - b_ub[i] - 1.0
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128])
def test_sub_1d_multichunk(npu, lanes):
    _require_npu_runtime()
    n = 300
    p = _vl_padded(n, lanes)
    a = torch.randn(p, device=npu)
    b = torch.randn(p, device=npu)
    kernel = tilelang.compile(_sub_1d(n, lanes), target="pto", out_idx=-1)
    out = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a[:n] - b[:n] - 1.0, rtol=1e-5, atol=1e-5)


def _div_1d(n, lanes, dtype):
    p = _vl_padded(n, lanes)

    @T.prim_func
    def main(A: T.Tensor((p,), dtype), B: T.Tensor((p,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((p,), dtype)
            b_ub = T.alloc_shared((p,), dtype)
            c_ub = T.alloc_shared((n,), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] / (b_ub[i] + T.cast(2, dtype))
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_div_f32_1d(npu):
    _require_npu_runtime()
    n = 256
    a = torch.randn(n, device=npu)
    b = torch.randn(n, device=npu)
    kernel = tilelang.compile(_div_1d(n, 128, "float32"), target="pto", out_idx=-1)
    out = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a / (b + 2.0), rtol=1e-4, atol=1e-4)


def _unary_1d(n, lanes, kind):
    p = _vl_padded(n, lanes)

    @T.prim_func
    def main(A: T.Tensor((p,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((p,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    if kind == 0:
                        c_ub[i] = T.abs(a_ub[i] - 0.5)
                    elif kind == 1:
                        c_ub[i] = T.exp(a_ub[i])
                    elif kind == 2:
                        c_ub[i] = T.log(T.abs(a_ub[i]) + 1.0)
                    else:
                        c_ub[i] = T.sqrt(T.abs(a_ub[i]) + 0.25)
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", [0, 1, 2, 3])
def test_unary_1d(npu, kind):
    _require_npu_runtime()
    n = 256
    a = torch.randn(n, device=npu)
    kernel = tilelang.compile(_unary_1d(n, 128, kind), target="pto", out_idx=-1)
    out = kernel(a)
    torch.npu.synchronize()
    if kind == 0:
        ref = (a - 0.5).abs()
    elif kind == 1:
        ref = a.exp()
    elif kind == 2:
        ref = (a.abs() + 1.0).log()
    else:
        ref = (a.abs() + 0.25).sqrt()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def _f32_narrow(n, lanes, to_dtype):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), to_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), to_dtype)
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c_ub[i] = T.cast(a_ub[i] * 0.5 - 0.25, to_dtype)
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("to_dtype", ["float16", "bfloat16"])
def test_f32_to_fp16_bf16_narrow(npu, to_dtype):
    _require_npu_runtime()
    n = 256
    a = torch.randn(n, device=npu) * 4.0
    kernel = tilelang.compile(_f32_narrow(n, 128, to_dtype), target="pto", out_idx=-1)
    out = kernel(a)
    torch.npu.synchronize()
    ref = (a * 0.5 - 0.25).to(getattr(torch, to_dtype))
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-3)


def test_pipeline_routes_to_new_passes(npu):
    """Pass-instrument check: the pto pipeline must run Verify/Vectorize and
    must NOT run AscendSimdVFLowerParallel; auto-schedule must inject sync."""

    names = []

    @tvm.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            names.append(info.name.split(".")[-1])

    with tvm.transform.PassContext(opt_level=3, instruments=[Capture()]):
        lower(_vecadd(512, 128), target="pto")

    assert "VerifyParallelToPTO" in names, f"VerifyParallelToPTO not in pipeline: {names}"
    assert "VectorizeParallelToPTO" in names, f"VectorizeParallelToPTO not in pipeline: {names}"
    assert "AscendSimdVFLowerParallel" not in names, f"legacy AscendC pass ran on target=pto: {names}"
    assert "InsertSync" in names, f"InsertSync not in pipeline (auto-schedule off?): {names}"


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()


# ---------------------------------------------------------------------------
# Dtype-matrix positives and tail coverage
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_div_f16_1d(npu):
    _require_npu_runtime()
    n = 256
    a = torch.randn(n, device=npu).half()
    b = (torch.randn(n, device=npu) + 2.5).half()
    kernel = tilelang.compile(_div_1d(n, 128, "float16"), target="pto", out_idx=-1)
    out = kernel(a, b)
    torch.npu.synchronize()
    # The device adds the constant in f16: the denominator rounds to f16
    # *before* the division. Mirror that in the reference.
    denominator = (b.float() + 2.0).half()
    ref = (a.float() / denominator.float()).half()
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.pto
def test_sub_bf16_1d(npu):
    _require_npu_runtime()
    n = 256
    a = torch.randn(n, device=npu).bfloat16()
    b = torch.randn(n, device=npu).bfloat16()
    kernel = tilelang.compile(_sub_1d_dtype(n, 128, "bfloat16"), target="pto", out_idx=-1)
    out = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.float(), (a.float() - b.float()), rtol=1e-2, atol=1e-2)


@pytest.mark.pto
def test_sub_i32_1d(npu):
    _require_npu_runtime()
    n = 256
    a = torch.randint(-1000, 1000, (n,), device=npu, dtype=torch.int32)
    b = torch.randint(-1000, 1000, (n,), device=npu, dtype=torch.int32)
    kernel = tilelang.compile(_sub_1d_dtype(n, 128, "int32"), target="pto", out_idx=-1)
    out = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a - b)


@pytest.mark.pto
def test_div_f32_tail(npu):
    _require_npu_runtime()
    n = 300
    p = _vl_padded(n, 128)
    a = torch.randn(p, device=npu)
    b = torch.randn(p, device=npu)
    kernel = tilelang.compile(_div_1d(n, 128, "float32"), target="pto", out_idx=-1)
    out = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a[:n] / (b[:n] + 2.0), rtol=1e-4, atol=1e-4)


@pytest.mark.pto
def test_unary_f32_tail(npu):
    _require_npu_runtime()
    n = 300
    p = _vl_padded(n, 128)
    a = torch.randn(p, device=npu)
    kernel = tilelang.compile(_unary_1d(n, 128, 1), target="pto", out_idx=-1)
    out = kernel(a)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a[:n].exp(), rtol=1e-4, atol=1e-4)


def _unary_f16_kernel(n, lanes):
    @T.prim_func
    def main(A: T.Tensor((n,), "float16"), C: T.Tensor((n,), "float16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float16")
            c_ub = T.alloc_shared((n,), "float16")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c_ub[i] = T.abs(a_ub[i] - T.cast(0.5, "float16"))
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_unary_f16_abs(npu):
    _require_npu_runtime()
    n = 256
    a = torch.randn(n, device=npu).half()
    kernel = tilelang.compile(_unary_f16_kernel(n, 128), target="pto", out_idx=-1)
    out = kernel(a)
    torch.npu.synchronize()
    ref = (a.float() - 0.5).abs().half()
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-3)


def _sub_1d_dtype(n, lanes, dtype):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            b_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] - b_ub[i]
            T.copy(c_ub, C)

    return main


# ---------------------------------------------------------------------------
# Narrowing-cast attribute/boundary coverage, f16 unary
# ---------------------------------------------------------------------------


def _narrow_kernel(n, lanes, to_dtype):
    p = _vl_padded(n, lanes)

    @T.prim_func
    def main(A: T.Tensor((p,), "float32"), C: T.Tensor((n,), to_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((p,), "float32")
            c_ub = T.alloc_shared((n,), to_dtype)
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c_ub[i] = T.cast(a_ub[i], to_dtype)
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_narrow_cast_attributes(npu):
    kernel = tilelang.compile(_narrow_kernel(256, 128, "float16"), target="pto", out_idx=-1)
    line = [ln.strip() for ln in kernel.get_kernel_source().split("\n") if "vmi.vcvt(" in ln and "=" in ln][0]
    assert "to_dtype=pto.f16" in line, line
    assert 'rounding="R"' in line, line
    assert 'saturate="NOSAT"' in line, line


@pytest.mark.pto
def test_narrow_cast_f16_boundary_values(npu):
    _require_npu_runtime()
    # f16: 2^-14 is the smallest normal; 2049/2051 are ties-to-even
    # midpoints; 65519.99 rounds to 65504 (largest finite) under R; 65520
    # overflows to +inf under NOSAT.
    values = [
        0.0,
        -0.0,
        5.960464477539063e-08,  # smallest f16 subnormal (2^-24)
        6.097555160522461e-05,  # mid-subnormal, nonzero
        6.103515625e-05,  # 2^-14, smallest f16 normal
        2049.0,  # tie between 2048 and 2052 -> even (2048)
        2051.0,  # tie between 2048 and 2052 -> even (2052)
        65504.0,  # largest finite
        65519.99,  # rounds to 65504
        65520.0,  # overflow -> +inf (NOSAT)
        -65520.0,  # overflow -> -inf (NOSAT)
    ]
    n = 256
    x = torch.zeros(n, dtype=torch.float32)
    x[: len(values)] = torch.tensor(values, dtype=torch.float32)
    kernel = tilelang.compile(_narrow_kernel(n, 128, "float16"), target="pto", out_idx=-1)
    out = kernel(x.to(npu))
    torch.npu.synchronize()
    got = out.float()[: len(values)].cpu()

    ref = x[: len(values)].half().float()
    torch.testing.assert_close(got, ref, rtol=0, atol=0, equal_nan=True)
    assert got[2] != 0.0 and got[3] != 0.0, "f16 subnormals must not flush to zero"
    assert got[1] == 0.0 and torch.signbit(got[1]), "negative zero must keep the sign bit"
    assert torch.isinf(got[9]) and got[9] > 0, "NOSAT f16 overflow must be +inf"
    assert torch.isinf(got[10]) and got[10] < 0, "NOSAT f16 negative overflow must be -inf"


@pytest.mark.pto
def test_narrow_cast_bf16_boundary_values(npu):
    _require_npu_runtime()
    # bf16: 2^-126 is the smallest normal; 7-bit mantissa means ties at
    # half-ULP of the 128-spaced binade. f32 sources cannot exceed bf16
    # max finite (~3.39e38 vs f32 max ~3.40e38), so no bf16 overflow case.
    values = [
        0.0,
        -0.0,
        9.183549615799121e-41,  # smallest bf16 subnormal (2^-133)
        1.1754943508222875e-38,  # 2^-126, smallest bf16 normal
        1.9921875,  # tie -> rounds to 2.0 (even)
        2.0078125,  # tie -> rounds to 2.0 (even)
        3.3895313892515355e38,  # near bf16 max finite
    ]
    n = 256
    x = torch.zeros(n, dtype=torch.float32)
    x[: len(values)] = torch.tensor(values, dtype=torch.float32)
    kernel = tilelang.compile(_narrow_kernel(n, 128, "bfloat16"), target="pto", out_idx=-1)
    out = kernel(x.to(npu))
    torch.npu.synchronize()
    got = out.float()[: len(values)].cpu()

    ref = x[: len(values)].bfloat16().float()
    torch.testing.assert_close(got, ref, rtol=0, atol=0, equal_nan=True)
    assert got[2] != 0.0, "bf16 subnormal must not flush to zero"
    assert got[1] == 0.0 and torch.signbit(got[1]), "negative zero must keep the sign bit"


@pytest.mark.pto
def test_narrow_cast_bf16_attributes(npu):
    kernel = tilelang.compile(_narrow_kernel(256, 128, "bfloat16"), target="pto", out_idx=-1)
    line = [ln.strip() for ln in kernel.get_kernel_source().split("\n") if "vmi.vcvt(" in ln and "=" in ln][0]
    assert "to_dtype=pto.bf16" in line, line
    assert 'rounding="R"' in line, line
    assert 'saturate="NOSAT"' in line, line


@pytest.mark.pto
def test_narrow_cast_bf16_tail(npu):
    _require_npu_runtime()
    n = 300
    p = _vl_padded(n, 128)
    a = torch.randn(p, device=npu) * 1000
    kernel = tilelang.compile(_narrow_kernel(n, 128, "bfloat16"), target="pto", out_idx=-1)
    out = kernel(a)
    torch.npu.synchronize()
    torch.testing.assert_close(out.float(), a[:n].bfloat16().float(), rtol=1e-3, atol=1e-3)


def _unary_f16_math_kernel(n, lanes, kind):
    @T.prim_func
    def main(A: T.Tensor((n,), "float16"), C: T.Tensor((n,), "float16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float16")
            c_ub = T.alloc_shared((n,), "float16")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    if kind == 0:
                        c_ub[i] = T.exp(a_ub[i])
                    elif kind == 1:
                        c_ub[i] = T.log(T.abs(a_ub[i]) + T.cast(1.0, "float16"))
                    else:
                        c_ub[i] = T.sqrt(T.abs(a_ub[i]) + T.cast(0.25, "float16"))
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_unary_f16_math(npu, kind):
    _require_npu_runtime()
    n = 256
    a = (torch.randn(n, device=npu) * 2).half()
    kernel = tilelang.compile(_unary_f16_math_kernel(n, 128, kind), target="pto", out_idx=-1)
    out = kernel(a)
    torch.npu.synchronize()
    if kind == 0:
        ref = a.exp()
    elif kind == 1:
        ref = (a.abs() + 1.0).log()
    else:
        ref = (a.abs() + 0.25).sqrt()
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
