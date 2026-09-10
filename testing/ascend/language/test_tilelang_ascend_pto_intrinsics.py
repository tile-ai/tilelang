"""Idiomatic PTO VMI twin of test_tilelang_ascend_intrinsics.py (SimdVF smoke).

ASC file mixes: (1) ASC sync/pipe/codegen string checks, (2) a broad SimdVF
op smoke, (3) packing / dist-mode / pad_value host details.

This twin covers the **algorithm purposes** of (2)/(3) that PTO VMI owns:
arith + compare/select, reduce+broadcast, vintlv/vdintlv rearrange, cast.
ASC-only host paths (PipeBarrier, CrossCore*, pad_value, E2B, vsstb
POST_UPDATE, vld2 dist tokens) are intentionally omitted — not PTO VMI ISA.
"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


VL = 64
N = 128


def simulator_safe_randn(shape, *, dtype=torch.float32, device="npu"):
    return torch.randn(shape, dtype=dtype, device="cpu").to(device)


def arith_select_kernel():
    """E2E: load → clamp-ish arith → compare/select → store."""

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            b_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                full = T.vmi.create_mask(VL, size=VL)
                one = T.vmi.vbrc(T.float32(1), size=VL)
                two = T.vmi.vbrc(T.float32(2), size=VL)
                for i in range(N // VL):
                    a = T.vmi.vload(a_ub[i * VL], size=VL)
                    b = T.vmi.vload(b_ub[i * VL], size=VL)
                    x = T.vmi.vadd(a, b, full)
                    x = T.vmi.vsub(x, one, full)
                    x = T.vmi.vmul(x, two, full)
                    x = T.vmi.vmax(x, b, full)
                    x = T.vmi.vmin(x, a, full)
                    x = T.vmi.vabs(x, full)
                    x = T.vmi.vrelu(x, full)
                    x = T.vmi.vmaxs(x, T.float32(0.25), full)
                    x = T.vmi.vmins(x, T.float32(8.0), full)
                    ge = T.vmi.vcmp(a, b, full, "ge")
                    gt0 = T.vmi.vcmps(a, T.float32(0), full, "gt")
                    # Keep x where (a>=b) else use b — seed of gt0 unused as AND;
                    # compose via nested sel: where ge keep x else b.
                    x = T.vmi.vsel(ge, x, b)
                    x = T.vmi.vsel(gt0, x, T.vmi.vbrc(T.float32(0), size=VL))
                    T.vmi.vstore(x, c_ub[i * VL], full)
            T.copy(c_ub, C)

    return main


def reduce_broadcast_kernel():
    """E2E: per-tile max-reduce, broadcast, add into output (ASC vcmax+vdupv)."""

    @T.prim_func
    def main(A: T.Buffer((N,), "float32"), C: T.Buffer((N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                full = T.vmi.create_mask(VL, size=VL)
                for i in range(N // VL):
                    a = T.vmi.vload(a_ub[i * VL], size=VL)
                    mx = T.vmi.vcmax(a, full, group=1)
                    mx_brc = T.vmi.vbrc(mx, size=VL)
                    T.vmi.vstore(T.vmi.vadd(a, mx_brc, full), c_ub[i * VL], full)
            T.copy(c_ub, C)

    return main


def intlv_roundtrip_kernel():
    """E2E: vintlv → vdintlv recovers originals (ASC rearrange smoke)."""

    @T.prim_func
    def main(
        X: T.Buffer((N,), "float32"),
        Y: T.Buffer((N,), "float32"),
        X_OUT: T.Buffer((N,), "float32"),
        Y_OUT: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), "float32")
            y_ub = T.alloc_shared((N,), "float32")
            xo_ub = T.alloc_shared((N,), "float32")
            yo_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            T.copy(Y, y_ub)
            with T.SimdVF():
                full = T.vmi.create_mask(VL, size=VL)
                for i in range(N // VL):
                    x = T.vmi.vload(x_ub[i * VL], size=VL)
                    y = T.vmi.vload(y_ub[i * VL], size=VL)
                    a0, a1 = T.vmi.vintlv(x, y, full)
                    xb, yb = T.vmi.vdintlv(a0, a1, full)
                    T.vmi.vstore(xb, xo_ub[i * VL], full)
                    T.vmi.vstore(yb, yo_ub[i * VL], full)
            T.copy(xo_ub, X_OUT)
            T.copy(yo_ub, Y_OUT)

    return main


def cast_roundtrip_kernel():
    """E2E: f32 → f16 → f32 continuous cast (ASC UNPK/PK purpose, PTO VMI idiom)."""

    @T.prim_func
    def main(A: T.Buffer((N,), "float32"), C: T.Buffer((N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                full = T.vmi.create_mask(VL, size=VL)
                for i in range(N // VL):
                    x = T.vmi.vload(a_ub[i * VL], size=VL)
                    # In-register roundtrip; dedicated cast suite covers UB stores.
                    h = T.vmi.vcvt(x, "float16")
                    T.vmi.vstore(T.vmi.vcvt(h, "float32"), c_ub[i * VL], full)
            T.copy(c_ub, C)

    return main


def pairwise_sum_kernel():
    """E2E: pairwise adjacent sum via vdintlv+vadd (ASC vcpadd purpose)."""

    @T.prim_func
    def main(A: T.Buffer((64,), "float32"), B: T.Buffer((32,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                full = T.vmi.create_mask(VL, size=VL)
                src = T.vmi.vload(a_ub[0], size=VL)
                zeros = T.vmi.vbrc(T.float32(0), size=VL)
                even, odd = T.vmi.vdintlv(src, zeros, full)
                low32 = T.vmi.create_mask(32, size=VL)
                T.vmi.vstore(T.vmi.vadd(even, odd, full), b_ub[0], low32)
            T.copy(b_ub[:32], B)

    return main


def vdup_float_to_int_kernel():
    """PTO physical SIMD vdup followed by f32->s32 conversion."""

    @T.prim_func
    def main(OUT: T.Buffer((128,), "int32")):
        with T.Kernel(1):
            out_ub = T.alloc_shared((128,), "int32")
            with T.SimdVF():
                mask = T.simd.pset(32)
                positive = T.simd.vdup(T.float32(1.5), "int32", mask)
                negative = T.simd.vdup(T.float32(-1.5), "int32", mask)
                T.simd.vsts(out_ub[0], positive, mask)
                T.simd.vsts(out_ub[VL], negative, mask)
            T.copy(out_ub, OUT)

    return main


def vdup_int_to_float_kernel():
    """PTO physical SIMD vdup followed by s32->f32 conversion."""

    @T.prim_func
    def main(OUT: T.Buffer((VL,), "float32")):
        with T.Kernel(1):
            out_ub = T.alloc_shared((VL,), "float32")
            with T.SimdVF():
                mask = T.simd.pset(32)
                value = T.simd.vdup(T.int32(7), "float32", mask)
                T.simd.vsts(out_ub[0], value, mask)
            T.copy(out_ub, OUT)

    return main


@pytest.mark.pto
def test_pto_arith_select_pipeline():
    kernel = tilelang.compile(arith_select_kernel(), target="pto", out_idx=-1)
    torch.manual_seed(0)
    a = simulator_safe_randn(N)
    b = simulator_safe_randn(N)
    out = kernel(a, b)
    torch.npu.synchronize()

    a_c, b_c = a.cpu(), b.cpu()
    x = (a_c + b_c - 1.0) * 2.0
    x = torch.maximum(x, b_c)
    x = torch.minimum(x, a_c)
    x = torch.relu(torch.abs(x))
    x = torch.clamp(x, 0.25, 8.0)
    x = torch.where(a_c >= b_c, x, b_c)
    x = torch.where(a_c > 0, x, torch.zeros_like(x))
    torch.testing.assert_close(out.cpu(), x, rtol=1e-5, atol=1e-5)


@pytest.mark.pto
def test_pto_reduce_broadcast():
    kernel = tilelang.compile(reduce_broadcast_kernel(), target="pto", out_idx=-1)
    torch.manual_seed(1)
    a = simulator_safe_randn(N)
    out = kernel(a)
    torch.npu.synchronize()
    a_c = a.cpu()
    ref = torch.empty_like(a_c)
    for i in range(N // VL):
        tile = a_c[i * VL : (i + 1) * VL]
        ref[i * VL : (i + 1) * VL] = tile + tile.max()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.pto
def test_pto_intlv_roundtrip():
    kernel = tilelang.compile(intlv_roundtrip_kernel(), target="pto")
    torch.manual_seed(2)
    x = simulator_safe_randn(N)
    y = simulator_safe_randn(N)
    xo = torch.empty(N, dtype=torch.float32, device="npu")
    yo = torch.empty(N, dtype=torch.float32, device="npu")
    kernel(x, y, xo, yo)
    torch.npu.synchronize()
    assert torch.equal(xo.cpu(), x.cpu())
    assert torch.equal(yo.cpu(), y.cpu())


@pytest.mark.pto
def test_pto_cast_roundtrip():
    kernel = tilelang.compile(cast_roundtrip_kernel(), target="pto", out_idx=-1)
    # Values that survive f16 round-trip exactly within a modest range.
    a = torch.linspace(-8.0, 8.0, N, dtype=torch.float32, device="cpu").to("npu")
    out = kernel(a)
    torch.npu.synchronize()
    ref = a.cpu().half().float()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@pytest.mark.pto
def test_pto_pairwise_sum():
    kernel = tilelang.compile(pairwise_sum_kernel(), target="pto", out_idx=-1)
    a = torch.arange(64, dtype=torch.float32, device="cpu").to("npu")
    out = kernel(a)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), a.cpu().reshape(32, 2).sum(dim=1), rtol=0, atol=0)


@pytest.mark.pto
def test_pto_vdup_float_to_int():
    kernel = tilelang.compile(vdup_float_to_int_kernel(), target="pto", out_idx=-1)
    out = kernel()
    torch.npu.synchronize()
    expected = torch.cat(
        (
            torch.ones(VL, dtype=torch.int32),
            -torch.ones(VL, dtype=torch.int32),
        )
    )
    assert torch.equal(out.cpu(), expected)


@pytest.mark.pto
def test_pto_vdup_int_to_float():
    kernel = tilelang.compile(vdup_int_to_float_kernel(), target="pto", out_idx=-1)
    out = kernel()
    torch.npu.synchronize()
    expected = torch.full((VL,), 7.0, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
