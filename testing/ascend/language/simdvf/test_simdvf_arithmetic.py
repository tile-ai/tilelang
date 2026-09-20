"""Small numerical kernels for Ascend SIMD arithmetic instructions."""

import pytest
import torch
import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.engine.lower import lower


def bitwise_kernel(n):
    @T.prim_func
    def main(
        A: T.Tensor((n,), "int32"),
        B: T.Tensor((n,), "int32"),
        Xor: T.Tensor((n,), "int32"),
        Not: T.Tensor((n,), "int32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "int32")
            b_ub = T.alloc_shared((n,), "int32")
            xor_ub = T.alloc_shared((n,), "int32")
            not_ub = T.alloc_shared((n,), "int32")

            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                T.simd.vsts(xor_ub[0], T.simd.vxor(a, b), mask)
                T.simd.vsts(not_ub[0], T.simd.vnot(a), mask)
            T.copy(xor_ub, Xor)
            T.copy(not_ub, Not)

    return main


def test_simdvf_vxor_vnot():
    n = 64
    kernel = tilelang.compile(bitwise_kernel(n), target="ascend", out_idx=[2, 3])
    device = torch.device("npu")
    a = torch.randint(-(2**30), 2**30, (n,), dtype=torch.int32, device="cpu").to(device)
    b = torch.randint(-(2**30), 2**30, (n,), dtype=torch.int32, device="cpu").to(device)

    xor, not_ = kernel(a, b)
    torch.npu.synchronize()

    torch.testing.assert_close(xor.cpu(), torch.bitwise_xor(a.cpu(), b.cpu()), rtol=0, atol=0)
    torch.testing.assert_close(not_.cpu(), torch.bitwise_not(a.cpu()), rtol=0, atol=0)


def vcpadd_kernel():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((32,), "float32")

            T.copy(A, a_ub)
            with T.SimdVF():
                low_half = T.simd.pset(32, "PAT_VL32")
                src = T.simd.vld(a_ub[0])
                result = T.simd.vcpadd(src)
                T.simd.vsts(b_ub[0], result, low_half, extent=32)
            T.copy(b_ub, B)

    return main


def test_simdvf_vcpadd():
    kernel = tilelang.compile(vcpadd_kernel(), target="ascend", out_idx=-1)
    a = torch.arange(64, dtype=torch.float32, device="cpu").to("npu")

    result = kernel(a)
    torch.npu.synchronize()

    torch.testing.assert_close(result.cpu(), a.cpu().reshape(32, 2).sum(dim=1), rtol=0, atol=0)


N = 64
NUM_ITERS = 4
LANES = 64


def serial_vreg_accum_kernel(backend="ascend"):
    @T.prim_func
    def main(
        A: T.Tensor((NUM_ITERS * LANES,), T.float32),
        B: T.Tensor((NUM_ITERS * LANES,), T.float32),
        Out: T.Tensor((LANES,), T.float32),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((NUM_ITERS * LANES,), T.float32)
            b_ub = T.alloc_shared((NUM_ITERS * LANES,), T.float32)
            out_ub = T.alloc_shared((LANES,), T.float32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                acc = T.simd.alloc_var(T.float32)
                acc = T.simd.vdup(0.0, T.float32)
                for i in T.serial(NUM_ITERS):
                    a = T.simd.vld(a_ub[i * LANES])
                    b = T.simd.vld(b_ub[i * LANES])
                    T.simd.vmula(acc, a, b, mask)
                T.simd.vsts(out_ub[0], acc, mask)
            T.copy(out_ub, Out)

    return main


@pytest.mark.pto
def test_serial_vreg_accum_pto_codegen_carries_local_var():
    source = lower(serial_vreg_accum_kernel("pto"), target="pto").kernel_source
    assert "pto.vmula" in source
    assert "T.unroll" not in source
    assert "pto.static_range" not in source
    has_python_range = " in range(" in source
    has_explicit_carry = ".carry(" in source
    assert has_python_range or has_explicit_carry, source
    assert "with pto.for_(" not in source or has_explicit_carry, source


@pytest.mark.parametrize("backend", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_serial_vreg_accum(backend):
    kernel = tilelang.compile(serial_vreg_accum_kernel(backend), target=backend, out_idx=[2])
    device = torch.device("npu")
    torch.manual_seed(0)
    a = torch.randn(NUM_ITERS * LANES, dtype=torch.float32, device=device)
    b = torch.randn(NUM_ITERS * LANES, dtype=torch.float32, device=device)
    out = kernel(a, b)
    torch.npu.synchronize()

    expected = (a.view(NUM_ITERS, LANES) * b.view(NUM_ITERS, LANES)).sum(dim=0)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


# --- vlrelu / vprelu ---


@tilelang.jit(target="ascend")
def relu_kernel():
    @T.prim_func
    def main(x: T.Tensor((N,), "float32"), o1: T.Tensor((N,), "float32"), o2: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), "float32")
            o1_ub = T.alloc_shared((N,), "float32")
            o2_ub = T.alloc_shared((N,), "float32")
            T.copy(x, x_ub)
            with T.SimdVF():
                full = S.pset(32)
                v = S.vld(x_ub[0])
                S.vsts(o1_ub[0], S.vlrelu(v, T.float32(0.1), full), full)
                slope = S.vdup(T.float32(0.2), T.float32, full)
                S.vsts(o2_ub[0], S.vprelu(v, slope, full), full)
            T.copy(o1_ub, o1)
            T.copy(o2_ub, o2)

    return main


def test_vlrelu_vprelu():
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32, device="npu") * 2
    o1 = torch.empty(N, dtype=torch.float32, device="npu")
    o2 = torch.empty(N, dtype=torch.float32, device="npu")
    relu_kernel()(x, o1, o2)
    torch.npu.synchronize()
    xc = x.cpu()
    torch.testing.assert_close(o1.cpu(), torch.where(xc > 0, xc, xc * 0.1), rtol=0, atol=1e-6)
    torch.testing.assert_close(o2.cpu(), torch.where(xc > 0, xc, xc * 0.2), rtol=0, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_absolute_difference(dtype):
    lanes = 2048 // tilelang.tvm.DataType(dtype).bits

    @T.prim_func
    def kernel(a: T.Tensor((lanes,), dtype), b: T.Tensor((lanes,), dtype), out: T.Tensor((lanes,), dtype)):
        with T.Kernel(1):
            left = T.alloc_shared((lanes,), dtype)
            right = T.alloc_shared((lanes,), dtype)
            result = T.alloc_shared((lanes,), dtype)
            T.copy(a, left)
            T.copy(b, right)
            with T.SimdVF():
                mask = T.simd.pset(2048 // lanes)
                T.simd.vsts(result[0], T.simd.vabsdif(T.simd.vld(left[0]), T.simd.vld(right[0]), mask), mask)
            T.copy(result, out)

    run = tilelang.compile(kernel, target="ascend", out_idx=-1)
    a = torch.randn(lanes, dtype=getattr(torch, dtype), device="npu")
    b = torch.randn_like(a)
    torch.testing.assert_close(run(a, b), (a - b).abs(), rtol=0, atol=0)


@pytest.mark.parametrize("op", ["vmadd", "vmula"])
@pytest.mark.parametrize("dtype,atol", [("float32", 1e-5), ("float16", 1e-2), ("bfloat16", 1e-1)])
def test_fused_multiply_add(op, dtype, atol):
    bits = tilelang.tvm.DataType(dtype).bits
    lanes = 2048 // bits
    n = lanes * 2

    @T.prim_func
    def kernel(a: T.Tensor((n,), dtype), b: T.Tensor((n,), dtype), out: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            left = T.alloc_shared((n,), dtype)
            right = T.alloc_shared((n,), dtype)
            result = T.alloc_shared((n,), dtype)
            T.copy(a, left)
            T.copy(b, right)
            T.copy(out, result)
            with T.SimdVF():
                mask = T.simd.pset(bits)
                for i in T.serial(2):
                    acc = T.simd.alloc_var(dtype)
                    acc = T.simd.vld(result[i * lanes])
                    getattr(T.simd, op)(acc, T.simd.vld(left[i * lanes]), T.simd.vld(right[i * lanes]), mask)
                    T.simd.vsts(result[i * lanes], acc, mask)
            T.copy(result, out)

    run = tilelang.compile(kernel, target="ascend")
    a, b, out = [torch.randn(n, dtype=getattr(torch, dtype), device="npu") for _ in range(3)]
    expected = out * a + b if op == "vmadd" else out + a * b
    run(a, b, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=atol)
