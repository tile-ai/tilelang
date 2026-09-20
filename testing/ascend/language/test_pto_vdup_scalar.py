"""Regression tests for pto.vdup with non-immediate scalar operands.

Covers three scenarios that were broken or untested before the fix in
codegen_pto.cc:

1. Integer scalar from a runtime expression (e.g. T.min(N, VEC))
   — PTODSL infers 'index' type which pto.vdup rejects.  The fix applies
   _tl_coerce_iXX based on the return dtype's element width.

2. 8-bit integer (int8/uint8) scalar — must use _tl_coerce_i8 (not the
   default _tl_coerce_i32) so the scalar width matches the b8 mask
   granularity expected by pto.vdup.

3. Float scalar (float32/float16/bfloat16) from a runtime expression —
   must NOT be passed through integer coercion; PTODSL handles float
   types natively.
"""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing


VEC = 64


@tilelang.jit(out_idx=[1], target="pto")
def vdup_runtime_scalar_i32(N: int):
    @T.prim_func
    def main(
        X: T.Tensor((N,), T.int32),
        Y: T.Tensor((N,), T.int32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), T.int32)
            y_ub = T.alloc_shared((N,), T.int32)
            T.copy(X, x_ub)

            with T.SimdVF():
                full = T.simd.pset(32)
                # rem is a runtime expression (depends on N), inferred as
                # 'index' by PTODSL without coercion.
                rem = T.min(N, VEC)
                rem_v = T.simd.vdup(rem, "int32", full)
                for i in range(N // VEC):
                    x = T.simd.vld(x_ub[i * VEC])
                    y = T.simd.vmul(x, rem_v, full)
                    T.simd.vsts(y_ub[i * VEC], y)

            T.copy(y_ub, Y)

    return main


@tilelang.jit(out_idx=[1], target="pto")
def vdup_runtime_scalar_i8(N: int, dtype_str: str):
    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype_str),
        Y: T.Tensor((N,), dtype_str),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), dtype_str)
            y_ub = T.alloc_shared((N,), dtype_str)
            T.copy(X, x_ub)

            with T.SimdVF():
                full = T.simd.pset(8)
                # N is a compile-time constant but T.min(N, 1) is still
                # a non-immediate PrimExpr.  vdup with int8/uint8 dtype
                # must coerce to i8 (not i32) to match b8 mask granularity.
                val = T.cast(T.min(N, 1), dtype_str)
                val_v = T.simd.vdup(val, dtype_str, full)
                for i in range(N // VEC):
                    x = T.simd.vld(x_ub[i * VEC], dist="UNPK4_B8")
                    y = T.simd.vadd(x, val_v, full)
                    T.simd.vsts(y_ub[i * VEC], y, dist="PK4_B32")

            T.copy(y_ub, Y)

    return main


@tilelang.jit(out_idx=[1], target="pto")
def vdup_runtime_scalar_f32(N: int):
    @T.prim_func
    def main(
        X: T.Tensor((N,), T.float32),
        Y: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), T.float32)
            y_ub = T.alloc_shared((N,), T.float32)
            T.copy(X, x_ub)

            with T.SimdVF():
                full = T.simd.pset(32)
                # scale is a runtime expression but float — must NOT go
                # through integer _tl_coerce_iXX.
                scale = T.min(T.cast(N, T.float32), 2.0)
                scale_v = T.simd.vdup(scale, "float32", full)
                for i in range(N // VEC):
                    x = T.simd.vld(x_ub[i * VEC])
                    y = T.simd.vmul(x, scale_v, full)
                    T.simd.vsts(y_ub[i * VEC], y)

            T.copy(y_ub, Y)

    return main


@tilelang.jit(out_idx=[1], target="pto")
def vdup_runtime_scalar_f16(N: int, dtype_str: str):
    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype_str),
        Y: T.Tensor((N,), dtype_str),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), dtype_str)
            y_ub = T.alloc_shared((N,), dtype_str)
            T.copy(X, x_ub)

            with T.SimdVF():
                full = T.simd.pset(16)
                # Runtime float scalar — must NOT go through integer
                # _tl_coerce_iXX.  The scalar is float32 but vdup
                # broadcasts it as float16/bfloat16.
                scale = T.min(T.cast(N, T.float32), 2.0)
                scale_v = T.simd.vdup(scale, dtype_str, full)
                for i in range(N // VEC):
                    x = T.simd.vld(x_ub[i * VEC], dist="UNPK_B16")
                    y = T.simd.vmul(x, scale_v, full)
                    T.simd.vsts(y_ub[i * VEC], y, dist="PK_B32")

            T.copy(y_ub, Y)

    return main


def test_vdup_runtime_i32():
    N = 256
    kernel = vdup_runtime_scalar_i32(N)
    x = torch.arange(N, dtype=torch.int32, device="npu")
    y = kernel(x)
    torch.npu.synchronize()
    ref = x * min(N, VEC)
    assert torch.equal(y, ref), f"i32 vdup mismatch: {y} vs {ref}"


@pytest.mark.parametrize("dtype_str,torch_dtype", [("int8", torch.int8), ("uint8", torch.uint8)])
def test_vdup_runtime_i8(dtype_str, torch_dtype):
    N = 256
    kernel = vdup_runtime_scalar_i8(N, dtype_str)
    x = torch.arange(N, dtype=torch.int32, device="npu").to(torch_dtype)
    y = kernel(x)
    torch.npu.synchronize()
    ref = x + 1
    assert torch.equal(y, ref), f"{dtype_str} vdup mismatch: {y} vs {ref}"


def test_vdup_runtime_f32():
    N = 256
    kernel = vdup_runtime_scalar_f32(N)
    x = torch.arange(N, dtype=torch.float32, device="npu")
    y = kernel(x)
    torch.npu.synchronize()
    ref = x * min(float(N), 2.0)
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype_str,torch_dtype", [("float16", torch.float16), ("bfloat16", torch.bfloat16)])
def test_vdup_runtime_f16(dtype_str, torch_dtype):
    N = 256
    kernel = vdup_runtime_scalar_f16(N, dtype_str)
    x = torch.arange(N, dtype=torch.float32, device="npu").to(torch_dtype)
    y = kernel(x)
    torch.npu.synchronize()
    ref = x * 2
    torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tilelang.testing.main()
