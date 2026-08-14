"""Test Metal code generation for bfloat16.

These tests verify that TileLang can compile kernels down to Metal shader
source code while correctly handling both float16 and bfloat16.
"""

import pytest
import torch

import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


@T.prim_func
def bf16_numeric_vector(
    A: T.Tensor((8,), "bfloat16"),
    B: T.Tensor((8,), "bfloat16"),
    C: T.Tensor((8,), "bfloat16"),
):
    with T.Kernel(1, threads=1):
        for i in T.vectorized(8):
            C[i] = T.max(A[i] + B[i], T.bfloat16(0))


@T.prim_func
def bf16_six_lane_copy(
    A: T.Tensor((1,), "bfloat16x6"),
    B: T.Tensor((1,), "bfloat16x6"),
):
    with T.Kernel(1, threads=1):
        B[0] = A[0]


@tilelang.jit(out_idx=[2], target="metal", execution_backend="torch")
def repro_gemm(dtype: str):
    M = 32
    N = 64
    K = 16
    threads = 64

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1, threads=threads) as (_bx, _by):
            A_shared = T.alloc_shared((M, K), dtype, "shared")
            B_shared = T.alloc_shared((K, N), dtype, "shared")
            C_local = T.alloc_fragment((M, N), "float32")

            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    return main


def lower_to_metal(dtype: str) -> str:
    prim_func = repro_gemm.get_tir(dtype)
    target = tvm.target.Target("metal", tvm.target.Target("llvm"))
    with target:
        artifact = tilelang.lower(
            prim_func,
            target=target,
            target_host="llvm",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return artifact.kernel_source or ""


def lower_prim_to_metal(prim_func) -> str:
    target = tvm.target.Target("metal", tvm.target.Target("llvm"))
    with target:
        artifact = tilelang.lower(
            prim_func,
            target=target,
            target_host="llvm",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return artifact.kernel_source or ""


def test_metal_bf16_vectorized_copy_uses_packed_uint_type():
    src = lower_to_metal("bfloat16")

    assert "simdgroup_bfloat8x8" in src
    assert "*(threadgroup bfloat*)" not in src
    assert "*(threadgroup uint4*)" in src


def test_metal_fp16_vectorized_copy_still_uses_packed_uint_type():
    src = lower_to_metal("float16")

    assert "*(threadgroup uint4*)" in src


def test_metal_bf16_numeric_vector_uses_native_bfloat2():
    src = lower_prim_to_metal(bf16_numeric_vector)

    assert "bfloat2" in src
    assert "uint4" not in src
    assert "select(" in src


def test_metal_bf16_six_lane_packed_copy_is_rejected():
    with pytest.raises(Exception, match=r"bf16x6|6 lanes|not representable"):
        lower_prim_to_metal(bf16_six_lane_copy)


@tilelang.testing.requires_metal
def test_metal_bf16_numeric_vector_runtime():
    kernel = tilelang.compile(
        bf16_numeric_vector,
        target="metal",
        execution_backend="torch",
    )
    a = torch.linspace(-2.0, 1.5, 8, dtype=torch.bfloat16, device="mps")
    b = torch.linspace(0.5, 2.0, 8, dtype=torch.bfloat16, device="mps")
    result = torch.empty_like(a)
    kernel(a, b, result)
    torch.mps.synchronize()

    reference = torch.maximum(a.cpu() + b.cpu(), torch.zeros(8, dtype=torch.bfloat16))
    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
