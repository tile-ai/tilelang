"""Test Metal code generation for bfloat16.

These tests verify that TileLang can compile kernels down to Metal shader
source code while correctly handling both float16 and bfloat16.
"""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from metal_test_utils import lower_prim_to_metal


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


@T.prim_func
def bf16_fp16_literals(
    A: T.Tensor((4,), "bfloat16"),
    B: T.Tensor((4,), "bfloat16"),
    C: T.Tensor((4,), "float16"),
    D: T.Tensor((4,), "bfloat16"),
    E: T.Tensor((4,), "float16"),
):
    with T.Kernel(1, threads=4):
        for i in T.Parallel(4):
            B[i] = A[i] + T.bfloat16(1.5)
            C[i] = T.float16(2.5)
            D[i] = T.bfloat16(float("inf"))
            E[i] = T.float16(float("inf"))


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
    return lower_prim_to_metal(prim_func)


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


def test_metal_fp16_bf16_literals_use_explicit_cast():
    """fp16/bf16 FloatImm literals (finite and INFINITY) must be emitted with
    an explicit ``(half)``/``(bfloat)`` cast: MSL has no implicit conversion
    into bfloat from float/half, and bare INFINITY/NAN or `h`-suffixed
    literals break bf16 assignments and half select() overloads. This also
    pins the unified form shared with the reduce PR: no ``bfloat(...)``
    wrapper and no ``h`` suffix on bf16 finite literals."""
    src = lower_prim_to_metal(bf16_fp16_literals)
    assert "(bfloat)(1.500000e+00)" in src
    assert "1.500000e+00h" not in src
    assert "bfloat(1.500000e+00)" not in src
    assert "(half)(2.500000e+00h)" in src
    assert "(bfloat)(INFINITY)" in src
    assert "(half)(INFINITY)" in src


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
