"""Test Metal gemm_v2 with actual execution on Metal hardware.

These tests verify correctness of T.gemm (gemm_v2) using simdgroup matrix
operations by comparing results against torch.matmul.
"""

import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import torch
import pytest


@tilelang.jit
def matmul_gemm_v2(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared")

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                T.gemm_v2(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


def assert_gemm_v2(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float16,
    accum_dtype=T.float32,
    atol=1e-2,
):
    jit_kernel = matmul_gemm_v2(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)

    torch_dtype = dtype.as_torch()
    torch_accum_dtype = accum_dtype.as_torch()
    a = torch.randn(M, K, dtype=torch_dtype, device="mps")
    b = torch.randn(K, N, dtype=torch_dtype, device="mps")
    c = torch.zeros(M, N, dtype=torch_accum_dtype, device="mps")

    jit_kernel(a, b, c)

    ref = a.to(torch_accum_dtype) @ b.to(torch_accum_dtype)
    assert torch.allclose(ref, c, atol=atol), (
        f"Result mismatch for M={M}, N={N}, K={K}, "
        f"block=({block_M},{block_N},{block_K}), dtype={dtype}\n"
        f"max diff: {(ref - c).abs().max().item()}"
    )


@tilelang.testing.requires_metal
def test_gemm_v2_16x16x16():
    assert_gemm_v2(128, 128, 128, 16, 16, 16)


@tilelang.testing.requires_metal
def test_gemm_v2_16x16x8():
    assert_gemm_v2(128, 128, 128, 16, 16, 8)


@pytest.mark.xfail(reason="TODO: codegen not support float16x8")
@tilelang.testing.requires_metal
def test_gemm_v2_large():
    assert_gemm_v2(128, 128, 128, 32, 32, 32)


@tilelang.testing.requires_metal
def test_gemm_v2_1024():
    assert_gemm_v2(1024, 1024, 1024, 16, 16, 16, atol=1.0)


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
