"""Regression test for issue #2760.

T.const() variables that are used only in grid dimensions or kernel-body
computations (not in any buffer shape/stride) should be resolvable via an
explicit keyword argument at call time, instead of raising RuntimeError.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
def test_const_grid_only():
    """A T.const() used only in T.Kernel(...) extent should be resolvable."""

    @tilelang.jit
    def kernel(A, B, block_N: int = 32):
        """Copy A into B."""
        N = T.const("N")
        num_blocks = T.const("num_blocks")  # grid-only, not in any buffer shape
        A: T.Tensor((N,), T.float16)
        B: T.Tensor((N,), T.float16)
        with T.Kernel(num_blocks, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_N,), T.float16)
            T.copy(A[bx * block_N], A_shared)
            T.copy(A_shared, B[bx * block_N])

    A = torch.randn(128, dtype=torch.float16, device="cuda")
    B = torch.zeros(128, dtype=torch.float16, device="cuda")
    kernel(A, B, 32, num_blocks=4)
    torch.testing.assert_close(B, A)


@tilelang.testing.requires_cuda
def test_const_computation_only():
    """A T.const() used only in the kernel body should be resolvable."""

    @tilelang.jit
    def kernel(A, B, block_N: int = 32):
        """Copy A into B."""
        N = T.const("N")
        scale = T.const("scale")  # computation-only, not in any buffer shape
        A: T.Tensor((N,), T.float16)
        B: T.Tensor((N,), T.float16)
        with T.Kernel(T.ceildiv(N, block_N), threads=128) as (bx,):
            A_local = T.alloc_fragment((block_N,), T.float16)
            T.copy(A[bx * block_N], A_local)
            for i in T.Parallel(block_N):
                A_local[i] = A_local[i] * T.cast(scale, T.float16)
            T.copy(A_local, B[bx * block_N])

    A = torch.randn(128, dtype=torch.float16, device="cuda")
    B = torch.zeros(128, dtype=torch.float16, device="cuda")
    kernel(A, B, 32, scale=3)
    torch.testing.assert_close(B, (A * 3).half())


@tilelang.testing.requires_cuda
def test_const_mixed():
    """Mix of shape-derived and explicit-kwarg const should work together."""

    @tilelang.jit
    def kernel(A, B, block_N: int = 32):
        """Copy A into B."""
        N = T.const("N")
        num_blocks = T.const("num_blocks")
        scale = T.const("scale")
        A: T.Tensor((N,), T.float16)
        B: T.Tensor((N,), T.float16)
        with T.Kernel(num_blocks, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_N,), T.float16)
            T.copy(A[bx * block_N], A_shared)
            for i in T.Parallel(block_N):
                A_shared[i] = A_shared[i] * T.cast(scale, T.float16)
            T.copy(A_shared, B[bx * block_N])

    A = torch.randn(128, dtype=torch.float16, device="cuda")
    B = torch.zeros(128, dtype=torch.float16, device="cuda")
    kernel(A, B, 32, num_blocks=4, scale=2)
    torch.testing.assert_close(B, (A * 2).half())


@tilelang.testing.requires_cuda
def test_const_pipelined_gemm():
    """Grid-only const should work with T.Pipelined + T.gemm (real kernel pattern)."""

    @tilelang.jit
    def kernel(A, B, C, block_M: int = 64, block_N: int = 64, block_K: int = 32):
        """Pipelined GEMM: C = A @ B with grid-only const extents."""
        M, N, K = T.const("M, N, K")
        num_blocks_m = T.const("num_blocks_m")  # grid-only
        num_blocks_n = T.const("num_blocks_n")  # grid-only
        A: T.Tensor((M, K), T.float16)
        B: T.Tensor((K, N), T.float16)
        C: T.Tensor((M, N), T.float16)
        with T.Kernel(num_blocks_m, num_blocks_n, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * block_M, by * block_N])

    M, N, K = 128, 128, 128
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    kernel(A, B, C, num_blocks_m=2, num_blocks_n=2)
    torch.testing.assert_close(C, (A @ B).half(), atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
def test_const_cache_isolation():
    """Different kwarg values must produce distinct, correct kernels."""

    @tilelang.jit
    def kernel(A, B, block_N: int = 32):
        """Copy A into B."""
        N = T.const("N")
        num_blocks = T.const("num_blocks")
        A: T.Tensor((N,), T.float16)
        B: T.Tensor((N,), T.float16)
        with T.Kernel(num_blocks, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_N,), T.float16)
            T.copy(A[bx * block_N], A_shared)
            T.copy(A_shared, B[bx * block_N])

    A = torch.randn(128, dtype=torch.float16, device="cuda")

    # num_blocks=4 writes all 128 elements; num_blocks=2 writes only the first 64.
    B_full = torch.zeros(128, dtype=torch.float16, device="cuda")
    B_half = torch.zeros(128, dtype=torch.float16, device="cuda")
    kernel(A, B_full, 32, num_blocks=4)
    kernel(A, B_half, 32, num_blocks=2)

    assert len(kernel.func.p1_cache) == 2
    torch.testing.assert_close(B_full, A)
    torch.testing.assert_close(B_half[:64], A[:64])
    assert torch.equal(B_half[64:], torch.zeros(64, dtype=torch.float16, device="cuda"))


@tilelang.testing.requires_cuda
def test_const_float_value():
    """A float kwarg value should be accepted and produce correct results."""

    @tilelang.jit
    def kernel(A, B, block_N: int = 32):
        """Copy A into B."""
        N = T.const("N")
        scale = T.const("scale")
        A: T.Tensor((N,), T.float16)
        B: T.Tensor((N,), T.float16)
        with T.Kernel(T.ceildiv(N, block_N), threads=128) as (bx,):
            A_local = T.alloc_fragment((block_N,), T.float16)
            T.copy(A[bx * block_N], A_local)
            for i in T.Parallel(block_N):
                A_local[i] = A_local[i] * T.cast(scale, T.float16)
            T.copy(A_local, B[bx * block_N])

    A = torch.randn(128, dtype=torch.float16, device="cuda")
    B = torch.zeros(128, dtype=torch.float16, device="cuda")
    kernel(A, B, 32, scale=2.0)
    torch.testing.assert_close(B, (A * 2).half())


@tilelang.testing.requires_cuda
def test_const_missing_kwarg_errors():
    """A grid-only const without a value should raise a clear error."""

    @tilelang.jit
    def kernel(A, B, block_N: int = 32):
        """Copy A into B."""
        N = T.const("N")
        num_blocks = T.const("num_blocks")
        A: T.Tensor((N,), T.float16)
        B: T.Tensor((N,), T.float16)
        with T.Kernel(num_blocks, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_N,), T.float16)
            T.copy(A[bx * block_N], A_shared)
            T.copy(A_shared, B[bx * block_N])

    A = torch.randn(128, dtype=torch.float16, device="cuda")
    B = torch.zeros(128, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="Cannot find value for constexpr variable"):
        kernel(A, B, 32)  # num_blocks not provided


if __name__ == "__main__":
    tilelang.testing.main()
