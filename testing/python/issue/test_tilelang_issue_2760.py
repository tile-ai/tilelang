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
def test_const_cache_isolation():
    """Different kwarg values must produce distinct, correct kernels."""

    @tilelang.jit
    def kernel(A, B, block_N: int = 32):
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
