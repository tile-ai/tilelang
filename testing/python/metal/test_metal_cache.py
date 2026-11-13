"""Regression tests for issue #2722: TileLang on Metal.

Covers two behaviors that previously made Metal support opaque:
1. `@tilelang.jit` with no explicit `execution_backend` must resolve to a
   backend that is actually usable on Metal. #2722 resolved `auto` to the torch
   backend because `tvm_ffi` was not usable on Metal at the time; that has since
   inverted -- the torch adapter binds arguments positionally rather than in the
   emitted `[[buffer(N)]]` order, cannot launch a kernel whose grid depends on a
   `T.dynamic()` extent, and ignores `out_idx` -- so `auto` resolves to
   `tvm_ffi`, which is correct on all three.
2. The torch backend must not attempt to persist a compiled library to the disk
   cache (it has no `libpath` artifact; `torch.mps.compile_shader` compiles the
   MSL source in-process). Before the fix this raised
   `AttributeError: 'MetalKernelAdapter' object has no attribute 'libpath'`
   and logged "Error during atomic cache save" on every run.
"""

import logging
import os

import tilelang
import tilelang.testing
import tilelang.language as T
import torch

from tilelang.cache.kernel_cache import KernelCache


def _gemm_prim_func(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
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

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


@tilelang.jit
def _matmul_gemm_auto(M, N, K, block_M, block_N, block_K):
    return _gemm_prim_func(M, N, K, block_M, block_N, block_K)


@tilelang.jit(execution_backend="torch")
def _matmul_gemm_torch(M, N, K, block_M, block_N, block_K):
    return _gemm_prim_func(M, N, K, block_M, block_N, block_K)


def _run_gemm(jit_kernel, M, N, K):
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    c = torch.zeros(M, N, dtype=torch.float32, device="mps")
    jit_kernel(a, b, c)
    ref = a.to(torch.float32) @ b.to(torch.float32)
    assert torch.allclose(ref, c, atol=1e-2), f"max diff: {(ref - c).abs().max().item()}"


@tilelang.testing.requires_metal
def test_auto_backend_resolves_to_tvm_ffi():
    """Bare @tilelang.jit must resolve to a backend that works on Metal."""
    M, N, K = 64, 64, 64
    kernel = _matmul_gemm_auto(M, N, K, 16, 16, 8)
    assert kernel.execution_backend == "tvm_ffi", kernel.execution_backend
    _run_gemm(kernel, M, N, K)


@tilelang.testing.requires_metal
def test_auto_backend_launches_a_dynamic_shape_kernel():
    """A dynamic extent must not stop `auto` from launching.

    The torch adapter resolves the launch grid at compile time, so a
    `T.dynamic()` extent leaves it symbolic and the launch fails. Pinned here
    because it is the most visible consequence of which backend `auto` picks.
    """

    @tilelang.jit
    def increment():
        n = T.dynamic("n")

        @T.prim_func
        def kernel(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
            with T.Kernel(T.ceildiv(n, 64), threads=64) as bx:
                i = bx * 64 + T.get_thread_binding(0)
                if i < n:
                    B[i] = A[i] + 1.0

        return kernel

    compiled = increment()
    for size in (64, 192, 1000):
        a = torch.full((size,), 2.0, dtype=torch.float32, device="mps")
        b = torch.zeros(size, dtype=torch.float32, device="mps")
        compiled(a, b)
        torch.mps.synchronize()
        assert torch.equal(b.cpu(), torch.full((size,), 3.0)), size


@tilelang.testing.requires_metal
def test_torch_backend_no_disk_cache_write():
    """The torch backend must not write disk cache entries or log save errors."""
    cache_logger = logging.getLogger("tilelang.cache.kernel_cache")
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record)
    cache_logger.addHandler(handler)
    cache_logger.setLevel(logging.ERROR)

    cache_root = KernelCache._get_write_cache_root()
    before = set(os.listdir(cache_root)) if os.path.isdir(cache_root) else set()

    try:
        M, N, K = 64, 64, 64
        kernel = _matmul_gemm_torch(M, N, K, 16, 16, 8)
        assert kernel.execution_backend == "torch", kernel.execution_backend
        _run_gemm(kernel, M, N, K)
        # Second run in the same process: memory-cache hit, still no disk I/O.
        _run_gemm(kernel, M, N, K)
    finally:
        cache_logger.removeHandler(handler)

    error_messages = [r.getMessage() for r in records if r.levelno >= logging.ERROR]
    assert not any("Error during atomic cache save" in msg for msg in error_messages), error_messages

    after = set(os.listdir(cache_root)) if os.path.isdir(cache_root) else set()
    assert after == before, f"torch backend wrote disk cache entries: {after - before}"


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
