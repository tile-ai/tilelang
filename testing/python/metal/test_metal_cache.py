"""Regression tests for issue #2722: TileLang on Metal.

Covers two behaviors that previously made Metal support opaque:
1. `@tilelang.jit` with no explicit `execution_backend` must resolve to the
   torch backend (Metal adapter) instead of `tvm_ffi`.
2. The torch backend has no library artifact (`torch.mps.compile_shader`
   compiles the MSL source in-process). Saving its cache entry must neither
   raise `AttributeError: 'MetalKernelAdapter' object has no attribute
   'libpath'` nor log "Error during atomic cache save", and compiling a Metal
   kernel must not disable caching for the rest of the process. Entry
   contents and reloads are covered by `test_metal_kernel_cache.py`.
"""

import logging
import os

import tilelang
import tilelang.testing
import tilelang.language as T
import torch

from tilelang.cache.kernel_cache import KernelCache
from tilelang.env import env


@tilelang.jit
def _matmul_gemm_auto(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
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


def _run_gemm(jit_kernel, M, N, K):
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    c = torch.zeros(M, N, dtype=torch.float32, device="mps")
    jit_kernel(a, b, c)
    ref = a.to(torch.float32) @ b.to(torch.float32)
    assert torch.allclose(ref, c, atol=1e-2), f"max diff: {(ref - c).abs().max().item()}"


@tilelang.testing.requires_metal
def test_auto_backend_resolves_to_torch():
    """Bare @tilelang.jit must resolve to the torch (Metal) backend."""
    M, N, K = 64, 64, 64
    kernel = _matmul_gemm_auto(M, N, K, 16, 16, 8)
    assert kernel.execution_backend == "torch", kernel.execution_backend
    _run_gemm(kernel, M, N, K)


@tilelang.testing.requires_metal
def test_torch_backend_caches_without_save_errors_or_global_disable():
    """The torch backend persists entries cleanly and leaves the cache state alone."""
    cache_logger = logging.getLogger("tilelang.cache.kernel_cache")
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record)
    cache_logger.addHandler(handler)
    cache_logger.setLevel(logging.ERROR)

    enabled_before = env.is_cache_enabled()
    cache_root = KernelCache._get_cache_root()
    before = set(os.listdir(cache_root)) if os.path.isdir(cache_root) else set()

    try:
        M, N, K = 64, 64, 64
        kernel = _matmul_gemm_auto(M, N, K, 16, 16, 8)
        _run_gemm(kernel, M, N, K)
        # Second run in the same process: memory-cache hit.
        _run_gemm(kernel, M, N, K)
    finally:
        cache_logger.removeHandler(handler)

    error_messages = [r.getMessage() for r in records if r.levelno >= logging.ERROR]
    assert not any("Error during atomic cache save" in msg for msg in error_messages), error_messages
    assert env.is_cache_enabled() == enabled_before, "compiling a Metal kernel must not change the cache state"

    after = set(os.listdir(cache_root)) if os.path.isdir(cache_root) else set()
    if enabled_before:
        entry = getattr(kernel, "_tilelang_cache_path", None)
        assert entry is not None and os.path.isdir(entry), "an enabled cache must persist the Metal entry"
        assert os.path.exists(os.path.join(entry, "launch.json"))
    else:
        assert after == before, f"a disabled cache must not write entries: {after - before}"


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
