"""Default Metal Host IR execution/cache and explicit Torch cache behavior."""

import logging
import os

import tilelang
import tilelang.testing
import tilelang.language as T
import torch

from tilelang.cache.kernel_cache import KernelCache


def _matmul_gemm(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
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


_matmul_gemm_auto = tilelang.jit(_matmul_gemm)


def _run_gemm(jit_kernel, M, N, K):
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    c = torch.zeros(M, N, dtype=torch.float32, device="mps")
    jit_kernel(a, b, c)
    ref = a.to(torch.float32) @ b.to(torch.float32)
    assert torch.allclose(ref, c, atol=1e-2), f"max diff: {(ref - c).abs().max().item()}"


@tilelang.testing.requires_metal
def test_auto_backend_resolves_to_tvm_ffi():
    """Bare @tilelang.jit must resolve to the shared Host IR backend."""
    M, N, K = 64, 64, 64
    kernel = _matmul_gemm_auto(M, N, K, 16, 16, 8)
    assert kernel.execution_backend == "tvm_ffi", kernel.execution_backend
    _run_gemm(kernel, M, N, K)


@tilelang.testing.requires_metal
def test_torch_backend_no_disk_cache_write():
    """The torch backend must not write disk cache entries or log save errors."""
    cache_logger = logging.getLogger("tilelang.cache.kernel_cache")
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record)
    cache_logger.addHandler(handler)
    cache_logger.setLevel(logging.ERROR)

    cache_root = KernelCache._get_cache_root()
    before = set(os.listdir(cache_root)) if os.path.isdir(cache_root) else set()

    try:
        M, N, K = 64, 64, 64
        kernel = tilelang.compile(_matmul_gemm(M, N, K, 16, 16, 8), target="metal", execution_backend="torch")
        _run_gemm(kernel, M, N, K)
        # Second run in the same process: memory-cache hit, still no disk I/O.
        _run_gemm(kernel, M, N, K)
    finally:
        cache_logger.removeHandler(handler)

    error_messages = [r.getMessage() for r in records if r.levelno >= logging.ERROR]
    assert not any("Error during atomic cache save" in msg for msg in error_messages), error_messages

    after = set(os.listdir(cache_root)) if os.path.isdir(cache_root) else set()
    assert after == before, f"torch backend wrote disk cache entries: {after - before}"


@tilelang.testing.requires_metal
def test_mir_host_native_roundtrip(tmp_path):
    @T.prim_func
    def add_one(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = A[i] + 1

    kernel = tilelang.compile(add_one, target="metal")
    path = str(tmp_path / "metal.tlm")
    kernel.export_library(path)
    loaded = tilelang.tvm.runtime.load_module(path)
    a = torch.arange(128, device="mps", dtype=torch.float32)
    b = torch.empty_like(a)
    for call in (kernel, loaded):
        # Torch work before/after the launch exercises MPS command-buffer ordering.
        a = a * 2
        call(a, b)
        torch.testing.assert_close(b * 2, (a + 1) * 2)


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
