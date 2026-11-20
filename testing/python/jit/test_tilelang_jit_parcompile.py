import concurrent.futures
from pathlib import Path

import pytest
import tilelang.testing
import tilelang
import torch
from tilelang import env
from tilelang import language as T


@tilelang.jit(
    out_idx=-1,  # create the output tensor during runtime
    verbose=True,
)
def matmul_kernel_jit(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A=False,
    trans_B=True,
    in_dtype=T.float16,
    out_dtype=T.float32,
    accum_dtype=T.float32,
    num_stages=2,
    threads=128,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def test_par_compile():
    configs = [
        (1024, 1024, 1024, 128, 128, 64),
        (2048, 2048, 2048, 256, 256, 32),
        (4096, 4096, 4096, 64, 64, 128),
    ]
    kernels = matmul_kernel_jit.par_compile(configs)
    for (M, N, K, _, _, _), kernel in zip(configs, kernels):
        A = torch.randn(M, K, dtype=torch.float16).cuda()
        B = torch.randn(N, K, dtype=torch.float16).cuda()
        ref = (A @ B.T).float()
        C = kernel(A, B)
        tilelang.testing.torch_assert_close(C, ref, rtol=1e-2, atol=1e-2)


@tilelang.jit(out_idx=-1)
def vec_add_kernel_jit(N, block_N):
    @T.prim_func
    def main(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
        C: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            for i in T.Parallel(block_N):
                idx = bx * block_N + i
                if idx < N:
                    C[idx] = A[idx] + B[idx]

    return main


def _make_invalid_lowering_prim_func():
    @T.prim_func
    def broken(A: T.Tensor((7, 7), T.float16), B: T.Tensor((7, 7), T.float16), C: T.Tensor((7, 7), T.float32)):
        with T.Kernel(1, threads=32) as _:
            a_s = T.alloc_shared((7, 7), T.float16)
            b_s = T.alloc_shared((7, 7), T.float16)
            c_f = T.alloc_fragment((7, 7), T.float32)
            T.copy(A, a_s)
            T.copy(B, b_s)
            T.gemm(a_s, b_s, c_f)
            T.copy(c_f, C)

    return broken


def test_par_compile_process_executor(monkeypatch, tmp_path):
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))

    configs = [(256, 64), (512, 128), (256, 64)]
    kernels = vec_add_kernel_jit.par_compile(configs, num_workers=2, executor="process")
    assert len(kernels) == 3
    assert kernels[0] is kernels[2]

    for (N, _), kernel in zip(configs, kernels):
        assert Path(kernel._tilelang_cache_path).is_relative_to(tmp_path)
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        tilelang.testing.torch_assert_close(kernel(a, b), a + b)

    # Clear in-memory caches so the second call exercises parent-side disk-cache lookup
    # without spawning any worker processes.
    from tilelang.cache import _resolve_cache_dispatch

    cache, _, _ = _resolve_cache_dispatch(None, None, None, False)
    cache._memory_cache.clear()

    def _unexpected_pool(*args, **kwargs):
        raise AssertionError("ProcessPoolExecutor should not be created when all kernels hit disk cache")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _unexpected_pool)
    cached_kernels = vec_add_kernel_jit.par_compile(configs, num_workers=2, executor="process")
    for (N, _), kernel in zip(configs, cached_kernels):
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        tilelang.testing.torch_assert_close(kernel(a, b), a + b)


def test_par_compile_process_validation_and_error_handling(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="Invalid executor"):
        tilelang.par_compile([], executor="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_workers must be greater than 0"):
        tilelang.par_compile([], num_workers=0, executor="process")

    env.disable_cache()
    try:
        with pytest.raises(ValueError, match='executor="process" requires disk caching'):
            tilelang.par_compile([], executor="process")
    finally:
        env.enable_cache()

    assert tilelang.par_compile([], executor="process") == []

    from tilelang.cache import _resolve_cache_dispatch
    from tilelang.jit.adapter.torch.kernel_cache import TorchKernelCache

    assert TorchKernelCache().supports_disk_cache is False
    valid_func = vec_add_kernel_jit.get_tir(128, 64)
    cache, _, _ = _resolve_cache_dispatch(None, None, None, False)
    with monkeypatch.context() as mp:
        mp.setattr(cache, "supports_disk_cache", False)
        with pytest.raises(ValueError, match="does not persist kernels to the disk cache"):
            tilelang.par_compile([valid_func], out_idx=[-1], executor="process")
    assert env.is_cache_enabled()

    broken_func = _make_invalid_lowering_prim_func()
    with pytest.raises(RuntimeError, match="Traceback"):
        tilelang.par_compile([broken_func], out_idx=[-1], num_workers=1, executor="process")

    mixed = tilelang.par_compile(
        [valid_func, broken_func],
        out_idx=[-1],
        num_workers=2,
        ignore_error=True,
        executor="process",
    )
    assert mixed[0] is not None
    assert mixed[1] is None
    a = torch.randn(128, device="cuda", dtype=torch.float32)
    b = torch.randn(128, device="cuda", dtype=torch.float32)
    tilelang.testing.torch_assert_close(mixed[0](a, b), a + b)


if __name__ == "__main__":
    tilelang.testing.main()
