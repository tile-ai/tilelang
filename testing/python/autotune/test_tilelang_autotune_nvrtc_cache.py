"""Regression test for autotune save failure after an NVRTC kernel disk-cache hit."""

import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
import torch
from tilelang.autotuner import AutoTuner
from tilelang.cache import _dispatch_map
from tilelang.env import env


def _make_vec_add_autotuned():
    @tilelang.autotune(configs=[{"threads": t} for t in (128, 256)], warmup=3, rep=5)
    @tilelang.jit(out_idx=[-1], execution_backend="nvrtc")
    def vec_add(n: int, threads: int = 128):
        @T.prim_func
        def kernel(a: T.Tensor((n,), T.float32), b: T.Tensor((n,), T.float32), c: T.Tensor((n,), T.float32)):
            with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx:
                for i in T.Parallel(threads):
                    c[bx * threads + i] = a[bx * threads + i] + b[bx * threads + i]

        return kernel

    return vec_add


@tilelang.testing.requires_cuda
def test_nvrtc_cache_save(tmp_path):
    """Verify that disk-loaded NVRTC kernels have libpath so autotune results can be saved."""
    original_cache_dir = env.TILELANG_CACHE_DIR
    env.TILELANG_CACHE_DIR = str(tmp_path / "cache")
    original_cache_enabled = env.is_cache_enabled()
    tilelang.enable_cache()
    cache = _dispatch_map["nvrtc"]
    cache._memory_cache.clear()
    AutoTuner._memory_cache.clear()

    try:
        n = 256
        # Step 1: compile both configs, then remove them from memory caches.
        entry = _make_vec_add_autotuned()
        for threads in (128, 256):
            entry(n, threads=threads)
        cache._memory_cache.clear()
        AutoTuner._memory_cache.clear()

        # Step 2: autotune using disk-cached kernels and load the saved result.
        entry = _make_vec_add_autotuned()
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")
        kernel = entry(n)
        torch.testing.assert_close(kernel(a, b), a + b)

        (key,) = AutoTuner._memory_cache
        result = entry.get_tunner()._load_result_from_disk(key)
        # Before the fix, missing libpath prevented the autotune result from being saved.
        assert result is not None
        assert result.config == kernel.config
        torch.testing.assert_close(result.kernel(a, b), a + b)
    finally:
        env.TILELANG_CACHE_DIR = original_cache_dir
        cache._memory_cache.clear()
        AutoTuner._memory_cache.clear()
        if not original_cache_enabled:
            tilelang.disable_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
