"""Regression test for #1967: CuTeDSL autotune cache saved .py as .so → "invalid ELF header"."""

import os
import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang.autotuner.param import AutotuneResult
from tilelang.env import env


def test_cutedsl_save_creates_kernel_py(tmp_path):
    """_save_kernel_to_disk should write kernel.py (not kernel_lib.so) for CuTeDSL."""
    original_tmp_dir = env.TILELANG_TMP_DIR
    env.TILELANG_TMP_DIR = str(tmp_path / "tmp")
    os.makedirs(env.TILELANG_TMP_DIR, exist_ok=True)

    try:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "kernel.py").write_text("# cutedsl kernel\n")
        (src_dir / "kernel.cubin").write_bytes(b"fake_cubin")

        class FakeLibGen:
            launcher_libpath = None

        class FakeAdapter:
            libpath = str(src_dir / "kernel.py")
            lib_generator = FakeLibGen()

            def get_kernel_source(self, kernel_only=True):
                return "# src"

        class FakeKernel:
            execution_backend = "cutedsl"
            adapter = FakeAdapter()
            kernel_source = "# src"
            params = []

        cache = tmp_path / "cache"
        cache.mkdir()
        AutotuneResult()._save_kernel_to_disk(cache, FakeKernel())

        assert (cache / "kernel.py").exists()
        assert not (cache / "kernel_lib.so").exists()
        assert (cache / "kernel.cubin").exists()
    finally:
        env.TILELANG_TMP_DIR = original_tmp_dir


def _is_cutedsl_available():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
        return True
    except (ImportError, AssertionError):
        return False


# Define autotune kernel at module level so closures don't capture module objects
def _make_vec_add_autotuned():
    from tilelang.autotuner import autotune

    @autotune(configs=[{"threads": t} for t in (128, 256)], warmup=3, rep=5)
    @tilelang.jit(out_idx=[-1], target="cutedsl")
    def vec_add(n: int, dtype: str = "float32", threads: int = 128):
        num_blocks = n // threads

        @T.prim_func
        def kernel(a: T.Tensor((n,), dtype), b: T.Tensor((n,), dtype), c: T.Tensor((n,), dtype)):
            with T.Kernel(num_blocks, threads=threads) as bx:
                for i in T.Parallel(threads):
                    c[bx * threads + i] = a[bx * threads + i] + b[bx * threads + i]

        return kernel

    return vec_add


@tilelang.testing.requires_cuda
@pytest.mark.skipif(not _is_cutedsl_available(), reason="CuTeDSL not installed")
def test_cutedsl_autotune_cache_roundtrip(tmp_path):
    """Autotune + CuTeDSL: save → reload from disk → verify correctness."""
    import torch
    from tilelang.autotuner import AutoTuner

    original_cache_dir, original_tmp_dir = env.TILELANG_CACHE_DIR, env.TILELANG_TMP_DIR
    env.TILELANG_CACHE_DIR = str(tmp_path / "cache")
    env.TILELANG_TMP_DIR = str(tmp_path / "tmp")
    os.makedirs(env.TILELANG_CACHE_DIR, exist_ok=True)
    os.makedirs(env.TILELANG_TMP_DIR, exist_ok=True)
    original_cache_enabled = env.is_cache_enabled()
    tilelang.enable_cache()
    AutoTuner._memory_cache.clear()

    try:
        vec_add = _make_vec_add_autotuned()
        N = 256
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        ref = a + b

        # Pass 1: cache miss
        torch.testing.assert_close(vec_add(N)(a, b), ref, atol=1e-5, rtol=1e-5)

        # Pass 2: clear memory cache → force disk reload (was "invalid ELF header" before fix)
        AutoTuner._memory_cache.clear()
        vec_add._tuner_cache.clear()
        torch.testing.assert_close(vec_add(N)(a, b), ref, atol=1e-5, rtol=1e-5)
    finally:
        env.TILELANG_CACHE_DIR = original_cache_dir
        env.TILELANG_TMP_DIR = original_tmp_dir
        if not original_cache_enabled:
            tilelang.disable_cache()
        AutoTuner._memory_cache.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
