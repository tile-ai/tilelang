import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.autotuner import autotune
from tilelang.tileir import checks


def _vec_add_kernel(n: int):
    def kernel(threads: int = 128):
        @T.prim_func
        def main(
            A: T.Tensor((n,), "float32"),
            B: T.Tensor((n,), "float32"),
            C: T.Tensor((n,), "float32"),
        ):
            with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx:
                for i in T.Parallel(threads):
                    idx = bx * threads + i
                    if idx < n:
                        C[idx] = A[idx] + B[idx]

        return main

    return kernel


def _vec_add_num_ctas_kernel(n: int):
    """vec_add whose tunable knob is the cuTile ``num_ctas`` entry hint."""

    def kernel(num_ctas: int = 1):
        @T.prim_func
        def main(
            A: T.Tensor((n,), "float32"),
            B: T.Tensor((n,), "float32"),
            C: T.Tensor((n,), "float32"),
        ):
            with T.Kernel(T.ceildiv(n, 128), threads=128, num_ctas=num_ctas) as bx:
                for i in T.Parallel(128):
                    idx = bx * 128 + i
                    if idx < n:
                        C[idx] = A[idx] + B[idx]

        return main

    return kernel


def _make_autotuned_vec_add(target: str, supply_prog):
    @autotune(
        configs=[{"threads": 128}, {"threads": 256}],
        warmup=1,
        rep=1,
        timeout=60,
        supply_prog=supply_prog,
        skip_check=True,
    )
    @tilelang.jit(out_idx=[-1], target=target, execution_backend="tileir")
    def vec_add(n: int, threads: int = 128):
        return _vec_add_kernel(n)(threads)

    return vec_add


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.skipif(not checks.is_tileir_available(), reason="TileIR toolchain is not available")
def test_tileir_autotuner_from_kernel_compiles_and_runs(monkeypatch):
    torch = pytest.importorskip("torch")

    major, minor = torch.cuda.get_device_capability()
    n = 257
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    monkeypatch.setenv("TILELANG_AUTO_TUNING_CPU_COUNTS", "1")

    configs = [{"threads": 128}, {"threads": 256}]
    autotuner = (
        AutoTuner.from_kernel(_vec_add_kernel(n), configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target=f"tileir -arch=sm_{major}{minor}",
            execution_backend="tileir",
        )
        .set_profile_args(
            supply_prog=lambda _: [a, b],
            skip_check=True,
        )
    )

    result = autotuner.run(warmup=1, rep=1, timeout=60)
    out = result.kernel(a, b)

    assert result.kernel.execution_backend == "tileir"
    assert result.config in configs
    torch.testing.assert_close(out, a + b, rtol=1e-5, atol=1e-5)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.skipif(not checks.is_tileir_available(), reason="TileIR toolchain is not available")
def test_tileir_autotune_decorator_compiles_and_runs(monkeypatch):
    torch = pytest.importorskip("torch")

    major, minor = torch.cuda.get_device_capability()
    n = 257
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    monkeypatch.setenv("TILELANG_AUTO_TUNING_CPU_COUNTS", "1")

    vec_add = _make_autotuned_vec_add(
        target=f"tileir -arch=sm_{major}{minor}",
        supply_prog=lambda _: [a, b],
    )
    kernel = vec_add(n)
    out = kernel(a, b)

    assert kernel.execution_backend == "tileir"
    torch.testing.assert_close(out, a + b, rtol=1e-5, atol=1e-5)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.skipif(not checks.is_tileir_available(), reason="TileIR toolchain is not available")
def test_tileir_autotune_disk_cache_round_trip(monkeypatch, tmp_path):
    """Autotune a tunable cuTile entry hint with caching ON, then force the
    disk round-trip (``save_to_disk`` -> ``from_database``) the existing tests
    skip via ``TILELANG_DISABLE_CACHE``.
    """
    torch = pytest.importorskip("torch")

    major, minor = torch.cuda.get_device_capability()
    n = 257
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    # Caching ON, isolated to a temp dir. Single compile worker keeps it light.
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TILELANG_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("TILELANG_AUTO_TUNING_DISABLE_CACHE", raising=False)
    monkeypatch.setenv("TILELANG_AUTO_TUNING_CPU_COUNTS", "1")

    # The autotuner memory cache is process-wide; clear it so this run actually
    # persists and re-reads from disk rather than hitting an in-memory entry.
    AutoTuner._memory_cache.clear()

    configs = [{"num_ctas": 1}, {"num_ctas": 2}]

    saved_keys: list[str] = []
    orig_save = AutoTuner._save_result_to_disk

    def spy_save(self, key, result):
        saved_keys.append(key)
        return orig_save(self, key, result)

    monkeypatch.setattr(AutoTuner, "_save_result_to_disk", spy_save)

    tuner = (
        AutoTuner.from_kernel(_vec_add_num_ctas_kernel(n), configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target=f"tileir -arch=sm_{major}{minor}",
            execution_backend="tileir",
        )
        .set_profile_args(supply_prog=lambda _: [a, b], skip_check=True)
    )

    result = tuner.run(warmup=1, rep=1, timeout=60)
    torch.testing.assert_close(result.kernel(a, b), a + b, rtol=1e-5, atol=1e-5)
    assert saved_keys, "TileIR autotune result was not persisted to disk"

    # Drop the in-memory cache and reload purely from disk. This is the path that
    # rebuilds the kernel through TileIRKernelAdapter.from_database.
    AutoTuner._memory_cache.clear()
    reloaded = tuner._load_result_from_disk(saved_keys[0])
    assert reloaded is not None, "TileIR autotune disk cache failed to reload (from_database round-trip)"
    assert reloaded.kernel.execution_backend == "tileir"
    assert reloaded.config == result.config
    torch.testing.assert_close(reloaded.kernel(a, b), a + b, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
