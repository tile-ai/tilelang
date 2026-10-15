"""PTO autotune: tuning completes, the best kernel is correct, and results are cached.

Runs AutoTuner over a small vecadd with one tunable knob (the SIMT vector-core
tile size) on ``target="pto"`` (execution backend auto-resolves to tvm_ffi).
The tuner validates nothing by itself here (no ref_prog, so the on-disk
autotune cache stays usable); numerics are asserted directly against torch.
"""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang.autotuner import AutoTuner
from tilelang.env import env

pytestmark = pytest.mark.pto

N = 64 * 2048
CONFIGS = [{"tile": 1024}, {"tile": 2048}]


def _tunable_vecadd(tile):
    @T.prim_func
    def main(
        a: T.Tensor((N,), T.float32),
        b: T.Tensor((N,), T.float32),
        c: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(N // tile) as bx, T.SimtVF(threads=tile):
            for i in T.Parallel(tile):
                idx = bx * tile + i
                c[idx] = a[idx] + b[idx]

    return main


def _set_cache_dirs(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    monkeypatch.setattr(env, "TILELANG_AUTO_TUNING_DISABLE_CACHE", "0")
    tilelang.enable_cache()
    AutoTuner._memory_cache.clear()
    return cache_dir


def _deterministic_inputs(n):
    a = torch.arange(n, dtype=torch.float32, device="npu")
    b = torch.arange(n, dtype=torch.float32, device="npu") * 2
    return a, b


def test_pto_autotune_tunes_and_caches(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    tuner = (
        AutoTuner(_tunable_vecadd, configs=list(CONFIGS))
        .set_compile_args(out_idx=[2], target="pto")
        .set_profile_args(warmup=1, rep=3, timeout=120)
    )

    records = []
    result = tuner.run(warmup=1, rep=3, timeout=120, on_trial=records.append)

    # Every config compiled and benchmarked successfully.
    assert sorted(record["status"] for record in records) == ["ok", "ok"]
    assert result.kernel is not None
    assert result.config in CONFIGS
    assert result.latency > 0

    a, b = _deterministic_inputs(N)
    expected = a.cpu() + b.cpu()
    out = result.kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    # A second run with a cleared in-memory cache must hit the on-disk
    # autotune cache: no trials run, the same config and latency come back.
    AutoTuner._memory_cache.clear()
    cached_records = []
    cached = tuner.run(warmup=1, rep=3, timeout=120, on_trial=cached_records.append)

    assert [record["status"] for record in cached_records] == ["cache_hit"]
    assert cached.config == result.config
    assert cached.latency == result.latency

    cached_out = cached.kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(cached_out.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
