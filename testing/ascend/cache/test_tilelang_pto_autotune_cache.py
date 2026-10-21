"""PTO autotune cache round-trip: tune, persist the winner, reload from disk.

Timing is stubbed so the test stays on the cache chain: per-config compile
through the pto/tvm_ffi pipeline, the on-disk autotune result, and the
kernel rebuilt from it on the cache hit.
"""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang.autotuner import AutoTuner
from tilelang.env import CacheState, env
from tilelang.profiler import Profiler

pytestmark = pytest.mark.pto

N = 64
CONFIGS = [{"tile": 32}, {"tile": 64}]


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
    monkeypatch.setattr(CacheState, "_enabled", True)
    monkeypatch.setattr(AutoTuner, "_memory_cache", {})
    return cache_dir


def _deterministic_inputs(n):
    a = torch.arange(n, dtype=torch.float32, device="npu")
    b = torch.arange(n, dtype=torch.float32, device="npu") * 2
    return a, b


def _stub_do_bench(self, **kwargs):
    return 0.05


@tilelang.testing.requires_ascend
def test_pto_autotune_cache_roundtrip(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(Profiler, "do_bench", _stub_do_bench)

    tuner = AutoTuner(_tunable_vecadd, configs=list(CONFIGS)).set_compile_args(out_idx=[2], target="pto").set_profile_args(warmup=1, rep=3)

    records = []
    result = tuner.run(warmup=1, rep=3, on_trial=records.append)

    assert sorted(record["status"] for record in records) == ["ok", "ok"]
    assert result.kernel is not None
    assert result.config in CONFIGS

    a, b = _deterministic_inputs(N)
    expected = a.cpu() + b.cpu()
    out = result.kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    # A second run with a cleared in-memory cache must hit the on-disk cache.
    AutoTuner._memory_cache.clear()
    cached_records = []
    cached = tuner.run(warmup=1, rep=3, on_trial=cached_records.append)

    assert [record["status"] for record in cached_records] == ["cache_hit"]
    assert cached.config == result.config

    cached_out = cached.kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(cached_out.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
