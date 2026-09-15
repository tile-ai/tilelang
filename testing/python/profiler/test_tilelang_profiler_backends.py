from importlib import import_module
from types import SimpleNamespace

import pytest

from tilelang import profiler
from tilelang.profiler import _torch_gpu, bench


@pytest.mark.parametrize(
    "name",
    [
        "IS_CUDA",
        "_CACHE_FLUSH_ID",
        "_bench_with_cuda_events",
        "_bench_with_cudagraph",
        "_bench_with_cupti",
        "_cache_device",
        "_cuda_synchronize",
        "_do_bench_impl",
        "_normalize_cuda_device",
        "device",
        "do_bench",
        "logger",
        "suppress_stdout_stderr",
    ],
)
def test_legacy_benchmark_reexports(name):
    assert getattr(bench, name) is getattr(_torch_gpu, name)


def test_public_benchmark_reexports():
    assert profiler.do_bench is _torch_gpu.do_bench


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_gpu_backend_reexports_shared_benchmark(backend):
    implementation = import_module(f"tilelang.{backend}.profiler")
    assert implementation.do_bench is _torch_gpu.do_bench


@pytest.mark.parametrize("warmup,repeats", [(0, 1), (2, 4)])
def test_metal_preserves_single_batch_measurement(monkeypatch, warmup, repeats):
    implementation = import_module("tilelang.metal.profiler")
    calls = []
    clock = iter((3.0, 5.0))

    def perf_counter():
        calls.append("clock")
        return next(clock)

    monkeypatch.setattr(implementation, "time", SimpleNamespace(perf_counter=perf_counter))
    monkeypatch.setattr(implementation.torch.mps, "synchronize", lambda: calls.append("synchronize"))

    elapsed = implementation.do_bench(lambda: calls.append("call"), warmup=warmup, repeats=repeats)

    assert elapsed == 2.0 / repeats
    assert calls == ["call"] * warmup + ["synchronize", "clock"] + ["call"] * repeats + ["synchronize", "clock"]
