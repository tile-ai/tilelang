from dataclasses import replace
from importlib import import_module
from inspect import signature
from types import SimpleNamespace

import pytest

from tilelang import profiler, tvm
from tilelang.backend import BackendContext, ProfilerBackendSpec, get_backend
from tilelang.jit.kernel import JITKernel
from tilelang.profiler import bench


def test_public_benchmark_reexports():
    assert profiler.do_bench is bench.do_bench
    assert bench.do_bench.__module__ == "tilelang.profiler.bench"


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_gpu_backend_reexports_shared_benchmark(backend):
    implementation = import_module(f"tilelang.{backend}.profiler")
    assert implementation.do_bench is bench.do_bench


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


def _make_context(backend_module):
    return BackendContext(
        backend_module,
        tvm.target.Target({"kind": "cuda", "arch": "sm_80"}),
        tvm.target.Target("c"),
        backend_module.execution_backends[0],
    )


@pytest.mark.parametrize("method", ["event", "cupti", "cudagraph"])
def test_context_profiler_lazily_forwards_to_unchanged_gpu_bench(monkeypatch, method):
    calls = []
    result = [1.0, 2.0]

    def benchmark(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(bench, "do_bench", benchmark)
    context = _make_context(get_backend("cuda"))
    spec = context.profiler(method)
    assert calls == []

    def fn():
        pass

    options = {
        "warmup": 0,
        "rep": 30,
        "_n_warmup": 2,
        "_n_repeat": 3,
        "quantiles": [0.5, 0.95],
        "fast_flush": False,
        "return_mode": "median",
        "device": 1,
        "cache_size": 128,
        "early_stop_baseline": 1.0,
    }

    assert spec.do_bench(fn, **options) is result
    assert calls == [((fn,), {"backend": method, **options})]
    assert context.execution_backend.name == "tvm_ffi"


def test_kernel_profiler_uses_bound_context_and_selects_method_per_call(monkeypatch):
    calls = []

    def benchmark(fn, **kwargs):
        fn()
        calls.append(kwargs)
        return 1.0

    methods = tuple(ProfilerBackendSpec(name, benchmark) for name in ("event", "cupti"))
    context = _make_context(replace(get_backend("cuda"), profiler_backends=methods))
    kernel = SimpleNamespace(params=[], out_idx=[], adapter=lambda: None, backend_context=context)

    def unexpected(*args, **kwargs):
        raise AssertionError("Profiling must use the compiled context, not resolve or select a global backend again")

    monkeypatch.setattr("tilelang.backend.module.create_backend_context", unexpected)
    monkeypatch.setattr(import_module("tilelang.jit.kernel"), "create_backend_context", unexpected)
    monkeypatch.setattr(profiler, "do_bench", unexpected)

    instance = JITKernel.get_profiler(kernel)
    assert instance._backend_context is context
    assert instance.do_bench(input_tensors=[], warmup=0, rep=30) == 1.0
    assert instance.do_bench(input_tensors=[], warmup=0, rep=30, backend="cupti") == 1.0
    assert [call["backend"] for call in calls] == ["event", "cupti"]
    assert all(call["warmup"] == 0 and call["rep"] == 30 for call in calls)
    assert all(call["device"] is None for call in calls)
    with pytest.raises(ValueError, match="Allowed: event, cupti"):
        instance.do_bench(input_tensors=[], backend="cudagraph")


def test_manually_constructed_profiler_keeps_legacy_api_and_default(monkeypatch):
    calls = []

    def benchmark(fn, **kwargs):
        calls.append(kwargs)
        return 1.0

    monkeypatch.setattr(profiler, "do_bench", benchmark)
    instance = profiler.Profiler([], [], profiler.TensorSupplyType.Auto, lambda: None)

    assert tuple(signature(profiler.Profiler).parameters) == ("params", "result_idx", "supply_type", "adapter")
    assert instance._backend_context is None
    assert instance.do_bench(input_tensors=[]) == 1.0
    assert calls[0]["backend"] == "event"
