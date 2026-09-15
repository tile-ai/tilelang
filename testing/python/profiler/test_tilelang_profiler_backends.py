from contextlib import contextmanager
from functools import partial
from inspect import Parameter, signature

import pytest
import torch

from tilelang import profiler
from tilelang.profiler import bench, torch_bench, wall


@pytest.mark.parametrize(
    "original_name, implementation_name",
    [
        ("_CACHE_FLUSH_ID", "_CACHE_FLUSH_ID"),
        ("_bench_with_cuda_events", "bench_with_cuda_events"),
        ("_bench_with_cudagraph", "bench_with_cudagraph"),
        ("_bench_with_cupti", "bench_with_cupti"),
        ("_cuda_synchronize", "_cuda_synchronize"),
        ("suppress_stdout_stderr", "suppress_stdout_stderr"),
    ],
)
def test_legacy_benchmark_helpers(original_name, implementation_name):
    assert getattr(bench, original_name) is getattr(torch_bench, implementation_name)


@pytest.mark.parametrize("module", [torch_bench, wall])
def test_timing_modules_do_not_duplicate_entry_point(module):
    assert not hasattr(module, "do_bench")
    assert not hasattr(module, "_do_bench_impl")


def test_public_benchmark_api():
    assert profiler.do_bench is bench.do_bench
    current = signature(bench.do_bench).parameters
    original_defaults = {
        "fn": Parameter.empty,
        "warmup": 25,
        "rep": 100,
        "_n_warmup": 0,
        "_n_repeat": 0,
        "quantiles": None,
        "fast_flush": True,
        "backend": "event",
        "return_mode": "mean",
        "device": None,
        "cache_size": 256,
        "early_stop_baseline": None,
    }
    assert tuple(current) == tuple(original_defaults)
    assert {name: parameter.default for name, parameter in current.items()} == original_defaults
    assert tuple(signature(profiler.Profiler).parameters) == ("params", "result_idx", "supply_type", "adapter")
    assert signature(profiler.Profiler.do_bench).parameters["backend"].default == "event"


@pytest.mark.parametrize("method", ["event", "cupti", "cudagraph"])
@pytest.mark.parametrize("device", [None, 1, torch.device("cuda:1")])
def test_gpu_benchmark_forwarding(monkeypatch, method, device):
    calls = []
    contexts = []
    result = [1.0, 2.0]

    def benchmark(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    @contextmanager
    def device_context(selected_device):
        contexts.append(("enter", selected_device))
        try:
            yield
        finally:
            contexts.append(("exit", selected_device))

    def function():
        pass

    monkeypatch.setattr(bench, "_do_bench_impl", benchmark)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    options = {
        "warmup": 0,
        "rep": 30,
        "_n_warmup": 2,
        "_n_repeat": 3,
        "quantiles": [0.5, 0.95],
        "fast_flush": False,
        "backend": method,
        "return_mode": "median",
        "device": device,
        "cache_size": 128,
        "early_stop_baseline": 1.0,
    }

    assert profiler.do_bench(function, **options) is result
    expected_options = {name: value for name, value in options.items() if name != "device"}
    expected_options["device_idx"] = 1 if device is not None else None
    assert calls == [((function,), expected_options)]
    assert contexts == ([("enter", 1), ("exit", 1)] if device is not None else [])


def test_wall_benchmark_forwarding(monkeypatch):
    calls = []
    function_calls = []
    result = [1.0, 2.0]

    def benchmark(*args, **kwargs):
        calls.append((args, kwargs))
        return 1.0 if len(calls) == 1 else result

    def function():
        function_calls.append(None)

    def unexpected(*args, **kwargs):
        raise AssertionError("Wall timing must not call the GPU benchmark")

    monkeypatch.setattr(bench, "bench_with_wall", benchmark)
    monkeypatch.setattr(bench, "_do_bench_impl", unexpected)
    monkeypatch.setattr(bench, "_normalize_cuda_device", unexpected)
    options = {
        "warmup": 0,
        "rep": 30,
        "_n_warmup": 2,
        "_n_repeat": 3,
        "quantiles": [0.5, 0.95],
        "return_mode": "median",
        "device": torch.device("cpu"),
        "early_stop_baseline": 1.0,
    }

    assert profiler.do_bench(function, backend="wall", fast_flush=False, cache_size=128, **options) is result
    assert calls == [
        ((function,), {"n_repeat": 5, "synchronize": None}),
        ((function,), {"n_repeat": 3, "quantiles": [0.5, 0.95], "return_mode": "median", "synchronize": None}),
    ]
    assert len(function_calls) == 3


@pytest.mark.parametrize("device", [pytest.param(torch.device("cpu"), id="cpu"), pytest.param(torch.device("mps"), id="metal")])
def test_profiler_wall_non_cuda_device(monkeypatch, device):
    calls = []

    def synchronize():
        pass

    def benchmark(fn, **kwargs):
        fn()
        calls.append(kwargs)
        return 1.0

    def unexpected(*args, **kwargs):
        raise AssertionError("Non-CUDA wall timing must not enter a CUDA context")

    monkeypatch.setattr(bench, "bench_with_wall", benchmark)
    monkeypatch.setattr(torch.cuda, "device", unexpected)
    monkeypatch.setattr(torch.mps, "synchronize", synchronize)
    instance = profiler.Profiler([], [], profiler.TensorSupplyType.Auto, lambda: None)

    assert instance.do_bench(backend="wall", device=device, input_tensors=[], n_warmup=1, n_repeat=2) == 1.0
    expected_sync = calls[0]["synchronize"] if device.type == "mps" else None
    if expected_sync is not None:
        assert isinstance(expected_sync, partial)
        assert expected_sync.func is bench.device_synchronize
        assert expected_sync.args == (device,)
    assert calls == [
        {"n_repeat": 5, "synchronize": expected_sync},
        {"n_repeat": 2, "quantiles": None, "return_mode": "mean", "synchronize": expected_sync},
    ]


def test_profiler_event_metal_device(monkeypatch):
    calls = []

    def benchmark(fn, **kwargs):
        fn()
        calls.append(kwargs)
        return 1.0

    def unexpected(*args, **kwargs):
        raise AssertionError("MPS event timing must not enter a CUDA context")

    monkeypatch.setattr(bench, "_do_bench_impl", benchmark)
    monkeypatch.setattr(torch.cuda, "device", unexpected)
    instance = profiler.Profiler([], [], profiler.TensorSupplyType.Auto, lambda: None)

    assert instance.do_bench(backend="event", device=torch.device("mps"), input_tensors=[]) == 1.0
    assert calls[0]["backend"] == "event"
    assert calls[0]["device_idx"] == torch.device("mps")
