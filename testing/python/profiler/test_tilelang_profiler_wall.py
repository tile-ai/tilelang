from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tilelang.profiler import do_bench, wall


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0, calls=0, pending=0.0)
    monkeypatch.setattr(wall, "perf_counter", lambda: state.now)
    return state


def _timed_function(clock, durations):
    durations = iter(durations)

    def run():
        clock.calls += 1
        clock.now += next(durations)

    return run


@pytest.mark.parametrize("return_mode, expected", [("min", 1.0), ("max", 4.0), ("mean", 2.5), ("median", 2.0)])
def test_wall_aggregation_and_iteration_overrides(clock, return_mode, expected):
    function = _timed_function(clock, [0.001] * 8 + [0.001, 0.002, 0.003, 0.004])

    result = do_bench(function, backend="wall", warmup=1000, rep=1000, _n_warmup=2, _n_repeat=4, return_mode=return_mode)

    assert result == pytest.approx(expected)
    assert clock.calls == 12


@pytest.mark.parametrize("quantiles, expected", [([0.0, 0.5, 1.0], [1.0, 2.5, 4.0]), ([0.25], 1.75), ([], [])])
def test_wall_quantiles(clock, quantiles, expected):
    function = _timed_function(clock, [0.001] * 6 + [0.001, 0.004, 0.002, 0.003])

    result = do_bench(function, backend="wall", warmup=0, _n_repeat=4, quantiles=quantiles, return_mode="min")

    assert result == pytest.approx(expected)
    assert clock.calls == 10


def test_wall_time_budgets(clock):
    duration = 1 / 1024
    function = _timed_function(clock, [duration] * 11)

    result = do_bench(function, backend="wall", warmup=duration * 1e3 * 2.5, rep=duration * 1e3 * 3.5)

    assert result == pytest.approx(duration * 1e3)
    assert clock.calls == 11


def test_wall_zero_budgets_and_zero_elapsed_time(clock):
    function = _timed_function(clock, [0.0] * 7)

    assert do_bench(function, backend="wall", warmup=0, rep=0) == 0.0
    assert clock.calls == 7


@pytest.mark.parametrize("quantiles, expected", [(None, 2.0), ([0.5], 2.0), ([0.1, 0.9], [2.0, 2.0])])
def test_wall_early_stop(clock, quantiles, expected):
    function = _timed_function(clock, [0.002] * 6)

    result = do_bench(function, backend="wall", _n_warmup=100, _n_repeat=100, quantiles=quantiles, early_stop_baseline=1.0)

    assert result == pytest.approx(expected)
    assert clock.calls == 6


@pytest.mark.parametrize(
    "options",
    [
        {"return_mode": "sum"},
        {"warmup": -1},
        {"rep": -1},
        {"warmup": float("inf")},
        {"rep": float("nan")},
        {"_n_warmup": -1},
        {"_n_repeat": -1},
        {"quantiles": [-0.1]},
        {"quantiles": [1.1]},
        {"quantiles": [float("nan")]},
        {"device": torch.device("meta")},
    ],
)
def test_wall_rejects_invalid_arguments_before_running(options):
    def unexpected():
        raise AssertionError("Invalid arguments must be rejected before running the callable")

    with pytest.raises(ValueError):
        do_bench(unexpected, backend="wall", **options)


@pytest.mark.parametrize("device", [None, torch.device("cpu")])
def test_wall_cpu_does_not_use_gpu_timing_resources(monkeypatch, clock, device):
    def unexpected(*args, **kwargs):
        raise AssertionError("CPU wall timing must not use GPU timing or tensor allocation")

    for name in ("Event", "Stream", "device", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    monkeypatch.setattr(torch.mps, "synchronize", unexpected)
    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch, "tensor", unexpected)
    function = _timed_function(clock, [0.001] * 7)

    assert do_bench(function, backend="wall", device=device, warmup=0, _n_repeat=1) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(1, id="cuda-index"),
        pytest.param(torch.device("cuda:1"), id="cuda-device"),
        pytest.param(torch.device("mps"), id="metal"),
    ],
)
def test_wall_explicit_device_synchronization(monkeypatch, clock, device):
    contexts = []
    synchronized_devices = []
    resolved_device = torch.device("cuda", device) if isinstance(device, int) else device

    @contextmanager
    def device_context(selected_device):
        contexts.append(("enter", selected_device))
        try:
            yield
        finally:
            contexts.append(("exit", selected_device))

    def synchronize(selected_device):
        synchronized_devices.append(selected_device)
        clock.now += clock.pending
        clock.pending = 0.0

    def function():
        clock.calls += 1
        clock.now += 0.002
        clock.pending += 0.003

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: synchronize(torch.device("mps")))

    result = do_bench(function, backend="wall", device=device, warmup=0, _n_repeat=2)

    assert result == pytest.approx(5.0)
    assert clock.calls == 8
    assert synchronized_devices and all(selected == resolved_device for selected in synchronized_devices)
    expected_contexts = [("enter", resolved_device), ("exit", resolved_device)] if resolved_device.type == "cuda" else []
    assert contexts == expected_contexts


def test_wall_restores_device_context_on_error(monkeypatch):
    contexts = []

    @contextmanager
    def device_context(device):
        contexts.append("enter")
        try:
            yield
        finally:
            contexts.append("exit")

    def failure():
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(torch.cuda, "device", device_context)

    with pytest.raises(RuntimeError, match="kernel failure"):
        do_bench(failure, backend="wall", device=0)
    assert contexts == ["enter", "exit"]


def _check_device_wall_clock(device):
    source = torch.ones(1024, device=device)
    output = torch.empty_like(source)

    def function():
        torch.add(source, 1, out=output)

    latency = do_bench(function, backend="wall", device=source.device, warmup=0, _n_warmup=1, _n_repeat=3)

    assert latency > 0
    torch.testing.assert_close(output, source + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA or ROCm runtime")
def test_wall_gpu_runtime():
    _check_device_wall_clock(torch.device("cuda"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires an MPS runtime")
def test_wall_metal_runtime():
    _check_device_wall_clock(torch.device("mps"))
