from types import SimpleNamespace

import pytest
import torch

from tilelang.profiler import bench, do_bench, torch_bench
from tilelang.utils import device as device_utils


@pytest.fixture
def metal_events(monkeypatch):
    state = SimpleNamespace(now=0.0, calls=0, flushes=0, synchronizations=0, event_synchronizations=0, events=[], allocations=[])

    class Event:
        def __init__(self, enable_timing=False):
            assert enable_timing
            self.timestamp = None
            state.events.append(self)

        def record(self):
            self.timestamp = state.now

        def synchronize(self):
            state.event_synchronizations += 1

        def elapsed_time(self, end_event):
            return end_event.timestamp - self.timestamp

    def synchronize():
        state.synchronizations += 1

    def flush():
        state.flushes += 1
        state.now += 0.25

    def empty(numel, *, dtype, device):
        assert torch.device(device).type == "mps"
        state.allocations.append((numel, dtype, torch.device(device)))
        return SimpleNamespace(zero_=flush)

    def unexpected(*args, **kwargs):
        raise AssertionError("MPS event timing must not use CUDA APIs")

    for module in (bench, torch_bench, device_utils):
        monkeypatch.setattr(module, "Event", Event)
    monkeypatch.setattr(bench, "device", "mps:0")
    monkeypatch.setattr(device_utils, "IS_CUDA", False)
    monkeypatch.setattr(device_utils, "IS_NPU", False)
    monkeypatch.setattr(device_utils, "IS_MPS", True)
    monkeypatch.setattr(torch.mps, "synchronize", synchronize)
    monkeypatch.setattr(torch, "empty", empty)
    for name in ("Event", "device", "synchronize", "current_device", "Stream", "CUDAGraph"):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    return state


@pytest.mark.parametrize("device", [None, torch.device("mps"), "mps:0"])
@pytest.mark.parametrize(
    "return_mode, quantiles, expected",
    [
        ("min", None, 1.0),
        ("max", None, 3.0),
        ("mean", None, 2.0),
        ("median", None, 2.0),
        ("mean", [0.5], 2.0),
        ("mean", [0.25, 0.75], [1.5, 2.5]),
    ],
)
def test_event_metal_timing(metal_events, device, return_mode, quantiles, expected):
    durations = iter([1.0] * 8 + [1.0, 2.0, 3.0])

    def function():
        metal_events.calls += 1
        metal_events.now += next(durations)

    result = do_bench(
        function,
        backend="event",
        device=device,
        _n_warmup=2,
        _n_repeat=3,
        cache_size=1,
        return_mode=return_mode,
        quantiles=quantiles,
    )

    assert result == pytest.approx(expected)
    assert metal_events.calls == 11
    assert metal_events.flushes == 9
    assert metal_events.synchronizations == 3
    assert metal_events.event_synchronizations == 2
    assert len(metal_events.events) == 8
    cache_device = torch.device(device) if device is not None else torch.device("mps:0")
    assert metal_events.allocations == [(1024 * 1024 // 4, torch.int, cache_device)]


@pytest.mark.parametrize("device", [None, torch.device("mps"), "mps:0"])
@pytest.mark.parametrize("backend", ["cupti", "cudagraph"])
def test_metal_rejects_cuda_only_timing(metal_events, device, backend):
    def unexpected():
        raise AssertionError("Unsupported timing must fail before running the callable")

    with pytest.raises(ValueError, match='use "event" or "wall" for MPS'):
        do_bench(unexpected, backend=backend, device=device)
    assert not metal_events.events
    assert not metal_events.allocations
    assert metal_events.synchronizations == 0


def test_event_early_stop_excludes_cache_flush(metal_events):
    def function():
        metal_events.calls += 1
        metal_events.now += 0.01

    result = do_bench(function, backend="event", _n_warmup=1, _n_repeat=2, cache_size=1, early_stop_baseline=0.02)
    assert result == pytest.approx(0.01)
    assert metal_events.calls == 9


@pytest.mark.parametrize("quantiles,expected", [(None, 2.0), ([0.5], 2.0), ([0.25, 0.75], [2.0, 2.0])])
def test_event_early_stop_returns_kernel_estimate(metal_events, quantiles, expected):
    def function():
        metal_events.calls += 1
        metal_events.now += 2.0

    result = do_bench(
        function,
        backend="event",
        _n_warmup=100,
        _n_repeat=100,
        cache_size=1,
        quantiles=quantiles,
        early_stop_baseline=1.0,
    )
    assert result == pytest.approx(expected)
    assert metal_events.calls == 6


def test_event_early_stop_keeps_flush_in_iteration_budget(metal_events):
    def function():
        metal_events.calls += 1
        metal_events.now += 0.001

    result = do_bench(function, backend="event", warmup=0.6, rep=0.9, cache_size=1, early_stop_baseline=0.002)
    assert result == pytest.approx(0.001)
    # Each flushed iteration costs 0.251 ms: two warmups and three repeats.
    assert metal_events.calls == 11
    assert metal_events.flushes == 9


def test_event_zero_estimate_keeps_minimum_iterations(metal_events, monkeypatch):
    class ZeroElapsedEvent(bench.Event):
        def elapsed_time(self, end_event):
            return 0.0

    monkeypatch.setattr(bench, "Event", ZeroElapsedEvent)
    monkeypatch.setattr(torch_bench, "Event", ZeroElapsedEvent)

    def function():
        metal_events.calls += 1

    assert do_bench(function, backend="event", warmup=0, rep=0, cache_size=1, early_stop_baseline=1.0) == 0.0
    assert metal_events.calls == 8


def test_event_early_stop_keeps_equal_threshold_candidate(metal_events):
    def function():
        metal_events.calls += 1
        metal_events.now += 0.125

    result = do_bench(function, backend="event", _n_warmup=1, _n_repeat=2, cache_size=1, early_stop_baseline=0.125)
    assert result == 0.125
    assert metal_events.calls == 9
