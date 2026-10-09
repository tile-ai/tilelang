from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tilelang.profiler import Profiler, TensorSupplyType, bench, torch_bench
from tilelang.utils import device as device_utils


@pytest.fixture
def accelerator(monkeypatch):
    def create(device_type):
        if not hasattr(torch, device_type):
            pytest.skip(f"Requires PyTorch {device_type} device registration")
        api = getattr(torch, device_type)
        state = SimpleNamespace(current=0, contexts=[], calls=[], events=[], synchronizations=[], allocations=[], flushes=0)

        @contextmanager
        def device_context(device):
            selected = device if isinstance(device, int) else torch.device(device).index
            previous = state.current
            state.current = previous if selected is None else selected
            state.contexts.append(("enter", state.current))
            try:
                yield
            finally:
                state.contexts.append(("exit", state.current))
                state.current = previous

        class Event:
            def __init__(self, enable_timing=False):
                assert enable_timing

            def record(self):
                state.events.append(state.current)

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 4.0

        def synchronize(device=None):
            state.synchronizations.append((device, state.current))

        def empty(numel, *, dtype, device):
            requested = torch.device(device)
            assert requested.type == device_type
            selected = state.current if requested.index is None else requested.index
            tensor_device = torch.device(device_type, selected)
            state.allocations.append(tensor_device)

            def zero():
                state.flushes += 1

            return SimpleNamespace(device=tensor_device, dtype=dtype, numel=lambda: numel, zero_=zero)

        def function():
            state.calls.append(state.current)

        monkeypatch.setattr(api, "current_device", lambda: state.current)
        monkeypatch.setattr(api, "device", device_context)
        monkeypatch.setattr(api, "synchronize", synchronize)
        monkeypatch.setattr(api, "Event", Event)
        monkeypatch.setattr(torch, "empty", empty)
        monkeypatch.setattr(device_utils, "IS_CUDA", device_type == "cuda")
        monkeypatch.setattr(device_utils, "IS_NPU", device_type == "npu")
        monkeypatch.setattr(device_utils, "IS_MPS", False)
        monkeypatch.setattr(bench, "device", "cuda:0" if device_type == "cuda" else "npu")
        for module in (bench, torch_bench, device_utils):
            monkeypatch.setattr(module, "Event", Event)
        state.function = function
        return state

    return create


@pytest.mark.parametrize("backend", ["event", "cupti", "cudagraph"])
@pytest.mark.parametrize("device", [None, 0, 2, torch.device("cuda:2"), torch.device("cuda"), "cuda:2"])
def test_cuda_scope_covers_shared_path_and_backend(monkeypatch, accelerator, backend, device):
    state = accelerator("cuda")
    calls = []
    selected = 0 if device is None else device if isinstance(device, int) else torch.device(device).index or 0

    def implementation(function, cache, n_repeat, *options):
        calls.append((state.current, cache.device, n_repeat, options))
        function()
        return 1.5

    monkeypatch.setattr(bench, f"_bench_with_{'cuda_events' if backend == 'event' else backend}", implementation)
    result = bench.do_bench(state.function, device=device, backend=backend, _n_warmup=1, _n_repeat=2, cache_size=2)

    assert result == 1.5
    assert state.current == 0
    assert state.calls == [selected] * 8
    assert state.events == [selected, selected]
    assert state.allocations == [torch.device("cuda", selected)]
    device_idx = None if device is None else selected
    assert state.synchronizations == [(device_idx, selected)] * 2
    assert state.flushes == 6
    options = () if backend == "cupti" else (None, "mean", device_idx)
    assert calls == [(selected, torch.device("cuda", selected), 2, options)]


def test_cuda_scope_restored_after_failure(accelerator):
    state = accelerator("cuda")

    def failure():
        assert state.current == 2
        raise RuntimeError("kernel failure")

    with pytest.raises(RuntimeError, match="kernel failure"):
        bench.do_bench(failure, device=2)
    assert state.current == 0
    assert state.contexts == [("enter", 2), ("exit", 2)]
    assert not state.allocations


def test_cuda_graph_stream_uses_explicit_device(monkeypatch, accelerator):
    state = accelerator("cuda")
    streams = []
    replays = []

    def stream(device=None):
        streams.append(device)
        return SimpleNamespace(device=device)

    @contextmanager
    def stream_context(selected_stream):
        with torch.cuda.device(selected_stream.device):
            yield

    @contextmanager
    def capture(graph):
        yield

    monkeypatch.setattr(torch.cuda, "Stream", stream)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=lambda: replays.append(state.current)))
    monkeypatch.setattr(torch.cuda, "graph", capture)
    cache = SimpleNamespace(zero_=lambda: None)

    result = torch_bench.bench_with_cudagraph(state.function, cache, 2, None, "mean", 2)

    assert result == 2.0
    assert streams == [2]
    assert state.calls == [2, 2]
    assert replays == [2] * 10
    assert state.events == [2] * 20
    assert state.synchronizations == [(2, 2), (2, 2)]
    assert state.current == 0


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("backend", ["event", "msprof", "msprof_detail"])
def test_npu_shared_path_and_timing_backend(monkeypatch, accelerator, backend, explicit):
    state = accelerator("npu")
    state.current = 0 if explicit else 2
    selected_device = torch.device("npu:2") if explicit else None
    msprof_calls = []
    detailed_result = object()

    def msprof(function, cache, n_repeat, detailed=False):
        msprof_calls.append((state.current, cache.device, n_repeat, detailed))
        function()
        return detailed_result if detailed else 1.5

    def unexpected(*args, **kwargs):
        raise AssertionError("NPU profiling must not call CUDA APIs")

    if backend != "event":
        from tilelang.profiler import msprof as msprof_module

        monkeypatch.setattr(msprof_module, "bench_with_msprof", msprof)
    for name in ("Event", "device", "current_device", "synchronize", "Stream", "CUDAGraph"):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    result = bench.do_bench(state.function, device=selected_device, backend=backend, _n_warmup=1, _n_repeat=2, cache_size=2)

    assert state.current == (0 if explicit else 2)
    assert state.allocations == [torch.device("npu:2")]
    assert set(state.calls) == {2}
    assert set(state.events) == {2}
    assert all(context == 2 for _, context in state.synchronizations)
    if backend == "event":
        assert result == 4.0
        assert not msprof_calls
    else:
        if backend == "msprof_detail":
            assert result is detailed_result
        else:
            assert result == 1.5
        assert msprof_calls == [(2, torch.device("npu:2"), 2, backend == "msprof_detail")]


def test_npu_profiler_input_generation_is_scoped(monkeypatch, accelerator):
    state = accelerator("npu")
    input_devices = []
    instance = Profiler([], [], TensorSupplyType.Auto, lambda: state.function())

    def inputs():
        input_devices.append(state.current)
        return []

    monkeypatch.setattr(instance, "_get_inputs", inputs)
    assert instance.do_bench(device=torch.device("npu:2"), n_warmup=1, n_repeat=2) == 4.0
    assert input_devices == [2]
    assert set(state.calls) == {2}
    assert state.current == 0


def test_npu_wall_synchronizes_selected_device(monkeypatch, accelerator):
    state = accelerator("npu")
    samples = []

    def wall(function, n_repeat, synchronize, **kwargs):
        samples.append(n_repeat)
        synchronize()
        function()
        synchronize()
        return 1.5

    monkeypatch.setattr(bench, "bench_with_wall", wall)
    result = bench.do_bench(state.function, backend="wall", device=torch.device("npu:2"), _n_warmup=1, _n_repeat=2)
    assert result == 1.5
    assert samples == [5, 2]
    assert state.current == 0
    assert state.calls == [2] * 4
    assert state.synchronizations == [(torch.device("npu:2"), 2)] * 6
    assert not state.allocations
