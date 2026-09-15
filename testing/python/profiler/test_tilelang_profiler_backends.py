import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.kernel import JITKernel
from tilelang.profiler import Profiler, do_bench
from tilelang.profiler import _common
from tilelang.profiler.device import get_backend, resolve_device
from tilelang.utils.tensor import TensorSupplyType


def _unexpected_cuda_call(*args, **kwargs):
    pytest.fail("CPU/Metal profiling must not call the CUDA runtime")


def _make_profiler(target=None, shape=None):
    param = KernelParam(tvm.DataType("float32"), [8] if shape is None else shape)
    return Profiler([param], [], TensorSupplyType.One, adapter=lambda tensor: tensor + 1, target=target)


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu", 0)])
def test_cpu_device_does_not_probe_cuda(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", _unexpected_cuda_call)
    monkeypatch.setattr(torch.cuda, "current_device", _unexpected_cuda_call)
    assert resolve_device(device) == torch.device("cpu")


@pytest.mark.parametrize("kind", ["c", "llvm"])
def test_cpu_target_overrides_available_gpu(monkeypatch, kind):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", _unexpected_cuda_call)
    profiler = _make_profiler(target=tvm.target.Target(kind))
    assert profiler._get_inputs()[0].device == torch.device("cpu")


def test_cpu_inputs_override_available_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", _unexpected_cuda_call)
    assert resolve_device(inputs=[torch.ones(8), 32]) == torch.device("cpu")


def test_explicit_device_and_target_mismatches():
    inputs = [torch.ones(8)]
    with pytest.raises(ValueError, match="does not match input device"):
        resolve_device("cuda:1", inputs=inputs)
    with pytest.raises(ValueError, match="incompatible with kernel target"):
        resolve_device(inputs=inputs, target=SimpleNamespace(kind=SimpleNamespace(name="metal")))
    with pytest.raises(ValueError, match="inputs on one device"):
        resolve_device(inputs=[*inputs, torch.empty(8, device="meta")])


def test_cuda_current_device_is_resolved_at_call_time(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert resolve_device("cuda") == torch.device("cuda", 3)
    assert resolve_device(2) == torch.device("cuda", 2)


@pytest.mark.parametrize("hip, name", [(None, "cuda"), ("7.0", "rocm")])
def test_gpu_runtime_selects_profiling_module(monkeypatch, hip, name):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", _unexpected_cuda_call)
    implementation = get_backend(torch.device("cuda", 0))
    assert implementation.__name__ == f"tilelang.{name}.profiler"
    assert {"event", "cupti", "cudagraph"} == implementation.SUPPORTED_METHODS


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("method", ["event", "cupti", "cudagraph", "unknown"])
def test_unsupported_methods_fail_before_execution(device, method):
    with pytest.raises(ValueError, match="supported methods: wall"):
        do_bench(lambda: pytest.fail("Unsupported methods must not execute the callable"), device=device, backend=method)


@pytest.mark.parametrize("quantiles", [[], [-0.1], [1.1]])
def test_invalid_quantiles_fail_before_execution(quantiles):
    with pytest.raises(ValueError, match="quantiles"):
        do_bench(lambda: pytest.fail("Invalid options must not execute the callable"), device="cpu", backend="wall", quantiles=quantiles)


@pytest.mark.parametrize("return_mode", ["min", "max", "mean", "median"])
@pytest.mark.parametrize("quantiles", [None, [0.5], [0.25, 0.75]])
def test_wall_statistics_and_iteration_counts(monkeypatch, return_mode, quantiles):
    timestamps = [0.0, 0.005]
    for sample in range(10):
        timestamps.extend([0.01 * (sample + 1), 0.01 * (sample + 1) + 0.003])
    clock = iter(timestamps)
    monkeypatch.setattr(_common, "perf_counter", lambda: next(clock))
    calls = []
    result = do_bench(
        lambda: calls.append(None),
        backend="wall",
        device="cpu",
        _n_warmup=2,
        _n_repeat=3,
        quantiles=quantiles,
        return_mode=return_mode,
    )
    assert len(calls) == 1 + 5 + 2 + 10 * 3
    assert result == pytest.approx([1.0, 1.0] if quantiles is not None and len(quantiles) > 1 else 1.0)


@pytest.mark.parametrize("quantiles", [None, [0.5], [0.25, 0.75]])
def test_wall_early_stop(monkeypatch, quantiles):
    clock = iter([0.0, 0.005])
    monkeypatch.setattr(_common, "perf_counter", lambda: next(clock))
    calls = []
    result = do_bench(lambda: calls.append(None), device="cpu", backend="wall", early_stop_baseline=0.5, quantiles=quantiles)
    assert len(calls) == 6
    assert result == pytest.approx([1.0] * len(quantiles) if quantiles is not None else 1.0)


def test_cpu_profiler_checks_and_dynamic_inputs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", _unexpected_cuda_call)
    monkeypatch.setattr(torch.cuda, "device", _unexpected_cuda_call)
    profiler = _make_profiler(target=tvm.target.Target("c"))
    profiler.assert_allclose(lambda tensor: tensor + 1)
    profiler.manual_assert_close(
        lambda tensor: tensor + 1,
        manual_check_prog=lambda actual, expected: torch.testing.assert_close(actual, expected),
    )
    assert profiler.do_bench(backend="wall", n_warmup=1, n_repeat=2) > 0

    dynamic = _make_profiler(shape=[tvm.tirx.Var("length", "int32")])
    inputs = dynamic._get_inputs(device="cpu", dynamic_symbolic_constraints={"length": 16})
    assert inputs[0].shape == (16,)
    assert inputs[0].device.type == "cpu"
    assert dynamic.do_bench(input_tensors=inputs, backend="wall", n_warmup=1, n_repeat=2) > 0


def test_jit_kernel_passes_its_resolved_target():
    kernel = SimpleNamespace(
        params=[KernelParam(tvm.DataType("float32"), [8])],
        out_idx=[],
        target=tvm.target.Target("c"),
        adapter=lambda tensor: tensor + 1,
    )
    profiler = JITKernel.get_profiler(kernel)
    assert profiler.target.same_as(kernel.target)
    assert profiler._get_inputs()[0].device.type == "cpu"


def test_cpu_profiling_does_not_import_gpu_timer():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tilelang.profiler import do_bench; "
            "do_bench(lambda: None, device='cpu', backend='wall', _n_warmup=1, _n_repeat=1); "
            "assert 'tilelang.profiler._torch_gpu' not in sys.modules",
        ],
        check=True,
    )


def test_cpu_wall_statistics_ignore_default_tensor_device():
    with torch.device("meta"):
        result = do_bench(lambda: None, device="cpu", backend="wall", _n_warmup=1, _n_repeat=1, quantiles=[0.5])
    assert result > 0


def test_metal_wall_synchronizes_pending_work(monkeypatch):
    state = {"pending": 0, "clock": 0.0, "synchronizations": 0}

    def launch():
        state["pending"] += 1

    def synchronize():
        state["clock"] += state["pending"] * 0.001
        state["pending"] = 0
        state["synchronizations"] += 1

    monkeypatch.setattr(torch.cuda, "synchronize", _unexpected_cuda_call)
    monkeypatch.setattr(torch.cuda, "device", _unexpected_cuda_call)
    monkeypatch.setattr(torch.cuda, "Event", _unexpected_cuda_call)
    monkeypatch.setattr(torch.mps, "synchronize", synchronize)
    monkeypatch.setattr(_common, "perf_counter", lambda: state["clock"])
    assert do_bench(launch, device="mps", backend="wall", _n_warmup=2, _n_repeat=3) == pytest.approx(1.0)
    assert state["pending"] == 0
    assert state["synchronizations"] == 13


@tilelang.testing.requires_cuda_or_cdna
@pytest.mark.parametrize("method", ["event", "cudagraph"])
@pytest.mark.parametrize("quantiles", [None, [0.5], [0.25, 0.75]])
def test_gpu_benchmark_methods(method, quantiles):
    tensor = torch.ones(1024, device="cuda")
    result = do_bench(lambda: tensor.add_(1), backend=method, quantiles=quantiles, _n_warmup=1, _n_repeat=3, cache_size=1)
    if quantiles is not None and len(quantiles) > 1:
        assert len(result) == len(quantiles)
        assert 0 < result[0] <= result[1]
    else:
        assert isinstance(result, float)
        assert result > 0


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("method", ["event", "cudagraph"])
def test_cuda_device_scope_and_profiler_inputs(method):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    with torch.cuda.device(0):
        seen_devices = []

        def run(tensor):
            seen_devices.append(torch.cuda.current_device())
            assert tensor.device == torch.device("cuda", 1)
            return tensor + 1

        profiler = _make_profiler()
        profiler.adapter = run
        assert profiler.do_bench(device=1, backend=method, n_warmup=1, n_repeat=2) > 0
        inputs = [torch.ones(8, device="cuda:1")]
        profiler.assert_allclose(run, input_tensors=inputs)
        assert set(seen_devices) == {1}
        assert torch.cuda.current_device() == 0


@tilelang.testing.requires_metal
def test_metal_jit_profiler():
    @tilelang.jit(out_idx=[1], target="metal")
    def copy_kernel():
        @T.prim_func
        def copy(source: T.Tensor((256,), "float32"), destination: T.Tensor((256,), "float32")):
            with T.Kernel(1, threads=32):
                for element in T.Parallel(256):
                    destination[element] = source[element]

        return copy

    profiler = copy_kernel().get_profiler()
    assert profiler._get_inputs()[0].device.type == "mps"
    profiler.assert_allclose(lambda tensor: tensor.clone())
    assert profiler.do_bench(backend="wall", n_warmup=1, n_repeat=2) > 0


@tilelang.testing.requires_cuda
def test_cuda_implicit_device_scopes_cache_and_restores_on_error(monkeypatch):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    cache_devices = []
    original_empty = torch.empty

    def allocate(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        cache_devices.append(tensor.device)
        return tensor

    with torch.cuda.device(1):
        tensor = torch.ones(8, device="cuda")
        monkeypatch.setattr(torch, "empty", allocate)
        assert do_bench(lambda: tensor.add_(1), _n_warmup=1, _n_repeat=2, cache_size=1) > 0
    assert cache_devices == [torch.device("cuda", 1)]

    def fail():
        assert torch.cuda.current_device() == 1
        raise RuntimeError("benchmark failure")

    with torch.cuda.device(0):
        with pytest.raises(RuntimeError, match="benchmark failure"):
            do_bench(fail, device=1)
        assert torch.cuda.current_device() == 0


if __name__ == "__main__":
    tilelang.testing.main()
