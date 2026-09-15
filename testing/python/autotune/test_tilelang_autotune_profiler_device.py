from types import SimpleNamespace

import torch

import tilelang.testing
from tilelang import tvm
from tilelang.autotuner import AutoTuner, set_autotune_inputs
from tilelang.autotuner.param import CompileArgs
from tilelang.autotuner.tuner import _BenchmarkWorkerState
from tilelang.engine.param import KernelParam
from tilelang.profiler import Profiler
from tilelang.utils.tensor import TensorSupplyType


def test_cpu_captured_inputs_follow_resolved_target(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    inputs = [torch.ones(8)]
    tuner = AutoTuner(lambda: None, [])
    tuner.compile_args = CompileArgs(target=tvm.target.Target("c"))
    with set_autotune_inputs(inputs):
        tuner.set_profile_args(backend="wall")
    supplied = tuner.profile_args.supply_prog([])
    assert supplied[0].device.type == "cpu"
    assert supplied[0].data_ptr() != inputs[0].data_ptr()
    torch.testing.assert_close(supplied[0], inputs[0])


def test_cpu_autotuner_profiles_candidate_and_reference():
    profiler = Profiler(
        [KernelParam(tvm.DataType("float32"), [8])],
        [],
        TensorSupplyType.One,
        adapter=lambda tensor: tensor + 1,
        target=tvm.target.Target("c"),
    )
    kernel = SimpleNamespace(get_profiler=lambda tensor_supply_type: profiler)
    tuner = AutoTuner(lambda: None, [])
    tuner.set_profile_args(backend="wall", ref_prog=lambda tensor: tensor + 1)
    state = _BenchmarkWorkerState()
    latency, reference = tuner._benchmark_target(kernel, warmup=1, rep=2, early_stop_factor=0, benchmark_state=state)
    assert latency > 0
    assert reference > 0
    assert state.jit_input_tensors[0].device.type == "cpu"
    assert state.ref_input_tensors[0].device.type == "cpu"


@tilelang.testing.requires_cuda
def test_cuda_autotuner_passes_worker_device():
    if torch.cuda.device_count() < 2:
        import pytest

        pytest.skip("Requires two CUDA devices")
    profiler = Profiler(
        [KernelParam(tvm.DataType("float32"), [8])],
        [],
        TensorSupplyType.One,
        adapter=lambda tensor: tensor + 1,
    )
    kernel = SimpleNamespace(get_profiler=lambda tensor_supply_type: profiler)
    tuner = AutoTuner(lambda: None, [])
    tuner.set_profile_args(ref_prog=lambda tensor: tensor + 1)
    state = _BenchmarkWorkerState()
    with torch.cuda.device(0):
        latency, reference = tuner._benchmark_target(
            kernel, warmup=1, rep=2, early_stop_factor=0, benchmark_state=state, benchmark_device=1
        )
        assert torch.cuda.current_device() == 0
    assert latency > 0
    assert reference > 0
    assert state.jit_input_tensors[0].device == torch.device("cuda", 1)
    assert state.ref_input_tensors[0].device == torch.device("cuda", 1)


@tilelang.testing.requires_metal
def test_metal_captured_inputs_follow_resolved_target():
    inputs = [torch.ones(8, device="mps")]
    tuner = AutoTuner(lambda: None, [])
    tuner.compile_args = CompileArgs(target=tvm.target.Target("metal"))
    with set_autotune_inputs(inputs):
        tuner.set_profile_args(backend="wall")
    supplied = tuner.profile_args.supply_prog([])
    assert supplied[0].device.type == "mps"
    assert supplied[0].data_ptr() != inputs[0].data_ptr()
    torch.testing.assert_close(supplied[0], inputs[0])


if __name__ == "__main__":
    tilelang.testing.main()
