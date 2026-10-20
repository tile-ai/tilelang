"""CUDA integration and callback transport for isolated benchmarking."""

import json
import os
from pathlib import Path
import threading
from functools import partial

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.autotuner import AutoTuner, set_autotune_inputs
from tilelang.autotuner.benchmark import deserialize_profile, export_kernel, serialize_profile
from tilelang.autotuner.param import AutotuneResult, CompileArgs, ProfileArgs


@pytest.mark.parametrize("field", ["ref_prog", "supply_prog", "manual_check_prog"])
def test_unserializable_callback_is_explicit(field):
    lock = threading.Lock()

    def callback(*args):
        with lock:
            return args

    with pytest.raises(ValueError, match="serializable.*timeout=0"):
        serialize_profile(ProfileArgs(**{field: callback}))


def test_cpu_callback_tensors_are_snapshots():
    value = torch.arange(20).reshape(4, 5).T

    def supply(_params):
        return [value]

    profile = deserialize_profile(serialize_profile(ProfileArgs(supply_prog=supply)))
    copied = profile.supply_prog(None)[0]
    assert copied.stride() == value.stride()
    torch.testing.assert_close(copied, value)
    copied.zero_()
    assert value.count_nonzero() == 19


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_callback_tensor_aliases_survive_transport(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    base = torch.arange(20, device=device).reshape(4, 5)
    view = base.T[1:]

    def supply(_params):
        return [base, view]

    copied_base, copied_view = deserialize_profile(serialize_profile(ProfileArgs(supply_prog=supply))).supply_prog(None)
    assert copied_view.stride() == view.stride()
    assert copied_view.storage_offset() == view.storage_offset()
    torch.testing.assert_close(copied_view, view)
    copied_base.fill_(7)
    torch.testing.assert_close(copied_view, torch.full_like(view, 7))
    assert base.count_nonzero() == 19


_GLOBAL_INPUT = torch.arange(4)


def _global_supply(params, scale=1):
    return [_GLOBAL_INPUT]


class _GlobalSupplier:
    def __call__(self, params):
        return [_GLOBAL_INPUT]


@pytest.mark.parametrize("callback", [_global_supply, partial(_global_supply, scale=2), _GlobalSupplier()])
def test_importable_callbacks_snapshot_globals(callback):
    profile = deserialize_profile(serialize_profile(ProfileArgs(supply_prog=callback)))
    copied = profile.supply_prog(None)[0]
    torch.testing.assert_close(copied, _GLOBAL_INPUT)
    assert copied.data_ptr() != _GLOBAL_INPUT.data_ptr()


def _add_one(n=128, threads=64):
    @T.prim_func
    def kernel(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        with T.Kernel(1, threads=threads):
            for i in T.Parallel(n):
                B[i] = A[i] + 1

    return kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_cuda_timeout_then_recovery(tmp_path, execution_backend):
    events = str(tmp_path / "events.jsonl")
    parent_tensor = torch.ones(128, device="cuda")

    def supply(params):
        import time

        n = int(params[0].shape[0])
        with open(events, "a") as f:
            f.write(json.dumps([n, os.getpid()]) + "\n")
        # Exercise real CUDA before blocking in Python, without wedging the GPU.
        parent_tensor.add_(1)
        torch.cuda.synchronize()
        if n == 64:
            time.sleep(60)
        return [parent_tensor[:n]]

    trials = []

    def observe(record):
        trials.append(record)
        if record["status"] == "timeout":
            records = [json.loads(line) for line in Path(events).read_text().splitlines()]
            slow_pid = next(pid for n, pid in records if n == 64)
            with pytest.raises(ProcessLookupError):
                os.kill(slow_pid, 0)

    tuner = (
        AutoTuner(_add_one, [{"n": 64}, {"n": 128}])
        .set_compile_args(out_idx=[1], target="cuda", execution_backend=execution_backend)
        .set_profile_args(supply_prog=supply, ref_prog=lambda x: x + 1)
    )
    result = tuner.run(warmup=1, rep=1, timeout=2, use_pipeline=True, on_trial=observe)
    assert result.config == {"n": 128}
    assert sorted(trial["status"] for trial in trials) == ["ok", "timeout"]
    torch.testing.assert_close(parent_tensor, torch.ones_like(parent_tensor))
    torch.testing.assert_close(result.kernel(parent_tensor), parent_tensor + 1)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("grouped,multi_gpu", [(False, False), (True, False), (False, True)])
def test_process_input_cache_and_device_binding(tmp_path, grouped, multi_gpu):
    if multi_gpu and torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    events = str(tmp_path / "events.jsonl")
    expected = 3
    # A captured CUDA tensor must be copied onto each worker's device.
    captured = torch.full((128,), expected, dtype=torch.float32, device="cuda")

    def supply(params):
        assert captured.device.index == torch.cuda.current_device()
        with open(events, "a") as f:
            f.write(json.dumps([os.getpid(), torch.cuda.current_device(), expected]) + "\n")
        return [captured.clone()]

    def ref(x):
        torch.testing.assert_close(x, torch.full_like(x, expected))
        return x + 1

    def check(actual, expected):
        torch.testing.assert_close(actual, expected)

    tuner = (
        AutoTuner(_add_one, [{"threads": t} for t in (32, 64, 128, 256)])
        .set_compile_args(out_idx=[1], target="cuda")
        .set_profile_args(supply_prog=supply, ref_prog=ref, manual_check_prog=check, cache_input_tensors=True)
    )
    result = tuner.run(
        warmup=1,
        rep=1,
        timeout=20,
        use_pipeline=True,
        enable_grouped_compile=grouped,
        benchmark_multi_gpu=multi_gpu,
        benchmark_devices=[0, 1],
        early_stop=True,
    )
    records = [json.loads(line) for line in Path(events).read_text().splitlines()]
    # One supply for the cached benchmark inputs, one for reference timing.
    assert len(records) == (4 if multi_gpu else 2)
    assert len({pid for pid, _, _ in records}) == (2 if multi_gpu else 1)
    assert {device for _, device, _ in records} == ({0, 1} if multi_gpu else {torch.cuda.current_device()})
    torch.testing.assert_close(result.kernel(captured), captured + 1)

    if not grouped and not multi_gpu:
        expected = 7
        captured.fill_(expected)
        result = tuner.run(warmup=1, rep=1, timeout=20)
        records = [json.loads(line) for line in Path(events).read_text().splitlines()]
        assert [value for _, _, value in records] == [3, 3, 7, 7]
        assert len({pid for pid, _, _ in records}) == 2
        torch.testing.assert_close(result.kernel(captured), captured + 1)


@tilelang.testing.requires_cuda
def test_captured_inputs_are_isolated():
    x = torch.ones(128, device="cuda")
    before = x.clone()
    with set_autotune_inputs(x):
        tuner = (
            AutoTuner(_add_one, [{"threads": 32}, {"threads": 64}])
            .set_compile_args(out_idx=[1], target="cuda")
            .set_profile_args(ref_prog=lambda x: x + 1, cache_input_tensors=True)
        )
    result = tuner.run(warmup=1, rep=1, timeout=20)
    torch.testing.assert_close(x, before)
    torch.testing.assert_close(result.kernel(x), x + 1)


@tilelang.testing.requires_cuda
def test_unresolved_compile_defaults():
    tuner = AutoTuner(_add_one, [{"threads": 32}, {"threads": 64}]).set_profile_args(ref_prog=lambda x: x + 1)
    tuner.compile_args = CompileArgs(out_idx=[1])
    result = tuner.run(warmup=1, rep=1, timeout=20, enable_grouped_compile=True)
    x = torch.ones(128, device="cuda")
    torch.testing.assert_close(result.kernel(x), x + 1)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("timeout", [0, 20])
def test_each_trial_restores_its_assigned_device(timeout):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")

    def supply(params):
        assert torch.cuda.current_device() == 0
        inputs = [torch.ones(tuple(params[0].shape), device="cuda:0")]
        torch.cuda.set_device(1)
        return inputs

    tuner = AutoTuner(_add_one, [{"n": 64}, {"n": 128}]).set_compile_args(out_idx=[1], target="cuda").set_profile_args(supply_prog=supply)
    trials = []
    with torch.cuda.device(0):
        tuner.run(warmup=1, rep=1, timeout=timeout, on_trial=trials.append)
    assert [trial["status"] for trial in trials] == ["ok", "ok"]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_cached_kernel_can_be_exported_again(tmp_path, execution_backend):
    kernel = tilelang.compile(_add_one(), out_idx=[1], target="cuda", execution_backend=execution_backend)
    export_kernel(kernel, tmp_path / "first")
    loaded = AutotuneResult()._load_kernel_from_disk(
        tmp_path / "first",
        kernel.backend_context,
        out_idx=kernel.out_idx,
        pass_configs=kernel.pass_configs,
        compile_flags=kernel.compile_flags,
        func=kernel.prim_func,
    )
    assert loaded is not None
    export_kernel(loaded, tmp_path / "second")
    x = torch.ones(128, device="cuda")
    torch.testing.assert_close(loaded(x), x + 1)
