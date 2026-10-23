"""Artifact and callback transport for isolated autotuner benchmarks."""

from __future__ import annotations

from functools import partial
import io
import pickle
from pathlib import Path

import cloudpickle
import torch

from tilelang import tvm
from tilelang.autotuner.param import AutotuneResult
from tilelang.backend.module import create_backend_context
from tilelang.utils.device import device_synchronize


class _ProfilePickler(cloudpickle.CloudPickler):
    def __init__(self, file, callback_modules):
        super().__init__(file)
        self.callback_modules = callback_modules
        self.tensors = []
        self.tensor_ids = {}

    def _function_reduce(self, obj):
        # Snapshot user callback globals too, including CUDA tensors, rather
        # than importing the caller's module and recreating its global state.
        if obj.__module__ in self.callback_modules:
            return self._dynamic_function_reduce(obj)
        return super()._function_reduce(obj)

    def persistent_id(self, obj):
        if isinstance(obj, torch.Tensor):
            if type(obj) not in (torch.Tensor, torch.nn.Parameter):
                raise TypeError("Tensor subclasses are not supported in isolated benchmark callbacks")
            if obj.device.type not in ("cpu", "cuda"):
                raise TypeError(f"Cannot snapshot benchmark tensor on {obj.device}")
            if id(obj) not in self.tensor_ids:
                self.tensor_ids[id(obj)] = len(self.tensors)
                self.tensors.append(obj)
            return ("tensor", self.tensor_ids[id(obj)])
        return None

    def reducer_override(self, obj):
        if isinstance(obj, type) and obj.__module__ in self.callback_modules:
            return cloudpickle.cloudpickle._dynamic_class_reduce(obj)
        return super().reducer_override(obj)


def serialize_profile(profile_args):
    callbacks = (profile_args.ref_prog, profile_args.supply_prog, profile_args.manual_check_prog)
    modules = set()
    for callback in callbacks:
        while isinstance(callback, partial):
            callback = callback.func
        if callback is not None:
            modules.add(getattr(callback, "__module__", type(callback).__module__))
    # Library functions are already importable and have no user closure to copy.
    modules = {module for module in modules if module and not module.startswith(("torch", "numpy", "tilelang"))}
    data = io.BytesIO()
    try:
        pickler = _ProfilePickler(data, modules)
        pickler.dump(profile_args)
        tensors = io.BytesIO()
        # One torch archive preserves storage aliases, strides, and views across
        # callbacks. torch.save copies CUDA storage; it does not create CUDA IPC
        # handles or share the caller's allocations with the child.
        torch.save(pickler.tensors, tensors)
    except Exception as exc:
        raise ValueError(
            "Isolated autotuning requires serializable ref_prog, supply_prog, and manual_check_prog callbacks. "
            "Create locks, streams, and other runtime resources inside the callback, or use timeout=0 "
            "for execution in the current process without a timeout."
        ) from exc
    return pickle.dumps((data.getvalue(), tensors.getvalue()))


def deserialize_profile(payload):
    data, tensor_data = pickle.loads(payload)

    def map_storage(storage, location):
        return storage.cuda(torch.cuda.current_device()) if location.startswith("cuda") else storage

    tensors = torch.load(io.BytesIO(tensor_data), map_location=map_storage, weights_only=True)

    class ProfileUnpickler(pickle.Unpickler):
        def persistent_load(self, token):
            kind, index = token
            if kind != "tensor":
                raise pickle.UnpicklingError(f"Unexpected profile reference: {kind}")
            return tensors[index]

    return ProfileUnpickler(io.BytesIO(data)).load()


def export_kernel(kernel, path):
    """Export compiled code and its ABI, never a live adapter or CUDA context."""
    if kernel.execution_backend not in ("tvm_ffi", "cython", "nvrtc", "cutedsl", "pto"):
        raise ValueError(f"Isolated autotuning does not support backend {kernel.execution_backend!r}; use timeout=0")
    AutotuneResult()._save_kernel_to_disk(path, kernel)
    metadata = {
        "func": tvm.ir.save_json(kernel.prim_func),
        "target": str(kernel.target),
        "target_host": str(kernel.target_host) if kernel.target_host is not None else None,
        "execution_backend": kernel.execution_backend,
        "out_idx": kernel.out_idx,
        "pass_configs": kernel.pass_configs,
        "compile_flags": kernel.compile_flags,
    }
    (path / "benchmark.pkl").write_bytes(cloudpickle.dumps(metadata))


def collect_benchmark_artifacts(kernel, path):
    # CuTeDSL generates its cubin lazily during the first call. Keep that result
    # for the returned kernel and the autotune cache, even though it was built
    # in the benchmark process. No runtime handles cross the boundary.
    if kernel.execution_backend == "cutedsl":
        cubin = path / "kernel.cubin"
        if cubin.exists():
            destination = Path(kernel.adapter.libpath).with_suffix(".cubin")
            AutotuneResult._safe_write_file(str(destination), "wb", lambda f: f.write(cubin.read_bytes()))
            kernel.adapter.pymodule._cubin_needs_generation = False


def make_worker_factory(profile, device, target_kind, warmup, rep, early_stop_factor):
    return cloudpickle.dumps(partial(_create_runner, profile, device, target_kind, warmup, rep, early_stop_factor))


def bind_benchmark_device(device, target_kind):
    if target_kind in ("cuda", "hip"):
        torch.cuda.set_device(device)
        return device
    if target_kind == "ascend":
        torch.npu.set_device(device)
        return torch.device("npu", device)
    return None


def _create_runner(profile, device, target_kind, warmup, rep, early_stop_factor):
    from tilelang.autotuner.tuner import AutoTuner, _BenchmarkWorkerState

    benchmark_device = bind_benchmark_device(device, target_kind)
    if benchmark_device is not None:
        # Startup, including context creation/recovery, has a separate deadline.
        torch.empty(1, device=benchmark_device)
        device_synchronize(benchmark_device)
    tuner = AutoTuner(lambda: None, [])
    tuner.profile_args = deserialize_profile(profile)
    state = _BenchmarkWorkerState()

    def run(request):
        # Callbacks may change the thread's current device. Each trial starts
        # on its assigned device, including after reusing a healthy process.
        bind_benchmark_device(device, target_kind)
        path = Path(request["path"])
        metadata = cloudpickle.loads((path / "benchmark.pkl").read_bytes())
        context = create_backend_context(metadata["target"], metadata["target_host"], metadata["execution_backend"])
        kernel = AutotuneResult()._load_kernel_from_disk(
            path,
            context,
            out_idx=metadata["out_idx"],
            pass_configs=metadata["pass_configs"],
            compile_flags=metadata["compile_flags"],
            func=tvm.ir.load_json(metadata["func"]),
        )
        if kernel is None:
            raise RuntimeError("Unable to reload benchmark kernel")
        best = request["best_latency"]
        state.shared_best_latency = [best] if best is not None else None
        latency, ref_latency = tuner._benchmark_target(kernel, warmup, rep, early_stop_factor, state, benchmark_device=benchmark_device)
        # Include outstanding work from user callbacks in the timed trial.
        if benchmark_device is not None:
            device_synchronize(benchmark_device)
        return float(latency), float(ref_latency) if ref_latency is not None else None

    return run
