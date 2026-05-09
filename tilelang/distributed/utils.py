from __future__ import annotations

import torch
import torch.distributed as dist
import os
import inspect
from typing import Callable, Literal
from collections.abc import Sequence
from contextlib import contextmanager

import importlib.metadata

cuda_python_version = importlib.metadata.version("cuda-python")
from packaging import version

if version.parse(cuda_python_version) >= version.parse("12.8.0"):
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
else:
    from cuda import cuda, cudart

import ctypes
from tilelang.distributed.shared_memory import (
    _create_tensor,
    _create_ipc_handle,
    _sync_ipc_handles,
    _create_vmm_handle,
    _sync_vmm_handles,
    create_host_device_tensor,
)
import functools
from functools import lru_cache
from threading import Lock
import subprocess
import warnings

dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
    "float32": torch.float32,
}


def init_dist(local_rank: int, num_local_ranks: int, master_port: int | None = None):
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_DEBUG", "ERROR")

    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = (
        master_port
        if master_port is not None
        else int(os.getenv("TILESCALE_MASTER_PORT", os.getenv("MASTER_PORT", "8361")))
    )
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_nodes * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in sig.parameters:
        # noinspection PyTypeChecker
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


def create_tensor(shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    # NOTE(wt): We discovered that IPC only works with tensors explicitly allocated by `cudaMalloc` somehow.
    return _create_tensor(shape, dtype)


# IPC related functions
def get_local_ipc_handle(data: torch.Tensor):
    p = ctypes.c_void_p(data.data_ptr())
    handle = _create_ipc_handle(p.value)
    return handle


def _resolve_use_vmm(use_vmm: bool | None) -> bool:
    """Resolve whether to use VMM based on env var and hardware support."""
    import os

    env_val = os.environ.get("TILESCALE_USE_VMM", None)
    if env_val is not None:
        return env_val == "1"
    if use_vmm is not None:
        return use_vmm
    return False


def create_dist_tensor(
    local_rank: int,
    num_local_ranks: int,
    data: torch.Tensor,
    rank: int,
    group: dist.ProcessGroup,
    use_vmm: bool | None = None,
):
    assert num_local_ranks == group.size()
    _use_vmm = _resolve_use_vmm(use_vmm)

    # Synchronize device IDs
    device_ids = [
        None,
    ] * group.size()
    local_device_id = local_rank
    dist.all_gather_object(device_ids, local_device_id, group)

    # Synchronize handles (VMM or IPC)
    handles = [
        None,
    ] * group.size()
    if _use_vmm:
        local_handle = _create_vmm_handle(ctypes.c_void_p(data.data_ptr()).value)
    else:
        local_handle = get_local_ipc_handle(data)
    dist.all_gather_object(handles, local_handle, group)

    buffer_ptrs_gpu = torch.empty(group.size(), dtype=torch.uint64, device="cuda")
    if _use_vmm:
        _sync_vmm_handles(rank, device_ids, ctypes.c_void_p(buffer_ptrs_gpu.data_ptr()).value, handles)
    else:
        _sync_ipc_handles(rank, device_ids, ctypes.c_void_p(buffer_ptrs_gpu.data_ptr()).value, handles, None)
    return buffer_ptrs_gpu


@contextmanager
def with_torch_deterministic(mode: bool, warn_only: bool = True):
    old_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_mode, warn_only=warn_only)


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 1 and dtype.is_floating_point


def _make_tensor(
    shape: list[int | Callable[[], int]],
    dtype: torch.dtype,
    init_args: tuple[float, float] | tuple[int, int],
    device: str = "cuda",
):
    """
    rand() * scale + bias
    randint(-scale, scale) + bias
    """
    if isinstance(shape, Sequence):
        shape = tuple([x() if isinstance(x, Callable) else x for x in shape])
    elif isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, Callable):
        shape = shape()
    else:
        raise ValueError(f"unsupported shape {shape}")

    scale, bias = init_args
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        out = (torch.rand(shape, dtype=dtype, device=device) * 2 - 1) * scale + bias
    elif dtype == torch.int8:
        out = torch.randint(-scale, scale, shape, dtype=torch.int8, device=device)
        out = out + bias
    elif is_fp8_dtype(dtype):
        out = (torch.rand(shape, dtype=torch.float16, device=device) * 2 - 1) * scale + bias
        with with_torch_deterministic(False):
            out = out.to(dtype)
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    return out


def generate_data(configs):
    while True:
        yield (_make_tensor(*args) if args else None for args in configs)


def dist_print(*args, **kwargs):
    """A wrapped distributed version of the built-in `print` function.
    Args:
        allowed_ranks (list[int] or "all"): The ranks that are allowed to print. Default: [0].
        prefix (bool): Whether to add a prefix indicating the rank number. Default: False.
        need_sync (bool): Whether to synchronize all ranks before printing. Default: False.
    Note:
        This function requires the environment variables "RANK" and "WORLD_SIZE" to be set.
    Example:
    ```
    dist_print("Hello, world!", allowed_ranks=[0, 1], prefix=True, need_sync=True)
    ```
    """
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    prefix = False
    if "allowed_ranks" in kwargs:
        allowed_ranks = kwargs["allowed_ranks"]
        if isinstance(allowed_ranks, str) and allowed_ranks == "all":
            allowed_ranks = list(range(world_size))

        del kwargs["allowed_ranks"]
    else:
        allowed_ranks = [0]
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]

        del kwargs["prefix"]

    need_sync = False
    if "need_sync" in kwargs:
        need_sync = kwargs["need_sync"]

        del kwargs["need_sync"]

    for allowed in allowed_ranks:
        if need_sync:
            torch.distributed.barrier()
        if rank == allowed:
            if prefix:
                print(f"[rank:{rank}]", end="")
            print(*args, **kwargs)


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _resolve_bench_group(group):
    if not _dist_ready():
        return None
    return group if group is not None else dist.group.WORLD


def _reduce_benchmark_times(
    times: "np.ndarray",
    *,
    group=None,
    aggregate: Literal["max", "mean", "min", "none"] = "max",
    device: str | torch.device = "cuda",
) -> "np.ndarray":
    if aggregate == "none" or not _dist_ready():
        return times

    group = _resolve_bench_group(group)
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return times

    time_tensor = torch.as_tensor(times, dtype=torch.float64, device=device)
    if aggregate == "max":
        op = dist.ReduceOp.MAX
    elif aggregate == "min":
        op = dist.ReduceOp.MIN
    elif aggregate == "mean":
        op = dist.ReduceOp.SUM
    else:
        raise ValueError(f"unsupported benchmark aggregate mode: {aggregate}")

    dist.all_reduce(time_tensor, op=op, group=group)
    if aggregate == "mean":
        time_tensor /= world_size
    return time_tensor.cpu().numpy()


def do_bench(
    fn,
    warmup: int = 50,
    rep: int = 50,
    post_fn=None,
    *,
    group=None,
    aggregate: Literal["max", "mean", "min", "none"] = "max",
    flush_l2: bool = True,
    l2_flush_bytes: int = int(256e6),
    barrier_comm_profiling: bool = True,
    barrier: Callable | None = None,
    discard_first: bool = True,
    return_mode: Literal["mean", "median", "min", "max", "all"] = "mean",
):
    """Benchmark a CUDA function and normalize timing across distributed ranks.

    For distributed runs, the default result is the mean of per-iteration max
    latency across ranks. This is the latency visible to a collective/multi-GPU
    operation, and avoids reporting an arbitrary rank's local CUDA event time.

    Args:
        fn: the function to benchmark.
        warmup: number of warmup iterations.
        rep: number of measured iterations.
        post_fn: optional function to call after each measured iteration.
        group: process group used for cross-rank timing reduction.
        aggregate: cross-rank reduction for each iteration.
        flush_l2: whether to flush L2 before each measured iteration.
        l2_flush_bytes: size of the flush buffer in bytes.
        barrier_comm_profiling: insert a DeepEP-style GPU sleep + comm barrier
            before each measured iteration to reduce CPU launch skew. This only
            takes effect for distributed groups with more than one rank.
        barrier: custom barrier used when barrier_comm_profiling is enabled.
        discard_first: drop the first measured iteration when rep > 1.
        return_mode: aggregate over measured iterations.

    Returns:
        Runtime in milliseconds. If return_mode is "all", returns the measured
        per-iteration runtimes after cross-rank aggregation.
    """
    import numpy as np

    if rep <= 0:
        raise ValueError("rep must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if l2_flush_bytes <= 0:
        raise ValueError("l2_flush_bytes must be positive")

    group = _resolve_bench_group(group)

    torch.cuda.synchronize()
    if group is not None:
        dist.barrier(group)
    world_size = dist.get_world_size(group) if group is not None else 1
    use_barrier_comm_profiling = barrier_comm_profiling and group is not None and world_size > 1

    cache = None
    if flush_l2:
        cache = torch.empty(int(l2_flush_bytes // 4), dtype=torch.int, device="cuda")

    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    if group is not None:
        dist.barrier(group)

    dummy = None
    if use_barrier_comm_profiling and barrier is None:
        dummy = torch.ones(1, dtype=torch.float32, device="cuda")

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        if cache is not None:
            cache.zero_()
        if use_barrier_comm_profiling:
            if hasattr(torch.cuda, "_sleep"):
                torch.cuda._sleep(int(2e7))
            if barrier is None:
                dist.all_reduce(dummy, group=group)
            else:
                barrier()
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    local_times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=np.float64)
    if discard_first and len(local_times) > 1:
        local_times = local_times[1:]
    times = _reduce_benchmark_times(local_times, group=group, aggregate=aggregate)
    if group is not None:
        dist.barrier(group)

    if return_mode == "all":
        return times
    if return_mode == "mean":
        return np.average(times).item()
    if return_mode == "median":
        return np.median(times).item()
    if return_mode == "min":
        return np.min(times).item()
    if return_mode == "max":
        return np.max(times).item()
    raise ValueError(f"unsupported benchmark return_mode: {return_mode}")


perf_fn = do_bench  # backward compatibility


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


@lru_cache
def supports_p2p_native_atomic():
    """Check if native atomic operations are supported for peer-to-peer (P2P) access between CUDA devices 0 and 1."""

    assert torch.cuda.is_available() and torch.cuda.device_count() > 1

    # force create CUDA context
    (err,) = cudart.cudaFree(0)
    CUDA_CHECK(err)

    (err, support) = cudart.cudaDeviceGetP2PAttribute(cudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported, 0, 1)
    CUDA_CHECK(err)
    return support == 1


def set_signal(signal_tensor: torch.Tensor, signal: int, stream: torch.cuda.Stream | None = None):
    # host side
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype in (torch.int32, torch.uint32):
        (err,) = cuda.cuStreamWriteValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            signal,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        CUDA_CHECK(err)
    else:
        raise Exception(f"Unsupported signal dtype {signal_tensor.dtype}")


def wait_eq(signal_tensor: torch.Tensor, signal: int, stream: torch.cuda.Stream | None = None, require_i64=False):
    # host side
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        (err,) = cuda.cuStreamWaitValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            signal,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)
    else:
        raise Exception(f"Unsupported signal dtype {signal_tensor.dtype}")


def cuda_stream_max_priority():
    ret = cudart.cudaDeviceGetStreamPriorityRange()
    CUDA_CHECK(ret[0])
    return ret[2]  # (leastPriority, greatestPriority) -> greatestPriority is max priority


_pynvml_initialized = False
_lock = Lock()


def ensure_nvml_initialized():
    global _pynvml_initialized
    if not _pynvml_initialized:
        with _lock:
            if not _pynvml_initialized:
                import pynvml

                pynvml.nvmlInit()
                _pynvml_initialized = True


@functools.lru_cache
def has_fullmesh_nvlink_pynvml():
    num_devices = torch.cuda.device_count()

    ensure_nvml_initialized()
    import pynvml

    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_devices)]
        for cur_device in range(num_devices):
            cur_handle = handles[cur_device]
            for remote_device in range(num_devices):
                if remote_device == cur_device:
                    continue
                remote_handle = handles[remote_device]
                p2p_status = pynvml.nvmlDeviceGetP2PStatus(cur_handle, remote_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                    return False
        return True
    except pynvml.NVMLError_NotSupported:
        return False


class NvidiaSmiUtil:
    @staticmethod
    def get_nvlink_adjacency_matrix():
        output = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        lines = [line.strip() for line in output.split("\n") if line.startswith("GPU")]

        device_count = len(lines)
        matrix = [[-1 for _ in range(device_count)] for _ in range(device_count)]

        # 解析每行数据
        for i, line in enumerate(lines):
            parts = line.split()
            for j in range(1, len(parts)):
                if "NV" in parts[j]:
                    matrix[i][j - 1] = 1

        return matrix

    @staticmethod
    def get_gpu_numa_node(gpu_index=0):
        try:
            cmd = f"nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader,nounits -i {gpu_index}"
            pci_id = subprocess.check_output(cmd, shell=True).decode().strip()
            pci_address = pci_id.replace("00000000:", "").lower()

            numa_node_path = f"/sys/bus/pci/devices/0000:{pci_address}/numa_node"
            with open(numa_node_path) as f:
                numa_node = int(f.read().strip())

            assert numa_node >= 0
            return numa_node if numa_node >= 0 else 0

        except Exception as e:
            print(f"Error: {e}")
            return -1


@functools.lru_cache
def has_fullmesh_nvlink():
    try:
        return has_fullmesh_nvlink_pynvml()
    except Exception:
        nvlink_matrix = NvidiaSmiUtil.get_nvlink_adjacency_matrix()
        has_nvlink = any([any(x == 1 for x in row) for row in nvlink_matrix])
        _has_fullmesh_nvlink = all([i == j or v == 1 for i, row in enumerate(nvlink_matrix) for j, v in enumerate(row)])
        if has_nvlink and not _has_fullmesh_nvlink:
            warnings.warn(
                "⚠️ found NVLink but not fullmesh NVLink, this may cause undefined behavior, please check your GPU topology",
                stacklevel=2,
            )
        return _has_fullmesh_nvlink


def create_mapped_tensor(shape: list[int], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    return create_host_device_tensor(shape, dtype)
