from __future__ import annotations

import torch
import torch.distributed as dist
import datetime
import os
import inspect
from typing import Callable, Sequence
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
from tilescale_ext import _create_tensor, _create_ipc_handle, _sync_ipc_handles, _get_device_tensor
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


def init_dist(local_rank: int, num_local_ranks: int):
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(
        list(range(num_local_ranks * num_nodes)))


def init_distributed(return_tp_group=False, init_nvshmem=True, return_lc_group=False):
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    torch.distributed.init_process_group(
        backend="nccl",
        device_id=torch.device(f'cuda:{LOCAL_RANK}'),
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    torch.cuda.set_device(LOCAL_RANK)
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    torch.cuda.synchronize()
    if init_nvshmem:
        import pynvshmem
        pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    if return_lc_group:
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        base = (RANK // local_world_size) * local_world_size
        LC_GROUP = torch.distributed.new_group(
            list(range(base, base + local_world_size)), backend="nccl")

        return WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP, LC_GROUP
    elif return_tp_group:
        return WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP
    else:
        return WORLD_SIZE, RANK, LOCAL_RANK


def create_tensor(shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    # NOTE(wt): We discovered that IPC only works with tensors explicitly allocated by `cudaMalloc` somehow.
    return _create_tensor(shape, dtype)


# IPC related functions
def get_local_ipc_handle(data: torch.Tensor):
    p = ctypes.c_void_p(data.data_ptr())
    handle = _create_ipc_handle(p.value)
    return handle


def create_dist_tensor(local_rank: int, num_local_ranks: int, data: torch.Tensor, rank: int,
                       group: dist.ProcessGroup):
    assert num_local_ranks == group.size()
    # Synchronize device IDs
    device_ids = [
        None,
    ] * group.size()
    local_device_id = local_rank
    dist.all_gather_object(device_ids, local_device_id, group)

    # Synchronize IPC handles
    ipc_handles = [
        None,
    ] * group.size()
    local_ipc_handle = get_local_ipc_handle(data)
    dist.all_gather_object(ipc_handles, local_ipc_handle, group)
    buffer_ptrs_gpu = torch.empty(group.size(), dtype=torch.uint64, device="cuda")
    _sync_ipc_handles(rank, device_ids,
                      ctypes.c_void_p(buffer_ptrs_gpu.data_ptr()).value, ipc_handles, None)
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


def perf_fn(fn, rep, warmup):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    for n in range(rep + warmup):
        if n == warmup:
            start_event.record()
        output = fn()
    stop_event.record()
    start_event.wait()
    stop_event.wait()
    torch.cuda.current_stream().synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / rep


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

    (err, support) = cudart.cudaDeviceGetP2PAttribute(
        cudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported, 0, 1)
    CUDA_CHECK(err)
    return support == 1


def set_signal(signal_tensor: torch.Tensor, signal: int, stream: torch.cuda.Stream | None = None):
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


def wait_eq(signal_tensor: torch.Tensor,
            signal: int,
            stream: torch.cuda.Stream | None = None,
            require_i64=False):
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
                p2p_status = pynvml.nvmlDeviceGetP2PStatus(cur_handle, remote_handle,
                                                           pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
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
        _has_fullmesh_nvlink = all(
            [i == j or v == 1 for i, row in enumerate(nvlink_matrix) for j, v in enumerate(row)])
        if has_nvlink and not _has_fullmesh_nvlink:
            warnings.warn(
                "⚠️ found NVLink but not fullmesh NVLink, this may cause undefined behavior, please check your GPU topology",
                stacklevel=2,
            )
        return _has_fullmesh_nvlink


def get_device_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Get the device tensor from the host tensor.
    This is implemented via `cudaHostGetDevicePointer`

    Args:
        tensor: The host tensor.

    Returns:
        The device tensor with same meta-data.
    """
    assert tensor.device.type == "cpu"
    assert tensor.is_pinned()
    return _get_device_tensor(tensor)
