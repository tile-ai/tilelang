from __future__ import annotations

import importlib.metadata
import inspect
import os
from functools import lru_cache

import torch
import torch.distributed as dist
from packaging import version

cuda_python_version = importlib.metadata.version("cuda-python")
if version.parse(cuda_python_version) >= version.parse("12.8.0"):
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
else:
    from cuda import cuda, cudart


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


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
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


@lru_cache
def supports_p2p_native_atomic():
    """Check native atomic support for peer-to-peer access between CUDA devices 0 and 1."""

    assert torch.cuda.is_available() and torch.cuda.device_count() > 1

    (err,) = cudart.cudaFree(0)
    CUDA_CHECK(err)

    (err, support) = cudart.cudaDeviceGetP2PAttribute(cudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported, 0, 1)
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


def wait_eq(signal_tensor: torch.Tensor, signal: int, stream: torch.cuda.Stream | None = None, require_i64=False):
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
    return ret[2]
