from __future__ import annotations

import sys
import torch
import torch.distributed

import importlib.metadata

cuda_python_version = importlib.metadata.version("cuda-python")
from packaging import version

if version.parse(cuda_python_version) >= version.parse("12.8.0"):
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
else:
    from cuda import cuda, cudart

from enum import IntEnum

try:
    from _pynvshmem import *  # noqa: F403
except Exception as e:
    print(
        "please add NVSHMEM library path to LD_LIBRARY_PATH and try again",
        flush=True,
        file=sys.stderr,
    )
    raise e


def _CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def broadcast_cpu(tensor: torch.Tensor, src: int, group: torch.distributed.ProcessGroup):
    if not tensor.is_cuda:
        tensor_gpu = tensor.cuda()
        torch.distributed.broadcast(tensor_gpu, src=src, group=group)
        tensor.copy_(tensor_gpu)
    else:
        torch.distributed.broadcast(tensor, src=src, group=group)
    torch.cuda.synchronize()


def init_nvshmem_by_uniqueid(group: torch.distributed.ProcessGroup):
    rank, nranks = group.rank(), group.size()
    if rank == 0:
        unique_id: bytes = nvshmemx_get_uniqueid()  # noqa: F405
        unique_id = torch.frombuffer(unique_id, dtype=torch.uint8).clone()
    else:
        unique_id = torch.empty(128, dtype=torch.uint8)

    broadcast_cpu(tensor=unique_id, group=group, src=0)

    unique_id = unique_id.numpy().tobytes()
    nvshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)  # noqa: F405
    nvshmem_barrier_all()
    torch.cuda.synchronize()


"""Host-side signaling functions."""


def write32_on_stream(tensor: torch.Tensor, value: int, stream: torch.cuda.Stream | None = None):
    """Atomic write an 32-bit value to a tensor.
    Args:
        tensor (torch.Tensor): The tensor to write to.
        value (int): The value to write.
        stream (torch.cuda.Stream | None): The CUDA stream to use for the operation.
            If None, the current stream will be used.
    """
    assert isinstance(tensor, torch.Tensor) and tensor.dtype in (torch.int32, torch.uint32), (
        f"tensor must be a torch.Tensor with 32-bit dtype, but got {tensor.dtype}"
    )
    assert tensor.numel() == 1, "tensor must have exactly one element"
    if stream is None:
        stream = torch.cuda.current_stream()
    (err,) = cuda.cuStreamWriteValue32(
        stream.cuda_stream,
        tensor.data_ptr(),
        value,
        cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    _CUDA_CHECK(err)


def write64_on_stream(tensor: torch.Tensor, value: int, stream: torch.cuda.Stream | None = None):
    """Atomic write an 64-bit value to a tensor.
    Args:
        tensor (torch.Tensor): The tensor to write to.
        value (int): The value to write.
        stream (torch.cuda.Stream | None): The CUDA stream to use for the operation.
            If None, the current stream will be used.
    """
    assert isinstance(tensor, torch.Tensor) and tensor.dtype in (torch.int64, torch.uint64), (
        f"tensor must be a torch.Tensor with 64-bit dtype, but got {tensor.dtype}"
    )
    assert tensor.numel() == 1, "tensor must have exactly one element"
    if stream is None:
        stream = torch.cuda.current_stream()
    (err,) = cuda.cuStreamWriteValue64(
        stream.cuda_stream,
        tensor.data_ptr(),
        value,
        cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    _CUDA_CHECK(err)


### Distributed enums ###


class Team(IntEnum):
    INVALID = -1
    WORLD = 0
    WORLD_INDEX = 0
    SHARED = 1
    SHARED_INDEX = 1
    NODE = 2
    NODE_INDEX = 2
    SAME_MYPE_NODE = 3
    SAME_MYPE_NODE_INDEX = 3
    SAME_GPU = 4
    SAME_GPU_INDEX = 4
    GPU_LEADERS = 5
    GPU_LEADERS_INDEX = 5
    TEAMS_MIN = 6
    TEAM_INDEX_MAX = sys.maxsize


class CmpType(IntEnum):
    EQ = 0
    NE = 1
    GT = 2
    LE = 3
    LT = 4
    GE = 5
    SENTINEL = sys.maxsize


class Amo(IntEnum):
    """Atomic Memory Operation (AMO) types.
    Note: Signal ops (NVSHMEMI_AMO_SIGNAL_SET and NVSHMEMI_AMO_SIGNAL_ADD) are
    included as a part of the AMO operations.
    """

    AMO_ACK = 1
    AMO_INC = 2
    AMO_SET = 3
    AMO_ADD = 4
    AMO_AND = 5
    AMO_OR = 6
    AMO_XOR = 7
    AMO_SIGNAL = 8
    SIGNAL_SET = 9
    SIGNAL_ADD = 10
    AMO_SIGNAL_SET = 9  # the same as SIGNAL_SET
    AMO_SIGNAL_ADD = 10  # the same as SIGNAL_ADD
    AMO_END_OF_NONFETCH = 11  # end of nonfetch atomics
    AMO_FETCH = 12
    AMO_FETCH_INC = 13
    AMO_FETCH_ADD = 14
    AMO_FETCH_AND = 15
    AMO_FETCH_OR = 16
    AMO_FETCH_XOR = 17
    AMO_SWAP = 18
    AMO_COMPARE_SWAP = 19
    AMO_OP_SENTINEL = sys.maxsize
