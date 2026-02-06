from __future__ import annotations

import ctypes
import ctypes.util
import torch
import torch.distributed as dist
from tilescale_ext import tensor_from_ptr, _create_ipc_handle, _sync_ipc_handles
from tilelang.utils.target import parse_device
import contextlib

_dtype_to_str = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.bool: "bool",
}


def _element_size_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _prod_shape(shape: tuple[int, ...] | int) -> int:
    if isinstance(shape, int):
        return shape
    p = 1
    for d in shape:
        if d < 0:
            raise ValueError("negative dimension in shape")
        p *= int(d)
    return p


def _align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


# helper: load CUDA runtime library
def _load_cudart():
    name = ctypes.util.find_library("cudart") or ctypes.util.find_library("cuda")
    if not name:
        # fallback common linux name
        name = "libcudart.so"
    try:
        lib = ctypes.CDLL(name)
    except OSError as e:
        raise RuntimeError(f"cannot load cudart ({name}): {e}") from e
    return lib


_libcudart = _load_cudart()
# setup signatures
_libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_libcudart.cudaMalloc.restype = ctypes.c_int
_libcudart.cudaFree.argtypes = [ctypes.c_void_p]
_libcudart.cudaFree.restype = ctypes.c_int
_libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
_libcudart.cudaGetErrorString.restype = ctypes.c_char_p
# optional set device
if hasattr(_libcudart, "cudaSetDevice"):
    _libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
    _libcudart.cudaSetDevice.restype = ctypes.c_int


class BaseAllocator:
    func: callable | None = None

    def __init__(
        self,
        size: int,
        device: str | torch.device | int | None = None,
        is_distributed: bool = False,
        local_rank: int | None = None,
        num_local_ranks: int | None = None,
        group: dist.ProcessGroup | None = None,
        align: int = 256,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be > 0")
        self.size = int(size)
        self._base_ptr = ctypes.c_void_p(0)
        self._ptr = ctypes.c_void_p(0)
        self._device = parse_device(device)
        self._is_distributed = is_distributed
        self._local_rank = local_rank
        self._num_local_ranks = num_local_ranks
        self._group = group
        self._align = align
        # table items:
        # 1. local_rank, size: 8 bytes
        # 2. num_local_ranks, size: 8 bytes
        # 3. buffer_ptrs, size: 8 bytes * num_local_ranks
        # total size: 16 + 8 * num_local_ranks
        self._table = None
        self._buffer_ptrs = None
        self._device_ids = None
        self._initialized = False
        if self._is_distributed:
            assert self._group is not None, "group must be provided when is_distributed is True"
            assert self._local_rank is not None, "local_rank must be provided when is_distributed is True"
            assert self._num_local_ranks is not None, "num_local_ranks must be provided when is_distributed is True"
            assert self._group.size() == self._num_local_ranks, "group.size() must be equal to num_local_ranks"

        self._alloc()
        if self._is_distributed:
            self._init_table()
        self._initialized = True

    @property
    def device(self) -> int:
        return self._device

    def _alloc(self):
        # optionally set device
        if self._device is not None:
            rc = _libcudart.cudaSetDevice(int(self._device))
            if rc != 0:
                raise RuntimeError(f"cudaSetDevice failed: {rc} {_libcudart.cudaGetErrorString(rc).decode()}")
        rc = _libcudart.cudaMalloc(ctypes.byref(self._base_ptr), ctypes.c_size_t(self.size))
        if rc != 0:
            msg = _libcudart.cudaGetErrorString(rc)
            raise RuntimeError(f"cudaMalloc failed: {rc} {msg.decode() if msg else ''}")
        self._ptr.value = self._base_ptr.value

    def _free(self):
        if getattr(self, "_base_ptr", None) and self._base_ptr.value:
            rc = _libcudart.cudaFree(self._base_ptr)
            # mark freed even if error to avoid double free in destructor
            self._base_ptr = ctypes.c_void_p(0)
            if rc != 0:
                msg = _libcudart.cudaGetErrorString(rc)
                raise RuntimeError(f"cudaFree failed: {rc} {msg.decode() if msg else ''}")

    def _init_table(self):
        device_ids = [
            None,
        ] * self._group.size()
        local_device_id = self._local_rank
        dist.all_gather_object(device_ids, local_device_id, self._group)
        self._device_ids = device_ids

        # Synchronize IPC handles
        ipc_handles = [
            None,
        ] * self._group.size()
        local_ipc_handle = _create_ipc_handle(self._base_ptr.value)
        dist.all_gather_object(ipc_handles, local_ipc_handle, self._group)
        buffer_ptrs = torch.empty(self._group.size(), dtype=torch.uint64, device="cuda")
        _sync_ipc_handles(self._local_rank, device_ids, ctypes.c_void_p(buffer_ptrs.data_ptr()).value, ipc_handles, None)
        buffer_ptrs[self._local_rank] = self._base_ptr.value
        self._buffer_ptrs = buffer_ptrs
        self._table_size = 2 + self._group.size()
        self._table = torch.empty(self._table_size, dtype=torch.uint64, device="cpu")
        self._table[0] = self._local_rank
        self._table[1] = self._group.size()
        self._table[2:] = buffer_ptrs

    def initialized(self) -> bool:
        return self._initialized

    def _allocate_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype, return_peers=False, take_ownership: bool = False
    ) -> torch.Tensor:
        numel = _prod_shape(shape)
        itemsize = _element_size_bytes(dtype)
        bytes_needed = numel * itemsize

        bytes_alloc = _align_up(bytes_needed, self._align)

        current_offset = int(self._ptr.value) - int(self._base_ptr.value)
        if current_offset + bytes_alloc > self.size:
            bytes_available = self.size - current_offset
            raise MemoryError(
                f"Allocation failed: Requesting {bytes_alloc} bytes, but only "
                f"{bytes_available} bytes are available in the pre-allocated buffer "
                f"(total size: {self.size} bytes)."
            )

        if not isinstance(self._ptr, ctypes.c_void_p):
            raise TypeError("self._ptr must be ctypes.c_void_p")
        cur_ptr_val = int(self._ptr.value)
        if cur_ptr_val == 0:
            raise RuntimeError("null device pointer")

        dtype_str = _dtype_to_str.get(dtype)
        if dtype_str is None:
            dtype_str = str(dtype).split(".")[-1]

        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            shape = [shape]

        t = tensor_from_ptr(cur_ptr_val, shape, dtype_str, self._device, take_ownership)

        if return_peers:
            peer_ts = []
            for i in range(self._group.size()):
                if i == self._local_rank:
                    peer_ts.append(t)
                else:
                    peer_ptr_val = int(self._buffer_ptrs[i]) + current_offset
                    peer_device = self._device_ids[i]
                    peer_t = tensor_from_ptr(peer_ptr_val, shape, dtype_str, peer_device, False)
                    peer_ts.append(peer_t)

        if take_ownership:
            self._ptr = ctypes.c_void_p(0)
        else:
            new_ptr_val = cur_ptr_val + bytes_alloc
            self._ptr.value = new_ptr_val

        return peer_ts if return_peers else t

    @property
    def ptr(self) -> int:
        return int(self._ptr.value) if self._ptr and self._ptr.value else 0

    @property
    def table(self) -> torch.Tensor:
        return self._table

    @property
    def table_size(self) -> int:
        return self._table_size

    def __del__(self):
        with contextlib.suppress(Exception):
            self._free()


def get_allocator(
    size: int = 2**30,
    device: str = "cuda",
    is_distributed: bool = True,
    local_rank: int = 0,
    num_local_ranks: int = 1,
    group: dist.ProcessGroup | None = None,
) -> BaseAllocator:
    return BaseAllocator(
        size, device=device, is_distributed=is_distributed, local_rank=local_rank, num_local_ranks=num_local_ranks, group=group
    )
