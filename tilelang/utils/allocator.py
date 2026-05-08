from __future__ import annotations

import ctypes
import ctypes.util
import os
import torch
import torch.distributed as dist
from tilelang.distributed.shared_memory import (
    tensor_from_ptr,
    _create_ipc_handle,
    _sync_ipc_handles,
    _vmm_malloc,
    _vmm_free,
    _create_vmm_handle,
    _sync_vmm_handles,
    _supports_vmm_fabric,
    _supports_multicast,
    _mc_create,
    _mc_export_handle,
    _mc_import_handle,
    _mc_add_device,
    _mc_bind_mem,
    _mc_map,
    _mc_release_handle,
    _mc_unmap,
    _mc_get_aligned_size,
)
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
    torch.uint16: "uint16",
    torch.uint32: "uint32",
    torch.uint64: "uint64",
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


def _resolve_use_vmm(use_vmm: bool | None, is_distributed: bool = False) -> bool:
    """Resolve whether to use VMM based on env var and hardware support."""
    env_val = os.environ.get("TILESCALE_USE_VMM", None)
    if env_val is not None:
        return env_val == "1"
    if use_vmm is not None:
        return use_vmm
    return is_distributed and _supports_vmm_fabric()


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
        use_vmm: bool | None = None,
        mcast_size: int | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be > 0")
        self.size = int(size)
        self._use_vmm = _resolve_use_vmm(use_vmm, is_distributed)
        self._base_ptr = ctypes.c_void_p(0)
        self._ptr = ctypes.c_void_p(0)
        self._device = parse_device(device)
        self._is_distributed = is_distributed
        self._local_rank = local_rank
        self._num_local_ranks = num_local_ranks
        self._group = group
        self._align = align
        self._mcast_size_requested = mcast_size
        # table items:
        # 1. local_rank, size: 8 bytes
        # 2. num_local_ranks, size: 8 bytes
        # 3. buffer_ptrs, size: 8 bytes * num_local_ranks
        # total size: 16 + 8 * num_local_ranks
        self._table = None
        self._buffer_ptrs = None
        self._device_ids = None
        self._initialized = False
        self._closed = False
        # Multicast state
        self._mcast_base_ptr = 0
        self._mcast_ptr = 0
        self._mcast_phys_ptr = 0
        self._mcast_aligned_size = 0
        self._use_multicast = False

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

        if self._use_vmm:
            ptr_val = _vmm_malloc(self.size)
            self._base_ptr.value = ptr_val
        else:
            rc = _libcudart.cudaMalloc(ctypes.byref(self._base_ptr), ctypes.c_size_t(self.size))
            if rc != 0:
                msg = _libcudart.cudaGetErrorString(rc)
                raise RuntimeError(f"cudaMalloc failed: {rc} {msg.decode() if msg else ''}")
        self._ptr.value = self._base_ptr.value

        # Multicast buffer (only when explicitly requested via mcast_size)
        if self._mcast_size_requested is not None:
            assert self._use_vmm, "mcast_size requires use_vmm=True"
            assert self._is_distributed, "mcast_size requires is_distributed=True"
            if _supports_multicast():
                self._init_multicast_buffer()
            else:
                raise RuntimeError("Multicast not supported on this hardware")

    def _init_multicast_buffer(self):
        """Create multicast object and map, following multi-process fabric pattern."""
        mcast_size = self._mcast_size_requested if self._mcast_size_requested else self.size
        num_devices = self._num_local_ranks
        aligned = _mc_get_aligned_size(mcast_size, num_devices)
        self._mcast_aligned_size = aligned

        # Allocate physical memory (reuses vmm_malloc, same fabric handle type)
        self._mcast_phys_ptr = _vmm_malloc(aligned)

        # Rank 0 creates MC object, exports fabric handle; broadcast to all
        if self._local_rank == 0:
            mcast_handle = _mc_create(aligned, num_devices)
            mcast_fabric_bytes = _mc_export_handle(mcast_handle)
        else:
            mcast_handle = 0
            mcast_fabric_bytes = None

        obj_list = [mcast_fabric_bytes]
        dist.broadcast_object_list(obj_list, src=0, group=self._group)
        mcast_fabric_bytes = obj_list[0]

        # Non-rank-0 import the MC handle
        if self._local_rank != 0:
            mcast_handle = _mc_import_handle(mcast_fabric_bytes)

        # Each rank adds its own device
        _mc_add_device(mcast_handle, self._local_rank)

        # Barrier: all devices must be added before binding
        dist.barrier(self._group)

        # Each rank binds its own physical memory
        _mc_bind_mem(mcast_handle, self._mcast_phys_ptr, aligned)

        # Barrier: all binds must complete before mapping
        dist.barrier(self._group)

        # Each rank maps the MC object to a local VA
        self._mcast_base_ptr = _mc_map(mcast_handle, aligned, num_devices)
        self._mcast_ptr = self._mcast_base_ptr

        # Release handle (backing persists due to mapping)
        _mc_release_handle(mcast_handle)
        self._use_multicast = True

    def _allocate_mcast_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate from multicast buffer (bump-pointer).

        Returns:
            (mcast_tensor, local_tensor):
                mcast_tensor: backed by MC VA, for multimem read instructions
                local_tensor: backed by physical VA, for writing data
        """
        if not self._use_multicast:
            raise RuntimeError("Multicast buffer not initialized")

        numel = _prod_shape(shape)
        itemsize = _element_size_bytes(dtype)
        bytes_needed = numel * itemsize
        bytes_alloc = _align_up(bytes_needed, self._align)

        current_offset = self._mcast_ptr - self._mcast_base_ptr
        if current_offset + bytes_alloc > self._mcast_aligned_size:
            raise MemoryError(
                f"Mcast allocation failed: Requesting {bytes_alloc} bytes, but only "
                f"{self._mcast_aligned_size - current_offset} bytes available "
                f"(total mcast size: {self._mcast_aligned_size} bytes)."
            )

        dtype_str = _dtype_to_str.get(dtype)
        if dtype_str is None:
            dtype_str = str(dtype).split(".")[-1]
        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            shape = [shape]

        mcast_t = tensor_from_ptr(self._mcast_ptr, shape, dtype_str, self._device, False)
        local_t = tensor_from_ptr(self._mcast_phys_ptr + current_offset, shape, dtype_str, self._device, False)
        self._mcast_ptr += bytes_alloc
        return mcast_t, local_t

    def close(self):
        """Explicitly free resources with proper distributed coordination.

        Must be called collectively by all ranks before process group destruction.
        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        # Barrier before multicast teardown to ensure no rank is still using MC VA
        if getattr(self, "_use_multicast", False) and self._group is not None:
            with contextlib.suppress(Exception):
                dist.barrier(self._group)
        self._free()

    def _free(self):
        # Free multicast resources
        if getattr(self, "_mcast_base_ptr", 0) and self._mcast_base_ptr:
            _mc_unmap(self._mcast_base_ptr, self._mcast_aligned_size, self._num_local_ranks)
            self._mcast_base_ptr = 0
            self._mcast_ptr = 0
        if getattr(self, "_mcast_phys_ptr", 0) and self._mcast_phys_ptr:
            _vmm_free(self._mcast_phys_ptr)
            self._mcast_phys_ptr = 0
        self._use_multicast = False

        # Free main buffer
        if getattr(self, "_base_ptr", None) and self._base_ptr.value:
            if getattr(self, "_use_vmm", False):
                _vmm_free(self._base_ptr.value)
                self._base_ptr = ctypes.c_void_p(0)
            else:
                rc = _libcudart.cudaFree(self._base_ptr)
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

        # Synchronize handles (VMM or IPC)
        handles = [
            None,
        ] * self._group.size()
        if self._use_vmm:
            local_handle = _create_vmm_handle(self._base_ptr.value)
        else:
            local_handle = _create_ipc_handle(self._base_ptr.value)
        dist.all_gather_object(handles, local_handle, self._group)

        buffer_ptrs = torch.empty(self._group.size(), dtype=torch.uint64, device="cuda")
        if self._use_vmm:
            _sync_vmm_handles(self._local_rank, device_ids, ctypes.c_void_p(buffer_ptrs.data_ptr()).value, handles)
        else:
            _sync_ipc_handles(self._local_rank, device_ids, ctypes.c_void_p(buffer_ptrs.data_ptr()).value, handles, None)
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
                    # NOTE: This is a workaround, as different CUDA driver versions have different behaviors
                    # on the device of the peer tensor.
                    try:
                        peer_t = tensor_from_ptr(peer_ptr_val, shape, dtype_str, peer_device, False)
                    except Exception as e:
                        if isinstance(e, ValueError) and "does not match device of data" in str(e):
                            peer_t = tensor_from_ptr(peer_ptr_val, shape, dtype_str, self._device, False)
                        else:
                            raise e
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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()


def get_allocator(
    size: int = 2**30,
    device: str = "cuda",
    is_distributed: bool = True,
    local_rank: int = 0,
    num_local_ranks: int = 1,
    group: dist.ProcessGroup | None = None,
    use_vmm: bool | None = None,
    mcast_size: int | None = None,
) -> BaseAllocator:
    return BaseAllocator(
        size,
        device=device,
        is_distributed=is_distributed,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
        use_vmm=use_vmm,
        mcast_size=mcast_size,
    )
