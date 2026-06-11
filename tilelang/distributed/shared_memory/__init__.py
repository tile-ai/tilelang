"""Shared memory allocator for distributed communication (IPC + VMM/fabric).

All ops registered via TVM FFI under tl.shared_memory.* namespace.
"""

import ctypes
import ctypes.util

import torch
import tvm_ffi

# ---------- TVM FFI function handles ----------

def _missing_shared_memory_func(name):
    def _missing(*args, **kwargs):
        raise RuntimeError(
            f"TileScale shared-memory FFI function '{name}' is unavailable. "
            "This usually means TileLang was built without CUDA shared-memory support. "
            "Rebuild with CUDA enabled to use distributed shared-memory allocations."
        )

    return _missing


def _get_required_global_func(name):
    func = tvm_ffi.get_global_func(name, allow_missing=True)
    if func is None:
        return _missing_shared_memory_func(name)
    return func


def _get_capability_global_func(name):
    func = tvm_ffi.get_global_func(name, allow_missing=True)
    if func is None:
        return lambda *args, **kwargs: False
    return func


_vmm_malloc = _get_required_global_func("tl.shared_memory.vmm_malloc")
_vmm_free = _get_required_global_func("tl.shared_memory.vmm_free")
_create_vmm_handle = _get_required_global_func("tl.shared_memory.create_vmm_handle")
_open_vmm_handle = _get_required_global_func("tl.shared_memory.open_vmm_handle")
_close_vmm_handle = _get_required_global_func("tl.shared_memory.close_vmm_handle")
_sync_vmm_handles_raw = _get_required_global_func("tl.shared_memory.sync_vmm_handles")

_create_ipc_handle = _get_required_global_func("tl.shared_memory.create_ipc_handle")
_open_ipc_handle = _get_required_global_func("tl.shared_memory.open_ipc_handle")
_close_ipc_handle = _get_required_global_func("tl.shared_memory.close_ipc_handle")
_sync_ipc_handles_raw = _get_required_global_func("tl.shared_memory.sync_ipc_handles")

_supports_vmm_fabric = _get_capability_global_func("tl.shared_memory.supports_vmm_fabric")
_supports_multicast = _get_capability_global_func("tl.shared_memory.supports_multicast")

# Multicast (NVSwitch) ops
_mc_create = _get_required_global_func("tl.shared_memory.mc_create")
_mc_export_handle = _get_required_global_func("tl.shared_memory.mc_export_handle")
_mc_import_handle = _get_required_global_func("tl.shared_memory.mc_import_handle")
_mc_add_device = _get_required_global_func("tl.shared_memory.mc_add_device")
_mc_bind_mem = _get_required_global_func("tl.shared_memory.mc_bind_mem")
_mc_map = _get_required_global_func("tl.shared_memory.mc_map")
_mc_release_handle = _get_required_global_func("tl.shared_memory.mc_release_handle")
_mc_unmap = _get_required_global_func("tl.shared_memory.mc_unmap")
_mc_get_aligned_size = _get_required_global_func("tl.shared_memory.mc_get_aligned_size")


# ---------- tensor_from_ptr (pure Python, no C++ torch dependency) ----------

_dtype_str_to_torch = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "uint8": torch.uint8,
    "byte": torch.uint8,
    "int8": torch.int8,
    "bool": torch.bool,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}

# __cuda_array_interface__ typestr mapping
_torch_dtype_to_typestr = {
    torch.float32: "<f4",
    torch.float64: "<f8",
    torch.float16: "<f2",
    torch.int8: "<i1",
    torch.int16: "<i2",
    torch.int32: "<i4",
    torch.int64: "<i8",
    torch.uint8: "<u1",
    torch.bool: "|b1",
    # bfloat16/uint32/uint64 handled specially
}


class _ExternalCUDAArray:
    """Minimal __cuda_array_interface__ provider for zero-copy tensor creation."""

    def __init__(self, ptr: int, shape: tuple, typestr: str):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": typestr,
            "version": 3,
            "strides": None,
        }


def tensor_from_ptr(
    ptr_val: int,
    shape: list,
    dtype_str: str = "float32",
    device: int = 0,
    take_ownership: bool = False,
) -> torch.Tensor:
    """Create a CUDA tensor viewing external device memory (zero-copy)."""
    if ptr_val == 0:
        raise RuntimeError("Received null pointer (0).")

    dtype = _dtype_str_to_torch.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype string: '{dtype_str}'")

    if isinstance(shape, (list, tuple)):
        shape = tuple(shape)
    else:
        shape = (shape,)

    numel = 1
    for s in shape:
        numel *= s
    if numel == 0:
        return torch.empty(shape, dtype=dtype, device=f"cuda:{device}")

    typestr = _torch_dtype_to_typestr.get(dtype)
    if typestr is not None:
        # Standard path via __cuda_array_interface__
        arr = _ExternalCUDAArray(ptr_val, shape, typestr)
        return torch.as_tensor(arr, device=f"cuda:{device}")
    else:
        # bfloat16 / uint32 / uint64: create as matching-size int type, then view
        element_size = torch.empty((), dtype=dtype).element_size()
        if element_size == 2:
            # proxy_dtype = torch.int16
            proxy_typestr = "<i2"
        elif element_size == 4:
            # proxy_dtype = torch.int32
            proxy_typestr = "<i4"
        elif element_size == 8:
            # proxy_dtype = torch.int64
            proxy_typestr = "<i8"
        else:
            raise ValueError(f"Cannot handle dtype {dtype} with element_size={element_size}")

        arr = _ExternalCUDAArray(ptr_val, shape, proxy_typestr)
        t = torch.as_tensor(arr, device=f"cuda:{device}")
        return t.view(dtype)


# ---------- Higher-level Python wrappers ----------


def _sync_vmm_handles(rank, device_ids, buffer_ptrs_gpu_addr, all_gathered_handles):
    """Compatibility wrapper: packs handles into a single bytes blob and calls FFI."""
    num = len(device_ids)
    # all_gathered_handles is a list of bytearrays (or bytes)
    # Pack into single contiguous bytes blob
    # handle_size = len(all_gathered_handles[0]) if all_gathered_handles[0] is not None else 0
    packed = b""
    for h in all_gathered_handles:
        packed += bytes(h)
    _sync_vmm_handles_raw(rank, num, buffer_ptrs_gpu_addr, packed)


def _sync_ipc_handles(rank, device_ids, buffer_ptrs_gpu_addr, all_gathered_handles, root_unique_id_opt=None):
    """Compatibility wrapper for IPC handle sync."""
    num = len(device_ids)
    packed = b""
    for h in all_gathered_handles:
        packed += bytes(h)
    _sync_ipc_handles_raw(rank, num, buffer_ptrs_gpu_addr, packed)


def _create_tensor(shape, dtype):
    """Create a CUDA tensor (simple cudaMalloc-backed)."""
    return torch.empty(shape, dtype=dtype, device="cuda")


def create_host_device_tensor(shape, dtype):
    """Create host/device tensor views backed by one CUDA managed allocation."""
    if not isinstance(shape, (list, tuple)):
        shape = (shape,)
    else:
        shape = tuple(shape)

    numel = 1
    for s in shape:
        numel *= s

    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    cudart = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    cudart.cudaMallocManaged.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    cudart.cudaMallocManaged.restype = ctypes.c_int
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int

    ptr = ctypes.c_void_p()
    rc = cudart.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(nbytes), 1)
    if rc != 0:
        raise RuntimeError(f"cudaMallocManaged failed with error code {rc}")

    class _ManagedAllocation:
        def __init__(self, ptr):
            self.ptr = ptr
            self.cudart = cudart
            self.buffer = (ctypes.c_byte * nbytes).from_address(ptr.value)

        def __del__(self):
            if getattr(self, "ptr", None) and self.ptr.value:
                self.cudart.cudaFree(self.ptr)
                self.ptr = ctypes.c_void_p()

    allocation = _ManagedAllocation(ptr)
    host = torch.frombuffer(allocation.buffer, dtype=dtype, count=numel).reshape(shape)
    device = tensor_from_ptr(ptr.value, list(shape), str(dtype).split(".")[-1], torch.cuda.current_device(), False)
    host._tilelang_managed_allocation = allocation
    device._tilelang_managed_allocation = allocation
    return host, device


__all__ = [
    "tensor_from_ptr",
    "_create_tensor",
    "_create_ipc_handle",
    "_open_ipc_handle",
    "_close_ipc_handle",
    "_sync_ipc_handles",
    "create_host_device_tensor",
    "_supports_vmm_fabric",
    "_vmm_malloc",
    "_vmm_free",
    "_create_vmm_handle",
    "_open_vmm_handle",
    "_close_vmm_handle",
    "_sync_vmm_handles",
    "_supports_multicast",
]
