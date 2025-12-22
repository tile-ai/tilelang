from __future__ import annotations
import ctypes
import sys

try:
    import torch.cuda._CudaDeviceProperties as _CudaDeviceProperties
except ImportError:
    _CudaDeviceProperties = type("DummyCudaDeviceProperties", (), {})


class cudaDeviceAttrNames:
    r"""
    refer to https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49e2f8c2c0bd6fe264f2fc970912e5cd
    """

    cudaDevAttrMaxThreadsPerBlock: int = 1
    cudaDevAttrMaxRegistersPerBlock: int = 12
    cudaDevAttrMaxSharedMemoryPerMultiprocessor: int = 81
    cudaDevAttrMaxPersistingL2CacheSize: int = 108


def get_cuda_device_properties(device_id: int = 0) -> _CudaDeviceProperties | None:
    try:
        import torch.cuda

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(torch.device(device_id))
    except ImportError:
        return None


def get_device_name(device_id: int = 0) -> str | None:
    prop = get_cuda_device_properties(device_id)
    if prop:
        return prop.name


def get_shared_memory_per_block(device_id: int = 0, format: str = "bytes") -> int | None:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    prop = get_cuda_device_properties(device_id)
    if prop is None:
        raise RuntimeError("Failed to get device properties.")
    shared_mem = int(prop.shared_memory_per_block)
    if format == "bytes":
        return shared_mem
    elif format == "kb":
        return shared_mem // 1024
    elif format == "mb":
        return shared_mem // (1024 * 1024)
    else:
        raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")


def get_device_attribute(attr: int, device_id: int = 0) -> int:
    try:
        if sys.platform == "win32":
            libcudart = ctypes.windll.LoadLibrary("cudart64_110.dll")
        else:
            libcudart = ctypes.cdll.LoadLibrary("libcudart.so")

        value = ctypes.c_int()
        cudaDeviceGetAttribute = libcudart.cudaDeviceGetAttribute
        cudaDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        cudaDeviceGetAttribute.restype = ctypes.c_int

        ret = cudaDeviceGetAttribute(ctypes.byref(value), attr, device_id)
        if ret != 0:
            raise RuntimeError(f"cudaDeviceGetAttribute failed with error {ret}")

        return value.value
    except Exception as e:
        print(f"Error getting device attribute: {str(e)}")
        return None


def get_max_dynamic_shared_size_bytes(device_id: int = 0, format: str = "bytes") -> int | None:
    """
    Get the maximum dynamic shared memory size in bytes, kilobytes, or megabytes.
    """
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    shared_mem = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id)
    if format == "bytes":
        return shared_mem
    elif format == "kb":
        return shared_mem // 1024
    elif format == "mb":
        return shared_mem // (1024 * 1024)
    else:
        raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")


def get_persisting_l2_cache_max_size(device_id: int = 0) -> int:
    prop = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxPersistingL2CacheSize, device_id)
    return prop


def get_num_sms(device_id: int = 0) -> int:
    """
    Get the number of streaming multiprocessors (SMs) on the CUDA device.

    Args:
        device_id (int, optional): The CUDA device ID. Defaults to 0.

    Returns:
        int: The number of SMs on the device.

    Raises:
        RuntimeError: If unable to get the device properties.
    """
    prop = get_cuda_device_properties(device_id)
    if prop is None:
        raise RuntimeError("Failed to get device properties.")
    return prop.multi_processor_count


def get_registers_per_block(device_id: int = 0) -> int:
    """
    Get the maximum number of 32-bit registers available per block.
    """
    prop = get_device_attribute(
        cudaDeviceAttrNames.cudaDevAttrMaxRegistersPerBlock,
        device_id,
    )
    return prop
