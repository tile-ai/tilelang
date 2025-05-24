# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import ctypes
import sys
from typing import Optional


class cudaDeviceProp(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("uuid", ctypes.c_byte * 16),  # cudaUUID_t
        ("luid", ctypes.c_char * 8),
        ("luidDeviceNodeMask", ctypes.c_uint),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("textureAlignment", ctypes.c_size_t),
        ("texturePitchAlignment", ctypes.c_size_t),
        ("deviceOverlap", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("kernelExecTimeoutEnabled", ctypes.c_int),
        ("integrated", ctypes.c_int),
        ("canMapHostMemory", ctypes.c_int),
        ("computeMode", ctypes.c_int),
        ("maxTexture1D", ctypes.c_int),
        ("maxTexture1DMipmap", ctypes.c_int),
        ("maxTexture1DLinear", ctypes.c_int),
        ("maxTexture2D", ctypes.c_int * 2),
        ("maxTexture2DMipmap", ctypes.c_int * 2),
        ("maxTexture2DLinear", ctypes.c_int * 3),
        ("maxTexture2DGather", ctypes.c_int * 2),
        ("maxTexture3D", ctypes.c_int * 3),
        ("maxTexture3DAlt", ctypes.c_int * 3),
        ("maxTextureCubemap", ctypes.c_int),
        ("maxTexture1DLayered", ctypes.c_int * 2),
        ("maxTexture2DLayered", ctypes.c_int * 3),
        ("maxTextureCubemapLayered", ctypes.c_int * 2),
        ("maxSurface1D", ctypes.c_int),
        ("maxSurface2D", ctypes.c_int * 2),
        ("maxSurface3D", ctypes.c_int * 3),
        ("maxSurface1DLayered", ctypes.c_int * 2),
        ("maxSurface2DLayered", ctypes.c_int * 3),
        ("maxSurfaceCubemap", ctypes.c_int),
        ("maxSurfaceCubemapLayered", ctypes.c_int * 2),
        ("surfaceAlignment", ctypes.c_size_t),
        ("concurrentKernels", ctypes.c_int),
        ("ECCEnabled", ctypes.c_int),
        ("pciBusID", ctypes.c_int),
        ("pciDeviceID", ctypes.c_int),
        ("pciDomainID", ctypes.c_int),
        ("tccDriver", ctypes.c_int),
        ("asyncEngineCount", ctypes.c_int),
        ("unifiedAddressing", ctypes.c_int),
        ("memoryClockRate", ctypes.c_int),
        ("memoryBusWidth", ctypes.c_int),
        ("l2CacheSize", ctypes.c_int),
        ("persistingL2CacheMaxSize", ctypes.c_int),
        ("maxThreadsPerMultiProcessor", ctypes.c_int),
        ("streamPrioritiesSupported", ctypes.c_int),
        ("globalL1CacheSupported", ctypes.c_int),
        ("localL1CacheSupported", ctypes.c_int),
        ("sharedMemPerMultiprocessor", ctypes.c_size_t),
        ("regsPerMultiprocessor", ctypes.c_int),
        ("managedMemory", ctypes.c_int),
        ("isMultiGpuBoard", ctypes.c_int),
        ("multiGpuBoardGroupID", ctypes.c_int),
        ("reserved2", ctypes.c_int * 2),
        ("reserved1", ctypes.c_int * 1),
        ("reserved", ctypes.c_int * 60)
    ]


def get_cuda_device_properties(device_id: int = 0) -> Optional[cudaDeviceProp]:

    if sys.platform == "win32":
        libcudart = ctypes.windll.LoadLibrary("cudart64_110.dll")
    else:
        libcudart = ctypes.cdll.LoadLibrary("libcudart.so")

    prop = cudaDeviceProp()
    cudaGetDeviceProperties = libcudart.cudaGetDeviceProperties
    cudaGetDeviceProperties.argtypes = [ctypes.POINTER(cudaDeviceProp), ctypes.c_int]
    cudaGetDeviceProperties.restype = ctypes.c_int
    ret = cudaGetDeviceProperties(ctypes.byref(prop), device_id)
    if ret == 0:
        return prop
    else:
        raise RuntimeError(f"cudaGetDeviceProperties failed with error {ret}")


def get_device_name(device_id: int = 0) -> Optional[str]:
    prop = get_cuda_device_properties(device_id)
    if prop:
        return prop.name.decode()
    else:
        raise RuntimeError("Failed to get device properties.")


def get_shared_memory_per_block(device_id: int = 0, format: str = "bytes") -> Optional[int]:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    prop = get_cuda_device_properties(device_id)
    if prop:
        # Convert size_t to int to avoid overflow issues
        shared_mem = int(prop.sharedMemPerBlock)
        if format == "bytes":
            return shared_mem
        elif format == "kb":
            return shared_mem // 1024
        elif format == "mb":
            return shared_mem // (1024 * 1024)
        else:
            raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")
    else:
        raise RuntimeError("Failed to get device properties.")


def get_device_attribute(attr: int, device_id: int = 0) -> int:
    try:
        if sys.platform == "win32":
            libcudart = ctypes.windll.LoadLibrary("cudart64_110.dll")
        else:
            libcudart = ctypes.cdll.LoadLibrary("libcudart.so")

        value = ctypes.c_int()
        cudaDeviceGetAttribute = libcudart.cudaDeviceGetAttribute
        cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        cudaDeviceGetAttribute.restype = ctypes.c_int

        ret = cudaDeviceGetAttribute(ctypes.byref(value), attr, device_id)
        if ret != 0:
            raise RuntimeError(f"cudaDeviceGetAttribute failed with error {ret}")

        return value.value
    except Exception as e:
        print(f"Error getting device attribute: {str(e)}")
        return None


def get_max_dynamic_shared_size_bytes(device_id: int = 0, format: str = "bytes") -> Optional[int]:
    """
    Get the maximum dynamic shared memory size in bytes, kilobytes, or megabytes.
    """
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    prop = get_cuda_device_properties(device_id)
    if prop:
        # Convert size_t to int to avoid overflow issues
        shared_mem = int(prop.sharedMemPerMultiprocessor)
        if format == "bytes":
            return shared_mem
        elif format == "kb":
            return shared_mem // 1024
        elif format == "mb":
            return shared_mem // (1024 * 1024)
        else:
            raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")
    else:
        raise RuntimeError("Failed to get device properties.")


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
    if prop:
        return prop.multiProcessorCount
    else:
        raise RuntimeError("Failed to get device properties.")
