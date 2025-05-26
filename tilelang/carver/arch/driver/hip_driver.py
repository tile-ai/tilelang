# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import ctypes
import sys
from typing import Optional


# hipDeviceProp, checked from hip_runtime_api.h's hipDeviceProp_t
class hipDeviceProp(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 256), ("uuid", ctypes.c_ubyte * 16),
                ("luid", ctypes.c_char * 8), ("luidDeviceNodeMask", ctypes.c_uint),
                ("totalGlobalMem", ctypes.c_size_t), ("sharedMemPerBlock", ctypes.c_size_t),
                ("regsPerBlock", ctypes.c_int), ("warpSize", ctypes.c_int),
                ("memPitch", ctypes.c_size_t), ("maxThreadsPerBlock", ctypes.c_int),
                ("maxThreadsDim", ctypes.c_int * 3), ("maxGridSize", ctypes.c_int * 3),
                ("clockRate", ctypes.c_int), ("totalConstMem", ctypes.c_size_t),
                ("major", ctypes.c_int), ("minor", ctypes.c_int),
                ("textureAlignment", ctypes.c_size_t), ("texturePitchAlignment", ctypes.c_size_t),
                ("deviceOverlap", ctypes.c_int), ("multiProcessorCount", ctypes.c_int),
                ("kernelExecTimeoutEnabled", ctypes.c_int), ("integrated", ctypes.c_int),
                ("canMapHostMemory", ctypes.c_int), ("computeMode", ctypes.c_int),
                ("sharedMemPerMultiprocessor", ctypes.c_size_t),
                ("regsPerMultiprocessor", ctypes.c_int), ("managedMemory", ctypes.c_int),
                ("isMultiGpuBoard", ctypes.c_int), ("multiGpuBoardGroupID", ctypes.c_int),
                ("gcnArchName", ctypes.c_char * 256),
                ("maxSharedMemoryPerMultiProcessor", ctypes.c_size_t),
                ("clockInstructionRate", ctypes.c_int), ("isLargeBar", ctypes.c_int),
                ("asicRevision", ctypes.c_int)]


def get_hip_device_properties(device_id: int = 0) -> Optional[hipDeviceProp]:
    try:
        if sys.platform == "win32":
            libhip = ctypes.windll.LoadLibrary("amdhip64.dll")
        else:
            libhip = ctypes.cdll.LoadLibrary("libamdhip64.so")

        hipGetDeviceProperties = libhip.hipGetDeviceProperties
        hipGetDeviceProperties.argtypes = [ctypes.POINTER(hipDeviceProp), ctypes.c_int]
        hipGetDeviceProperties.restype = ctypes.c_int

        prop = hipDeviceProp()
        ret = hipGetDeviceProperties(ctypes.byref(prop), device_id)
        if ret == 0:
            return prop
        else:
            raise RuntimeError(f"hipGetDeviceProperties failed with error code {ret}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def get_hip_device_name(device_id: int = 0) -> Optional[str]:
    prop = get_hip_device_properties(device_id)
    if prop:
        return prop.name.decode()
    else:
        return None


def get_hip_shared_memory_per_block(device_id: int = 0, format: str = "bytes") -> Optional[int]:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    prop = get_hip_device_properties(device_id)
    if prop:
        shared_mem = int(prop.sharedMemPerBlock)
        if format == "bytes":
            return shared_mem
        elif format == "kb":
            return shared_mem // 1024
        elif format == "mb":
            return shared_mem // (1024 * 1024)
    return None


def get_hip_device_attribute(attr: int, device_id: int = 0) -> Optional[int]:
    try:
        if sys.platform == "win32":
            libhip = ctypes.windll.LoadLibrary("amdhip64.dll")
        else:
            libhip = ctypes.cdll.LoadLibrary("libamdhip64.so")

        value = ctypes.c_int()
        hipDeviceGetAttribute = libhip.hipDeviceGetAttribute
        hipDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        hipDeviceGetAttribute.restype = ctypes.c_int

        ret = hipDeviceGetAttribute(ctypes.byref(value), attr, device_id)
        if ret != 0:
            raise RuntimeError(f"hipDeviceGetAttribute failed with error code {ret}")

        return value.value

    except Exception as e:
        print(f"Error getting HIP device attribute: {str(e)}")
        return None


def get_hip_max_dynamic_shared_size_bytes(device_id: int = 0,
                                          format: str = "bytes") -> Optional[int]:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    prop = get_hip_device_properties(device_id)
    if prop:
        shared_mem = int(prop.sharedMemPerMultiprocessor)
        if format == "bytes":
            return shared_mem
        elif format == "kb":
            return shared_mem // 1024
        elif format == "mb":
            return shared_mem // (1024 * 1024)
    else:
        raise RuntimeError("Failed to get device properties.")


def get_hip_num_sms(device_id: int = 0) -> int:
    prop = get_hip_device_properties(device_id)
    if prop:
        return prop.multiProcessorCount
    else:
        raise RuntimeError("Failed to get device properties.")


def get_hip_registers_per_block(device_id: int = 0) -> int:
    prop = get_hip_device_properties(device_id)
    if prop:
        return prop.regsPerBlock
    else:
        raise RuntimeError("Failed to get device properties.")
