from __future__ import annotations


def get_ascend_device_properties(device_id: int = 0):
    """Get raw torch_npu device properties object for the given device.

    Returns None if torch_npu or the NPU device is not available.
    """
    try:
        import torch_npu

        if not torch_npu.npu.is_available():
            return None
        return torch_npu.npu.get_device_properties(device_id)
    except ImportError:
        return None


def get_device_name(device_id: int = 0) -> str | None:
    """Get the device name (e.g. 'Ascend950PR_9589')."""
    prop = get_ascend_device_properties(device_id)
    if prop is not None:
        return prop.name
    return None


def get_num_cube_cores(device_id: int = 0) -> int | None:
    """Get the number of AI Cube cores (matmul units) on the device."""
    prop = get_ascend_device_properties(device_id)
    if prop is None:
        raise RuntimeError("Failed to get Ascend device properties.")
    return prop.cube_core_num


def get_num_vector_cores(device_id: int = 0) -> int | None:
    """Get the number of AI Vector cores (SIMD units) on the device."""
    prop = get_ascend_device_properties(device_id)
    if prop is None:
        raise RuntimeError("Failed to get Ascend device properties.")
    return prop.vector_core_num


def get_total_memory(device_id: int = 0, format: str = "bytes") -> int | None:
    """Get the total HBM memory of the device in bytes, kb, mb, or gb."""
    assert format in ["bytes", "kb", "mb", "gb"], "Invalid format. Must be one of: bytes, kb, mb, gb"
    prop = get_ascend_device_properties(device_id)
    if prop is None:
        raise RuntimeError("Failed to get Ascend device properties.")
    total = int(prop.total_memory)
    if format == "bytes":
        return total
    elif format == "kb":
        return total // 1024
    elif format == "mb":
        return total // (1024 * 1024)
    elif format == "gb":
        return total // (1024 * 1024 * 1024)
    else:
        raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb, gb")


def get_l2_cache_size(device_id: int = 0, format: str = "bytes") -> int | None:
    """Get the L2 cache size of the device in bytes, kb, or mb."""
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    prop = get_ascend_device_properties(device_id)
    if prop is None:
        raise RuntimeError("Failed to get Ascend device properties.")
    size = int(prop.L2_cache_size)
    if format == "bytes":
        return size
    elif format == "kb":
        return size // 1024
    elif format == "mb":
        return size // (1024 * 1024)
    else:
        raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")
