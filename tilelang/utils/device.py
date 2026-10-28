import os

import torch

IS_CUDA = torch.cuda.is_available()

IS_NPU = False
try:
    if not hasattr(torch, "npu"):
        import torch_npu  # noqa: F401
    IS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    IS_NPU = False
except Exception as e:
    print(f"An unexpected error occurred while checking NPU availability: {e}")

IS_MPS = False
try:
    IS_MPS = torch.backends.mps.is_available()
except AttributeError:
    print("MPS backend is not available in this PyTorch build.")
except Exception as e:
    print(f"An unexpected error occurred while checking MPS availability: {e}")


if IS_CUDA:
    Event = torch.cuda.Event
elif IS_NPU:
    Event = torch.npu.Event
elif IS_MPS:
    Event = torch.mps.Event
else:
    Event = None


def get_current_device():
    device = None
    if IS_CUDA:
        device = torch.cuda.current_device()
    elif IS_NPU:
        device = torch.npu.current_device()
    elif IS_MPS:
        device = "mps:0"

    return device


# Opt-in dynamic shared memory per block, in KiB, as documented for
# cudaDevAttrMaxSharedMemoryPerBlockOptin. Used only when PyTorch does not
# expose the limit itself (torch < 2.7).
_OPTIN_SMEM_KIB = {
    70: 96,
    75: 96,
    80: 163,
    86: 99,
    87: 163,
    89: 99,
    90: 227,
    100: 227,
    103: 227,
    110: 227,
    120: 99,
    121: 99,
}


def get_max_dynamic_shared_memory(device: int | torch.device | None = None) -> int:
    """Largest dynamic shared-memory allocation a thread block may request, in bytes.

    Kernels that request more than this fail at launch with
    ``Failed to set the allowed dynamic shared memory size to N``.
    """
    if not IS_CUDA:
        return 48 * 1024
    index = device if isinstance(device, int) else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    limit = getattr(props, "shared_memory_per_block_optin", None)
    if limit is not None:
        return int(limit)
    major, minor = torch.cuda.get_device_capability(index)
    return 1024 * _OPTIN_SMEM_KIB.get(major * 10 + minor, 99)


def device_synchronize(device: int | torch.device | None = None) -> None:
    """Synchronize CUDA/HIP, MPS, or NPU, preferring CUDA when no device is given."""
    if device is None:
        if IS_CUDA:
            torch.cuda.synchronize()
        elif IS_NPU:
            torch.npu.synchronize()
        elif IS_MPS:
            torch.mps.synchronize()
        else:
            raise RuntimeError("No device is available")
        return

    device_type = "cuda" if isinstance(device, int) else torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "npu":
        torch.npu.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()
    else:
        raise ValueError(f"device_synchronize only supports CUDA/HIP, MPS, or NPU devices, got {device}")


def get_available_cpu_count() -> int:
    """CPU cores available to this process (cpuset affinity), at least 1.

    Falls back to ``os.cpu_count`` where affinity is unavailable (macOS/Windows).
    """
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return max(1, cpu_count or 1)
