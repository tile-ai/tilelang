import os

import torch

IS_CUDA = torch.cuda.is_available()

IS_MPS = False
try:
    IS_MPS = torch.backends.mps.is_available()
except AttributeError:
    print("MPS backend is not available in this PyTorch build.")
except Exception as e:
    print(f"An unexpected error occurred while checking MPS availability: {e}")


if IS_CUDA:
    Event = torch.cuda.Event
elif IS_MPS:
    Event = torch.mps.Event
else:
    Event = None


def get_current_device():
    device = None
    if IS_CUDA:
        device = torch.cuda.current_device()
    elif IS_MPS:
        device = "mps:0"

    return device


def device_synchronize(device: int | torch.device | None = None) -> None:
    """Synchronize CUDA/HIP or MPS, preferring CUDA when no device is given."""
    if device is None:
        if IS_CUDA:
            torch.cuda.synchronize()
        elif IS_MPS:
            torch.mps.synchronize()
        else:
            raise RuntimeError("No device is available")
        return

    device_type = "cuda" if isinstance(device, int) else torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()
    else:
        raise ValueError(f"device_synchronize only supports CUDA/HIP or MPS devices, got {device}")


def get_available_cpu_count() -> int:
    """CPU cores available to this process (cpuset affinity), at least 1.

    Falls back to ``os.cpu_count`` where affinity is unavailable (macOS/Windows).
    """
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return max(1, cpu_count or 1)
