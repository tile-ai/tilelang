"""Runtime-only Torch NPU stream integration for TVM-FFI execution."""

from __future__ import annotations


def install_torch_npu_stream_exchange() -> bool:
    """Make TVM-FFI submit Ascend kernels to Torch's current NPU stream."""

    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "torch_npu":
            raise
        raise RuntimeError("Torch NPU stream integration requires torch_npu") from error

    try:
        from tilelang_ascend_npu_exchange import (
            install_torch_npu_stream_exchange as install,
        )
    except ModuleNotFoundError as error:
        if error.name != "tilelang_ascend_npu_exchange":
            raise
        raise RuntimeError("Torch NPU stream integration requires tilelang_ascend_npu_exchange") from error

    return bool(install())


def is_torch_npu_stream_exchange_installed() -> bool:
    """Return whether TileLang owns Torch's current Exchange API table."""

    try:
        from tilelang_ascend_npu_exchange import (
            is_torch_npu_stream_exchange_installed as is_installed,
        )
    except ModuleNotFoundError as error:
        if error.name != "tilelang_ascend_npu_exchange":
            raise
        raise RuntimeError("Torch NPU stream integration requires tilelang_ascend_npu_exchange") from error

    return bool(is_installed())


def npu_current_device():
    """Torch device for the current NPU, for kernel output allocation."""
    import torch

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError("npu_current_device: Ascend NPU is unavailable; install torch_npu and ensure an NPU device is present.")
    return torch.device("npu", torch.npu.current_device())


def npu_current_raw_stream():
    """Raw stream of Torch's current NPU stream.

    Uses the low-level accessor when available: torch.npu.current_stream()
    goes through a Python wrapper that internally probes
    torch.cuda.is_available(), which costs ~150us per call on a CUDA-less NPU
    host; the _C accessor returns the raw stream in <1us.
    """
    import torch

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError("npu_current_raw_stream: Ascend NPU is unavailable; install torch_npu and ensure an NPU device is present.")

    try:
        import torch_npu

        return torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device())
    except (ImportError, AttributeError):
        return torch.npu.current_stream().npu_stream


__all__ = [
    "install_torch_npu_stream_exchange",
    "is_torch_npu_stream_exchange_installed",
    "npu_current_device",
    "npu_current_raw_stream",
]
