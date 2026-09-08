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

    from tilelang_cython_wrapper import (
        install_torch_npu_stream_exchange as install,
    )

    return bool(install())


def is_torch_npu_stream_exchange_installed() -> bool:
    """Return whether TileLang owns Torch's current Exchange API table."""

    from tilelang_cython_wrapper import (
        is_torch_npu_stream_exchange_installed as is_installed,
    )

    return bool(is_installed())


__all__ = [
    "install_torch_npu_stream_exchange",
    "is_torch_npu_stream_exchange_installed",
]
