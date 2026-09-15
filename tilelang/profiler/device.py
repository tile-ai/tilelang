"""Runtime device selection for profiling, independent of compilation backends."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tvm.target import Target


def _normalize_device(device: int | str | torch.device) -> torch.device:
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.type == "mps":
        if device.index not in (None, 0):
            raise ValueError(f"Profiling requires the default MPS device, got {device}")
        return torch.device("mps", 0)
    if device.type == "cpu":
        return torch.device("cpu")
    return device


def resolve_device(
    device: int | str | torch.device | None = None,
    inputs: Sequence[object] | None = None,
    target: Target | None = None,
) -> torch.device:
    """Resolve a device once, checking explicit devices, inputs, and kernel targets."""
    input_devices = {_normalize_device(tensor.device) for tensor in inputs or () if isinstance(tensor, torch.Tensor)}
    if len(input_devices) > 1:
        raise ValueError(f"Profiling requires inputs on one device, got {sorted(map(str, input_devices))}")

    target_device = None
    if target is not None:
        target_device = {"cuda": "cuda", "rocm": "cuda", "metal": "mps", "llvm": "cpu", "c": "cpu"}.get(target.kind.name)
        if target_device is None:
            raise ValueError(f"Profiling is not supported for target {target.kind.name!r}")

    if device is not None:
        resolved = _normalize_device(device)
    elif input_devices:
        resolved = next(iter(input_devices))
    elif target_device is not None:
        resolved = _normalize_device(target_device)
    elif torch.cuda.is_available():
        resolved = _normalize_device("cuda")
    elif torch.backends.mps.is_available():
        resolved = _normalize_device("mps")
    else:
        resolved = torch.device("cpu")

    if input_devices and resolved not in input_devices:
        raise ValueError(f"Profiling device {resolved} does not match input device {next(iter(input_devices))}")
    if target_device is not None and resolved.type != target_device:
        raise ValueError(f"Profiling device {resolved} is incompatible with kernel target {target.kind.name!r}")
    return resolved


def get_backend(device: torch.device) -> ModuleType:
    """Lazily load the profiling implementation for the actual tensor runtime."""
    if device.type == "cuda":
        name = "rocm" if torch.version.hip else "cuda"
    elif device.type == "mps":
        name = "metal"
    elif device.type == "cpu":
        name = "cpu"
    else:
        raise ValueError(f"Profiling is not supported on device {device}")
    return import_module(f"tilelang.{name}.profiler")
