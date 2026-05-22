from __future__ import annotations

import torch

__all__ = ["tensor"]


def tensor(
    shape,
    dtype,
    device=None,
    allocator=None,
    return_peers=None,
):
    """Allocate a tensor, optionally using a distributed allocator."""

    if not isinstance(dtype, torch.dtype):
        try:
            dtype = dtype.as_torch()
        except AttributeError:
            pass

    if allocator is None:
        assert return_peers is None, "return_peers must be None when allocator is not provided"
        return torch.empty(shape, dtype=dtype, device=device)

    from tilelang.distributed.allocator import BaseAllocator
    from tilelang.utils.target import parse_device

    assert isinstance(allocator, BaseAllocator) and allocator.initialized(), "Allocator must be an initialized BaseAllocator"
    if device is not None:
        device_idx = parse_device(device)
        assert allocator.device == device_idx, f"Allocator device mismatch: {allocator.device} != {device_idx}"
    return allocator._allocate_tensor(shape, dtype, return_peers)
