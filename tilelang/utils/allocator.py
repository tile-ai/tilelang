"""Compatibility exports for distributed allocation helpers.

New code should import from `tilelang.distributed.allocator`.
"""

from tilelang.distributed import allocator as _allocator
from tilelang.distributed.allocator import BaseAllocator, get_allocator  # noqa: F401

__all__ = ["BaseAllocator", "get_allocator"]


def __getattr__(name):
    return getattr(_allocator, name)
