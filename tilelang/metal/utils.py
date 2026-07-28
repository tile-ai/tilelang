from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.utils.language import _get_buffer


def is_metal_cooperative_tensor(buffer: BufferLikeType) -> bool:
    """Check if the buffer is in the Metal cooperative tensor scope."""
    buffer = _get_buffer(buffer)
    return buffer.scope() == "metal.cooperative_tensor"


def is_metal_simdgroup(buffer: BufferLikeType) -> bool:
    """Check if the buffer is in the Metal simdgroup scope."""
    buffer = _get_buffer(buffer)
    return buffer.scope() == "metal.simdgroup"
