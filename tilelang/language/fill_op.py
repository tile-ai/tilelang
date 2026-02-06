"""Fill operations exposed on the TileLang language surface."""

from __future__ import annotations
from tvm import tir
from tilelang.language import has_let_value, get_let_value
from tilelang.utils.language import get_buffer_region_from_load, to_buffer_region


def fill(buffer: tir.Buffer | tir.BufferRegion | tir.BufferLoad, value: tir.PrimExpr):
    """Fill a buffer or buffer region with a specified value.

    Args:
        buffer: Either a TVM buffer or buffer region to be filled
        value: The value to fill the buffer with

    Returns:
        A TVM intrinsic call that performs the fill operation
    """
    # Normalize Var with let value to its underlying object
    if isinstance(buffer, tir.Var) and has_let_value(buffer):
        buffer = get_let_value(buffer)

    # Build tl.region as argument
    if isinstance(buffer, tir.Buffer):
        extents = list(buffer.shape)
    elif isinstance(buffer, tir.BufferRegion):
        extents = [r.extent for r in buffer.region]
    elif isinstance(buffer, tir.BufferLoad):
        region = get_buffer_region_from_load(buffer)
        if region is not None:
            extents = [r.extent for r in region.region]
        else:
            extents = [tir.IntImm("int32", 1) for _ in buffer.indices]
    else:
        extents = []
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.fill"), to_buffer_region(buffer, access_type="w", extents=extents), value)


def clear(buffer: tir.Buffer | tir.Var):
    """Clear a buffer by filling it with zeros.

    Args:
        buffer: Either a TVM buffer or a variable that contains a buffer region

    Returns:
        A fill operation that sets the buffer contents to zero

    Raises:
        ValueError: If the buffer variable contains an invalid buffer region
    """
    if isinstance(buffer, tir.Var) and has_let_value(buffer):
        buffer_region = get_let_value(buffer)  # Get the actual buffer region from variable
        if isinstance(buffer_region, tir.BufferRegion):
            return fill(buffer_region, 0)
        elif isinstance(buffer_region, tir.BufferLoad):
            region = get_buffer_region_from_load(buffer_region)
            if region is None:
                raise ValueError(f"Invalid buffer region: {buffer_region}, type: {type(buffer_region)}")
            return fill(region, 0)
        else:
            raise ValueError(f"Invalid buffer region: {buffer_region}, type: {type(buffer_region)}")
    return fill(buffer, 0)
