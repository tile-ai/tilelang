# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm import tir
from typing import Union
from tilelang.language import has_let_value, get_let_value


def fill(buffer: Union[tir.Buffer, tir.BufferRegion], value: tir.PrimExpr):
    """Fill a buffer or buffer region with a specified value.
    
    Args:
        buffer: Either a TVM buffer or buffer region to be filled
        value: The value to fill the buffer with
    
    Returns:
        A TVM intrinsic call that performs the fill operation
    """
    if isinstance(buffer, tir.Buffer):
        buffer = buffer.access_ptr("w")  # Get write pointer if input is a Buffer
    return tir.call_intrin("handle", tir.op.Op.get("tl.fill"), buffer, value)


def clear(buffer: Union[tir.Buffer, tir.Var]):
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
        else:
            raise ValueError(f"Invalid buffer region: {buffer_region}")
    return fill(buffer, 0)
