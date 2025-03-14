# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm.script import tir as T
from tvm.tir import PrimExpr, Buffer
from typing import List, Union


def atomic_add(dst: Buffer, value: PrimExpr) -> PrimExpr:
    return T.call_extern("handle", "AtomicAdd", T.address_of(dst), value)


def atomic_addx2(dst: Buffer, value: PrimExpr) -> PrimExpr:
    return T.call_extern("handle", "AtomicAddx2", T.address_of(dst), T.address_of(value))


def dp4a(A: Buffer, B: Buffer, C: Buffer) -> PrimExpr:
    return T.call_extern("handle", "DP4A", T.address_of(A), T.address_of(B), T.address_of(C))


def clamp(dst: PrimExpr, min_val: PrimExpr, max_val: PrimExpr) -> PrimExpr:
    """Clamps the input value dst between [min_val, max_val]
    
    Args:
        dst: Input value to be clamped
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Value clamped to the specified range
    """
    dst = T.max(dst, min_val)  # Ensure value is not less than minimum
    dst = T.min(dst, max_val)  # Ensure value is not greater than maximum
    return dst


def reshape(src: Buffer, shape: List[PrimExpr]) -> Buffer:
    """Reshapes the input buffer to the specified shape.
    
    Args:
        src: Input buffer to be reshaped
        shape: New shape for the buffer
    """
    return T.Buffer(shape, src.dtype, src.data)


def view(src: Buffer,
         shape: Union[List[PrimExpr], None] = None,
         dtype: Union[str, None] = None) -> Buffer:
    """Views the input buffer to the specified shape.
    
    Args:
        src: Input buffer to be viewed
        shape: New shape for the buffer
        dtype: New dtype for the buffer
    """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    return T.Buffer(shape, dtype, src.data)
