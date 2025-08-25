# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

import tilelang.language as T
from tvm import ir
from tvm.tir import PrimExpr, Buffer, BufferLoad, BufferRegion, Var, op
from typing import List, Union

_MEMORY_ORDER_ID_MAP = {
    "relaxed": 0,
    "consume": 1,
    "acquire": 2,
    "release": 3,
    "acq_rel": 4,
    "seq_cst": 5,
}


def region(buffer: BufferLoad, access_type: str, *args: PrimExpr):
    """Create a memory region descriptor for tile operations.

    Args:
        buffer (tir.BufferLoad): The buffer to create a region for
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write
        *args (tir.PrimExpr): Extent expressions defining the region size

    Returns:
        tir.Call: A region descriptor for tile operations
    """
    access_type = {"r": 1, "w": 2, "rw": 3}[access_type]
    return T.call_intrin("handle", op.Op.get("tl.region"), buffer, access_type, *args)


def buffer_to_tile_region(buffer: Buffer, access_type: str):
    """Convert a TVM buffer to a tile region descriptor.

    Args:
        buffer (tir.Buffer): The buffer to convert
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write

    Returns:
        tir.Call: A region descriptor covering the entire buffer
    """
    mins = [0 for _ in buffer.shape]
    extents = [x for x in buffer.shape]
    return region(T.BufferLoad(buffer, mins), access_type, *extents)


def buffer_load_to_tile_region(load: BufferLoad, access_type: str, extents: List[PrimExpr]):
    """Convert a buffer load operation to a tile region descriptor.

    Args:
        load (tir.BufferLoad): The buffer load operation
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write
        extents (List[tir.PrimExpr]): List of expressions defining the region size

    Returns:
        tir.Call: A region descriptor for the loaded area
    """
    indices = load.indices
    if len(indices) > len(extents):
        # (f"mismatch between indices and extents for buffer load {load}: indices = {indices}, extents = {extents}, "
        # f"region will be expanded in the last 2 dimensions")
        new_extents = []
        for _ in range(len(indices) - len(extents)):
            new_extents.append(1)
        for extent in extents:
            new_extents.append(extent)
        extents = new_extents
    assert len(indices) == len(extents), f"indices = {indices}, extents = {extents}"
    return region(load, access_type, *extents)


def buffer_region_to_tile_region(buffer_region: BufferRegion, access_type: str,
                                 extents: List[PrimExpr]):
    """Convert a buffer region to a tile region descriptor.

    Args:
        buffer_region (tir.BufferRegion): The buffer region to convert
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write

    Returns:
        tir.Call: A region descriptor for the specified buffer region
    """
    mins = [x.min for x in buffer_region.region]
    region_extents = [x.extent for x in buffer_region.region]
    assert len(region_extents) >= len(
        extents
    ), f"region_extents must be >= extents, region_extents = {region_extents}, extents = {extents}"

    return region(T.BufferLoad(buffer_region.buffer, mins), access_type, *region_extents)


def atomic_max(dst: Buffer, value: PrimExpr, memory_order: str | None = None) -> PrimExpr:
    """Perform an atomic maximum operation.

    Args:
        dst (Buffer): Destination buffer where the atomic maximum will be performed
        value (PrimExpr): Value to be atomically added

    Returns:
        PrimExpr: Handle to the atomic maximum operation
    """
    if memory_order is None:
        return T.call_extern("handle", "AtomicMax", T.address_of(dst), value)
    else:
        return T.call_extern("handle", "AtomicMax", T.address_of(dst), value,
                             _MEMORY_ORDER_ID_MAP[memory_order])


def atomic_min(dst: Buffer, value: PrimExpr, memory_order: str | None = None) -> PrimExpr:
    """Perform an atomic minimum operation.

    Args:
        dst (Buffer): Destination buffer where the atomic minimum will be performed
        value (PrimExpr): Value to be atomically added

    Returns:
        PrimExpr: Handle to the atomic minimum operation
    """
    if memory_order is None:
        return T.call_extern("handle", "AtomicMin", T.address_of(dst), value)
    else:
        return T.call_extern("handle", "AtomicMin", T.address_of(dst), value,
                             _MEMORY_ORDER_ID_MAP[memory_order])


def atomic_add(dst: Buffer, value: PrimExpr, memory_order: str | None = None) -> PrimExpr:
    """Perform an atomic addition operation.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added

    Returns:
        PrimExpr: Handle to the atomic addition operation
    """

    def get_extent(data):
        if isinstance(data, Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, Buffer):
            return data.shape
        elif isinstance(data, BufferRegion):
            return [x.extent for x in data.region]
        else:
            return None

    src_extent = get_extent(value)
    dst_extent = get_extent(dst)

    if dst_extent is None and src_extent is None:
        if memory_order is None:
            return T.call_extern("handle", "AtomicAdd", T.address_of(dst), value)
        else:
            return T.call_extern("handle", "AtomicAdd", T.address_of(dst), value,
                                 _MEMORY_ORDER_ID_MAP[memory_order])

    if isinstance(dst, Buffer) and isinstance(value, Buffer):
        ir.assert_structural_equal(dst.shape, value.shape)

    assert src_extent or dst_extent, "Can't deduce atomicadd extents from args"
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
    extent = max(src_extent, dst_extent)

    def _to_region(data, access_type):
        if isinstance(data, Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, Buffer):
            return buffer_to_tile_region(data, access_type)
        elif isinstance(data, BufferRegion):
            return buffer_region_to_tile_region(data, access_type, extent)
        else:
            return buffer_load_to_tile_region(data, access_type, extent)

    value = _to_region(value, "r")
    dst = _to_region(dst, "w")
    return T.call_intrin("handle", op.Op.get("tl.atomicadd"), value, dst)


def atomic_addx2(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation with double-width operands.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added (double-width)

    Returns:
        PrimExpr: Handle to the double-width atomic addition operation
    """
    return T.call_extern("handle", "AtomicAddx2", T.address_of(dst), T.address_of(value))


def atomic_addx4(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation with quad-width operands.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added (quad-width)

    Returns:
        PrimExpr: Handle to the quad-width atomic addition operation
    """
    return T.call_extern("handle", "AtomicAddx4", T.address_of(dst), T.address_of(value))


def dp4a(A: Buffer, B: Buffer, C: Buffer) -> PrimExpr:
    """Perform a 4-element dot product with accumulation (DP4A).

    Args:
        A (Buffer): First input buffer
        B (Buffer): Second input buffer
        C (Buffer): Accumulation buffer

    Returns:
        PrimExpr: Handle to the DP4A operation
    """
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
        src (Buffer): Input buffer to be reshaped
        shape (List[PrimExpr]): New shape for the buffer

    Returns:
        Buffer: A new buffer view with the specified shape
    """
    return T.Tensor(shape, src.dtype, src.data)


def view(src: Buffer,
         shape: Union[List[PrimExpr], None] = None,
         dtype: Union[str, None] = None) -> Buffer:
    """Views the input buffer with optionally modified shape and dtype.
    
    Args:
        src (Buffer): Input buffer to be viewed
        shape (Union[List[PrimExpr], None], optional): New shape for the buffer. Defaults to None.
        dtype (Union[str, None], optional): New dtype for the buffer. Defaults to None.

    Returns:
        Buffer: A new buffer view with the specified shape and dtype
    """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    return T.Tensor(shape, dtype, src.data)


def atomic_load(src: Buffer, memory_order: str = "seq_cst") -> PrimExpr:
    """Loads a value from the input buffer with specified memory_order.

    Args:
        src (Buffer): Input buffer to load from
        memory_order (str, optional): Atomicity level for the load operation. Defaults to "seq_cst".

    Returns:
        PrimExpr: The loaded value from the buffer
    """
    return T.call_extern(src.dtype, "AtomicLoad", T.address_of(src),
                         _MEMORY_ORDER_ID_MAP[memory_order])


def atomic_store(dst: Buffer, src: PrimExpr, memory_order: str = "seq_cst") -> PrimExpr:
    """Stores a value to the input buffer with specified memory_order.

    Args:
        dst (Buffer): Input buffer to store to
        src (PrimExpr): Value to store
        memory_order (str, optional): Atomicity level for the load operation. Defaults to "seq_cst".

    Returns:
        PrimExpr: The handle of the store operation
    """
    return T.call_extern("handle", "AtomicStore", T.address_of(dst), src,
                         _MEMORY_ORDER_ID_MAP[memory_order])
