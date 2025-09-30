"""The language interface for tl programs."""

import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferLoad, BufferRegion, op
from typing import List, Union
from .atomic import atomic_max, atomic_min, atomic_add, atomic_addx2, atomic_addx4, atomic_load, atomic_store  # noqa: F401


def region(buffer: BufferLoad, access_type: str, *args: PrimExpr):
    """
    Create a tile memory-region descriptor for a BufferLoad.

    Maps access_type ('r', 'w', 'rw') to the numeric codes expected by the `tl.region` intrinsic
    (1, 2, 3 respectively) and returns a tir.Call representing the region with the provided extents.

    Parameters:
        buffer (tir.BufferLoad): The BufferLoad that identifies the underlying buffer and indices.
        access_type (str): One of 'r', 'w', or 'rw' indicating read, write, or read-write access.
        *args (tir.PrimExpr): Extent expressions for each region dimension.

    Returns:
        tir.Call: A call to the `tl.region` intrinsic describing the memory region.

    Raises:
        KeyError: If access_type is not one of 'r', 'w', or 'rw'.
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
    """
                                 Create a tl region descriptor for the given BufferRegion.

                                 Parameters:
                                     buffer_region (tir.BufferRegion): Source buffer region whose `region` items provide mins and extents.
                                     access_type (str): Access mode: "r", "w", or "rw".
                                     extents (List[PrimExpr]): Requested extents; must have length <= the number of extents in buffer_region.region.

                                 Returns:
                                     tir.Call: A tile-region descriptor (tl.region) covering the buffer_region.

                                 Raises:
                                     AssertionError: If the number of extents in buffer_region.region is smaller than len(extents).
                                 """
    mins = [x.min for x in buffer_region.region]
    region_extents = [x.extent for x in buffer_region.region]
    assert len(region_extents) >= len(
        extents
    ), f"region_extents must be >= extents, region_extents = {region_extents}, extents = {extents}"

    return region(T.BufferLoad(buffer_region.buffer, mins), access_type, *region_extents)


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
    """
         Return a Tensor view of the input buffer with an optional new shape and dtype.

         If `shape` is None the source buffer's shape is used; if `dtype` is None the source buffer's dtype is used. The returned buffer shares the same underlying data as `src` (no copy).
         """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    return T.Tensor(shape, dtype, src.data)


def loop_break():
    """Break out of the current loop.

    Returns:
        tir.Call: A call to the `tl.loop_break` intrinsic.
    """
    return T.call_intrin("handle", op.Op.get("tl.loop_break"))
