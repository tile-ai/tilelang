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


def atomic_max(dst: Buffer, value: PrimExpr, memory_order: str | None = None) -> PrimExpr:
    """
    Perform an atomic maximum on the value stored at dst with an optional memory-order.

    If memory_order is None the runtime extern "AtomicMax" is called without an explicit memory-order id; otherwise the provided memory_order string is mapped to a numeric id using the module's memory-order map and passed to the extern.

    Parameters:
        dst (Buffer): Destination buffer/address to apply the atomic max.
        value (PrimExpr): Value to compare/store atomically.
        memory_order (str | None): Optional memory-order name (e.g. "relaxed", "acquire", "seq_cst").
            If provided, it is translated to the corresponding numeric memory-order id before the call.

    Returns:
        PrimExpr: A handle/expression representing the issued atomic maximum operation.
    """
    if memory_order is None:
        return T.call_extern("handle", "AtomicMax", T.address_of(dst), value)
    else:
        return T.call_extern("handle", "AtomicMax", T.address_of(dst), value,
                             _MEMORY_ORDER_ID_MAP[memory_order])


def atomic_min(dst: Buffer, value: PrimExpr, memory_order: str | None = None) -> PrimExpr:
    """
    Atomically update the value at dst to the minimum of its current value and value.

    If memory_order is provided, it selects the memory-order semantic used by the underlying extern call;
    allowed names are "relaxed", "consume", "acquire", "release", "acq_rel", and "seq_cst" (mapped internally
    to integer IDs). If memory_order is None, the extern is invoked without an explicit memory-order argument.

    Parameters:
        memory_order (str | None): Optional memory-order name controlling the atomic operation's ordering.

    Returns:
        PrimExpr: A handle expression representing the atomic-min operation.
    """
    if memory_order is None:
        return T.call_extern("handle", "AtomicMin", T.address_of(dst), value)
    else:
        return T.call_extern("handle", "AtomicMin", T.address_of(dst), value,
                             _MEMORY_ORDER_ID_MAP[memory_order])


def atomic_add(dst: Buffer, value: PrimExpr, memory_order: str | None = None) -> PrimExpr:
    """
    Atomically add `value` into `dst`, returning a handle to the operation.

    Supports scalar/addressed extern atomic add when neither argument exposes extents, or tile-region-based atomic add for Buffer/BufferRegion/BufferLoad inputs. If both arguments are plain Buffers their shapes must be structurally equal. If at least one side exposes extents, extents are aligned (missing dimensions are treated as size 1); an assertion is raised if extents cannot be deduced. The optional `memory_order` (one of "relaxed","consume","acquire","release","acq_rel","seq_cst") is used only for the direct extern `AtomicAdd` path when no extents are available — otherwise the tile-region path ignores `memory_order`.

    Returns:
        PrimExpr: A handle representing the atomic addition operation.
    """

    def get_extent(data):
        """
        Return the inferred extent (shape) of a buffer-like object.

        If `data` is a Var bound to a let value, the let value is resolved before inspection.
        Parameters:
            data: A Var, Buffer, or BufferRegion to inspect.

        Returns:
            The shape/extents as a list-like of PrimExpr (Buffer.shape or list of region item extents), or None if the extent cannot be determined.
        """
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
    """
         Return a Tensor view of the input buffer with an optional new shape and dtype.

         If `shape` is None the source buffer's shape is used; if `dtype` is None the source buffer's dtype is used. The returned buffer shares the same underlying data as `src` (no copy).
         """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    return T.Tensor(shape, dtype, src.data)


def atomic_load(src: Buffer, memory_order: str = "seq_cst") -> PrimExpr:
    """
    Load a value from the given buffer using the specified atomic memory ordering.

    Performs an atomic load from `src` and returns a PrimExpr representing the loaded value.
    memory_order selects the ordering and must be one of: "relaxed", "consume", "acquire",
    "release", "acq_rel", or "seq_cst" (default).
    Raises KeyError if an unknown memory_order is provided.
    """
    return T.call_extern(src.dtype, "AtomicLoad", T.address_of(src),
                         _MEMORY_ORDER_ID_MAP[memory_order])


def atomic_store(dst: Buffer, src: PrimExpr, memory_order: str = "seq_cst") -> PrimExpr:
    """
    Perform an atomic store of `src` into `dst` with the given memory ordering.

    Parameters:
        dst (Buffer): Destination buffer to store into.
        src (PrimExpr): Value to store.
        memory_order (str, optional): Memory ordering name; one of "relaxed", "consume",
            "acquire", "release", "acq_rel", or "seq_cst". Defaults to "seq_cst".
            The name is mapped to an internal numeric ID used by the underlying runtime.

    Returns:
        PrimExpr: A handle representing the issued atomic store operation.

    Raises:
        KeyError: If `memory_order` is not one of the supported names.
    """
    return T.call_extern("handle", "AtomicStore", T.address_of(dst), src,
                         _MEMORY_ORDER_ID_MAP[memory_order])
