# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from typing import Union, List, Optional
from tilelang import language as T
from tvm import ir, tir


def region(buffer: tir.BufferLoad, access_type: str, *args: tir.PrimExpr):
    """Create a memory region descriptor for tile operations.

    Args:
        buffer (tir.BufferLoad): The buffer to create a region for
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write
        *args (tir.PrimExpr): Extent expressions defining the region size

    Returns:
        tir.Call: A region descriptor for tile operations
    """
    access_type = {"r": 1, "w": 2, "rw": 3}[access_type]
    return tir.call_intrin("handle", tir.op.Op.get("tl.region"), buffer, access_type, *args)


def buffer_to_tile_region(buffer: tir.Buffer, access_type: str):
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


def buffer_load_to_tile_region(load: tir.BufferLoad, access_type: str, extents: List[tir.PrimExpr]):
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
        for i in range(len(extents)):
            new_extents.append(extents[i])
        extents = new_extents
    assert len(indices) == len(extents), f"indices = {indices}, extents = {extents}"
    return region(load, access_type, *extents)


def buffer_region_to_tile_region(buffer_region: tir.BufferRegion, access_type: str):
    """Convert a buffer region to a tile region descriptor.

    Args:
        buffer_region (tir.BufferRegion): The buffer region to convert
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write

    Returns:
        tir.Call: A region descriptor for the specified buffer region
    """
    mins = [x.min for x in buffer_region.region]
    extents = [x.extent for x in buffer_region.region]
    return region(T.BufferLoad(buffer_region.buffer, mins), access_type, *extents)


def copy(
    src: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
    dst: Union[tir.Buffer, tir.BufferLoad],
    coalesced_width: Optional[int] = None,
):
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination memory region
        coalesced_width (Optional[int], optional): Width for coalesced memory access. Defaults to None.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation
    """
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    def get_extent(data):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return data.shape
        elif isinstance(data, tir.BufferRegion):
            return [x.extent for x in data.region]
        else:
            return None

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    if src_extent:
        extent = src_extent
    elif dst_extent:
        extent = dst_extent
    else:
        raise TypeError("Can't deduce copy extents from args")

    def _to_region(data, access_type):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return buffer_to_tile_region(data, access_type)
        elif isinstance(data, tir.BufferRegion):
            return buffer_region_to_tile_region(data, access_type)
        else:
            return buffer_load_to_tile_region(data, access_type, extent)

    src = _to_region(src, "r")
    dst = _to_region(dst, "w")
    if coalesced_width is not None:
        return tir.call_intrin("handle", tir.op.Op.get("tl.copy"), src, dst, coalesced_width)
    else:
        return tir.call_intrin("handle", tir.op.Op.get("tl.copy"), src, dst)


def c2d_im2col(
    img: tir.Buffer,
    col: tir.Buffer,
    nhw_step: tir.PrimExpr,
    c_step: tir.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
):
    """Perform im2col transformation for 2D convolution.

    Args:
        img (tir.Buffer): Input image buffer
        col (tir.Buffer): Output column buffer
        nhw_step (tir.PrimExpr): Step size for batch and spatial dimensions
        c_step (tir.PrimExpr): Step size for channel dimension
        kernel (int): Kernel size
        stride (int): Stride of the convolution
        dilation (int): Dilation rate
        pad (int): Padding size

    Returns:
        tir.Call: A handle to the im2col operation
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.c2d_im2col"),
        img.access_ptr("r"),
        col.access_ptr("w"),
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
    )
