from __future__ import annotations
from tilelang import tvm as tvm
from tvm import tir
from tvm.tir import PrimExpr, Buffer, BufferLoad, op
from tilelang import language as T


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


def buffer_load_to_tile_region(load: BufferLoad, access_type: str, extents: list[PrimExpr]):
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


def buffer_region_to_tile_region(buffer_region: tir.BufferRegion, access_type: str,
                                 extents: list[tir.PrimExpr]):
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


def index_to_coordinates(index, shape) -> list[PrimExpr]:
    """
    Convert a flat (linear) index into multi-dimensional coordinates for a given shape.

    Given a linear index and a shape (sequence of dimension extents), returns a list of coordinates (one per dimension) such that converting those coordinates back to a linear index using the usual row-major / C-order formula yields the original index. The computation iterates from the last dimension to the first using modulo and integer division, then reverses the collected coordinates.

    Parameters:
        index (int or PrimExpr): The flat index to convert.
        shape (Sequence[int]): The extents of each dimension (length >= 1).

    Returns:
        List[PrimExpr]: Coordinates for each dimension in the same order as `shape`.
    """
    coordinates = []
    dims = len(shape)
    for i in range(dims):
        coordinates.append(index % shape[dims - i - 1])
        index = index // shape[dims - i - 1]
    coordinates.reverse()
    return coordinates


def linear_index(*args: PrimExpr) -> PrimExpr:
    """
    Compute a flat (linear) index from multi-dimensional coordinates and strides.

    The function accepts a sequence of PrimExpr arguments where the first portion are coordinates
    and the trailing portion are the corresponding strides. The number of strides must equal
    (number of coordinates - 1). The linear index is computed as:

        linear = coords[0]
        for each (coord, stride) in zip(coords[1:], strides):
            linear = linear * stride + coord

    Examples:
        - linear_index(i) -> i
        - linear_index(i, j) -> i * j_stride + j  (requires j_stride provided as stride when needed)
        - linear_index(i, j, stride_j) -> i * stride_j + j
        - linear_index(i, j, k, stride_j, stride_k) -> i*stride_j*stride_k + j*stride_k + k
        - linear_index(i, tx, v, threads, local_size) -> i*threads*local_size + tx*local_size + v

    Raises:
        ValueError: If called with no arguments, or if the number of strides is not one less than
                    the number of coordinates.

    Returns:
        PrimExpr: The computed linear index expression.
    """
    n = len(args)
    if n == 0:
        raise ValueError("At least one index is required")

    if n == 1:
        return args[0]

    # The first part is indices, the second part is strides (starting from the second dimension)
    # A simpler way: the number of strides = total number of arguments - number of indices
    # Actually, the args are designed as indices... + strides..., and the number of strides = number of indices - 1
    num_coords = (n + 1) // 2
    coords = args[:num_coords]
    strides = args[num_coords:]

    if len(strides) != len(coords) - 1:
        raise ValueError("Stride count must be one less than coordinate count")

    linear = coords[0]
    for idx, stride in zip(coords[1:], strides):
        linear = linear * stride + idx
    return linear
