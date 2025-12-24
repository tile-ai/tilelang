from tilelang import tvm as tvm
from tvm import tir
from tvm.tir import PrimExpr, BufferLoad, op
from tilelang import language as T


def region(buffer: BufferLoad, access_type: str, *args: PrimExpr):
    """Create a tl.region call for a BufferLoad and extents."""
    access_type = {"r": 1, "w": 2, "rw": 3}[access_type]
    return T.call_intrin("handle", op.Op.get("tl.tileop.region"), buffer, access_type, *args)


def buffer_load_to_tile_region(load: BufferLoad, access_type: str, extents: list[PrimExpr]):
    """Convert a BufferLoad to a tl.region call with explicit extents."""
    indices = list(load.indices)
    if len(indices) > len(extents):
        extents = [tir.IntImm("int32", 1) for _ in range(len(indices) - len(extents))] + list(extents)
    assert len(indices) == len(extents), f"indices = {indices}, extents = {extents}"
    return region(load, access_type, *extents)


def buffer_region_to_tile_region(buffer_region: tir.BufferRegion, access_type: str, extents: list[tir.PrimExpr]):
    """Clamp extents and return a tl.region call."""
    mins = [r.min for r in buffer_region.region]
    region_extents = [r.extent for r in buffer_region.region]
    assert len(region_extents) >= len(extents), f"region_extents must be >= extents, region_extents = {region_extents}, extents = {extents}"
    clamped_extents = [
        tir.min(region_extents[i], extents[i]) if i < len(extents) else region_extents[i] for i in range(len(region_extents))
    ]
    return region(tir.BufferLoad(buffer_region.buffer, mins), access_type, *clamped_extents)


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
