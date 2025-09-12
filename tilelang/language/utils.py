from tilelang import tvm as tvm
from tvm.tir import PrimExpr


def index_to_coordinates(index, shape) -> list[PrimExpr]:
    """
    Convert a flat (linear) index into multi-dimensional coordinates for a given shape.

    Given a linear index and a shape (sequence of dimension extents), returns a list of coordinates (one per dimension) such that converting those coordinates back to a linear index using the usual row-major / C-order formula yields the original index. The computation iterates from the last dimension to the first using modulo and integer division, then reverses the collected coordinates.

    Parameters:
        index (int or PrimExpr): The flat index to convert.
        shape (Sequence[int]): The extents of each dimension (length >= 1).

    Returns:
        list[PrimExpr]: Coordinates for each dimension in the same order as `shape`.
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
