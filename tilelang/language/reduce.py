# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm import tir
from typing import Optional
from tilelang.language import copy, macro, alloc_shared


def _legalize_dim(buffer: tir.Buffer, dim: int):
    if dim < 0:
        dim = len(buffer.shape) + dim
    return dim


def reduce(buffer: tir.Buffer, out: tir.Buffer, reduce_type: str, dim: int, clear: bool):
    """Perform a reduction operation on a buffer along a specified dimension.

    Args:
        buffer (tir.Buffer): Input buffer to reduce
        out (tir.Buffer): Output buffer to store results
        reduce_type (str): Type of reduction ('max', 'min', 'sum', 'abssum')
        dim (int): Dimension along which to perform reduction
        clear (bool): Whether to initialize the output buffer before reduction

    Returns:
        tir.Call: Handle to the reduction operation
    """
    buffer = buffer.access_ptr("r")
    out = out.access_ptr("w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.reduce"),
        buffer,
        out,
        reduce_type,
        dim,
        clear,
    )


def reduce_max(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce max on input buffer, store the result to output buffer

    Parameters
    ----------
    buffer : Buffer
        The input buffer.
    out : Buffer
        The output buffer.
    dim : int
        The dimension to perform reduce on
    clear : bool
        If set to True, the output buffer will first be initialized to -inf.
    Returns
    -------
    handle : PrimExpr
    """
    dim = _legalize_dim(buffer, dim)
    return reduce(buffer, out, "max", dim, clear)


def reduce_min(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce min on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        clear (bool, optional): If True, output buffer will be initialized to inf. Defaults to True.

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return reduce(buffer, out, "min", dim, clear)


def reduce_sum(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce sum on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        clear (bool, optional): If True, output buffer will be cleared before reduction.
                              If False, results will be accumulated on existing values.
                              Defaults to True.
    Note: When clear=True, reduce_sum will not compute directly on the output buffer. This is because 
          during warp reduction, the same value would be accumulated multiple times (number of threads 
          in the warp). Therefore, the implementation with clear=True follows these steps:
        1. create a temp buffer with same shape and dtype as out
        2. copy out to temp buffer
        3. call reduce_sum with temp buffer and out
        4. Add temp buffer to out

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return reduce(buffer, out, "sum", dim, clear)


def reduce_abssum(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1):
    """Perform reduce absolute sum on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return reduce(buffer, out, "abssum", dim, True)


def reduce_absmax(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce absolute max on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return reduce(buffer, out, "absmax", dim, clear)


@macro
def cumsum_fragment(src: tir.Buffer, dst: tir.Buffer, dim: int, reverse: bool) -> tir.PrimExpr:
    cumsum_smem = alloc_shared(src.shape, src.dtype, "shared.dyn")
    copy(src, cumsum_smem)
    tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.cumsum"),
        cumsum_smem.access_ptr("r"),
        cumsum_smem.access_ptr("w"),
        dim,
        reverse,
    )
    copy(cumsum_smem, dst)


def cumsum(src: tir.Buffer, dst: Optional[tir.Buffer] = None, dim: int = 0, reverse: bool = False):
    """Perform cumulative sum on input buffer, store the result to output buffer.

    Args:
        src (tir.Buffer): The input buffer
        dst (tir.Buffer, optional): The output buffer. Defaults to None.
        dim (int, optional): The dimension to perform cumulative sum on. Defaults to 0.
        reverse (bool, optional): Whether to perform reverse cumulative sum. Defaults to False.

    Returns:
        tir.Call: Handle to the cumulative sum operation
    """

    shape = src.shape
    if dim >= len(shape) or dim <= -len(shape):
        raise ValueError(f"Dimension {dim} is out of bounds for buffer with shape {shape}")
    if dim < 0:
        dim = len(shape) + dim

    if dst is None:
        dst = src
    if src.scope() == "local.fragment":
        return cumsum_fragment(src, dst, dim, reverse)
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.cumsum"),
        src.access_ptr("r"),
        dst.access_ptr("w"),
        dim,
        reverse,
    )
