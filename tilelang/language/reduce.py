# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm import tir


def reduce(buffer: tir.Buffer, out: tir.Buffer, reduce_type: str, dim: int, clear: bool):
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


def reduce_max(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True):
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
    return reduce(buffer, out, "max", dim, clear)


def reduce_min(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True):
    return reduce(buffer, out, "min", dim, clear)


def reduce_sum(buffer: tir.Buffer, out: tir.Buffer, dim: int):
    return reduce(buffer, out, "sum", dim, True)


def reduce_abssum(buffer: tir.Buffer, out: tir.Buffer, dim: int):
    return reduce(buffer, out, "abssum", dim, True)
