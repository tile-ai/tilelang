# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tvm.script.ir_builder.tir.ir as _ir
from tvm.script.ir_builder.tir import frame
from tvm.tir import PrimExpr
from typing import Any, Dict


def serial(start: PrimExpr,
           stop: PrimExpr = None,
           *,
           annotations: Dict[str, Any] = None) -> frame.ForFrame:
    """The serial For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.serial(start=start, stop=stop, annotations=annotations)


def parallel(start: PrimExpr,
             stop: PrimExpr = None,
             *,
             annotations: Dict[str, Any] = None) -> frame.ForFrame:
    """The parallel For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.parallel(start=start, stop=stop, annotations=annotations)


def vectorized(start: PrimExpr,
               stop: PrimExpr = None,
               *,
               annotations: Dict[str, Any] = None) -> frame.ForFrame:
    """The vectorized For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.vectorized(start=start, stop=stop, annotations=annotations)


def unroll(start: PrimExpr,
           stop: PrimExpr = None,
           *,
           annotations: Dict[str, Any] = None) -> frame.ForFrame:
    """The unrolled For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.unroll(start=start, stop=stop, annotations=annotations)


def thread_binding(
    start: PrimExpr,
    stop: PrimExpr = None,
    thread: str = None,
    *,
    annotations: Dict[str, Any] = None,
) -> frame.ForFrame:
    """The thread-binding For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    thread : str
        The thread for loop variable to bind.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.thread_binding(start=start, stop=stop, thread=thread, annotations=annotations)


def grid(*extents: PrimExpr) -> frame.ForFrame:
    """The grid For statement.

    Parameters
    ----------
    extents : PrimExpr
        The extents of the iteration.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.grid(*extents)
