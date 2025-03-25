# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from typing import List, Optional
from tvm import tir
from tvm.tir import IntImm
from tilelang import _ffi_api


def Pipelined(
    start: tir.PrimExpr,
    stop: tir.PrimExpr = None,
    num_stages: int = 0,
    order: Optional[List[int]] = None,
    stage: Optional[List[int]] = None,
    sync: Optional[List[List[int]]] = None,
    group: Optional[List[List[int]]] = None,
):
    """Tools to construct pipelined for loop.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.
    stop : PrimExpr
        The maximum value of iteration.
    num_stages : int
        The max number of buffer used between pipeline producers and consumers.
        if num_stages is 0, pipeline will not be enabled.
    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        start = IntImm(start.dtype, 0) if hasattr(start, "dtype") else 0
    if order is None:
        order = []
    if stage is None:
        stage = []
    if sync is None:
        sync = []
    if group is None:
        group = []
    # type: ignore[attr-defined] # pylint: disable=no-member
    return _ffi_api.Pipelined(start, stop, num_stages, order, stage, sync, group)
