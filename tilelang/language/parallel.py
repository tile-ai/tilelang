"""The language interface for tl programs."""
from __future__ import annotations

from typing import Any
from tvm import tir
from tilelang import _ffi_api


def Parallel(*extents: tir.PrimExpr, coalesced_width: int | None = None):
    """Tools to construct nested parallel for loop.
       This can be used to create element-wise tensor expression.

    Parameters
    ----------
    extents : PrimExpr
        The extents of the iteration.

    coalesced_width : Optional[int]
        The coalesced width of the parallel loop.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    annotations: dict[str, Any] = {}
    if coalesced_width is not None:
        annotations.update({"coalesced_width": coalesced_width})
    return _ffi_api.Parallel(extents, annotations)  # type: ignore[attr-defined] # pylint: disable=no-member
