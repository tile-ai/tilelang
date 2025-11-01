"""The language interface for tl programs."""
from __future__ import annotations

from tvm import tir
from tilelang import _ffi_api


def Persistent(
    domain: list[tir.PrimExpr],
    wave_size: tir.PrimExpr,
    index: tir.PrimExpr,
    group_size: tir.PrimExpr | None = 8,
):
    """Tools to construct persistent for loop.

    Parameters
    ----------
    domain : List[tir.PrimExpr]
        The list of dominators.
    wave_size : int
        The wave size.
    index : int
        The tile index in one wave.
    group_size : tir.PrimExpr
        The group size.
    """
    return _ffi_api.Persistent(domain, wave_size, index, group_size)
