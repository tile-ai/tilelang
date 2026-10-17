"""CUDA dialect of the loop constructs: the common loops plus CUDA hints."""

from __future__ import annotations

from typing import Any

from tvm import tirx
from tvm.tirx.script.builder import frame

from tilelang.language.loop import (
    Parallel as _common_Parallel,
    unroll as _common_unroll,
)

__all__ = ["Parallel", "Unroll", "unroll"]


def Parallel(
    *extents: int | tirx.PrimExpr,
    coalesced_width: int | None = None,
    loop_layout: Any | None = None,
    prefer_async: bool | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """Construct a nested parallel loop, with CUDA lowering hints.

    Same semantics as the common :func:`tilelang.language.loop.Parallel`.
    ``prefer_async`` requests the PTX cp.async rewrite for copies in this loop
    subtree even outside pipelined loops; it is a performance hint ignored by
    targets without async copy.

    Parameters
    ----------
    extents : int | PrimExpr
        Extents of the parallel loop nest.
    coalesced_width : Optional[int]
        Width for coalesced memory access.
    loop_layout : Optional[Fragment]
        Layout annotation for the parallel loop nest.
    prefer_async : Optional[bool]
        When True, requests cp.async injection for this subtree; when False,
        forbids it. Lowered as the ``"parallel_prefer_async"`` annotation.
    annotations : Optional[Dict[str, Any]]
        Additional loop annotations; values in it take precedence.
    """
    ann: dict[str, Any] = dict(annotations) if annotations is not None else {}
    if prefer_async is not None:
        ann.setdefault("parallel_prefer_async", prefer_async)
    return _common_Parallel(
        *extents,
        coalesced_width=coalesced_width,
        loop_layout=loop_layout,
        annotations=ann or None,
    )


def unroll(
    start: tirx.PrimExpr,
    stop: tirx.PrimExpr | None = None,
    step: tirx.PrimExpr | None = None,
    *,
    explicit: bool = False,
    unroll_factor: int | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """The unrolled For statement, with the CUDA unroll-factor pragma.

    Same semantics as the common :func:`tilelang.language.loop.unroll`.
    ``unroll_factor`` emits ``#pragma unroll N``, which only the CUDA codegen
    honors; it is mutually exclusive with ``explicit``.

    Parameters
    ----------
    start, stop, step : PrimExpr
        Iteration range.
    explicit : bool
        Whether to explicitly unroll the loop at compile time.
    unroll_factor : Optional[int]
        Partial unroll factor, lowered as the ``"pragma_unroll_factor"``
        annotation.
    annotations : Optional[Dict[str, Any]]
        Additional loop annotations; values in it take precedence.
    """
    ann: dict[str, Any] = dict(annotations) if annotations is not None else {}
    if unroll_factor is not None:
        ann.setdefault("pragma_unroll_factor", unroll_factor)
    return _common_unroll(start, stop, step, explicit=explicit, annotations=ann or None)


def Unroll(
    start: tirx.PrimExpr,
    stop: tirx.PrimExpr | None = None,
    step: tirx.PrimExpr | None = None,
    *,
    explicit: bool = False,
    unroll_factor: int | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """Alias of the CUDA dialect's :func:`unroll`."""

    return unroll(start, stop, step, explicit=explicit, unroll_factor=unroll_factor, annotations=annotations)
