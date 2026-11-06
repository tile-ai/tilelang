"""Ascend dialect of the loop constructs: the common ops plus Ascend knobs."""

from __future__ import annotations

from tvm_tirx import ir as _ir
from tvm_tirx.script.ir_builder import frame as _stmt_frame

from typing import Any


pass

from tilelang.language.loop import unroll as _common_unroll

__all__ = ["unroll"]


def unroll(
    start: _ir.Expr,
    stop: _ir.Expr | None = None,
    step: _ir.Expr | None = None,
    *,
    explicit: bool = False,
    unroll_factor: int | None = None,
    annotations: dict[str, Any] | None = None,
) -> _stmt_frame.ForFrame:
    """The unrolled For statement, with the Ascend unroll-factor pragma.

    Same semantics as the common :func:`tilelang.language.loop.unroll`.
    ``unroll_factor`` is lowered as the ``"pragma_unroll_factor"`` annotation,
    which the Ascend codegen emits as ``#pragma unroll N``.

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
