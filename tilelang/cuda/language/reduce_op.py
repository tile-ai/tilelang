"""CUDA dialect of the reduction operators: the common ops plus CUDA knobs."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.reduce_op import (
    reduce_absmax as _common_reduce_absmax,
    reduce_max as _common_reduce_max,
    reduce_min as _common_reduce_min,
)

__all__ = ["reduce_absmax", "reduce_max", "reduce_min"]


def _with_nan_propagate(annotations: dict | None, nan_propagate: bool) -> dict | None:
    if not nan_propagate:
        return annotations
    ann = dict(annotations) if annotations is not None else {}
    ann.setdefault("nan_propagate", True)
    return ann


def reduce_max(
    buffer: tirx.Buffer,
    out: tirx.Buffer,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> None:
    """Perform reduce max, with the CUDA NaN-propagation knob.

    Same semantics as the common :func:`tilelang.language.reduce_op.reduce_max`.
    ``nan_propagate`` is meaningful for float16/bfloat16 only: when True the
    reduction lowers to ``__hmax_nan`` so NaNs propagate; when False (default)
    ``__hmax`` returns the non-NaN operand. Targets without these intrinsics
    reject the annotation at compile time.
    """
    _common_reduce_max(buffer, out, dim, clear, batch=batch, annotations=_with_nan_propagate(annotations, nan_propagate))


def reduce_min(
    buffer: tirx.Buffer,
    out: tirx.Buffer,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> None:
    """Perform reduce min, with the CUDA NaN-propagation knob.

    See :func:`reduce_max`; this lowers to ``__hmin_nan``/``__hmin``.
    """
    _common_reduce_min(buffer, out, dim, clear, batch=batch, annotations=_with_nan_propagate(annotations, nan_propagate))


def reduce_absmax(
    buffer: tirx.Buffer,
    out: tirx.Buffer,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> None:
    """Perform reduce absolute max, with the CUDA NaN-propagation knob.

    See :func:`reduce_max`.
    """
    _common_reduce_absmax(buffer, out, dim, clear, batch=batch, annotations=_with_nan_propagate(annotations, nan_propagate))
