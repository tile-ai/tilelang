"""Loop related language interfaces in TileLang."""

from __future__ import annotations
from typing import Any
from tvm import tir
from tvm.tir import IntImm
import tvm.script.ir_builder.tir as tb_tir
from .v2.builder import SerialForWithStep, UnrollForWithStep
from tilelang import _ffi_api
from tvm.script.ir_builder.tir import frame


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


def Pipelined(
    start: tir.PrimExpr,
    stop: tir.PrimExpr = None,
    num_stages: int = 0,
    order: list[int] | None = None,
    stage: list[int] | None = None,
    sync: list[list[int]] | None = None,
    group: list[list[int]] | None = None,
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


def serial(
    start: tir.PrimExpr, stop: tir.PrimExpr | None = None, step: tir.PrimExpr | None = None, *, annotations: dict[str, Any] | None = None
) -> frame.ForFrame:
    """The serial For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    step : PrimExpr
        The step size of the iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """

    step_is_one = False
    step_is_one |= isinstance(step, int) and step == 1
    step_is_one |= isinstance(step, IntImm) and step.value == 1
    if step is None or step_is_one:
        return tb_tir.serial(start, stop, annotations=annotations)
    else:
        if stop is None:
            stop = start
            start = IntImm(start.dtype, 0) if hasattr(start, "dtype") else 0
        return SerialForWithStep(start, stop, step, annotations=annotations)


def unroll(
    start: tir.PrimExpr,
    stop: tir.PrimExpr | None = None,
    step: tir.PrimExpr | None = None,
    *,
    explicit: bool = False,
    unroll_factor: int | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """The unrolled For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    step : PrimExpr
        The step size of the iteration.

    explicit : bool
        Whether to explicitly unroll the loop.

    unroll_factor : int
        The unroll factor of the loop.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """

    step_is_one = False
    if stop is None:
        stop = start
        if hasattr(start, "dtype"):
            start = IntImm(start.dtype, 0)
        else:
            start = 0

    # Ensure annotations has {"pragma_unroll_explicit": True} by default
    if annotations is None:
        annotations = {"pragma_unroll_explicit": explicit}
    else:
        # Add "pragma_unroll_explicit": True if not already present
        annotations = dict(annotations)
        annotations.setdefault("pragma_unroll_explicit", explicit)

    if unroll_factor is not None:
        # check pragma_unroll_explicit must be False
        if annotations.get("pragma_unroll_explicit", True):
            raise ValueError("pragma_unroll_explicit must be True when unroll_factor is not None")
        annotations.update({"pragma_unroll_factor": unroll_factor})

    if step is None or step_is_one:
        return tb_tir.unroll(start, stop, annotations=annotations)
    else:
        return UnrollForWithStep(start, stop, step, annotations=annotations)


# "Serial" and "Unroll" are aliases of "T.serial" and "T.unroll". We use uppercase to emphasize that they are tile-level loops.


def Serial(
    start: tir.PrimExpr,
    stop: tir.PrimExpr | None = None,
    step: tir.PrimExpr | None = None,
    *,
    annotations: dict[str, Any] | None = None,
):
    """Alias of T.serial."""

    return serial(start, stop, step, annotations=annotations)


def Unroll(
    start: tir.PrimExpr,
    stop: tir.PrimExpr | None = None,
    step: tir.PrimExpr | None = None,
    *,
    explicit: bool = False,
    unroll_factor: int | None = None,
    annotations: dict[str, Any] | None = None,
):
    """Alias of T.unroll."""

    return unroll(start, stop, step, explicit=explicit, unroll_factor=unroll_factor, annotations=annotations)
