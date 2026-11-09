"""Ascend schedule-family selection for graph-generated scalar TIR."""

from __future__ import annotations

from tvm import tirx

from .elementwise import ElementwiseScheduleError, schedule_elementwise


class UnsupportedScheduleError(ValueError):
    """The initial Ascend graph schedule families cannot lower this function."""


def schedule_primfunc(func: tirx.PrimFunc, config=None) -> tirx.PrimFunc:
    """Select an Ascend schedule without falling through to GPU templates."""
    try:
        return schedule_elementwise(func, config)
    except ElementwiseScheduleError as error:
        elementwise_error = error

    from .operators import try_schedule_gemm, try_schedule_reduction

    for schedule in (try_schedule_reduction, try_schedule_gemm):
        try:
            scheduled = schedule(func, config)
        except ValueError:
            continue
        if scheduled is not None:
            return scheduled
    raise UnsupportedScheduleError(f"No Ascend graph schedule applies: {elementwise_error}") from elementwise_error
