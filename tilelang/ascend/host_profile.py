"""Optional Torch host ranges for Ascend kernel calls."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

from tilelang.env import env

if TYPE_CHECKING:
    from tvm.tirx import PrimFunc


def maybe_wrap_host_profile(func: Callable[..., Any], prim_func: PrimFunc) -> Callable[..., Any]:
    """Optionally wrap an Ascend callable in a Torch host profiling range.

    Resolve the switch and range name once when the callable is created or
    restored from disk. Disabled profiling and missing ``global_symbol``
    attributes leave the original callable unchanged.
    """
    if str(env.TILELANG_ASCEND_PROFILE).strip().lower() not in ("1", "true", "yes", "on"):
        return func

    attrs = prim_func.attrs
    kernel_name = attrs.get("global_symbol") if attrs is not None else None
    if kernel_name is None:
        return func

    from torch.profiler import record_function

    range_name = f"tilelang::{kernel_name}"

    @wraps(func)
    def profiled_func(*args, **kwargs):
        with record_function(range_name):
            return func(*args, **kwargs)

    return profiled_func
