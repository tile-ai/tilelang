"""Optional Torch host ranges for Ascend kernel calls."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from tvm.target import Target

from tilelang.jit.adapter.utils import is_ascend_target


def maybe_wrap_host_profile(func: Callable[..., Any], adapter: Any) -> Callable[..., Any]:
    """Return the callable, optionally wrapped in an Ascend host profiling range.

    The profiling switch and range name are resolved when this wrapper is
    created. Disabled profiling, non-Ascend targets, and kernels without a
    resolvable ``global_symbol`` all return the original callable unchanged.
    """
    target = getattr(adapter, "target", None)
    if not isinstance(target, Target) or not is_ascend_target(target):
        return func

    from tilelang import env

    if not env.is_ascend_profile_enabled():
        return func

    prim_func = getattr(adapter, "prim_func", None)
    attrs = getattr(prim_func, "attrs", None) if prim_func is not None else None
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
