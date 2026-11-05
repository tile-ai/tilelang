"""TileLang simplification entry points over native Function and Bind nodes."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import tvm_ffi
from tvm_tirx import tirx
from tvm_tirx.ir import IRModule


def LetInline():
    """Inline native Bind values through TileLang's owning native pass.

    This is a distinct transformation from target StmtSimplify, which keeps
    flat Bind definitions while propagating known values. A compiler missing
    the inlining pass raises at lookup; it is never treated as a no-op.
    """
    return tvm_ffi.get_global_func("tl.transform.LetInline")()


def Simplify(simplify_arguments: bool = False):
    """Apply native rules with the six ``tl.Simplify`` PassContext controls.

    Optional argument cleanup only drops provably unused tensor parameters,
    including their shape/stride definition role. Scalar and pointer parameters
    remain in the public ABI. Tensor metadata and TileLang operators remain
    explicit until their owning lowering passes consume them.

    Known tensor values are propagated only within a straight-line sequence;
    writes, effectful calls and control-flow boundaries invalidate the facts.
    Metadata-only binding uses remain live, and volatile reads are retained.
    The target analyzer owns numeric rewrites. Transitive condition proving
    uses its safe default proof policy, without forcing SYMBOLIC_BOUND.
    """
    return tvm_ffi.get_global_func("tl.transform.Simplify")(simplify_arguments)


def _Simplify(stmt: tirx.Function | IRModule, inline_let: bool = False) -> tirx.Function | IRModule:
    if not isinstance(stmt, (tirx.Function, IRModule)):
        raise ValueError(f"Unsupported type: {type(stmt)}")
    module = IRModule.from_expr(stmt) if isinstance(stmt, tirx.Function) else stmt
    if inline_let:
        module = LetInline()(module)
    module = Simplify(simplify_arguments=True)(module)
    if len(module.functions) != 1:
        raise ValueError("Simplify should return a single function")
    return next(iter(module.functions.values()))


def simplify_prim_func(func: Callable) -> Callable:
    """Simplify a callable's native Function result while preserving metadata."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return _Simplify(func(*args, **kwargs))

    return wrapper


def apply_simplify(stmt: tirx.Function | IRModule, inline_let: bool = False) -> tirx.Function | IRModule:
    """Apply native simplification to a Function or single-function module."""
    return _Simplify(stmt, inline_let)
