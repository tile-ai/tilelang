"""TileLang script operations delegated to native TIRx builders."""

import functools

from tvm_tirx import ir
from tvm_tirx.ir import prim
from tvm_tirx.tirx.script import ir_builder as _native

from tilelang._compat.builder import alloc_buffer, attr, buffer_store, decl_buffer, match_buffer, meta_var as meta_var
from tilelang._compat.tirx import PrimExpr as PrimExpr
from tilelang._compat.tirx import call_extern as call_extern, call_intrin as call_intrin, call_pure_extern as call_pure_extern, tvm_tuple
from tilelang._compat.intrinsics import (
    ptx_arrive_barrier,
    ptx_arrive_barrier_expect_tx,
    ptx_commit_group,
    ptx_cp_async_barrier,
    ptx_init_barrier_thread_count,
    ptx_wait_group,
)
from .op import ptx_cp_async, reinterpret as reinterpret, round as _round_op
from .exports import BACKEND_ONLY_TIR_EXPORTS
from ..dtypes import __all__ as _dtype_exports


_VALID_CAST_ROUNDING_MODES = {"", "rn", "rz", "rp", "rm", "rs"}
While = _native.while_
round = _round_op


def infinity(dtype="float32", span=None):
    return ir.const(float("inf"), dtype, **({} if span is None else {"loc": span}))


class _LetBinding(type(_native.let)):
    def __call__(self, variable, value):
        from ..frame import register_let_value

        if not isinstance(variable, ir.Var):
            raise TypeError("T.let expects a native variable as its first argument")
        value = _native.cast(value, ty=variable.ty)
        _native.add_to_parent(ir.Bind(variable, value))
        register_let_value(variable, value)


let = _LetBinding()


def extract_lane(value, lane):
    """Extract a compile-time lane from a native vector expression."""
    if not ir.is_prim_expr(value):
        raise TypeError("extract_lane expects a PrimExpr")
    lanes = value.ty.dtype.lanes
    if lanes <= 1:
        raise ValueError("extract_lane expects a vector expression")
    if isinstance(lane, ir.IntImm):
        lane = int(lane.value)
    if not isinstance(lane, int):
        raise TypeError("extract_lane requires a compile-time integer lane")
    if not 0 <= lane < lanes:
        raise IndexError(f"extract_lane index {lane} is out of bounds for {lanes} lanes")
    return prim.Shuffle([value], [lane])


def cast(value, dtype=None, round: str = "", sat: bool = True, rbits=None, span=None, *, ty=None, loc=None):
    """Cast a native value, preserving explicit rounding and saturation options."""
    if round not in _VALID_CAST_ROUNDING_MODES:
        raise ValueError(f"Invalid round '{round}'. Must be one of: {sorted(_VALID_CAST_ROUNDING_MODES)}")
    if not isinstance(sat, bool):
        raise ValueError(f"Invalid sat '{sat}'. Must be a bool")
    if round == "rs" and rbits is None:
        raise ValueError("rbits is required when round='rs' (stochastic rounding)")
    if round != "rs" and rbits is not None:
        raise ValueError("rbits is only valid with round='rs' (stochastic rounding)")
    location = loc if loc is not None else span
    options = {} if location is None else {"loc": location}
    converted = _native.cast(value, dtype, ty=ty, **options)
    if not round and sat:
        return converted
    if not isinstance(value, ir.Expr):
        value = ir.const(value)
    arguments = [value]
    if rbits is not None:
        arguments.append(rbits)
    return ir.Call(
        "tl.cast_with_rounding",
        arguments,
        attrs={"dtype": str(converted.ty.dtype.with_lanes(1)), "round": round, "sat": sat},
        ty=converted.ty,
        **options,
    )


def _dtype_forward(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            args = (kwargs.pop("dtype"),) + args
        return func(*args, **kwargs)

    return wrapped


def _op_wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        kwargs.pop("dtype", None)
        return func(*args, **kwargs)

    return wrapped


__all__ = tuple(
    name for name in _native.__all__ if not name.startswith("_") and name not in _dtype_exports and name not in BACKEND_ONLY_TIR_EXPORTS
) + (
    "alloc_buffer",
    "attr",
    "buffer_store",
    "decl_buffer",
    "extract_lane",
    "PrimExpr",
    "match_buffer",
    "tvm_tuple",
    "While",
    "ptx_arrive_barrier",
    "ptx_arrive_barrier_expect_tx",
    "ptx_commit_group",
    "ptx_cp_async",
    "ptx_cp_async_barrier",
    "ptx_init_barrier_thread_count",
    "ptx_wait_group",
)


def __getattr__(name):
    return getattr(_native, name)
