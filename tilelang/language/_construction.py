"""Expression conversion, source locations, and statement construction for the DSL."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Any

import tvm_ffi
from tvm_tirx import ir
from tvm_tirx import tirx as _tirx
from tvm_tirx.ir import prim as _prim
from tvm_tirx.ir._prim_expr_overload import EqualOp, NotEqualOp
from tvm_tirx.script.ir_builder.base import AlreadyEmitted
from tvm_tirx.tirx.script import ir_builder as _builder


def unwrap(value):
    if isinstance(value, AlreadyEmitted):
        return value.value
    if isinstance(value, (EqualOp, NotEqualOp)):
        return value.__tvm_ffi_object__()
    return value


def as_expr(value):
    value = unwrap(value)
    if isinstance(value, ir.Expr):
        return value
    if isinstance(value, str):
        return ir.StringImm(value)
    if isinstance(value, (tuple, list, tvm_ffi.Array)):
        return ir.Tuple([as_expr(item) for item in value])
    # Python shape/stride/loop literals used the source language's automatic
    # int64 promotion. Native ir.const deliberately defaults to int32, so
    # supply the established width at this TileLang construction boundary.
    if isinstance(value, builtins.int) and not -(1 << 31) <= value < (1 << 31):
        return ir.const(value, "int64")
    return ir.const(value)


def tvm_tuple(*values):
    return ir.Tuple([as_expr(value) for value in values])


def current_location():
    # Lazy import avoids a dependency cycle while language exports initialize.
    from .eager.builder import Builder

    current = Builder.current()
    return current.current_location if current is not None else ir.UnknownLoc()


def region_as_load(region, location):
    """Realize a scalar or fixed vector slice at a primitive-value boundary.

    Keep the original tensor and logical coordinates so its layout, offset,
    and aliases still determine the addresses. Native TensorLoad permits a
    vector index only in the last dimension; a larger tile remains a region.
    """
    indices = []
    for dimension, bounds in enumerate(region.region):
        extent = bounds.extent
        if not isinstance(extent, ir.IntImm) or extent.value <= 0:
            raise ValueError("A primitive tensor-region value requires a positive constant extent in each dimension")
        if extent.value == 1:
            indices.append(bounds.begin)
        elif dimension != len(region.region) - 1:
            raise ValueError("A primitive tensor-region value may vectorize only the last dimension")
        else:
            indices.append(_prim.Ramp(bounds.begin, ir.const(1, bounds.begin.ty.dtype), extent, location))
    return ir.TensorLoad(region.source, indices, location)


def buffer_store(buffer, value, indices, predicate=None):
    value = unwrap(value)
    if isinstance(value, ir.TensorRegion):
        value = region_as_load(value, current_location())
    if predicate is None:
        return _builder.setitem_(buffer, indices, value, loc=current_location())
    with _builder.if_(predicate, loc=current_location()), _builder.then_(loc=current_location()):
        return _builder.setitem_(buffer, indices, value, loc=current_location())


def bind(value, type_annotation=None, *, var=None):
    """Emit one immutable binding and return its native variable.

    Explicit ``T.bind`` materializes its value even when the value is already
    a Var. Ordinary assignments keep the parser's alias policy. An optional
    ``var`` supplies the destination identity, as in the original builder API.
    """
    value = as_expr(value)
    if var is not None:
        if not isinstance(var, ir.Var):
            raise TypeError("bind var must be an ir.Var")
        _builder.emit_(ir.Bind(var, value, current_location()), loc=current_location())
        return var
    annotation = value.ty if type_annotation is None else type_annotation
    return unwrap(_builder.bind_(value, ty=annotation, loc=current_location()))


def let(v, value, body=None):
    """Preserve the callable source spelling for Bind and Let expressions."""
    from tilelang.utils.deprecated import deprecated_warning

    if body is None:
        deprecated_warning("T.let", "T.bind")
        return bind(value, var=v)
    deprecated_warning("T.let", "T.Let")
    return _prim.Let(v, as_expr(value), as_expr(body), current_location())


def side_effect(value):
    """Return target-recognized expression effects using the owning IR analysis."""
    return tvm_ffi.get_global_func("tl.SideEffect")(unwrap(value))


def dtype_of(value):
    """Read primitive dtype from a native expression or a Python tensor."""
    value = unwrap(value)
    if isinstance(value, ir.Expr):
        if isinstance(value.ty, (ir.PrimType, _tirx.TensorType)):
            ty = value.ty.dtype
            value = ty.dtype if isinstance(ty, ir.PrimType) else ty
            return value
        if isinstance(value.ty, ir.PtrType | ir.StringType):
            return tvm_ffi.dtype("handle")
        raise TypeError(f"Expression of type {value.ty} has no primitive dtype")
    return value.dtype


def evaluate(value):
    if isinstance(value, ir.Return):
        _builder.emit_(value)
        return AlreadyEmitted(value)
    return _builder.evaluate(value)


def prim_func(is_private=False):
    return _builder.function(private=is_private)


@dataclass
class meta_var:
    value: Any


def Assert(condition, message):
    from tvm_tirx.script.ir_builder.base import at

    # The lower-case target helper emits an empty frame immediately. The
    # historical capitalized spelling is also a context manager, so return
    # the real native AssertFrame for the caller to enter and populate.
    kind = "RuntimeError"
    if isinstance(message, tuple):
        kind, message = message
    parts = [str(part) for part in message] if isinstance(message, (list, tuple)) else [str(message)]
    frame = tvm_ffi.get_global_func("script.ir_builder.Assert")(as_expr(condition), kind, parts)
    return at(current_location(), frame)


def break_loop():
    return _builder.break_(loc=current_location())


def continue_loop():
    return _builder.continue_(loc=current_location())


def thread_return():
    """Return from the calling GPU thread using the native control-flow op."""
    return _tirx.op.gpu_thread_return(loc=current_location())


def handle(dtype=None, storage_scope="global", *, is_size_var=False):
    """Construct a native pointer while retaining the source keyword contract.

    ``is_size_var`` is accepted for existing annotations.  Native pointers do
    not carry the integer size default used by TileLang's ABI decoder.
    """
    return _builder.handle(dtype, storage_scope)


def void(expr=None, *, is_size_var=False):
    """Retain the void constructor spelling without a numerical Var domain."""
    return _builder.void(expr)


def type_annotation(dtype):
    return ir.DataTypeImm(dtype)


def is_scalar_expr(value):
    """Whether a native expression is usable as a scalar DSL value."""
    return isinstance(value, ir.Expr) and isinstance(value.ty, (ir.PrimType, ir.PtrType, ir.StringType))


def as_indices(indices):
    """Normalize source scalar/sequence subscripts to native expressions."""
    if not isinstance(indices, (tuple, list, tvm_ffi.Array)):
        indices = [indices]
    return [as_expr(index) for index in indices]


def scalar_value(value, location=None):
    """Realize a fixed tensor region at a primitive-value boundary."""
    value = as_expr(value)
    if isinstance(value, ir.TensorRegion):
        return region_as_load(value, current_location() if location is None else location)
    return value
