"""Public source-constructor spellings over native IR classes.

Internal construction uses native classes and the explicit DSL policy helpers."""

from __future__ import annotations

import tvm_ffi
from tvm_tirx import ir
from tvm_tirx.ir import prim as _prim

from .._construction import as_expr, current_location, region_as_load
from .._tensor_spec import buffer, is_buffer_like
from .._intrinsics import call_intrin


class _PrimExprMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.Expr) and isinstance(value.ty, (ir.PrimType, ir.PtrType, ir.StringType))


class PrimExpr(metaclass=_PrimExprMeta):
    """DSL membership predicate, without recreating the old IR class tree."""


class _BufferMeta(type):
    def __instancecheck__(cls, value):
        return is_buffer_like(value)


class Buffer(metaclass=_BufferMeta):
    """Construction-time TensorSpec factory and tensor membership predicate."""

    def __new__(cls, shape, dtype="float32", **kwargs):
        return buffer(shape, dtype, **kwargs)


class _BufferRegionMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.TensorRegion)


class BufferRegion(metaclass=_BufferRegionMeta):
    def __new__(cls, source, region, span=None):
        return ir.TensorRegion(source, region, ir.TensorRegionType(), span or current_location())


class _BufferLoadMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.TensorLoad)


class BufferLoad(metaclass=_BufferLoadMeta):
    def __new__(cls, source, indices, span=None):
        if not isinstance(indices, (tuple, list, tvm_ffi.Array)):
            indices = [indices]
        return ir.TensorLoad(source, [as_expr(index) for index in indices], span or current_location())


class _CastMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, _prim.Cast)


class Cast(metaclass=_CastMeta):
    """Construct a native cast, realizing a fixed tensor slice as a load."""

    def __new__(cls, dtype, value, span=None, *, loc=None):
        if span is not None and loc is not None:
            raise TypeError("Cast accepts only one of span and loc")
        location = loc if loc is not None else span if span is not None else current_location()
        value = as_expr(value)
        if isinstance(value, ir.TensorRegion):
            value = region_as_load(value, location)
        return _prim.Cast(dtype, value, location)


class _BufferStoreMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.TensorStore)


class BufferStore(metaclass=_BufferStoreMeta):
    def __new__(cls, buffer, value, indices, predicate=None, span=None):
        if predicate is not None:
            raise TypeError("Predicated stores require an explicit native If region")
        if not isinstance(indices, (tuple, list, tvm_ffi.Array)):
            indices = [indices]
        location = span or current_location()
        value = as_expr(value)
        if isinstance(value, ir.TensorRegion):
            value = region_as_load(value, location)
        return ir.TensorStore(buffer, [as_expr(i) for i in indices], value, location)


class _CallMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.Call)


class Call(metaclass=_CallMeta):
    def __new__(cls, dtype, op, args, annotations=None, span=None):
        return call_intrin(dtype, op, *args, annotations=annotations, span=span)
