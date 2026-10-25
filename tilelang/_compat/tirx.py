"""Former TIR entry points expressed using standalone TIRx objects."""

from __future__ import annotations

import sys

import tvm_ffi
from tvm_tirx import ir, tirx as _native
from tvm_tirx.ir import prim
from tvm_tirx.ir._prim_expr_overload import EqualOp as EqualOp, NotEqualOp as NotEqualOp
from tvm_tirx.tirx import analysis as analysis, transform as transform
from tvm_tirx.tirx import Function as _NativeFunction
from tvm_tirx.tirx import IndexMap as IndexMap

from . import functor as functor, stmt_functor as stmt_functor
from .functor import PyStmtExprMutator as PyStmtExprMutator, PyStmtExprVisitor as PyStmtExprVisitor
from .tensor import tensor_type

PrimFunc = _NativeFunction


class _PrimitiveMeta(type):
    def __instancecheck__(cls, value):
        return ir.is_prim_expr(value)


class PrimExpr(metaclass=_PrimitiveMeta):
    def __new__(cls, value, dtype=None):
        return ir.const(value, dtype) if dtype is not None else ir.const(value)


class _TensorMeta(type):
    def __instancecheck__(cls, value):
        return _native.is_tensor_var(value)


class Buffer(metaclass=_TensorMeta):
    def __new__(
        cls,
        shape,
        dtype="float32",
        name="buffer",
        data=None,
        strides=None,
        elem_offset=0,
        scope="global",
        data_alignment=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
        span=None,
    ):
        if data is not None:
            raise ValueError("A backing pointer must be introduced by a native decl_tensor Bind")
        if offset_factor or buffer_type or axis_separators:
            raise ValueError("Buffer geometry must be represented by a native TensorType layout")
        return ir.Var(
            name,
            tensor_type(
                shape,
                dtype,
                scope=scope,
                strides=strides,
                elem_offset=0 if elem_offset is None else elem_offset,
                data_alignment=data_alignment,
            ),
        )


class _RegionMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.TensorRegion)


class BufferRegion(metaclass=_RegionMeta):
    def __new__(cls, buffer, region, span=None):
        return ir.TensorRegion(buffer, list(region), ir.TensorRegionType())


class _StoreMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.TensorStore)


class BufferStore(metaclass=_StoreMeta):
    def __new__(cls, buffer, value, indices, predicate=None, span=None):
        if predicate is not None:
            raise ValueError("Predicated stores require native masked_store or an explicit If")
        return ir.TensorStore(buffer, list(indices), value)


BufferLoad = ir.TensorLoad
IfThenElse = ir.If
Var = ir.Var
Stmt = ir.Stmt
Call = ir.Call
IntImm = ir.IntImm
FloatImm = ir.FloatImm
StringImm = ir.StringImm
CallEffectKind = ir.CallEffectKind
const = ir.const
decl_buffer = Buffer
expr = sys.modules[__name__]
stmt = expr
op = expr


def expr_dtype(value):
    if _native.is_tensor_var(value):
        return value.dtype
    if ir.is_prim_expr(value):
        return value.ty.dtype
    if isinstance(value, ir.Expr) and isinstance(value.ty, ir.PtrType):
        return tvm_ffi.dtype("handle")
    raise TypeError(f"Expected a typed scalar or tensor expression, got {value!r}")


def call_intrin(dtype, name, *arguments, annotations=None, span=None):
    attributes = dict(annotations or {})
    result_type = None if dtype == "handle" else ir.PrimType(dtype)
    options = {} if span is None else {"loc": span}
    return ir.Call(name, list(arguments), attrs=attributes, ty=result_type, **options)


def call_extern(dtype, name, *arguments, span=None):
    options = {} if span is None else {"loc": span}
    arguments = [value.data if _native.is_tensor_var(value) else value for value in arguments]
    return ir.Call("tirx.call_extern", [name, *arguments], ty=dtype, **options)


def call_pure_extern(dtype, name, *arguments, span=None):
    options = {} if span is None else {"loc": span}
    arguments = [value.data if _native.is_tensor_var(value) else value for value in arguments]
    return ir.Call("tirx.call_pure_extern", [name, *arguments], ty=dtype, **options)


def reinterpret(dtype, value, span=None):
    result_type = dtype if isinstance(dtype, ir.Type) else ir.PtrType("void") if dtype == "handle" else ir.PrimType(dtype)
    options = {} if span is None else {"loc": span}
    return _native.reinterpret(value, ty=result_type, **options)


def tvm_tuple(*fields):
    return ir.Tuple(list(fields))


def convert(value):
    if isinstance(value, (EqualOp, NotEqualOp)):
        return value.__tvm_ffi_object__()
    if isinstance(value, ir.Expr):
        return value
    if isinstance(value, str):
        return ir.StringImm(value)
    if isinstance(value, (list, tuple)):
        return tvm_ffi.convert([convert(item) for item in value])
    if isinstance(value, dict):
        return tvm_ffi.convert({key: convert(item) for key, item in value.items()})
    if isinstance(value, tvm_ffi.Object):
        return value
    return ir.const(value)


def __getattr__(name):
    from . import intrinsics

    for namespace in (ir, prim, _native, intrinsics):
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise AttributeError(f"No native TIRx implementation for {name!r}")
