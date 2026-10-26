"""TileLang builder spellings backed exclusively by native lexical frames."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import tvm_ffi

from tvm_tirx import ir
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.script.ir_builder import frame as _shared_frames
from tvm_tirx.tirx.script import ir_builder as _native

from . import tirx

frame = SimpleNamespace(
    IRBuilderFrame=_shared_frames.IRBuilderFrame,
    TIRFrame=_shared_frames.StmtFrame,
    StmtFrame=_shared_frames.StmtFrame,
    ForFrame=_shared_frames.ForFrame,
    IfFrame=_shared_frames.IfFrame,
    WhileFrame=_shared_frames.WhileFrame,
    RegionFrame=_shared_frames.RegionFrame,
    PrimFuncFrame=_native.frame.FunctionFrame,
)

prim_func = _native.function
func_name = _native.func_name_
arg = _native.arg_
If = _native.if_
Then = _native.then_
Else = _native.else_
While = _native.while_
evaluate = _native.evaluate


@dataclass(frozen=True)
class meta_var:
    value: object


@dataclass(frozen=True)
class ThreadBinding:
    frame: _shared_frames.RegionFrame
    begin: object


def bind(value, type_annotation=None):
    if type_annotation is None and isinstance(value, ir.Var):
        type_annotation = value.ty
    return _native.bind_(value, ty=type_annotation)


def buffer(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    scope="global",
    align=0,
    offset_factor=0,
    buffer_type="",
    axis_separators=None,
    name="buffer",
):
    tensor = tirx.Buffer(
        shape,
        dtype,
        name=name,
        strides=strides,
        elem_offset=0 if elem_offset is None else elem_offset,
        scope=scope,
        data_alignment=align,
        offset_factor=offset_factor,
        buffer_type=buffer_type,
        axis_separators=axis_separators,
    )
    if data is not None:
        if isinstance(data.ty, ir.PrimType) and data.ty.dtype == "handle":
            data = ir.Call("tirx.reinterpret", [data], ty=ir.PtrType(tensor.ty.dtype, scope))
        _native.add_to_parent(ir.Bind(tensor, ir.Call("tirx.decl_tensor", [data], ty_args=[tensor.ty])))
    return tensor


def match_buffer(pointer, shape, dtype="float32", strides=None, **kwargs):
    if isinstance(pointer, tirx.Buffer):
        pointer = pointer.data
    return buffer(shape, dtype, data=pointer, strides=strides, **kwargs)


def alloc_buffer(shape, dtype="float32", scope="local", annotations=None, **geometry):
    from tilelang.language.dtypes import get_tvm_dtype

    dtype = get_tvm_dtype(dtype)
    tensor = tirx.Buffer(shape, dtype, scope=scope, **geometry)
    return _native.bind_(_native.alloc_tensor(ty_args=[tensor.ty], attrs=annotations or {}))


def scope_attr(attributes):
    return tvm_ffi.get_global_func("tl.AnnotateScope")(attributes)


def storage_identity(tensor):
    """Resolve a tensor's backing storage through visible native view bindings."""
    if not isinstance(tensor, tirx.Buffer):
        raise TypeError(f"Expected a tensor variable, got {type(tensor)}")
    if not IRBuilder.is_in_scope():
        return tensor
    definitions = {
        statement.var: statement.value
        for active_frame in IRBuilder.current().frames
        if isinstance(active_frame, _shared_frames.StmtFrame)
        for statement in active_frame.stmts
        if isinstance(statement, ir.Bind)
    }
    storage = tensor
    expression = tensor
    visited = set()
    while True:
        if isinstance(expression, ir.Var):
            if expression in visited:
                return storage
            visited.add(expression)
            storage = expression
            expression = definitions.get(expression)
        elif (
            isinstance(expression, ir.Call)
            and isinstance(expression.op, ir.Op)
            and expression.op.name
            in (
                "tirx.decl_tensor",
                "tirx.tensor_data_ptr",
                "tirx.ptr_byte_offset",
                "tirx.reinterpret",
            )
        ):
            expression = expression.args[0]
        else:
            return storage


def attr(node, key, value):
    if node is not None:
        raise ValueError("Node-specific annotations must be carried by native calls or tensor types")
    return _native.region("tl.scope", [], attrs={key: value})


def buffer_store(buffer, value, indices):
    return _native.setitem_(buffer, indices, value)


def decl_buffer(shape, dtype="float32", data=None, *, name="buffer", scope="global", **geometry):
    buffer = tirx.Buffer(shape, dtype, name=name, scope=scope, **geometry)
    if data is None:
        raise ValueError("decl_buffer requires a backing pointer; use alloc_buffer for allocation")
    value = ir.Call("tirx.decl_tensor", [data], ty_args=[buffer.ty])
    _native.add_to_parent(ir.Bind(buffer, value))
    return buffer


def __getattr__(name):
    if hasattr(_native, name):
        return getattr(_native, name)
    return getattr(tirx, name)
