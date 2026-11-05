"""TileLang construction policy over the standalone TIRx object model.

This module is the boundary between the Python DSL and native IR.  ``TensorSpec``
is an annotation, never a compiler node: materialized tensors are ordinary
``ir.Var`` objects carrying ``tirx.TensorType``.  The compatibility spellings in
this module are local to TileLang and do not modify the ``tvm_tirx`` namespace.
"""

from __future__ import annotations

import builtins
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import tvm_ffi
from tvm_tirx import ir
from tvm_tirx import tirx as _tirx
from tvm_tirx.ir import prim as _prim
from tvm_tirx.ir._prim_expr_overload import EqualOp, NotEqualOp
from tvm_tirx.tirx.layout import Axis, Iter
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.script.ir_builder import frame as _shared_frame
from tvm_tirx.script.ir_builder.base import AlreadyEmitted, IRBuilderFrame
from tvm_tirx.tirx.script import ir_builder as _builder
from tvm_tirx.tirx.script.ir_builder.frame import FunctionFrame

Var = ir.Var
IntImm = ir.IntImm
FloatImm = ir.FloatImm
StringImm = ir.StringImm
Range = ir.Range
Type = ir.Type
PrimFunc = _tirx.Function
Stmt = ir.Stmt
CallEffectKind = ir.CallEffectKind
TensorLoad = ir.TensorLoad
TensorRegion = ir.TensorRegion
TensorType = _tirx.TensorType
IndexMap = _tirx.IndexMap
DataType = tvm_ffi.dtype
convert = tvm_ffi.convert
const = ir.const
structural_equal = tvm_ffi.structural_equal
assert_structural_equal = ir.assert_structural_equal


class _PrimExprMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.Expr) and isinstance(value.ty, (ir.PrimType, ir.PtrType, ir.StringType))


class PrimExpr(metaclass=_PrimExprMeta):
    """DSL membership predicate, without recreating the old IR class tree."""


def is_tensor(value):
    return isinstance(value, TensorSpec) or _tirx.is_tensor_var(value)


class _BufferMeta(type):
    def __instancecheck__(cls, value):
        return is_tensor(value)


class Buffer(metaclass=_BufferMeta):
    """Construction-time TensorSpec factory and tensor membership predicate."""

    def __new__(cls, shape, dtype="float32", **kwargs):
        return buffer(shape, dtype, **kwargs)


def _unwrap(value):
    if isinstance(value, AlreadyEmitted):
        return value.value
    if isinstance(value, (EqualOp, NotEqualOp)):
        return value.__tvm_ffi_object__()
    return value


def _expr(value):
    value = _unwrap(value)
    if isinstance(value, ir.Expr):
        return value
    if isinstance(value, str):
        return ir.StringImm(value)
    if isinstance(value, (tuple, list, tvm_ffi.Array)):
        return ir.Tuple([_expr(item) for item in value])
    # Python shape/stride/loop literals used the source language's automatic
    # int64 promotion. Native ir.const deliberately defaults to int32, so
    # supply the established width at this TileLang construction boundary.
    if isinstance(value, builtins.int) and not -(1 << 31) <= value < (1 << 31):
        return ir.const(value, "int64")
    return ir.const(value)


def tvm_tuple(*values):
    return ir.Tuple([_expr(value) for value in values])


def _loc():
    # Lazy import avoids a dependency cycle while language exports initialize.
    from .eager.builder import Builder

    current = Builder.current()
    return current.current_location if current is not None else ir.UnknownLoc()


def _strides(shape):
    result = []
    stride = ir.const(1, shape[-1].ty.dtype) if shape else ir.const(1)
    for extent in reversed(shape):
        result.append(stride)
        stride = stride * extent
    return tuple(reversed(result))


@dataclass(eq=False)
class TensorSpec:
    """A source annotation whose shape/stride expressions retain their identities."""

    shape: tuple
    dtype: tvm_ffi.dtype
    strides: tuple
    elem_offset: ir.Expr
    storage_scope: str = "global"
    data_alignment: int = 0
    data: ir.Expr | None = None
    offset_factor: int = 0
    buffer_type: str = ""
    axis_separators: tuple = ()
    compact: bool = False

    def scope(self):
        return self.storage_scope

    def contract_metadata(self, *, parameter=False):
        """Return constraints consumed before generic tensor/ABI lowering.

        Physical strides and offset are already native layout expressions.
        These additional declarations describe checks and logical views; they
        must survive the temporary annotation object as typed IR operands.
        Only parameters need the explicit-stride source flag for ABI checks;
        a local view's physical strides and origin live entirely in its type.
        """
        if (not parameter or self.compact) and self.offset_factor <= 1 and not self.buffer_type and not self.axis_separators:
            return None
        return {
            "offset_factor": max(1, self.offset_factor),
            "buffer_type": self.buffer_type,
            "axis_separators": list(self.axis_separators),
            # Explicit contiguous strides still use the source strided
            # segmentation rule; native layout equivalence cannot recover
            # whether the user supplied strides.
            "compact": self.compact,
        }

    def as_type(self, *, allocation=False):
        # Compactness is an address contract, not an equation constraining
        # strides of empty or unit dimensions. Preserve explicit strided
        # annotations and nonzero/symbolic origins as actual layout operands.
        layout = None
        if not self.compact or not isinstance(self.elem_offset, ir.IntImm) or self.elem_offset.value != 0:
            layout = _tirx.TileLayout.from_iters(
                [Iter(extent, stride, "m") for extent, stride in zip(self.shape, self.strides)],
                offset={"m": self.elem_offset},
            )
        # A supplied tensor or pointer may start at a slice of its allocation.
        # Only owned storage receives the allocator's default alignment; a
        # source annotation promises more than byte alignment only explicitly.
        alignment = self.data_alignment or (0 if allocation else 1)
        return TensorType(
            self.shape,
            ir.PrimType(self.dtype),
            self.storage_scope,
            align=alignment,
            layout=layout,
            data=self.data,
        )


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
    name="",
):
    """Create a source tensor annotation, or bind a view of an explicit pointer."""
    from .dtypes import get_tvm_dtype

    if isinstance(shape, (int, PrimExpr)):
        shape = (shape,)
    shape = tuple(_expr(dim) for dim in shape)
    index_dtype = shape[0].ty.dtype if shape else "int32"
    buffer_type = str(buffer_type)
    if buffer_type not in ("", "default", "auto", "auto_broadcast"):
        raise ValueError("`buffer_type` must be `auto`, `default`, or empty")
    buffer_type = "auto" if buffer_type in ("auto", "auto_broadcast") else ""
    offset_factor = int(offset_factor)
    if offset_factor < 0:
        raise ValueError("offset_factor must be non-negative")
    axis_separators = tuple(int(value) for value in (axis_separators or ()))
    if any(value < 0 or value > len(shape) for value in axis_separators) or any(
        left > right for left, right in zip(axis_separators, axis_separators[1:])
    ):
        raise ValueError("axis_separators must be nondecreasing positions between 0 and the tensor rank")
    compact = not strides and buffer_type != "auto"
    if strides:
        strides = tuple(Var(value, index_dtype) if isinstance(value, str) else _expr(value) for value in strides)
    elif buffer_type == "auto":
        # The ABI binds these to physical strides. The TileLang tensor-spec
        # consumer creates a logical view with stride 0 for each unit axis.
        strides = tuple(Var(f"{name + '_' if name else ''}stride_{i}", index_dtype) for i in range(len(shape)))
    else:
        strides = _strides(shape)
    if len(strides) != len(shape):
        raise ValueError("Invalid shape/strides' dimensions")
    if elem_offset is None:
        elem_offset = Var(f"{name + '_' if name else ''}elem_offset", index_dtype) if offset_factor else ir.const(0, index_dtype)
    result = TensorSpec(
        shape,
        get_tvm_dtype(dtype),
        strides,
        _expr(elem_offset),
        str(scope),
        int(align),
        data,
        offset_factor,
        buffer_type,
        axis_separators,
        compact,
    )
    if data is not None and IRBuilder.is_in_scope():
        # Tensor/StridedTensor with an explicit pointer is a value-producing
        # declaration. Bind it here so immediate indexing, nested views and
        # named assignments share one native TensorVar definition.
        value = bind(_builder.decl_tensor(data, ty_args=[result.as_type()], loc=_loc()))
        _record_local_tensor_spec(value, result)
        return value
    return result


def decl_tensor(data, *, ty_args, attrs=None, loc=None, ty=None):
    """Declare a view of a native data pointer using a TensorType or TensorSpec.

    Native type arguments keep the upstream Call contract. A source TensorSpec
    is normalized without changing the pointer's meaning. Its extra source
    constraints need a bound TensorVar before their annotation is emitted.
    """
    annotations = list(ty_args)
    normalized = [value.as_type() if isinstance(value, TensorSpec) else value for value in annotations]
    value = _builder.decl_tensor(_expr(data), ty_args=normalized, attrs=attrs, loc=_loc() if loc is None else loc, ty=ty)
    if len(annotations) == 1 and isinstance(annotations[0], TensorSpec):
        spec = annotations[0]
        if spec.contract_metadata() is not None:
            if not IRBuilder.is_in_scope():
                raise ValueError("TensorSpec source constraints on decl_tensor require an active TileLang builder")
            value = bind(value)
            _record_local_tensor_spec(value, spec)
    return value


def decl_buffer(
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
    name="",
):
    """Emit a local tensor declaration in the current statement frame.

    Without ``data`` this owns a new allocation; with ``data`` it declares a
    view of that pointer. ``buffer``/``Tensor`` remain annotation factories.
    Each declaration returns the bound native TensorVar, so Python naming,
    source locations and subsequent views all refer to the same definition.
    """
    IRBuilder.current()
    spec = buffer(
        shape,
        dtype,
        strides=strides,
        elem_offset=elem_offset,
        scope=scope,
        align=align,
        offset_factor=offset_factor,
        buffer_type=buffer_type,
        axis_separators=axis_separators,
        name=name,
    )
    if data is None:
        value = _unwrap(_builder.alloc_tensor(ty_args=[spec.as_type(allocation=True)], loc=_loc()))
    else:
        spec.data = _expr(data)
        value = bind(_builder.decl_tensor(spec.data, ty_args=[spec.as_type()], loc=_loc()))
    _record_local_tensor_spec(value, spec)
    return value


def tensor_strides(value):
    """Read affine source strides from a tensor contract, without a side table."""
    if isinstance(value, TensorSpec):
        return value.strides
    if not _tirx.is_tensor_var(value):
        raise TypeError("Expected a native tensor variable")
    layout = value.ty.layout
    if layout is None:
        return _strides(tuple(value.ty.shape))
    if (
        isinstance(layout, _tirx.TileLayout)
        and len(layout.shard) == len(value.ty.shape)
        and all(str(item.axis.name) == "m" for item in layout.shard)
    ):
        return tuple(item.stride for item in layout.shard)
    raise ValueError("Tensor layout does not have one affine stride per logical dimension")


def tensor_offset(value):
    if isinstance(value, TensorSpec):
        return value.elem_offset
    layout = value.ty.layout
    if layout is None:
        return ir.const(0)
    if isinstance(layout, _tirx.TileLayout):
        return layout.offset.get(Axis.get("m"), ir.const(0))
    raise ValueError("Tensor layout does not have an affine element offset")


def tensor_storage_source(value):
    """Resolve direct DSL view aliases for construction-time diagnostics.

    Read already emitted definitions from active native statement frames.
    Emitted metadata retains TensorVar keys, so later passes can repeat alias
    analysis on their actual IR without construction-time Python mappings.
    """
    from .frame import get_let_value, has_let_value

    definitions = {}
    if IRBuilder.is_in_scope():
        for frame in IRBuilder.current().frames:
            if isinstance(frame, _shared_frame.StmtFrame):
                for statement in frame.stmts:
                    if isinstance(statement, ir.Bind):
                        definitions[statement.var] = statement.value
    visited = set()
    while value not in visited:
        visited.add(value)
        if isinstance(value, ir.Var):
            definition = definitions.get(value)
            if definition is None and has_let_value(value):
                definition = get_let_value(value)
            if isinstance(definition, ir.Var):
                value = definition
                continue
            if isinstance(definition, ir.Call) and isinstance(definition.op, ir.Op) and definition.op.name == "tirx.decl_tensor":
                value = definition.args[0]
                continue
        if isinstance(value, ir.Call) and isinstance(value.op, ir.Op) and value.op.name == "tirx.tensor_data_ptr":
            value = value.args[0]
            continue
        return value
    raise ValueError("Cyclic tensor view definition")


def arg(name, annotation):
    source = annotation
    spec = annotation if isinstance(annotation, TensorSpec) else None
    if isinstance(annotation, TensorSpec):
        annotation = annotation.as_type()
    value = _builder.arg_(name, annotation, loc=_loc())
    if isinstance(source, ir.Var) and tvm_ffi.get_global_func("tl.HasSizeVarDefault")(source):
        tvm_ffi.get_global_func("tl.MarkSizeVarDefault")(value)
    if spec is not None and (metadata := spec.contract_metadata(parameter=True)) is not None:
        from .eager.builder import Builder

        current = Builder.current()
        if current is None:
            _emit_annotation({"tl.tensor_spec": {value: metadata}})
        else:
            current._tensor_parameter_specs[value] = metadata
    return value


def _record_local_tensor_spec(value, spec):
    metadata = spec.contract_metadata()
    if metadata is not None:
        # This follows the allocation/view Bind. Hoisting its TensorVar key to
        # the function entry would create a use before the local definition.
        _emit_annotation({"tl.tensor_spec": {value: metadata}})


def attach_parameter_tensor_specs(function, specs):
    if not specs:
        return function
    schema, payload = encode_annotations({"node": None, "key": "tl.tensor_spec", "value": specs})
    body = ir.RegionStmt(
        "tl.attr",
        [payload],
        [],
        {"schema_version": 1, "annotation_schema": schema},
        function.body,
        loc=function.loc,
    )
    return function.with_body(body, loc=function.loc)


def attach_size_var_defaults(function):
    """Persist integer source defaults using native parameter/type identities."""
    return tvm_ffi.get_global_func("tl.AttachSizeVarDefaults")(function)


def deferred_arg():
    """Mark a source parameter whose native type is declared in the body.

    The marker is consumed by TileLang's builder before a Function is emitted.
    It is never an untyped parameter of the resulting compiler IR.
    """
    value = ir.Var("pending_parameter", ir.PtrType(ir.PrimType("void")))
    value._tilelang_deferred_parameter = True
    return value


def match_buffer(value, shape, dtype="float32", **kwargs):
    from .eager.builder import Builder

    # In a parameter match, explicit data links storage identities. Unlike a
    # local Tensor(data=...) declaration, it must not emit a view before the
    # final input tensor contract has been selected.
    data = kwargs.pop("data", None)
    spec = buffer(shape, dtype, **kwargs)
    if data is not None:
        spec.data = _expr(data)
    if isinstance(value, TensorSpec):
        raise TypeError("match_buffer requires a declared pointer or tensor parameter")
    current = Builder.current()
    if current is not None:
        parameter = current.match_parameter_buffer(value, spec)
        if parameter is not None:
            return parameter
    pointer = spec.data
    if pointer is None:
        pointer = _tirx.tensor_data_ptr(value) if _tirx.is_tensor_var(value) else value
    value = _unwrap(_builder.decl_tensor(pointer, ty_args=[spec.as_type()]))
    _record_local_tensor_spec(value, spec)
    return value


def sblock_alloc_buffer(shape, dtype="float32", data=None, **kwargs):
    spec = buffer(shape, dtype, **kwargs)
    if not isinstance(spec, TensorSpec):
        return spec
    if data is not None:
        value = _unwrap(_builder.decl_tensor(data, ty_args=[spec.as_type()]))
    else:
        # The public alloc_buffer/alloc_local helpers inherited SBlock-managed
        # storage placement. CPU lowering consumes this origin before grid
        # privacy analysis; a directly constructed native alloc_tensor keeps
        # its ordinary lexical lifetime.
        value = _unwrap(
            _builder.alloc_tensor(ty_args=[spec.as_type(allocation=True)], annotations={"tl.source_block_alloc": True}, loc=_loc())
        )
    _record_local_tensor_spec(value, spec)
    return value


alloc_buffer = sblock_alloc_buffer


class _BufferRegionMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.TensorRegion)


class BufferRegion(metaclass=_BufferRegionMeta):
    def __new__(cls, source, region, span=None):
        return ir.TensorRegion(source, region, ir.TensorRegionType(), span or _loc())


class _BufferLoadMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.TensorLoad)


class BufferLoad(metaclass=_BufferLoadMeta):
    def __new__(cls, source, indices, span=None):
        if not isinstance(indices, (tuple, list, tvm_ffi.Array)):
            indices = [indices]
        return ir.TensorLoad(source, [_expr(index) for index in indices], span or _loc())


def _region_as_load(region, location):
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


class _CastMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, _prim.Cast)


class Cast(metaclass=_CastMeta):
    """Construct a native cast, realizing a fixed tensor slice as a load."""

    def __new__(cls, dtype, value, span=None, *, loc=None):
        if span is not None and loc is not None:
            raise TypeError("Cast accepts only one of span and loc")
        location = loc if loc is not None else span if span is not None else _loc()
        value = _expr(value)
        if isinstance(value, ir.TensorRegion):
            value = _region_as_load(value, location)
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
        location = span or _loc()
        value = _expr(value)
        if isinstance(value, ir.TensorRegion):
            value = _region_as_load(value, location)
        return ir.TensorStore(buffer, [_expr(i) for i in indices], value, location)


def buffer_store(buffer, value, indices, predicate=None):
    value = _unwrap(value)
    if isinstance(value, ir.TensorRegion):
        value = _region_as_load(value, _loc())
    if predicate is None:
        return _builder.setitem_(buffer, indices, value, loc=_loc())
    with _builder.if_(predicate, loc=_loc()), _builder.then_(loc=_loc()):
        return _builder.setitem_(buffer, indices, value, loc=_loc())


def bind(value, type_annotation=None, *, var=None):
    """Emit one immutable binding and return its native variable.

    Explicit ``T.bind`` materializes its value even when the value is already
    a Var. Ordinary assignments keep the parser's alias policy. An optional
    ``var`` supplies the destination identity, as in the original builder API.
    """
    value = _expr(value)
    if var is not None:
        if not isinstance(var, ir.Var):
            raise TypeError("bind var must be an ir.Var")
        _builder.emit_(ir.Bind(var, value, _loc()), loc=_loc())
        return var
    annotation = value.ty if type_annotation is None else type_annotation
    return _unwrap(_builder.bind_(value, ty=annotation, loc=_loc()))


def let(v, value, body=None):
    """Preserve the callable source spelling for Bind and Let expressions."""
    from tilelang.utils.deprecated import deprecated_warning

    if body is None:
        deprecated_warning("T.let", "T.bind")
        return bind(value, var=v)
    deprecated_warning("T.let", "T.Let")
    return _prim.Let(v, _expr(value), _expr(body), _loc())


def side_effect(value):
    """Return target-recognized expression effects using the owning IR analysis."""
    return tvm_ffi.get_global_func("tl.SideEffect")(_unwrap(value))


def dtype_of(value):
    """Read primitive dtype from a native expression or a Python tensor."""
    value = _unwrap(value)
    if isinstance(value, ir.Expr):
        if isinstance(value.ty, (ir.PrimType, _tirx.TensorType)):
            ty = value.ty.dtype
            value = ty.dtype if isinstance(ty, ir.PrimType) else ty
            return value
        if isinstance(value.ty, ir.PtrType | ir.StringType):
            return tvm_ffi.dtype("handle")
        raise TypeError(f"Expression of type {value.ty} has no primitive dtype")
    return value.dtype


def isfinite(value, span=None, *, loc=None):
    """Keep floating finiteness visible for backend intrinsic selection.

    Native prim.isfinite expands during construction, before TileLang can
    select CUDA's isfinite instruction. The registered TileLang operation
    keeps exactly the same scalar/vector boolean type and has a portable
    lowering for targets without a dedicated intrinsic.
    """
    value = _expr(value)
    location = loc if loc is not None else span if span is not None else _loc()
    dtype = dtype_of(value)
    if dtype.type_code in (tvm_ffi.dtype("float32").type_code, tvm_ffi.dtype("bfloat16").type_code):
        return ir.Call("tl.isfinite", [value], loc=location)
    return _prim.isfinite(value, loc=location)


def encode_annotations(annotations):
    """Encode nested source metadata with every IR use in a native Tuple operand."""
    values = []

    def slot(value):
        values.append(value)
        return len(values) - 1

    def encode(value):
        value = _unwrap(value)
        if isinstance(value, ir.Expr):
            return ["expr", slot(value)]
        if value is None:
            return ["none"]
        if isinstance(value, builtins.bool):
            return ["bool", value]
        if isinstance(value, builtins.int):
            return ["int", value]
        if isinstance(value, builtins.float):
            return ["float", value]
        if isinstance(value, str):
            return ["str", value]
        if isinstance(value, dict | tvm_ffi.Map):
            return ["map", *[[encode(key), encode(item)] for key, item in value.items()]]
        if isinstance(value, tuple):
            return ["tuple", *map(encode, value)]
        if isinstance(value, list | tvm_ffi.Array):
            return ["list", *map(encode, value)]
        # Layout is a TL inference object. Its internal variables must be visible
        # to the shared IR visitor, serializer, and binder-aware rewrites.
        from tilelang.layout import Layout

        if isinstance(value, Layout):
            kind = tvm_ffi.get_global_func("tl.LayoutPayloadKind")(value)
            payload = tvm_ffi.get_global_func("tl.EncodeLayoutPayload")(value)
            return ["layout", kind, 1, slot(payload)]
        from .warp_specialize import (
            WSRole,
            WSPipeline,
            WSOpRef,
            WSSync,
            WSSyncKind,
            WSScope,
            WSSchedule,
        )

        if isinstance(value, WSSyncKind):
            return ["ws_sync_kind", value._str_index, int(value._int_index)]
        for cls, tag, fields in (
            (WSRole, "ws_role", ("name", "warp_lo", "warp_hi", "max_nreg")),
            (WSPipeline, "ws_pipeline", ("name", "buffers", "depth")),
            (WSOpRef, "ws_op_ref", ("id",)),
            (WSSync, "ws_sync", ("kind", "pipeline", "stage")),
            (WSScope, "ws_scope", ("id", "bodies")),
            (WSSchedule, "ws_schedule", ("num_warps", "roles", "pipelines", "scopes")),
        ):
            if isinstance(value, cls):
                return [tag, encode({name: getattr(value, name) for name in fields})]
        raise TypeError(f"Unsupported TileLang annotation value: {type(value).__name__}")

    return encode(annotations or {}), ir.Tuple(values)


def call_intrin(dtype, func_name, *args, annotations=None, span=None):
    """Construct a typed call; tile op effects and validation are registered natively."""
    if isinstance(func_name, ir.Op):
        name = func_name.name
    else:
        name = func_name
    operands = [_expr(value) for value in args]
    if name.startswith("tl.tileop."):
        schema, payload = encode_annotations(annotations)
        attrs = {"schema_version": 1, "operand_count": len(operands), "annotation_schema": schema}
        return ir.Call(name, [*operands, payload], attrs=attrs, loc=span or _loc())
    if annotations:
        schema, payload = encode_annotations(annotations)
        attrs = {"schema_version": 1, "operand_count": len(operands), "annotation_schema": schema}
        operands.append(payload)
    else:
        attrs = None
    ty = ir.PtrType(ir.PrimType("void")) if str(dtype) == "handle" else ir.PrimType(dtype)
    op = func_name if isinstance(func_name, ir.Op) else ir.Op.get(name)
    # A return annotation is a type argument only when the registered native
    # signature accepts one. External/LLVM calls require an explicit Call.ty
    # and zero type arguments; fixed-result intrinsics keep their own result.
    type_args = [ty] if op.ty_args_info or op.var_ty_args_info is not None else []
    value = ir.Call(op, operands, attrs=attrs, ty_args=type_args, loc=span or _loc())
    if isinstance(value.ty, ir.MissingType):
        # External calls and other operators without a registered result rule
        # need the public dtype as Call.ty, not merely as a type argument.
        # Retain inferred native types when the operator owns that contract.
        value = ir.Call(op, operands, attrs=attrs, ty_args=type_args, ty=ty, loc=span or _loc())
    return value


def tensor_descriptor_ptr(tensor, *, loc=None):
    """Borrow a matched input's original descriptor through TileLang's binder.

    This frontend operation is resolved before native layout/flattening passes.
    Ordinary pointer parameters and ``decl_tensor`` views do not need it.
    """
    return ir.Call("tl.tensor_descriptor_ptr", [_expr(tensor)], loc=loc or _loc())


def tvm_struct_get(arr, index, field, dtype):
    """Read a runtime ABI field using the existing public field numbering.

    The DLTensor and TVMFFIAny field indices are unchanged. A source handle
    matched to a tensor still denotes its original input descriptor; the
    TileLang binds that handle at function entry and resolves it in its own ABI.
    """
    from .dtypes import get_tvm_dtype

    ty = ir.PtrType(ir.PrimType("void")) if str(dtype) == "handle" else ir.PrimType(get_tvm_dtype(dtype))
    return _tirx.op.abi_field_get(_expr(arr), _expr(index), _expr(field), ty=ty, loc=_loc())


def tvm_struct_set(arr, index, field, value):
    """Write a runtime ABI field and retain the native update-state effect."""
    return _tirx.op.abi_field_set(_expr(arr), _expr(index), _expr(field), _expr(value), loc=_loc())


def tvm_storage_sync(storage_scope, *barrier_values):
    """Keep the public barrier spelling while constructing its native contract.

    Whole-block barriers belong to TIRx. Named barriers retain their ID and
    participant count in TileLang's operation through synchronization planning.
    CUDA consumes them as named barriers; the final WebGPU codegen boundary
    preserves its established whole-workgroup fallback.
    """
    if barrier_values:
        return ir.Call(
            "tl.storage_sync",
            [_expr(storage_scope), *[_expr(value) for value in barrier_values]],
            loc=_loc(),
        )
    return _tirx.op.gpu_storage_sync(storage_scope, loc=_loc())


def call_extern(dtype, func_name, *args, span=None):
    """Preserve TileLang's dtype-first external-call spelling on typed native calls."""
    ty = ir.PtrType(ir.PrimType("void")) if str(dtype) == "handle" else ir.PrimType(dtype)
    return _tirx.call_extern(func_name, *[_expr(value) for value in args], ty=ty, loc=span or _loc())


def call_pure_extern(dtype, func_name, *args, span=None):
    """Construct a native external call with an explicitly pure effect contract."""
    ty = ir.PtrType(ir.PrimType("void")) if str(dtype) == "handle" else ir.PrimType(dtype)
    return _tirx.call_pure_extern(func_name, *[_expr(value) for value in args], ty=ty, loc=span or _loc())


def _emit_annotation(annotations):
    """Record native metadata with every referenced variable in an IR operand.

    TileLang consumes these statements before generic simplification.  The
    native operation is opaque until its owning pass has handled the metadata.
    """
    schema, payload = encode_annotations(annotations)
    value = ir.Call(
        "tl.annotation",
        [payload],
        attrs={"schema_version": 1, "annotation_schema": schema},
        loc=_loc(),
    )
    return _builder.evaluate(value)


def _source_annotation_owner(*, required=True):
    """Find the innermost active source SBlock, including Kernel's block."""
    from .kernel import KernelLaunchFrame

    kernel = KernelLaunchFrame.Current()
    blocks = _sblock_stack.get()
    # BlockFrame uses a nested IRBuilder. Checking active native frames avoids
    # confusing an outer kernel with a block in that nested builder, and also
    # handles a kernel entered inside a source block.
    if IRBuilder.is_in_scope():
        for frame in reversed(IRBuilder.current().frames):
            if kernel is not None and frame.same_as(kernel._frame):
                return kernel
            for block in reversed(blocks):
                if frame.same_as(block._frame):
                    return block
    if not required:
        return None
    raise ValueError("T.sblock_attr must be called at the top of a T.sblock(), but it occurred outside of any T.sblock() frame")


def _source_annotation_values(value):
    value = _unwrap(value)
    if isinstance(value, dict | tvm_ffi.Map):
        return {_source_annotation_values(key): _source_annotation_values(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_source_annotation_values(item) for item in value]
    return value


def _source_sblock_attr(annotations, source_annotations=None):
    """Check the source declaration while emitting native TensorVar operands.

    A helper may supply data-identity keys for its source comparison. Emission
    continues to use tensor identities so definition/alias rewriting remains
    native. The short-lived owner state never enters serialized compiler IR.
    """
    owner = _source_annotation_owner()
    previous = getattr(owner, "_source_annotations", None)
    if previous is None:
        previous = _source_annotation_values(getattr(owner, "annotations", {}))
    declarations = _source_annotation_values(annotations if source_annotations is None else source_annotations)
    owner._source_annotations = tvm_ffi.get_global_func("tl.MergeSourceAnnotations")(declarations, previous)
    # The payload stays at its definition-safe source position. Consumers
    # collect marked declarations for this entire nearest source owner; they
    # do not give ordinary native metadata this whole-block interpretation.
    return _emit_annotation({**annotations, "tl.source_block_annotation": True})


def sblock_attr(annotations):
    """Recursively merge annotations on the current source block.

    Conflicting leaves raise ValueError during construction, using FFI value
    and object-identity equality. Loops and conditionals do not create a new
    annotation owner; an explicit SBlock or Kernel does.
    """
    return _source_sblock_attr(annotations)


def _allocation_annotation(annotations):
    """Attach internal metadata to a just-emitted allocation.

    Reducer and barrier helpers also work at function scope, where the old
    implicit allocation owner is not an explicit source SBlock. Keep their
    metadata next to the allocation as ordinary native operands. Public block
    annotation helpers continue to require a real SBlock or Kernel, and an
    allocation inside one still participates in that owner's merge checks.
    """
    if _source_annotation_owner(required=False) is None:
        return _emit_annotation(annotations)
    return _source_sblock_attr(annotations)


@dataclass(frozen=True)
class _ThreadReducer:
    """Construction policy carried explicitly by the native allreduce call."""

    combine: ir.LambdaExpr
    identity: tuple[ir.Expr, ...]


_thread_reducer_scope = ContextVar("tilelang_thread_reducer_scope", default=None)


class _ThreadReducerScope:
    def __init__(self, reducer):
        self.reducer = reducer
        self.token = None

    def __enter__(self):
        if self.token is not None:
            raise RuntimeError("A reduce_scope cannot be entered twice")
        self.token = _thread_reducer_scope.set((IRBuilder.current(), self.reducer))

    def __exit__(self, exc_type, exc_value, traceback):
        _thread_reducer_scope.reset(self.token)
        self.token = None


def comm_reducer(combiner, identity):
    """Keep the TileLang reducer spelling on a typed native combining lambda.

    The callable receives all left-hand values followed by all right-hand values.
    Its result must have one element of the matching type per identity. This is
    construction metadata; the resulting IR uses ``tirx.gpu_thread_allreduce``
    with explicit combiner, identity, values, predicate, outputs and thread axes.
    """
    if not isinstance(identity, (list, tuple, tvm_ffi.Array)) or not identity:
        raise ValueError("comm_reducer requires a non-empty list of identity values")
    values = tuple(_expr(value) for value in identity)
    types = [value.ty for value in values]
    if not all(isinstance(ty, ir.PrimType) for ty in types):
        raise TypeError("comm_reducer identity values must have primitive types")

    def combine(*args):
        result = _expr(combiner(*args))
        return result if isinstance(result, ir.Tuple) else ir.Tuple([result])

    function = ir.LambdaExpr(types + types, combine, ret_type=ir.TupleType(types), loc=_loc())
    return _ThreadReducer(function, values)


def tvm_thread_allreduce(*args):
    """Bind the legacy argument order to the current typed reduction scope."""
    active = _thread_reducer_scope.get()
    if active is None or not active[0].same_as(IRBuilder.current()):
        raise ValueError("tvm_thread_allreduce requires an enclosing reduce_scope in the same function")
    if not args:
        raise ValueError("tvm_thread_allreduce requires a constant result count")
    count = _expr(args[0])
    if not isinstance(count, ir.IntImm) or int(count) <= 0:
        raise ValueError("tvm_thread_allreduce requires a positive constant result count")
    count = int(count)
    reducer = active[1]
    if count != len(reducer.identity):
        raise ValueError("tvm_thread_allreduce result count does not match comm_reducer identities")
    # count, N values, predicate, N destinations, one or more thread axes.
    if len(args) < 2 * count + 3:
        raise ValueError("tvm_thread_allreduce requires values, predicate, destinations and thread axes")
    return _tirx.gpu_thread_allreduce(
        reducer.combine,
        reducer.identity,
        tuple(_expr(value) for value in args[1 : 1 + count]),
        _expr(args[1 + count]),
        tuple(_expr(value) for value in args[2 + count : 2 + 2 * count]),
        tuple(_expr(value) for value in args[2 + 2 * count :]),
        loc=_loc(),
    )


def attr(node, attr_key, value):
    """Scope an annotation over a body using a native zero-result region."""
    if attr_key == "reduce_scope":
        if not isinstance(node, _ThreadReducer):
            raise TypeError("reduce_scope requires a comm_reducer")
        return _ThreadReducerScope(node)
    schema, payload = encode_annotations({"node": node, "key": attr_key, "value": value})
    return _builder.region(
        "tl.attr",
        [payload],
        attrs={"schema_version": 1, "annotation_schema": schema},
    )


def _has_expr_annotation(value):
    if isinstance(value, ir.Expr):
        return True
    if isinstance(value, dict | tvm_ffi.Map):
        return any(_has_expr_annotation(key) or _has_expr_annotation(item) for key, item in value.items())
    if isinstance(value, list | tuple | tvm_ffi.Array):
        return any(_has_expr_annotation(item) for item in value)
    return not isinstance(value, (str, builtins.int, builtins.float, builtins.bool, type(None)))


def _normalize_pipeline_annotations(annotations):
    """Store compile-time scheduling integers as native host attributes.

    Source programs commonly spell these integers with ``T.int32``. Their
    type is irrelevant to the scheduling hint, and putting them in an operand
    region hides the stage/order fields from ordinary loop consumers.
    """

    def integer(value):
        if isinstance(value, ir.IntImm):
            return builtins.int(value.value)
        if isinstance(value, builtins.int):
            return value
        if isinstance(value, (tuple, list, tvm_ffi.Array)):
            return [integer(item) for item in value]
        raise ValueError("Pipeline annotations require compile-time integers")

    return {
        key: integer(value)
        if key.startswith("software_pipeline_")
        or key in ("num_stages", "tl_pipelined_num_stages", "tl_pipeline_stage", "tl_pipeline_order")
        else value
        for key, value in (annotations or {}).items()
    }


class LoopFrame:
    """Own a loop nest and optional typed metadata in the native frame stack."""

    def __init__(self, frames, annotations=None):
        self.frames = list(frames)
        self.vars = [var for current in self.frames for var in getattr(current, "vars", getattr(current, "body_params", ()))]
        self.annotations = annotations
        self._stack = None

    @property
    def stmts(self):
        return self.frames[-1].stmts

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.__enter__()
        try:
            if self.annotations:
                self._stack.enter_context(attr(None, "tl.loop.annotations", self.annotations))
            for current in self.frames:
                self._stack.enter_context(current)
            return self.vars[0] if len(self.vars) == 1 else self.vars
        except BaseException:
            import sys

            self._stack.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


class PersistentLoopFrame(LoopFrame):
    """Expand grouped persistent work into native loops, binds and tail guards.

    Coordinate binds are inside the wave loop, where their induction variable
    is defined. The public coordinates keep their domain dtypes; the generated
    condition guards both an incomplete group and an incomplete worker wave.
    """

    def __init__(self, domain, wave_size, index, group_size=8, num_stages=0, annotations=None):
        if not domain:
            raise ValueError("T.Persistent requires a nonempty domain")
        self.domain = [_expr(extent) for extent in domain]
        self.wave_size = _expr(wave_size)
        self.index = _expr(index)
        if isinstance(self.wave_size, ir.IntImm) and self.wave_size.value <= 0:
            raise ValueError("T.Persistent wave_size must be positive")
        if any(isinstance(extent, ir.IntImm) and extent.value < 0 for extent in self.domain):
            raise ValueError("T.Persistent domain extents must be nonnegative")
        last = self.domain[-1]
        self.group_size = _prim.max(1, _prim.min(_expr(8 if group_size is None else group_size), last))
        self.grouped_domain = [_prim.ceildiv(last, self.group_size), *self.domain[:-1], self.group_size]
        self.padded_size = self.grouped_domain[0]
        for extent in self.grouped_domain[1:]:
            self.padded_size = self.padded_size * extent
        waves = _prim.ceildiv(self.padded_size, self.wave_size)
        annotations = dict(annotations or {})
        if num_stages > 0:
            annotations["num_stages"] = num_stages
        annotations = _normalize_pipeline_annotations(annotations)
        dynamic = _has_expr_annotation(annotations)
        current = _builder.serial(waves, annotations={} if dynamic else annotations)
        super().__init__([current], annotations if dynamic else None)

    def __enter__(self):
        wave = super().__enter__()
        try:
            linear = wave * self.wave_size + self.index
            remainder = linear
            indices = [None] * len(self.grouped_domain)
            for axis in range(len(self.grouped_domain) - 1, 0, -1):
                # Empty dimensions produce zero waves. Keep the unused decode
                # well-formed as well, instead of constructing division by zero.
                extent = _prim.max(1, self.grouped_domain[axis])
                indices[axis] = _prim.truncmod(remainder, extent)
                remainder = _prim.truncdiv(remainder, extent)
            last = remainder * self.group_size + indices[-1]
            coordinates = [*indices[1:-1], last]
            self.vars = [bind(_prim.Cast(extent.ty.dtype, value)) for extent, value in zip(self.domain, coordinates)]
            condition = _prim.And(linear < self.padded_size, last < self.domain[-1])
            for current in (_builder.if_(condition, loc=_loc()), _builder.then_(loc=_loc())):
                self._stack.enter_context(current)
                self.frames.append(current)
            return self.vars[0] if len(self.vars) == 1 else self.vars
        except BaseException:
            import sys

            self._stack.__exit__(*sys.exc_info())
            raise


class ThreadBindingFrame(LoopFrame):
    """Bind a source loop index to an actual target thread region parameter."""

    def __init__(self, begin, end, thread, annotations=None, variable=None):
        self.begin, self.end = _expr(begin), _expr(end)
        self.annotations = annotations
        self.thread = str(thread)
        current = _builder.launch_thread(
            self.thread,
            self.end - self.begin,
            body_params=None if variable is None else [variable],
        )
        self.frames = [current]
        self.vars = list(current.body_params)
        self.doms = [ir.Range(self.begin, self.end)]
        self._stack = None

    def __enter__(self):
        value = super().__enter__()
        if isinstance(self.begin, ir.IntImm) and self.begin.value == 0:
            return value
        return bind(value + self.begin)


def thread_binding(start, stop=None, thread="", *, annotations=None):
    """Retain ``for``/concise thread syntax using ``tirx.launch_thread``."""
    if isinstance(stop, str) and not thread:
        thread, stop = stop, None
    if stop is None:
        start, stop = 0, start
    if not thread:
        raise ValueError("A thread binding requires a nonempty thread tag")
    return ThreadBindingFrame(start, stop, thread, annotations)


def env_thread(thread_tag, dtype="int32"):
    value = ir.Var(thread_tag.replace(".", "_"), ir.PrimType(dtype))
    value._tilelang_thread_tag = str(thread_tag)
    return value


def launch_thread(thread, extent):
    if isinstance(thread, str):
        return ThreadBindingFrame(0, extent, thread)
    tag = getattr(thread, "_tilelang_thread_tag", None)
    if tag is None:
        raise TypeError("launch_thread expects a thread tag or T.env_thread variable")
    return ThreadBindingFrame(0, extent, tag, variable=thread)


_sblock_stack = ContextVar("tilelang_sblock_stack", default=())


class BlockFrame:
    """Collect a lexical block so its predicate guards the complete body.

    A nested *native* builder keeps the finished body available until ``where``
    and axis metadata have been resolved. The emitted IR contains ordinary
    Bind, If and tl.attr nodes, with no legacy SBlock objects.
    """

    def __init__(self, name="", no_realize=False):
        self.name = str(name)
        self.no_realize = no_realize
        self.axes = []
        self.predicates = []
        self._builder = IRBuilder()
        self._frame = None
        self._token = None

    @property
    def stmts(self):
        return self._frame.stmts if self._frame is not None else []

    def __enter__(self):
        self._builder.__enter__()
        try:
            self._frame = attr(None, "tl.sblock", {"name": self.name, "no_realize": self.no_realize})
            self._frame.__enter__()
        except BaseException:
            import sys

            self._builder.__exit__(*sys.exc_info())
            raise
        self._token = _sblock_stack.set((*_sblock_stack.get(), self))
        return None

    def __exit__(self, exc_type, exc, trace):
        try:
            self._frame.__exit__(exc_type, exc, trace)
        finally:
            self._builder.__exit__(exc_type, exc, trace)
            _sblock_stack.reset(self._token)
        if exc_type is None:
            body = self._builder.get()
            # Block predicates belong to the enclosing scope. Axis aliases are
            # explicit Binds inside the block and must be replaced here.
            replacements = {var: binding for _, var, _, binding in self.axes}
            predicates = [
                tvm_ffi.structural_map(predicate, (ir.Var, lambda var: replacements.get(var, var))) for predicate in self.predicates
            ]
            if predicates:
                condition = predicates[0]
                for predicate in predicates[1:]:
                    condition = _prim.And(condition, predicate)
                body = ir.If(condition, body, None, _loc())
            _builder.add_to_parent(body)


def sblock(name="", no_realize=False):
    return BlockFrame(name, no_realize)


block = sblock


def _current_sblock():
    stack = _sblock_stack.get()
    if not stack:
        raise ValueError("This operation must appear inside T.sblock")
    return stack[-1]


def where(predicate):
    _current_sblock().predicates.append(_expr(predicate))


def _region(value):
    if isinstance(value, ir.TensorRegion):
        return value
    if isinstance(value, ir.TensorLoad):
        return BufferRegion(value.source, [Range.from_min_extent(index, 1) for index in value.indices])
    if _tirx.is_tensor_var(value):
        return BufferRegion(value, [Range.from_min_extent(0, extent) for extent in value.ty.shape])
    raise TypeError("Read/write regions require a tensor or a tensor subscript")


def reads(*regions):
    return sblock_attr({"tl.reads": [_region(value) for value in regions]})


def writes(*regions):
    return sblock_attr({"tl.writes": [_region(value) for value in regions]})


class InitFrame:
    """Execute a reduction initializer only on the first reduction coordinates."""

    def __init__(self):
        owner = _current_sblock()
        conditions = [var == domain.begin for kind, var, domain, _ in owner.axes if kind == "R"]
        self.condition = _expr(True)
        for condition in conditions:
            self.condition = _prim.And(self.condition, _expr(condition))
        self._stack = None

    @property
    def stmts(self):
        return self._frame.stmts

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.__enter__()
        try:
            self._stack.enter_context(_builder.if_(self.condition, loc=_loc()))
            self._stack.enter_context(_builder.then_(loc=_loc()))
            self._frame = attr(None, "tl.sblock.init", {})
            self._stack.enter_context(self._frame)
        except BaseException:
            import sys

            self._stack.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


def init():
    return InitFrame()


def _axis_domain(domain):
    if isinstance(domain, ir.Range):
        return domain
    if isinstance(domain, tuple | list | tvm_ffi.Array):
        if len(domain) != 2:
            raise ValueError("An axis domain must have begin/end bounds")
        return ir.Range(*domain)
    return ir.Range(0, domain)


class axis:
    """Block-axis annotations carried by actual native scalar bindings."""

    @staticmethod
    def _bind(kind, domain, binding, dtype="int32"):
        owner = _current_sblock()
        domain = _axis_domain(domain)
        value = _expr(binding)
        var = ir.Var("axis", ir.PrimType(dtype), _loc())
        _builder.add_to_parent(ir.Bind(var, value, _loc()))
        owner.axes.append((kind, var, domain, value))
        _emit_annotation({"tl.axis": {"kind": kind, "var": var, "begin": domain.begin, "extent": domain.extent}})
        return var

    @staticmethod
    def spatial(domain, binding, dtype="int32"):
        return axis._bind("S", domain, binding, dtype)

    @staticmethod
    def reduce(domain, binding, dtype="int32"):
        return axis._bind("R", domain, binding, dtype)

    @staticmethod
    def scan(domain, binding, dtype="int32"):
        return axis._bind("scan", domain, binding, dtype)

    @staticmethod
    def opaque(domain, binding, dtype="int32"):
        return axis._bind("opaque", domain, binding, dtype)

    @staticmethod
    def remap(kinds, bindings, dtype="int32"):
        from .eager.builder import Builder

        if len(kinds) != len(bindings) or any(kind not in "SR" for kind in kinds):
            raise ValueError("axis.remap requires one S/R kind per loop variable")
        builder = Builder.current()
        if builder is None:
            raise ValueError("axis.remap requires an active TileLang loop builder")
        domains = {}
        for current in builder.frames:
            frames = current.frames if isinstance(current, LoopFrame) else [current]
            for frame in frames:
                if isinstance(frame, _shared_frame.ForFrame):
                    domains.update(zip(frame.vars, frame.doms))
        result = []
        for kind, value in zip(kinds, bindings):
            if value not in domains:
                raise ValueError("axis.remap must refer to enclosing loop variables")
            result.append(axis._bind(kind, domains[value], value, dtype))
        return result[0] if len(result) == 1 else result

    S = spatial
    R = reduce


def _loop(constructor, start, stop=None, *, step=None, annotations=None):
    if not IRBuilder.is_in_scope():
        from tilelang.jit.exceptions import JITNoBuilderError

        raise JITNoBuilderError("TileLang loops require an active @tilelang.jit, @T.prim_func, or IRBuilder context.")
    annotations = _normalize_pipeline_annotations(annotations)
    dynamic = _has_expr_annotation(annotations)
    frame = constructor(start, stop, step=step, annotations={} if dynamic else annotations)
    return LoopFrame([frame], annotations) if dynamic else frame


def parallel_nest(*extents, annotations=None):
    if not extents:
        raise ValueError("T.Parallel requires at least one extent")
    annotations = _normalize_pipeline_annotations(annotations)
    dynamic = _has_expr_annotation(annotations)
    frames = [
        _builder.parallel(extent, annotations=annotations if index == 0 and not dynamic else {}) for index, extent in enumerate(extents)
    ]
    return LoopFrame(frames, annotations if dynamic else None)


class _CallMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, ir.Call)


class Call(metaclass=_CallMeta):
    def __new__(cls, dtype, op, args, annotations=None, span=None):
        return call_intrin(dtype, op, *args, annotations=annotations, span=span)


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


def tvm_access_ptr(ptype, data, offset, extent, rw_mask):
    """Preserve a raw pointer access interval using the native pointer contract.

    ``offset`` and ``extent`` count elements of ``ptype``, including when an
    explicit dtype differs from the input pointer's element type. The access
    keeps the input storage scope and never guesses a tensor's logical layout
    from its raw pointer. Tensor-coordinate accesses use ``T.access_ptr``.
    """
    data = _expr(data)
    if not isinstance(data.ty, ir.PtrType):
        raise TypeError("tvm_access_ptr requires a typed pointer expression")
    if not isinstance(ptype, ir.DataTypeImm):
        raise TypeError("tvm_access_ptr ptype must be T.type_annotation(dtype)")
    element_type = ir.PrimType(ptype.value)
    pointer_type = ir.PtrType(element_type, data.ty.scope)
    if not tvm_ffi.structural_equal(data.ty, pointer_type):
        data = _tirx.reinterpret(data, ty=pointer_type, loc=_loc())
    mask = _expr(rw_mask)
    if not isinstance(mask, ir.IntImm) or not 0 <= int(mask.value) <= 3:
        raise ValueError("tvm_access_ptr rw_mask must be a constant in [0, 3]")
    result = ir.Call("tl.ptr_access", [data, _expr(offset), _expr(extent), mask], loc=_loc())
    result.validate()
    return result


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
    frame = tvm_ffi.get_global_func("script.ir_builder.Assert")(_expr(condition), kind, parts)
    return at(_loc(), frame)


def break_loop():
    return _builder.break_(loc=_loc())


def continue_loop():
    return _builder.continue_(loc=_loc())


def thread_return():
    """Return from the calling GPU thread using the native control-flow op."""
    return _tirx.op.gpu_thread_return(loc=_loc())


func_name = _builder.func_name_
func_attr = _builder.func_attr
func_ret = _builder.func_ret_type_


def serial(start, stop=None, *, step=None, annotations=None):
    return _loop(_builder.serial, start, stop, step=step, annotations=annotations)


def parallel(start, stop=None, *, step=None, annotations=None):
    return _loop(_builder.parallel, start, stop, step=step, annotations=annotations)


def unroll(start, stop=None, *, step=None, annotations=None):
    return _loop(_builder.unroll, start, stop, step=step, annotations=annotations)


def vectorized(start, stop=None, *, step=None, annotations=None):
    return _loop(_builder.vectorized, start, stop, step=step, annotations=annotations)


grid = _builder.grid
If = _builder.if_
Then = _builder.then_
Else = _builder.else_
While = _builder.while_
frame = SimpleNamespace(
    IRBuilderFrame=IRBuilderFrame,
    TIRFrame=_shared_frame.StmtFrame,
    StmtFrame=_shared_frame.StmtFrame,
    PrimFuncFrame=FunctionFrame,
    FunctionFrame=FunctionFrame,
    ForFrame=_shared_frame.ForFrame,
    WhileFrame=_shared_frame.WhileFrame,
    IfFrame=_shared_frame.IfFrame,
    RegionFrame=_shared_frame.RegionFrame,
)


def __getattr__(name):
    # Only resolve capabilities that really exist in the target.  Unknown names
    # are not synthesized as arbitrary Call objects.
    for namespace in (_prim, ir, _tirx, _builder):
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise AttributeError(f"The native TileLang construction boundary has no {name!r}")


__all__ = tuple(
    dict.fromkeys(
        [
            *_builder.__all__,
            "Var",
            "IntImm",
            "FloatImm",
            "StringImm",
            "Range",
            "Call",
            "CallEffectKind",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Mod",
            "FloorDiv",
            "FloorMod",
            "Min",
            "Max",
            "EQ",
            "NE",
            "LT",
            "LE",
            "GT",
            "GE",
            "And",
            "Or",
            "Not",
            "Select",
            "Cast",
            "Ramp",
            "Broadcast",
            "Shuffle",
            "IndexMap",
            "TensorType",
            "PrimExpr",
            "PrimFunc",
            "Buffer",
            "BufferLoad",
            "BufferRegion",
            "BufferStore",
            "TensorSpec",
            "buffer",
            "decl_buffer",
            "match_buffer",
            "sblock_alloc_buffer",
            "alloc_buffer",
            "arg",
            "bind",
            "buffer_store",
            "call_intrin",
            "tvm_struct_get",
            "tvm_struct_set",
            "handle",
            "sblock_attr",
            "attr",
            "comm_reducer",
            "sblock",
            "block",
            "reads",
            "writes",
            "where",
            "init",
            "axis",
            "thread_binding",
            "launch_thread",
            "env_thread",
            "type_annotation",
            "func_name",
            "func_attr",
            "func_ret",
            "If",
            "Then",
            "Else",
            "While",
            "Assert",
            "break_loop",
            "continue_loop",
            "thread_return",
            "meta_var",
        ]
    )
)
