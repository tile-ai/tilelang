"""Source tensor declarations and their parameter, stride, and alias contracts."""

from __future__ import annotations

from dataclasses import dataclass

import tvm_ffi
from tvm_tirx import ir
from tvm_tirx import tirx as _tirx
from tvm_tirx.ir import Var
from tvm_tirx.tirx import TensorType
from tvm_tirx.tirx.layout import Axis, Iter
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.script.ir_builder import frame as _shared_frame
from tvm_tirx.tirx.script import ir_builder as _builder

from ._construction import as_expr, bind, current_location, is_scalar_expr, unwrap
from ._annotation_codec import emit_annotation, encode_annotations


def is_buffer_like(value):
    return isinstance(value, TensorSpec) or _tirx.is_tensor_var(value)


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

    if isinstance(shape, int) or is_scalar_expr(shape):
        shape = (shape,)
    shape = tuple(as_expr(dim) for dim in shape)
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
        strides = tuple(Var(value, index_dtype) if isinstance(value, str) else as_expr(value) for value in strides)
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
        as_expr(elem_offset),
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
        value = bind(_builder.decl_tensor(data, ty_args=[result.as_type()], loc=current_location()))
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
    value = _builder.decl_tensor(as_expr(data), ty_args=normalized, attrs=attrs, loc=current_location() if loc is None else loc, ty=ty)
    if len(annotations) == 1 and isinstance(annotations[0], TensorSpec):
        spec = annotations[0]
        if spec.contract_metadata() is not None:
            if not IRBuilder.is_in_scope():
                raise ValueError("TensorSpec source constraints on decl_tensor require an active TileLang builder")
            value = bind(value)
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
    value = _builder.arg_(name, annotation, loc=current_location())
    if isinstance(source, ir.Var) and tvm_ffi.get_global_func("tl.HasSizeVarDefault")(source):
        tvm_ffi.get_global_func("tl.MarkSizeVarDefault")(value)
    if spec is not None and (metadata := spec.contract_metadata(parameter=True)) is not None:
        from .eager.builder import Builder

        current = Builder.current()
        if current is None:
            emit_annotation({"tl.tensor_spec": {value: metadata}})
        else:
            current._tensor_parameter_specs[value] = metadata
    return value


def _record_local_tensor_spec(value, spec):
    metadata = spec.contract_metadata()
    if metadata is not None:
        # This follows the allocation/view Bind. Hoisting its TensorVar key to
        # the function entry would create a use before the local definition.
        emit_annotation({"tl.tensor_spec": {value: metadata}})


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
        spec.data = as_expr(data)
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
    value = unwrap(_builder.decl_tensor(pointer, ty_args=[spec.as_type()]))
    _record_local_tensor_spec(value, spec)
    return value


# Native tensor values are Vars carrying TensorType; source annotations are specs.
TensorValue = TensorSpec | ir.Var
