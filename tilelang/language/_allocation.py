"""Source allocation and pointer-view declarations with tensor constraints."""

from __future__ import annotations

from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as _builder

from ._construction import as_expr, bind, current_location, unwrap
from ._tensor_spec import TensorSpec, buffer, _record_local_tensor_spec


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
        value = unwrap(_builder.alloc_tensor(ty_args=[spec.as_type(allocation=True)], loc=current_location()))
    else:
        spec.data = as_expr(data)
        value = bind(_builder.decl_tensor(spec.data, ty_args=[spec.as_type()], loc=current_location()))
    _record_local_tensor_spec(value, spec)
    return value


def sblock_alloc_buffer(shape, dtype="float32", data=None, **kwargs):
    spec = buffer(shape, dtype, **kwargs)
    if not isinstance(spec, TensorSpec):
        return spec
    if data is not None:
        value = unwrap(_builder.decl_tensor(data, ty_args=[spec.as_type()]))
    else:
        # The public alloc_buffer/alloc_local helpers inherited SBlock-managed
        # storage placement. CPU lowering consumes this origin before grid
        # privacy analysis; a directly constructed native alloc_tensor keeps
        # its ordinary lexical lifetime.
        value = unwrap(
            _builder.alloc_tensor(
                ty_args=[spec.as_type(allocation=True)], annotations={"tl.source_block_alloc": True}, loc=current_location()
            )
        )
    _record_local_tensor_spec(value, spec)
    return value


alloc_buffer = sblock_alloc_buffer
