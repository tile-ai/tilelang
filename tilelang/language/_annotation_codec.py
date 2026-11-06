"""Emit typed layout declarations and operand-based source metadata."""

from __future__ import annotations

import builtins

import tvm_ffi
from tvm_tirx import ir
from tvm_tirx.tirx.script import ir_builder as _builder

from ._construction import current_location, unwrap


def encode_annotations(annotations):
    """Encode nested source metadata with every IR use in a native Tuple operand."""
    values = []

    def slot(value):
        values.append(value)
        return len(values) - 1

    def encode(value):
        value = unwrap(value)
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


def is_static_annotation(value):
    value = unwrap(value)
    if value is None or isinstance(value, (builtins.bool, builtins.int, builtins.float, str, ir.IntImm, ir.FloatImm, ir.StringImm)):
        return True
    if isinstance(value, dict | tvm_ffi.Map):
        return all(is_static_annotation(key) and is_static_annotation(item) for key, item in value.items())
    if isinstance(value, list | tuple | tvm_ffi.Array):
        return all(is_static_annotation(item) for item in value)
    return False


def emit_annotation(annotations):
    """Emit layouts as typed attrs and other metadata as explicit operands.

    Source declarations keep their original position and owner marker. Mixed
    maps become adjacent statements; both retain the same source-owner meaning.
    The operations stay opaque until their owning passes consume them.
    """
    if "layout_map" in annotations:
        remaining = dict(annotations)
        layouts = remaining.pop("layout_map")
        attrs = {"layout_map": {unwrap(tensor): unwrap(layout) for tensor, layout in layouts.items()}}
        if "tl.source_block_annotation" in remaining:
            attrs["tl.source_block_annotation"] = remaining["tl.source_block_annotation"]
        receipt = _builder.evaluate(ir.Call("tl.layout_map", [], attrs=attrs, loc=current_location()))
        if not remaining or set(remaining) == {"tl.source_block_annotation"}:
            return receipt
        annotations = remaining
    schema, payload = encode_annotations(annotations)
    value = ir.Call(
        "tl.annotation",
        [payload],
        attrs={"schema_version": 1, "annotation_schema": schema},
        loc=current_location(),
    )
    return _builder.evaluate(value)


def attr(node, attr_key, value):
    """Scope an annotation over a body using a native zero-result region."""
    if attr_key == "reduce_scope":
        from ._thread_reduction import _ThreadReducer, _ThreadReducerScope

        if not isinstance(node, _ThreadReducer):
            raise TypeError("reduce_scope requires a comm_reducer")
        return _ThreadReducerScope(node)
    schema, payload = encode_annotations({"node": node, "key": attr_key, "value": value})
    return _builder.region(
        "tl.attr",
        [payload],
        attrs={"schema_version": 1, "annotation_schema": schema},
    )
