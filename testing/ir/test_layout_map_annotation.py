"""Direct layout declarations preserve source ownership and typed identities."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.layout import Layout
from tilelang.language import _annotation_codec as codec


def calls(root):
    result = []
    tvm_ffi.structural_walk(root, (ir.Call, result.append))
    return result


def make_annotation(fields):
    return tvm_ffi.get_global_func("tl.MakeAnnotation")(fields, ir.UnknownLoc())


def read_annotation(call):
    return tvm_ffi.get_global_func("tl.ReadAnnotations")(call)


def tensor(name="A"):
    return ir.Var(name, tirx.TensorType((16,), "int32", scope="local"))


def test_layout_constructor_retains_objects_without_payload():
    a = tensor()
    layout = Layout((16,), lambda i: 15 - i)
    statement = make_annotation({"layout_map": {a: layout}})
    assert isinstance(statement, ir.Evaluate)
    call = statement.value
    assert call.op.same_as(ir.Op.get("tl.layout_map"))
    assert not call.args and not call.ty_args
    assert set(call.attrs.keys()) == {"layout_map"}
    assert call.attrs["layout_map"][a].same_as(layout)
    restored = tvm_ffi.get_global_func("tl.ReadLayoutMapAnnotation")(call)
    assert next(iter(restored)).same_as(a)
    assert restored[a].same_as(layout)


def test_mixed_metadata_splits_both_source_markers_without_layout_codec():
    a = tensor()
    layout = Layout((16,), lambda i: i)
    statement = make_annotation(
        {
            "layout_map": {a: layout},
            "safe_value_map": {a: 7},
            "tl.source_block_annotation": True,
        }
    )
    assert isinstance(statement, ir.SeqStmt)
    assert len(statement.seq) == 2
    layout_call, other_call = [stmt.value for stmt in statement.seq]
    assert layout_call.op.same_as(ir.Op.get("tl.layout_map"))
    assert other_call.op.same_as(ir.Op.get("tl.annotation"))
    assert layout_call.attrs["layout_map"][a].same_as(layout)
    assert read_annotation(layout_call)["tl.source_block_annotation"] is True
    other = read_annotation(other_call)
    assert "layout_map" not in other
    assert other["safe_value_map"][a] == 7
    assert other["tl.source_block_annotation"] is True
    collected = tvm_ffi.get_global_func("tl.CollectSourceBlockAnnotations")(statement)
    assert collected["layout_map"][a].same_as(layout)
    assert collected["safe_value_map"][a] == 7


def test_frontend_layout_only_does_not_call_python_codec(monkeypatch):
    original = codec.encode_annotations

    def forbid_layout(fields):
        if "layout_map" in fields:
            raise AssertionError("layout_map must not use the annotation codec")
        return original(fields)

    monkeypatch.setattr(codec, "encode_annotations", forbid_layout)
    layout = Layout((16,), lambda i: i)

    @T.prim_func
    def kernel():
        with T.sblock("owner"):
            a = T.alloc_buffer((16,), "int32", scope="local")
            a[0] = 1
            T.annotate_layout({a: layout})

    notes = [c for c in calls(kernel) if c.op.same_as(ir.Op.get("tl.layout_map"))]
    assert len(notes) == 1
    assert notes[0].attrs["tl.source_block_annotation"] is True
    assert next(iter(notes[0].attrs["layout_map"].values())).same_as(layout)


def test_frontend_mixed_sblock_attr_preserves_both_fields():
    layout = Layout((16,), lambda i: i)

    @T.prim_func
    def kernel():
        with T.sblock("owner"):
            a = T.alloc_buffer((16,), "int32", scope="local")
            T.sblock_attr({"layout_map": {a: layout}, "safe_value_map": {a: 7}})
            a[0] = 1

    notes = [c for c in calls(kernel) if c.op.same_as(ir.Op.get("tl.layout_map"))]
    assert len(notes) == 1
    others = [c for c in calls(kernel) if c.op.same_as(ir.Op.get("tl.annotation"))]
    safe = [read_annotation(c) for c in others if "safe_value_map" in read_annotation(c)]
    assert len(safe) == 1
    a = next(iter(notes[0].attrs["layout_map"]))
    assert safe[0]["safe_value_map"][a] == 7
    assert safe[0]["tl.source_block_annotation"] is True


def test_source_collection_does_not_cross_nested_owner():
    a, b = tensor("A"), tensor("B")
    first = Layout((16,), lambda i: i)
    second = Layout((16,), lambda i: 15 - i)
    outer_note = make_annotation({"layout_map": {a: first}, "tl.source_block_annotation": True})
    inner_note = make_annotation({"layout_map": {b: second}, "tl.source_block_annotation": True})
    schema, payload = codec.encode_annotations({"node": None, "key": "tl.sblock", "value": {"name": "inner"}})
    inner = ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, [inner_note])
    body = ir.SeqStmt([inner, outer_note])
    collected = tvm_ffi.get_global_func("tl.CollectSourceBlockAnnotations")(body)
    assert a in collected["layout_map"]
    assert b not in collected["layout_map"]
    assert collected["layout_map"][a].same_as(first)


@pytest.mark.parametrize("invalid", ["scalar_key", "non_layout", "extra_field", "false_marker", "missing_map", "operand"])
def test_layout_call_rejects_invalid_contract(invalid):
    a = tensor()
    attrs = {"layout_map": {a: Layout((16,), lambda i: i)}}
    args = []
    if invalid == "scalar_key":
        attrs["layout_map"] = {ir.Var("x", "int32"): next(iter(attrs["layout_map"].values()))}
    elif invalid == "non_layout":
        attrs["layout_map"] = {a: 1}
    elif invalid == "extra_field":
        attrs["unregistered"] = 1
    elif invalid == "false_marker":
        attrs["tl.source_block_annotation"] = False
    elif invalid == "missing_map":
        attrs = {}
    elif invalid == "operand":
        args = [a]
    with pytest.raises((TypeError, ValueError)):
        ir.Call("tl.layout_map", args, attrs=attrs).validate()
