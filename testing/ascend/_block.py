"""Construct and inspect native source scopes without hiding their IR operands."""

import tvm_ffi
from tvm_tirx import ir


TASK_REGION_KEYS = {
    "tl.ascend.task": "tl.ascend_task",
    "tl.ascend.per_core_task": "tl.ascend_per_core_task",
    "tl.ascend.schedule_unit": "tl.schedule_unit",
    "tl.ascend.stage": "tl.ascend_stage",
}
SEMANTIC_SCOPE_OPS = frozenset({"tl.attr", "tl.ascend_block", *TASK_REGION_KEYS})

ASCEND_BLOCK_NAMES = frozenset({"CUBE", "VECTOR", "SIMT_VF", "SIMD_VF"})


def ascend_block(name, body, annotations=None, *, loc=None):
    """Build execution configuration and independent metadata statements."""
    loc = ir.UnknownLoc() if loc is None else loc
    metadata = dict(annotations or {})
    attrs = {"name": name}
    for key in ("vector_count", "tl.vf_source_index", "tl.vf_latency"):
        if key in metadata:
            attrs[key] = metadata.pop(key)
    body = ir.SeqStmt(body)
    if metadata:
        note = tvm_ffi.get_global_func("tl.MakeAnnotation")({**metadata, "tl.source_block_annotation": True}, loc)
        body = ir.SeqStmt([note, body])
    return ir.RegionStmt("tl.ascend_block", [], [], attrs, body, loc=loc)


def ascend_block_annotations(block):
    """Inspect configuration and source declarations without altering their IR."""
    assert isinstance(block, ir.RegionStmt) and block.op.name == "tl.ascend_block"
    result = dict(block.attrs)
    del result["name"]
    result.update(tvm_ffi.get_global_func("tl.CollectSourceBlockAnnotations")(block.body))
    return result


def attribute(node, key, value, body, *, loc=None):
    """Construct the current native representation for a source-scope fixture.

    Execution blocks and task/schedule scopes use their dedicated ops.
    Ordinary source annotations retain the source annotation codec.
    """
    if key == "tl.sblock":
        assert node is None or isinstance(node, ir.IntImm) and int(node) == 0
        assert not (set(value) - {"name", "no_realize", "annotations"})
        if value.get("name") in ASCEND_BLOCK_NAMES:
            return ascend_block(value["name"], body, value.get("annotations", {}), loc=loc)
        body = ir.SeqStmt(body)
        metadata = value.get("annotations", {})
        if metadata:
            note = tvm_ffi.get_global_func("tl.MakeAnnotation")({**metadata, "tl.source_block_annotation": True}, ir.UnknownLoc())
            body = ir.SeqStmt([note, body])
        return ir.RegionStmt("tl.lexical_scope", [], [], {}, body)
    if key in {"tl.ascend_task", "tl.ascend_per_core_task"}:
        annotations = dict(node) if isinstance(node, (dict, tvm_ffi.Map)) else {}
        args, attrs = tvm_ffi.get_global_func("tl.EncodeAscendTask")(annotations)
        op = "tl.ascend.task" if key == "tl.ascend_task" else "tl.ascend.per_core_task"
        return ir.RegionStmt(op, args, [], attrs, body, loc=ir.UnknownLoc() if loc is None else loc)
    if key in {"tl.schedule_unit", "tl.ascend_stage"}:
        stage = node["stage"] if key == "tl.schedule_unit" else node
        op = "tl.ascend.schedule_unit" if key == "tl.schedule_unit" else "tl.ascend.stage"
        return ir.RegionStmt(op, [], [], {"stage": stage}, body, loc=ir.UnknownLoc() if loc is None else loc)
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": node, "key": key, "value": value})
    return ir.RegionStmt(
        "tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body, loc=ir.UnknownLoc() if loc is None else loc
    )


def attribute_fields(node):
    """Expose the semantic source-scope fields for existing test oracles.

    This is a decoded inspection view, not an IR compatibility object. Tests
    for representation inspect ``node.op``, ``node.attrs`` and ``node.args``.
    """
    assert isinstance(node, ir.RegionStmt)
    if node.op.name == "tl.ascend_block":
        return {"node": None, "key": "tl.sblock", "value": {"name": node.attrs["name"], "annotations": ascend_block_annotations(node)}}
    if node.op.name in TASK_REGION_KEYS:
        key = TASK_REGION_KEYS[node.op.name]
        if node.op.name in {"tl.ascend.task", "tl.ascend.per_core_task"}:
            metadata = dict(node.attrs["annotations"])
            if "annotation_schema" in node.attrs:
                assert len(node.args) == 1 and isinstance(node.args[0], ir.Tuple)
                dynamic = tvm_ffi.get_global_func("tl.DecodeAnnotations")(node.args[0], node.attrs["annotation_schema"])
                assert not set(metadata).intersection(dynamic)
                metadata.update(dynamic)
            else:
                assert not node.args
        else:
            assert not node.args and set(node.attrs.keys()) == {"stage"}
            metadata = {"stage": node.attrs["stage"]} if key == "tl.schedule_unit" else node.attrs["stage"]
        assert not node.body_params and not node.result_vars
        return {"node": metadata, "key": key, "value": 1}
    assert node.op.name == "tl.attr"
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(node.args[0], node.attrs["annotation_schema"])


def has_attribute(node, key):
    return isinstance(node, ir.RegionStmt) and node.op.name in SEMANTIC_SCOPE_OPS and attribute_fields(node)["key"] == key
