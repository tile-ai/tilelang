"""Ascend execution blocks keep constants in attrs and captures in IR operands."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.ascend import language as T
from tilelang.analysis._visitor import tilelang_metadata
from tilelang.layout import Layout
from tilelang.testing.ir import collect_nodes
from testing.ascend._block import ASCEND_BLOCK_NAMES, ascend_block, ascend_block_annotations


@pytest.mark.parametrize("name", sorted(ASCEND_BLOCK_NAMES))
def test_static_execution_metadata_needs_no_tuple_or_schema(name):
    typed = ir.IntImm("int16", 7)
    block = ascend_block(name, [], {"hint": 17, "typed": typed, "nested": {"values": [True, "tag", 1.25, None]}})
    assert block.op.name == "tl.ascend_block"
    assert not block.args and not block.body_params and not block.result_vars
    assert set(block.attrs.keys()) == {"name", "annotations"}
    assert block.attrs["name"] == name
    annotations = block.attrs["annotations"]
    assert int(annotations["hint"]) == 17
    assert annotations["typed"].same_as(typed)
    assert str(annotations["typed"].ty.dtype) == "int16"
    assert list(annotations["nested"]["values"]) == [True, "tag", 1.25, None]
    function = tirx.Function([], [block])
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))


def test_tensor_map_keys_and_layout_captures_remain_substitutable_operands():
    offset = ir.Var("offset", "int32")
    tensor = ir.Var("tensor", tirx.TensorType((8,), "float32"))
    replacement = ir.Var("replacement", tensor.ty)
    layout = Layout([8], lambda i: i + offset)
    block = ascend_block("VECTOR", [], {"vector_count": 2, "layout_map": {tensor: layout}, "aliases": {tensor: [tensor]}})
    assert len(block.args) == 1 and isinstance(block.args[0], ir.Tuple)
    assert set(block.attrs.keys()) == {"name", "annotations", "annotation_schema"}
    assert set(block.attrs["annotations"]) == {"vector_count"}
    metadata = tilelang_metadata(block)
    assert metadata["key"] == "tl.sblock" and metadata["value"]["name"] == "VECTOR"
    semantic_annotations = metadata["value"]["annotations"]
    assert int(semantic_annotations["vector_count"]) == 2
    assert semantic_annotations["aliases"][tensor][0].same_as(tensor)
    ir.assert_structural_equal(semantic_annotations["layout_map"][tensor], layout)
    variables = []
    tvm_ffi.structural_walk(block.attrs, (ir.Var, variables.append))
    assert not variables
    function = tirx.Function([offset, tensor], [block])
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)

    replaced = tvm_ffi.structural_mutate(block, (ir.Var, lambda var, _mutator: replacement if var.same_as(tensor) else var))
    annotations = ascend_block_annotations(replaced)
    (layout_tensor,) = annotations["layout_map"]
    (alias_tensor,) = annotations["aliases"]
    assert layout_tensor.same_as(replacement) and alias_tensor.same_as(replacement)
    assert annotations["aliases"][replacement][0].same_as(replacement)
    assert not any(var.same_as(tensor) for var in collect_nodes(replaced, ir.Var))
    rewritten = tirx.Function([offset, replacement], [replaced])
    assert tirx.analysis.check_well_formed(rewritten)
    assert not tirx.analysis.undefined_vars(rewritten)
    renewed = tirx.renew_def(rewritten)
    (renewed_block,) = collect_nodes(renewed, ir.RegionStmt)
    (renewed_tensor,) = ascend_block_annotations(renewed_block)["layout_map"]
    assert renewed_tensor.same_as(renewed.params[1]) and not renewed_tensor.same_as(replacement)
    assert not tirx.analysis.undefined_vars(renewed)
    ir.assert_structural_equal(renewed, ir.load_json(ir.save_json(renewed)))


def test_header_capture_cannot_use_a_definition_from_the_region_body():
    local = ir.Var("local", "int32")
    block = ascend_block("SIMD_VF", [ir.Bind(local, ir.const(7))], {"capture": local})
    function = tirx.Function([], [block])
    assert any(var.same_as(local) for var in tirx.analysis.undefined_vars(function))
    assert not tirx.analysis.check_well_formed(function, assert_mode=False)


def test_body_annotation_stays_after_its_definition_and_in_its_source_owner():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32")):
        with T.sblock("VECTOR"):
            local = T.alloc_local((4,), "int32")
            T.sblock_attr({"mapping": {local: [A]}})
            with T.sblock("SIMD_VF"):
                nested = T.alloc_local((4,), "int32")
                T.sblock_attr({"inner": {nested: [A]}})

    blocks = {node.attrs["name"]: node for node in collect_nodes(kernel, ir.RegionStmt) if node.op.name == "tl.ascend_block"}
    outer, inner = blocks["VECTOR"], blocks["SIMD_VF"]
    assert [outer.attrs["name"], inner.attrs["name"]] == ["VECTOR", "SIMD_VF"]
    assert all(not block.args and not block.attrs["annotations"] for block in (outer, inner))
    snapshot = ir.save_json(kernel)
    collect_annotations = tvm_ffi.get_global_func("tl.CollectSourceBlockAnnotations")
    for block, name, key in [(outer, "local", "mapping"), (inner, "nested", "inner")]:
        allocation_index, allocation = next(
            (index, statement)
            for index, statement in enumerate(block.body.seq)
            if isinstance(statement, ir.Bind) and statement.var.name == name
        )
        annotation_index, note = next(
            (index, statement.value)
            for index, statement in enumerate(block.body.seq)
            if isinstance(statement, ir.Evaluate)
            and isinstance(statement.value, ir.Call)
            and statement.value.op.name == "tl.annotation"
            and key in tilelang_metadata(statement.value)
        )
        assert allocation_index < annotation_index
        assert any(variable.same_as(allocation.var) for variable in collect_nodes(note.args[0], ir.Var))
        annotations = collect_annotations(block.body)
        assert set(annotations) == {key}
        assert list(annotations[key][allocation.var])[0].same_as(kernel.params[0])
    assert ir.save_json(kernel) == snapshot
    assert not tirx.analysis.undefined_vars(kernel)
    assert tirx.analysis.check_well_formed(kernel)


@pytest.mark.parametrize("name", sorted(ASCEND_BLOCK_NAMES))
@pytest.mark.parametrize("no_realize", [False, True])
def test_generic_sblock_keeps_axes_predicate_and_public_arguments(name, no_realize):
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32"), index: T.int32):
        with T.sblock(name, no_realize=no_realize):
            i = T.axis.spatial(4, index)
            T.where(i < 3)
            A[i] = 11

    (block,) = [node for node in collect_nodes(kernel, ir.RegionStmt) if node.op.name == "tl.ascend_block"]
    assert block.attrs["name"] == name
    assert not block.args and "no_realize" not in block.attrs
    assert len(collect_nodes(kernel, ir.If)) == 1
    assert len(collect_nodes(block, ir.TensorStore)) == 1
    assert not tirx.analysis.undefined_vars(kernel)
    assert tirx.analysis.check_well_formed(kernel)


@pytest.mark.parametrize(
    "case", ["missing_annotations", "unknown_name", "capture_in_attrs", "tuple_without_schema", "duplicate_annotation"]
)
def test_malformed_execution_headers_are_rejected(case):
    captured = ir.Var("captured", "int32")
    attrs = {"name": "VECTOR", "annotations": {}}
    args = []
    if case == "missing_annotations":
        del attrs["annotations"]
        message = "requires an annotations map"
    elif case == "unknown_name":
        attrs["name"] = "ordinary"
        message = "name must be"
    elif case == "capture_in_attrs":
        attrs["annotations"] = {"capture": captured}
        message = "static metadata"
    elif case == "tuple_without_schema":
        args = [ir.Tuple([captured])]
        message = "metadata Tuple"
    else:
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"capture": captured})
        args = [payload]
        attrs.update(annotations={"capture": 1}, annotation_schema=schema)
        message = "Duplicate static and dynamic"
    with pytest.raises((TypeError, ValueError), match=message):
        block = ir.RegionStmt("tl.ascend_block", args, [], attrs, [])
        tirx.analysis.check_well_formed(tirx.Function([captured], [block]))


def test_ordinary_block_and_task_still_keep_the_general_operand_codec():
    from testing.ascend._block import attribute, attribute_fields

    tensor = ir.Var("tensor", tirx.TensorType((8,), "float32"))
    for node, key, value in [
        (None, "tl.sblock", {"name": "tilelang_root", "annotations": {"mapping": {tensor: [tensor]}}}),
        ({"capture": tensor}, "tl.ascend_task", ir.const(1)),
    ]:
        region = attribute(node, key, value, [])
        assert region.op.name == "tl.attr"
        assert len(region.args) == 1 and "annotation_schema" in region.attrs
        assert any(var.same_as(tensor) for var in collect_nodes(region.args[0], ir.Var))
        assert attribute_fields(region)["key"] == key
        assert tirx.analysis.check_well_formed(tirx.Function([tensor], [region]))
