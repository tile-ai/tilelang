"""Execution domains use native statements and independent tensor annotations."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.ascend import language as T
from tilelang.analysis._visitor import tilelang_metadata
from tilelang.layout import Layout
from tilelang.testing.ir import collect_nodes
from testing.ascend._block import ASCEND_BLOCK_NAMES, ascend_block, ascend_block_annotations


@pytest.mark.parametrize("name", sorted(ASCEND_BLOCK_NAMES))
def test_execution_metadata_is_native_static_configuration(name):
    attrs = {"vector_count": 2} if name == "VECTOR" else {"tl.vf_source_index": 7} if name.endswith("_VF") else {}
    block = ascend_block(name, [], attrs)
    assert block.op.same_as(ir.Op.get("tl.ascend_block"))
    assert not block.args and not block.body_params and not block.result_vars
    assert set(block.attrs.keys()) == {"name", *attrs}
    for key, value in attrs.items():
        assert int(block.attrs[key]) == value
    assert tilelang_metadata(block) is None
    function = tirx.Function([], [block])
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))


def test_layout_declaration_uses_existing_attrs_rewrite_contract():
    offset = ir.Var("offset", "int32")
    tensor = ir.Var("tensor", tirx.TensorType((8,), "float32"))
    replacement = ir.Var("replacement", tensor.ty)
    layout = Layout([8], lambda i: i + offset)
    block = ascend_block("VECTOR", [], {"vector_count": 2, "layout_map": {tensor: layout}})
    assert not block.args and set(block.attrs.keys()) == {"name", "vector_count"}
    (note,) = collect_nodes(block, ir.Call)
    assert note.op.same_as(ir.Op.get("tl.layout_map")) and not note.args
    function = tirx.Function([offset, tensor], [block])
    tvm_ffi.get_global_func("tl.CheckLayoutMapScopes")(function)
    replaced = tvm_ffi.get_global_func("tl.SubstituteLayoutMap")(block, {tensor: replacement})
    (key,) = ascend_block_annotations(replaced)["layout_map"]
    assert key.same_as(replacement)
    rewritten = tirx.Function([offset, replacement], [replaced])
    renewed = tvm_ffi.get_global_func("tl.RenewDef")(rewritten)
    tvm_ffi.get_global_func("tl.CheckLayoutMapScopes")(renewed)
    (renewed_block,) = collect_nodes(renewed, ir.RegionStmt)
    (key,) = ascend_block_annotations(renewed_block)["layout_map"]
    assert key.same_as(renewed.params[1]) and not key.same_as(replacement)
    ir.assert_structural_equal(renewed, ir.load_json(ir.save_json(renewed)))


def test_layout_declaration_cannot_precede_its_tensor_definition():
    local = ir.Var("local", tirx.TensorType((8,), "float32", scope="local"))
    allocation = ir.Bind(local, ir.Call(ir.Op.get("tirx.alloc_tensor"), [], ty_args=[local.ty]))
    block = ascend_block("SIMD_VF", [allocation], {"layout_map": {local: Layout([8], lambda i: i)}})
    with pytest.raises(ValueError, match="layout_map"):
        tvm_ffi.get_global_func("tl.CheckLayoutMapScopes")(tirx.Function([], [block]))


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
    assert all(not block.args and set(block.attrs.keys()) == {"name"} for block in (outer, inner))
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


@pytest.mark.parametrize("case", ["missing_name", "unknown_name", "capture_in_attrs", "operands", "old_annotations"])
def test_malformed_execution_headers_are_rejected(case):
    captured = ir.Var("captured", "int32")
    attrs, args = {"name": "VECTOR"}, []
    if case == "missing_name":
        del attrs["name"]
    elif case == "unknown_name":
        attrs["name"] = "ordinary"
    elif case == "capture_in_attrs":
        attrs["vector_count"] = captured
    elif case == "operands":
        args = [ir.Tuple([captured])]
    else:
        attrs["annotations"] = {"vector_count": 2}
    with pytest.raises((TypeError, ValueError)):
        block = ir.RegionStmt("tl.ascend_block", args, [], attrs, [])
        tirx.analysis.check_well_formed(tirx.Function([captured], [block]))


def test_ordinary_source_block_has_only_a_native_definition_scope():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32")):
        with T.sblock("ordinary"):
            temporary = T.alloc_local((4,), "int32")
            T.annotate_layout({temporary: Layout([4], lambda i: i)})
            temporary[0] = A[0]

    (scope,) = collect_nodes(kernel, ir.RegionStmt)
    assert scope.op.name == "tl.lexical_scope"
    assert not scope.args and not scope.attrs.keys()
    assert not scope.body_params and not scope.result_vars
    (allocation,) = collect_nodes(kernel, ir.Bind)
    assert allocation.value.op.name == "tirx.alloc_tensor"
    tvm_ffi.get_global_func("tl.CheckLayoutMapScopes")(kernel)
    assert tirx.analysis.check_well_formed(kernel)
