"""Finalization preserves the execution domains consumed by Ascend emitters."""

import ast

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

import tilelang.ascend  # noqa: F401 -- Register the target and its backend-owned operators.
import tilelang.transform as tr
import tilelang.ascend.transform as ascend_transform
from tilelang.layout import Layout
from tilelang.testing.ir import collect_nodes
from testing.ascend._block import ascend_block, attribute as _scope_attribute
from testing.ascend._ir import annotation


def attribute(key, value, body):
    return _scope_attribute(None, key, value, body)


def _prepare_ascend_codegen(module):
    module = tr.LowerSourceAnnotations()(module)
    module = ascend_transform.LowerAscendAnnotations()(module)
    return tr.VerifyLoweredAnnotations()(module)


@pytest.mark.parametrize("name,scope", [("SIMT_VF", "tl.simtvf_scope"), ("SIMD_VF", "tl.simdvf_scope")])
@pytest.mark.parametrize("latency", [None, 77])
def test_vf_block_and_scope_are_kept_for_the_owning_emitter(name, scope, latency, tmp_path):
    annotations = {"tl.vf_source_index": 7}
    if latency is not None:
        annotations["tl.vf_latency"] = ir.IntImm("int32", latency)
    region = attribute(
        "tl.sblock",
        {"name": name, "no_realize": True, "annotations": annotations},
        [
            attribute(scope, 1, [ir.Evaluate(ir.IntImm("int32", 3))]),
        ],
    )
    function = tirx.Function([], [region], ir.NoneType()).with_attr("target", Target("ascend"))
    before = ir.save_json(function)
    actual = _prepare_ascend_codegen(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == before
    if latency is None:
        assert tvm_ffi.structural_equal(function, actual)
    (retained,) = actual.body.seq
    assert retained.op.name == "tl.ascend_block" and retained.attrs["name"] == name
    assert dict(retained.attrs["annotations"]) == {"tl.vf_source_index": 7}
    assert tvm_ffi.structural_equal(retained.body, region.body)
    assert tirx.analysis.check_well_formed(actual)
    # The terminal device-codegen entry uses the lowered void return ABI.
    # Keep the original NoneType finalization oracle above unchanged.
    device_function = tirx.Function(actual.params, actual.body, ret_type=ir.PrimType("void"), attrs=actual.attrs, loc=actual.loc)
    device_function = device_function.with_attr("global_symbol", "kernel").with_attr("calling_conv", 2)
    for backend, language in [("ascend", "asc"), ("pto", "py")]:
        target = Target("ascend") if backend == "ascend" else Target({"kind": "ascend", "keys": ["pto"]})
        module = tvm_ffi.get_global_func(f"target.build.tilelang_{backend}_without_compile")(
            ir.IRModule({"kernel": device_function}), target
        )
        source = module.inspect_source(language)
        (tmp_path / f"kernel.{language}").write_text(source)
        if backend == "ascend":
            assert f"kernel_{name.lower()}_7" in source
        else:
            ast.parse(source)
            assert "def kernel(" in source
            if name == "SIMT_VF":
                assert "with pto.simt(1, 1, 1):" in source
    foreign = function.with_attr("target", Target("c"))
    with pytest.raises(ValueError, match="requires an Ascend target"):
        _prepare_ascend_codegen(ir.IRModule({"main": foreign}))


def test_vf_source_index_rejects_invalid_annotation():
    region = attribute("tl.sblock", {"name": "SIMD_VF", "annotations": {"tl.vf_source_index": -1}}, [])
    function = tirx.Function([], [region], ir.NoneType()).with_attr("target", Target("ascend"))
    with pytest.raises(ValueError, match="source index|tl.vf_source_index"):
        _prepare_ascend_codegen(ir.IRModule({"main": function}))


@pytest.mark.parametrize("dynamic", [True, False])
def test_ascend_codegen_boundary_rejects_dynamic_latency_and_unconsumed_annotations(dynamic):
    latency = ir.Var("latency", "int32")
    annotations = {"tl.vf_latency": latency} if dynamic else {"tl.unconsumed_vf_hint": 1}
    region = attribute("tl.sblock", {"name": "SIMT_VF", "annotations": annotations}, [])
    function = tirx.Function([latency] if dynamic else [], [region], ir.NoneType()).with_attr("target", Target("ascend"))
    message = "constant integer|tl.vf_latency" if dynamic else "tl.unconsumed_vf_hint"
    with pytest.raises(ValueError, match=message):
        _prepare_ascend_codegen(ir.IRModule({"main": function}))


def _metadata_function(carrier, metadata, params, body):
    if carrier == "body":
        body = [annotation(metadata), *body]
    region = ascend_block("VECTOR", body, metadata if carrier == "header" else {})
    return tirx.Function(params, [region], ir.NoneType()).with_attr("target", Target("ascend"))


@pytest.mark.parametrize("disabled", [False, True])
def test_safe_value_carriers_share_owner_consumption_policy(disabled):
    a = ir.Var("a", tirx.TensorType((4,), "float32"))
    b = ir.Var("b", tirx.TensorType((1,), "float32"))
    index = ir.Var("index", "int32")
    metadata = {"safe_value_map": {a: ir.FloatImm("float32", -7.0)}}
    body = [ir.TensorStore(b, [0], ir.TensorLoad(a, [index]))]
    results = []
    for carrier in ("header", "body"):
        function = _metadata_function(carrier, metadata, [a, b, index], body)
        before = ir.save_json(function)
        assert tirx.analysis.check_well_formed(function)
        with PassContext(config={"tl.disable_safe_memory_legalize": disabled}):
            # Even when guarding is disabled, only the safety owner may
            # consume this metadata; source lowering must retain it.
            pending = tr.LowerSourceAnnotations()(ir.IRModule({"main": function}))
            check_pending = ascend_transform.LowerAscendAnnotations() if carrier == "header" else tr.VerifyLoweredAnnotations()
            with pytest.raises(ValueError, match="safe_value_map"):
                check_pending(pending)
            legalized = tr.LegalizeSafeMemoryAccess()(ir.IRModule({"main": function}))
            final = _prepare_ascend_codegen(legalized)["main"]
        assert ir.save_json(function) == before
        assert tirx.analysis.check_well_formed(final)
        assert not tirx.analysis.undefined_vars(final)
        (block,) = final.body.seq
        assert block.op.name == "tl.ascend_block" and not block.args
        assert not block.attrs["annotations"]
        (store,) = collect_nodes(final, ir.TensorStore)
        if disabled:
            assert isinstance(store.value, ir.TensorLoad)
        else:
            assert collect_nodes(store.value, ir.TensorLoad)
            guards = [call for call in collect_nodes(store.value, ir.Call) if call.op.name == "prim.if_then_else"]
            assert guards and any(isinstance(call.args[-1], ir.FloatImm) and float(call.args[-1].value) == -7.0 for call in guards)
            # Evaluate the generated guards with a fixed legal-load value.
            # Both bounds must choose padding outside [0, 4).
            for probe in (-1, 0, 3, 4):
                expression = tvm_ffi.structural_map(
                    store.value,
                    [
                        (ir.TensorLoad, lambda _load: ir.FloatImm("float32", 23.0)),
                        (ir.Var, lambda var, probe=probe: ir.IntImm("int32", probe) if var.same_as(index) else var),
                    ],
                )
                value = sym.Analyzer().simplify(expression)
                assert isinstance(value, ir.FloatImm)
                assert float(value.value) == (23.0 if 0 <= probe < 4 else -7.0)
        results.append(final)
    ir.assert_structural_equal(*results)


def test_layout_carriers_are_consumed_by_real_tile_operator_lowering():
    tensor = ir.Var("tensor", tirx.TensorType((4,), "float32"))
    metadata = {"layout_map": {tensor: Layout([4], lambda i: i)}}
    body = [ir.Evaluate(ir.TensorLoad(tensor, [0]))]
    results = []
    for carrier in ("header", "body"):
        function = _metadata_function(carrier, metadata, [tensor], body)
        before = ir.save_json(function)
        pending = tr.LowerSourceAnnotations()(ir.IRModule({"main": function}))
        check_pending = ascend_transform.LowerAscendAnnotations() if carrier == "header" else tr.VerifyLoweredAnnotations()
        with pytest.raises(ValueError, match="layout_map"):
            check_pending(pending)
        # Exercise the actual owner instead of forging its completion flag.
        lowered = tr.LowerTileOp()(ir.IRModule({"main": function}))
        final = _prepare_ascend_codegen(lowered)["main"]
        assert tirx.analysis.check_well_formed(final)
        # Physical parameter views can precede the execution region.
        (block,) = [node for node in collect_nodes(final, ir.RegionStmt) if node.op.name == "tl.ascend_block"]
        assert block.op.name == "tl.ascend_block" and not block.args
        assert not block.attrs["annotations"]
        assert len(collect_nodes(final, ir.TensorLoad)) == 1
        results.append(final)
        assert ir.save_json(function) == before
    ir.assert_structural_equal(*results)
