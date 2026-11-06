"""Source syntax lowers independently of backend and ABI annotation owners."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.transform as transform
from tilelang.language._intrinsics import call_intrin
from tilelang.language._annotation_codec import encode_annotations


def _attribute(key, value, body, node=None):
    schema, payload = encode_annotations({"node": node, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def _annotation(values):
    schema, payload = encode_annotations(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def _decode(node):
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(node.args[0], node.attrs["annotation_schema"])


def _lower(function):
    assert tirx.analysis.check_well_formed(function)
    result = transform.LowerSourceAnnotations()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    return result


def test_pending_alias_and_size_metadata_keep_their_native_definition_scope():
    size = ir.Var("size", "int32")
    tensor = ir.Var("tensor", tirx.TensorType([size], "float32"))
    local = ir.Var("local", tensor.ty)
    declaration = _annotation(
        {
            "tl.source_block_annotation": True,
            "tl.non_restrict_params": [local],
            "tl.pending_contract": size + 1,
            "pragma_unroll_factor": 4,
            "user_hint": size + 2,
        }
    )
    scope = _attribute("user_scope", size, [ir.Bind(local, tensor), declaration, ir.Evaluate(size)])
    defaults = _attribute("tl.size_var_defaults", [size], [scope])
    function = tirx.Function([size, tensor], [defaults], ir.NoneType())
    before = ir.save_json(function)
    result = _lower(function)

    default_region = result.body[0]
    assert default_region.op.name == "tl.attr"
    ir.assert_structural_equal(default_region.args, defaults.args)
    ir.assert_structural_equal(default_region.attrs, defaults.attrs)
    lexical = default_region.body[0]
    assert lexical.op.name == "tl.lexical_scope"
    assert lexical.body[0].var.same_as(local)
    retained = lexical.body[1].value
    assert retained.op.name == "tl.annotation"
    values = _decode(retained)
    assert set(values) == {"tl.non_restrict_params", "tl.pending_contract"}
    assert values["tl.non_restrict_params"][0].same_as(local)
    ir.assert_structural_equal(values["tl.pending_contract"], size + 1)
    codegen = lexical.body[2]
    assert _decode(codegen)["key"] == "pragma_unroll_factor"
    assert codegen.body[0].value.same_as(size)
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(tirx.renew_def(result))
    ir.assert_structural_equal(result, _lower(result))


@pytest.mark.parametrize("name", ["CUBE", "VECTOR", "SIMT_VF", "SIMD_VF", "ordinary"])
def test_legacy_source_name_never_creates_a_backend_execution_region(name):
    # Construct raw source syntax, bypassing the public Ascend block factory.
    source = _attribute(
        "tl.sblock",
        {"name": name, "no_realize": True, "annotations": {"lexical_alloc_scope": False, "pragma_import_c": "// prelude"}},
        [ir.Evaluate(ir.const(7))],
    )
    result = _lower(tirx.Function([], [source], ir.NoneType()))
    scope = result.body[0]
    assert scope.op.name == "tl.lexical_scope"
    assert scope.attrs["explicit_alloc_scope"] is True
    codegen = scope.body[0]
    assert codegen.op.name == "tl.attr"
    assert _decode(codegen)["key"] == "pragma_import_c"
    assert codegen.body[0].value.value == 7


def test_native_backend_block_retains_header_and_lowers_only_source_body_syntax():
    args, attrs = tvm_ffi.get_global_func("tl.EncodeAscendBlock")("SIMD_VF", {"tl.vf_source_index": 7, "tl.vf_latency": 3})
    source = _attribute("user_scope", 1, [ir.Evaluate(ir.const(9))])
    block = ir.RegionStmt("tl.ascend_block", args, [], attrs, [source])
    result = _lower(tirx.Function([], [block], ir.NoneType()))
    lowered = result.body[0]
    assert lowered.op.name == "tl.ascend_block"
    ir.assert_structural_equal(lowered.args, block.args)
    ir.assert_structural_equal(lowered.attrs, block.attrs)
    assert lowered.body[0].op.name == "tl.lexical_scope"
    assert lowered.body[0].body[0].value.value == 9


@pytest.mark.parametrize(
    "key", ["layout_map", "safe_value_map", "kWarpSpecializationScope", "tl.loop.annotations", "tl.simtvf_scope", "tl.simdvf_scope"]
)
def test_semantic_regions_keep_explicit_operands_until_their_owner_runs(key):
    value = ir.Var("value", "int32")
    region = _attribute(key, value, [ir.Evaluate(value)], node=value + 1)
    function = tirx.Function([value], [region], ir.NoneType())
    ir.assert_structural_equal(function, _lower(function))


def test_call_metadata_normalization_keeps_operands_and_does_not_evaluate_hints():
    value, debug = ir.Var("value", "int32"), ir.Var("debug", "int32")
    discarded = call_intrin("int32", "tirx.call_extern", "discarded_hint", debug)
    call = call_intrin("int32", "tirx.call_pure_extern", "identity", value, annotations={"source_tag": "tag", "debug_value": discarded})
    call = ir.Call(call.op, call.args, attrs={**dict(call.attrs), "native_semantics": 3}, ty_args=call.ty_args, ty=call.ty)
    function = tirx.Function([value, debug], [ir.Evaluate(call)], ir.NoneType())
    before = ir.save_json(function)
    result = _lower(function)
    lowered = result.body[0].value
    assert len(result.body) == 1
    assert len(lowered.args) == 2
    assert lowered.args[1].same_as(value)
    assert lowered.attrs["source_tag"] == "tag"
    assert lowered.attrs["native_semantics"] == 3
    assert list(lowered.attrs["tl.source_annotation_keys"]) == ["source_tag"]
    assert "debug_value" not in lowered.attrs
    assert "operand_count" not in lowered.attrs
    assert "annotation_schema" not in lowered.attrs
    assert ir.save_json(function) == before
    ir.assert_structural_equal(result, _lower(result))


def test_call_metadata_cannot_override_existing_native_attributes():
    value = ir.Var("value", "int32")
    call = call_intrin("int32", "tirx.call_pure_extern", "identity", value, annotations={"source_tag": "source", "hint": value})
    call = ir.Call(call.op, call.args, attrs={**dict(call.attrs), "source_tag": "native"}, ty_args=call.ty_args, ty=call.ty)
    function = tirx.Function([value], [ir.Evaluate(call)], ir.NoneType())
    with pytest.raises(ValueError, match="both native attributes and metadata operands"):
        _lower(function)


@pytest.mark.parametrize("key", ["precision", "round", "sat", "layout_map", "pragma_unroll_factor", "tl.pending_option"])
def test_dynamic_execution_options_cannot_disappear_as_descriptive_hints(key):
    value = ir.Var("value", "int32")
    call = call_intrin("int32", "tirx.call_pure_extern", "identity", value, annotations={key: value})
    function = tirx.Function([value], [ir.Evaluate(call)], ir.NoneType())
    with pytest.raises(ValueError, match="Unconsumed intrinsic annotation"):
        _lower(function)


def test_source_lowering_rejects_a_remaining_tile_operator():
    tensor = ir.Var("tensor", tirx.TensorType([4], "float32"))
    call = tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
        ir.PrimType("void"), ir.Op.get("tl.tileop.fill"), [tensor, ir.FloatImm("float32", 0.0)], {}, ir.UnknownLoc()
    )
    function = tirx.Function([tensor], [ir.Evaluate(call)], ir.NoneType())
    with pytest.raises(ValueError, match="cannot run before LowerTileOp"):
        _lower(function)


@pytest.mark.parametrize("marker", [False, 1, "true"])
def test_source_origin_marker_requires_the_actual_boolean(marker):
    function = tirx.Function([], [_annotation({"tl.source_block_annotation": marker})], ir.NoneType())
    with pytest.raises(ValueError, match="must be the source annotation marker"):
        _lower(function)


def test_source_shell_rejects_unlowered_structured_execution_fields():
    source = _attribute("tl.sblock", {"name": "ordinary", "init": 1}, [])
    with pytest.raises(ValueError, match="Unconsumed structured block field `init`"):
        _lower(tirx.Function([], [source], ir.NoneType()))
