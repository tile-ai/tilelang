"""C++ and Python metadata codecs share one schema with explicit native uses."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.language._native import encode_annotations
from tilelang.language.warp_specialize import (
    WSOpRef,
    WSPipeline,
    WSRole,
    WSSchedule,
    WSScope,
    WSSync,
)
from tilelang.layout import Fragment, Layout


def _cpp_encode(value):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(value)
    return schema, payload


def _decode(schema, payload):
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(payload, schema)


def _assert_static(value):
    assert not isinstance(value, ir.Expr)
    if isinstance(value, tvm_ffi.Array):
        for item in value:
            _assert_static(item)
    elif isinstance(value, tvm_ffi.Map):
        for key, item in value.items():
            _assert_static(key)
            _assert_static(item)


def _fixture():
    n = ir.Var("n", "int32")
    tensor = ir.Var("A", tirx.TensorType([n], "float32"))
    role = WSRole("producer", warps_lo=0, warps_hi=1)
    pipeline = WSPipeline("pipe", [tensor], 2)
    scope = WSScope("root", {"producer": [WSSync.producer_acquire("pipe"), WSOpRef("copy")]})
    schedule = WSSchedule(4, [role], [pipeline], [scope])
    return (
        n,
        tensor,
        {
            "scalars": [None, True, False, -7, 2**100 + 3, 1.25, "tag"],
            "uses": [n, n + 1, tensor],
            "layout": Layout([4], lambda i: i + n),
            "fragment": Fragment([32], forward_fn=lambda i: (i, 0)),
            "schedule": schedule,
        },
    )


def test_cpp_matches_python_codec_without_hiding_expression_uses():
    n, tensor, values = _fixture()
    py_schema, py_payload = encode_annotations(values)
    cpp_schema, cpp_payload = _cpp_encode(values)
    _assert_static(cpp_schema)
    ir.assert_structural_equal(_decode(py_schema, py_payload), _decode(cpp_schema, cpp_payload))
    assert _decode(cpp_schema, cpp_payload)["uses"][0].same_as(n)
    assert _decode(cpp_schema, cpp_payload)["schedule"].pipelines[0].buffers[0].same_as(tensor)


def _annotated_function():
    n, tensor, values = _fixture()
    call = tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
        ir.PrimType("void"),
        ir.Op.get("tl.tileop.fill"),
        [tensor, ir.FloatImm("float32", 0.0)],
        values,
        ir.UnknownLoc(),
    )
    call.validate()
    return tirx.Function([n, tensor], [ir.Evaluate(call)], ir.NoneType())


def _call_metadata(function):
    call = function.body[0].value
    return _decode(call.attrs["annotation_schema"], call.args[-1])


def test_annotation_json_and_renew_def_keep_layout_captures_and_tensor_identity():
    function = _annotated_function()
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)
    restored = ir.load_json(ir.save_json(function))
    ir.assert_structural_equal(restored, function)
    renewed = tirx.renew_def(function)
    assert not renewed.params[0].same_as(function.params[0])
    assert not renewed.params[1].same_as(function.params[1])
    assert tirx.analysis.check_well_formed(renewed)
    assert not tirx.analysis.undefined_vars(renewed)
    metadata = _call_metadata(renewed)
    assert metadata["uses"][0].same_as(renewed.params[0])
    assert metadata["uses"][2].same_as(renewed.params[1])
    assert metadata["schedule"].pipelines[0].buffers[0].same_as(renewed.params[1])
    ir.assert_structural_equal(function, renewed)


def test_encoder_rejects_unsupported_objects_instead_of_hiding_them_in_attrs():
    with pytest.raises(TypeError, match="Unsupported TileLang annotation value"):
        _cpp_encode({"type": ir.PrimType("float32")})


@pytest.mark.parametrize(
    "schema",
    [
        ["expr", 5],
        ["map", [["str", "a"], ["int", 1]], [["str", "a"], ["int", 2]]],
        ["layout", "layout", 2, 0],
        ["unknown", 0],
    ],
)
def test_decoder_rejects_malformed_or_unsupported_metadata(schema):
    with pytest.raises((TypeError, ValueError)):
        _decode(schema, ir.Tuple([ir.Tuple([])]))
