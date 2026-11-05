"""Check typed scalar calls and expression-bearing annotation payloads."""

import pytest
import tvm_tirx as tvm


I = tvm.ir


def test_native_result_inference_and_explicit_native_type():
    value = I.const(1, "float32")
    inferred = I.Call("tl.fmul", [value, value], ty_args=[I.PrimType("float32")])
    inferred.validate()
    assert inferred.ty.dtype == "float32"
    # Native C++ passes reconstruct calls with an explicit result type.
    explicit = I.Call("tl.fmul", [value, value], ty="float32")
    explicit.validate()
    assert explicit.ty.dtype == "float32"


@pytest.mark.parametrize("operands", [1, 3])
def test_intrinsic_rejects_wrong_runtime_arity(operands):
    value = I.const(1, "float32")
    call = I.Call("tl.fmul", [value] * operands, ty="float32")
    with pytest.raises(ValueError, match="runtime operands"):
        call.validate()


def test_result_type_mismatch_is_rejected():
    value = I.const(1, "float32")
    call = I.Call("tl.fmul", [value, value], ty="float32", ty_args=[I.PrimType("int32")])
    with pytest.raises(TypeError, match="result type disagrees"):
        call.validate()


def test_missing_result_type_is_not_silently_void():
    with pytest.raises(TypeError, match="explicit result type"):
        I.Call("tl.get_lane_idx", [])


def test_metadata_payload_does_not_change_runtime_arity():
    value = I.const(1, "float32")
    dynamic = I.Var("tag", "int32")
    attrs = {"schema_version": 1, "operand_count": 2, "annotation_schema": ["map", [["str", "tag"], ["expr", 0]]]}
    call = I.Call("tl.fmul", [value, value, I.Tuple([dynamic])], attrs=attrs, ty="float32")
    call.validate()
    assert call.args[-1].fields[0].same_as(dynamic)


@pytest.mark.parametrize("invalid", ["missing_schema", "missing_count", "wrong_version", "wrong_count", "non_tuple", "dangling_expr"])
def test_metadata_schema_errors_are_rejected(invalid):
    value = I.const(1, "float32")
    attrs = {"schema_version": 1, "operand_count": 2, "annotation_schema": ["map", [["str", "tag"], ["expr", 0]]]}
    payload = I.Tuple([I.const(1)])
    if invalid == "missing_schema":
        del attrs["annotation_schema"]
    elif invalid == "missing_count":
        del attrs["operand_count"]
    elif invalid == "wrong_version":
        attrs["schema_version"] = 2
    elif invalid == "wrong_count":
        attrs["operand_count"] = 1
    elif invalid == "non_tuple":
        payload = I.const(1)
    elif invalid == "dangling_expr":
        attrs["annotation_schema"] = ["expr", 1]
    call = I.Call("tl.fmul", [value, value, payload], attrs=attrs, ty="float32")
    with pytest.raises((ValueError, TypeError)):
        call.validate()
