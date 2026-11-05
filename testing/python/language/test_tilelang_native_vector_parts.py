"""The public vector spellings preserve operands and native result types."""

import inspect

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.language as T


@pytest.mark.parametrize("spelling", ["positional", "keyword"])
@pytest.mark.parametrize(
    "input_dtype, half_dtype, combined_dtype",
    [
        ("float32x4", "float32x2", "float32x8"),
        ("int8x16", "int8x8", "int8x32"),
        ("float16x2", "float16", "float16x4"),
        ("float32xvscalex4", "float32xvscalex2", "float32xvscalex8"),
    ],
)
@pytest.mark.parametrize(
    "name, native_name", [("vectorlow", "vector_low"), ("vectorhigh", "vector_high"), ("vectorcombine", "vector_combine")]
)
def test_public_vector_signatures_and_lane_types(name, native_name, input_dtype, half_dtype, combined_dtype, spelling):
    first = ir.Var("first", input_dtype)
    second = ir.Var("second", input_dtype)
    operands = [first, second] if name == "vectorcombine" else [first]
    result_dtype = combined_dtype if name == "vectorcombine" else half_dtype
    operator = getattr(T, name)
    result = operator(result_dtype, *operands) if spelling == "positional" else operator(*operands, dtype=result_dtype)
    assert isinstance(result, ir.Call)
    assert result.op.name == "tirx." + native_name
    assert result.ty == ir.PrimType(result_dtype)
    assert len(result.ty_args) == 0
    assert len(result.args) == len(operands)
    assert all(actual.same_as(expected) for actual, expected in zip(result.args, operands))
    assert int(result.op.get_attr("TCallEffectKind")) == int(ir.CallEffectKind.Pure)
    result.validate()
    restored = ir.load_json(ir.save_json(result))
    assert tvm_ffi.structural_equal(restored, result, map_free_vars=True)


@pytest.mark.parametrize("dtype", ["float32x2", T.float32x2, tvm_ffi.dtype("float32x2"), ir.PrimType("float32x2")])
def test_public_vector_result_dtype_values(dtype):
    value = ir.Var("value", "float32x4")
    result = T.vectorlow(value, dtype=dtype)
    assert result.ty == ir.PrimType("float32x2")
    assert result.args[0].same_as(value)


@pytest.mark.parametrize("name", ["vectorlow", "vectorhigh", "vectorcombine"])
@pytest.mark.parametrize("wrong_dtype", ["int32x2", "float32x4"])
def test_requested_result_dtype_must_match_native_lanes_and_element_type(name, wrong_dtype):
    value = ir.Var("value", "float32x4")
    operands = [value, value] if name == "vectorcombine" else [value]
    with pytest.raises(TypeError, match="requested dtype"):
        getattr(T, name)(*operands, dtype=wrong_dtype)


@pytest.mark.parametrize("name", ["vectorlow", "vectorhigh"])
@pytest.mark.parametrize("input_dtype", ["float32", "float32x3"])
def test_half_rejects_scalar_and_odd_lane_inputs(name, input_dtype):
    with pytest.raises(TypeError, match="even lane count"):
        getattr(T, name)("float32", ir.Var("value", input_dtype))


@pytest.mark.parametrize("second_dtype", ["int32x4", "float32x2"])
def test_combine_rejects_different_element_types_and_lane_counts(second_dtype):
    with pytest.raises(TypeError, match="equal input vector types"):
        T.vectorcombine("float32x8", ir.Var("first", "float32x4"), ir.Var("second", second_dtype))


@pytest.mark.parametrize("name", ["vectorlow", "vectorhigh", "vectorcombine"])
@pytest.mark.parametrize("input_type", [ir.PtrType("float32"), tirx.TensorType((4,), "float32")])
def test_vector_operators_reject_pointer_and_tensor_values(name, input_type):
    value = ir.Var("value", input_type)
    operands = [value, value] if name == "vectorcombine" else [value]
    with pytest.raises(TypeError):
        getattr(T, name)(*operands, dtype="float32x2")


def test_combine_scalar_lanes_keeps_native_contract():
    first, second = ir.Var("first", "int32"), ir.Var("second", "int32")
    result = T.vectorcombine(first, second, dtype="int32x2")
    assert result.ty == ir.PrimType("int32x2")
    assert result.args[0].same_as(first)
    assert result.args[1].same_as(second)


def test_vector_call_uses_current_user_source_location():
    def source(value: T.float32x4):
        T.evaluate(T.vectorlow(value, dtype="float32x2"))

    function = T.prim_func(source)
    calls = []
    tvm_ffi.structural_walk(function, (ir.Call, calls.append))
    parts = [call for call in calls if isinstance(call.op, ir.Op) and call.op.name == "tirx.vector_low"]
    assert len(parts) == 1
    assert parts[0].ty == ir.PrimType("float32x2")
    assert tirx.analysis.check_well_formed(function)
    _, first_line = inspect.getsourcelines(source)
    assert isinstance(parts[0].loc, ir.SourceLoc)
    assert parts[0].loc.start_line == first_line + 1
