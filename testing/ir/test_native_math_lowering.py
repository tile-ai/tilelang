"""Native scalar math lowering and C numeric oracles."""

from tilelang.testing.packed import compile_packed_function

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx


def _call(name, dtype, *args):
    result = ir.Call(name, args, ty_args=[ir.PrimType(dtype)])
    result.validate()
    return result


def _lower(call, target):
    return ir.Op.get_attr_map(target + ".FLowerIntrinsic")[call.op](call)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "uint32"])
def test_native_clamp_matches_numpy_at_boundaries(dtype):
    size = 7
    x = ir.Var("X", tirx.TensorType([size], dtype))
    y = ir.Var("Y", tirx.TensorType([size], dtype))
    i = ir.Var("i", "int32")
    imm = ir.FloatImm if dtype.startswith("float") else ir.IntImm
    value = _call("tl.clamp", dtype, ir.TensorLoad(x, [i]), imm(dtype, 2), imm(dtype, 6))
    lowered = _lower(value, "llvm")
    function = tirx.Function(
        [x, y],
        [ir.For(i, 0, size, 1, ir.ForKind.DEFAULT, [ir.TensorStore(y, [i], lowered)])],
        ir.NoneType(),
    ).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    compiled = compile_packed_function(function, tilelang_owned=False)
    values = np.array([0, 1, 2, 4, 6, 7, 15], dtype=dtype)
    if dtype.startswith("float"):
        values[-2:] = [np.inf, np.nan]
    source = tvm_tirx.runtime.tensor(values)
    destination = tvm_tirx.runtime.tensor(np.zeros(size, dtype=dtype))
    compiled(source, destination)
    np.testing.assert_equal(destination.numpy(), np.clip(values, 2, 6))


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "bfloat16", "float8_e5m2"])
def test_infinity_keeps_supported_native_dtype(dtype):
    result = _lower(_call("tl.infinity", dtype, ir.IntImm("int32", 0)), "default")
    assert result.ty == ir.PrimType(dtype)
    assert np.isinf(float(result.value))


def test_infinity_rejects_finite_only_fp8():
    with pytest.raises(ValueError, match="Cannot decide infinity"):
        _lower(_call("tl.infinity", "float8_e4m3fn", ir.IntImm("int32", 0)), "default")


def test_pow_of_int_rejects_dynamic_exponent():
    with pytest.raises((TypeError, ValueError)):
        _lower(_call("tl.pow_of_int", "float32", ir.FloatImm("float32", 2), ir.Var("e", "int32")), "cuda")


def test_logical_reductions_keep_typed_pointer_and_read_effect():
    pointer = ir.Var("data", ir.PtrType("bool", "global"))
    count = ir.Var("count", "int32")
    for name in ("tl.any_of", "tl.all_of"):
        call = _call(name, "bool", pointer, count)
        lowered = _lower(call, "cuda")
        assert lowered.args[1].same_as(pointer)
        assert lowered.args[2].same_as(count)
        assert lowered.ty == ir.PrimType("bool")
        assert ir.Op.get_attr_map("TCallEffectKind")[call.op] == ir.CallEffectKind.ReadState


def test_logical_reductions_reject_an_integer_as_a_pointer():
    with pytest.raises((TypeError, ValueError)):
        _lower(_call("tl.any_of", "bool", ir.IntImm("int64", 123), ir.IntImm("int32", 4)), "cuda")


def test_clamp_rejects_mismatched_operand_type_at_the_native_boundary():
    with pytest.raises(TypeError, match="matching types"):
        _call(
            "tl.clamp",
            "float32",
            ir.FloatImm("float32", 2.0),
            ir.IntImm("int32", 0),
            ir.FloatImm("float32", 3.0),
        )


def test_clamp_accepts_inferred_result_and_preserves_explicit_metadata_uses():
    x, hint = ir.Var("x", "float32"), ir.Var("n", "int32")
    plain = ir.Call("tl.clamp", [x, ir.FloatImm("float32", 0), ir.FloatImm("float32", 1)])
    plain.validate()
    assert plain.ty == x.ty
    annotated = tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(x.ty, plain.op, plain.args, {"tile_extent": hint}, ir.UnknownLoc())
    annotated.validate()
    assert annotated.args[-1].fields[0].same_as(hint)
    assert _lower(annotated, "llvm").ty == x.ty
