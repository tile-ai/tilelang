"""Native warp shuffle types remain valid across inference and serialization."""

from tilelang.language import _construction

import pytest
from tvm_tirx import ir

import tilelang.language as T


SHUFFLES = [
    (T.shfl_sync, "tl.shfl_sync"),
    (T.shfl_xor, "tl.shfl_xor_sync"),
    (T.shfl_up, "tl.shfl_up_sync"),
    (T.shfl_down, "tl.shfl_down_sync"),
]


@pytest.mark.parametrize("shuffle,op_name", SHUFFLES)
@pytest.mark.parametrize("dtype", ["int32", "float32", "bfloat16"])
def test_shuffle_native_value_type_roundtrip(shuffle, op_name, dtype):
    value = ir.Var("value", dtype)
    result = shuffle(value, 2, width=8, mask=0xFF)
    assert result.op.name == op_name
    ir.assert_structural_equal(result.ty, value.ty)
    result.validate()
    assert _construction.side_effect(result) == ir.CallEffectKind.Opaque
    assert result.args[1].same_as(value)
    assert int(result.args[0]) == 0xFF
    assert str(result.args[0].ty.dtype) == "uint32"
    assert int(result.args[2]) == 2
    assert int(result.args[3]) == 8

    restored = ir.load_json(ir.save_json(result))
    restored.validate()
    ir.assert_structural_equal(ir.reinfer_type(restored), value.ty)
    # A nested arithmetic consumer must work before an assignment or a pass
    # has a chance to attach a type annotation to the shuffle result.
    nested = T.max(result, value)
    ir.assert_structural_equal(nested.ty, value.ty)


@pytest.mark.parametrize("shuffle,op_name", SHUFFLES)
def test_shuffle_literal_value_normalized_once(shuffle, op_name):
    result = shuffle(7, 1)
    assert result.op.name == op_name
    assert str(result.ty.dtype) == "int32"
    assert int(result.args[1]) == 7
    result.validate()


@pytest.mark.parametrize("_,op_name", SHUFFLES)
@pytest.mark.parametrize("wrong_type_arg", [False, True])
def test_shuffle_rejects_result_type_different_from_value(_, op_name, wrong_type_arg):
    value = ir.Var("value", "float32")
    args = [ir.const(0xFFFFFFFF, "uint32"), value, ir.const(1), ir.const(8)]
    if wrong_type_arg:
        result = ir.Call(op_name, args, ty_args=[ir.PrimType("int32")])
    else:
        result = ir.Call(op_name, args, ty=ir.PrimType("int32"))
    with pytest.raises(TypeError, match="(result type|shuffled value type)"):
        result.validate()


@pytest.mark.parametrize("_,op_name", SHUFFLES)
def test_shuffle_raw_native_call_infers_without_explicit_result_type(_, op_name):
    value = ir.Var("value", "float64")
    result = ir.Call(op_name, [ir.const(0xFFFFFFFF, "uint32"), value, ir.const(1), ir.const(16)])
    result.validate()
    ir.assert_structural_equal(result.ty, value.ty)
    ir.assert_structural_equal(ir.reinfer_type(result), value.ty)
