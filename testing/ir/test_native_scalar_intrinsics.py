"""Native scalar intrinsics carry verifiable result types and effects."""

import pytest
from tvm_tirx import ir, tirx

import tilelang.cuda.language as T


@pytest.mark.parametrize("operation", ["sum", "min", "max", "bitand", "bitor"])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_warp_reduction_infers_from_the_operand(operation, dtype):
    value = ir.Var("value", dtype)
    for call in (
        ir.Call("tl.warp_reduce_" + operation, [value]),
        getattr(T, "warp_reduce_" + operation)(value),
    ):
        assert call.ty == value.ty
        assert ir.reinfer_type(call) == value.ty
        restored = ir.load_json(ir.save_json(call))
        assert restored.ty == value.ty
        call.validate()
        restored.validate()
        assert call.op.get_attr("TCallEffectKind") == ir.CallEffectKind.Opaque
        function = tirx.Function([value], [ir.Evaluate(call)], ir.NoneType())
        assert tirx.analysis.check_well_formed(function)


def test_explicit_warp_result_must_match_operand_type():
    value = ir.Var("value", "int64")
    with pytest.raises((TypeError, RuntimeError), match="type|Type"):
        call = ir.Call("tl.warp_reduce_sum", [value], ty=ir.PrimType("int32"))
        call.validate()
        tirx.analysis.check_well_formed(tirx.Function([value], [ir.Evaluate(call)], ir.NoneType()))


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_rng_types_survive_native_serialization_and_inference(bits, distribution):
    calls = [T.rng_init(17, seq=5, off=3), T.rng_rand(), T.rng_rand_float(bits, distribution)]
    expected = ["void", "uint32", f"float{bits}"]
    for call, dtype in zip(calls, expected):
        assert call.ty == ir.PrimType(dtype)
        assert ir.reinfer_type(call) == call.ty
        assert ir.load_json(ir.save_json(call)).ty == call.ty
        call.validate()
        assert call.op.get_attr("TCallEffectKind") == ir.CallEffectKind.Opaque
    function = tirx.Function([], [ir.Evaluate(call) for call in calls], ir.NoneType())
    assert tirx.analysis.check_well_formed(function)


@pytest.mark.parametrize(
    "name,args,ty",
    [
        ("tl.warp_reduce_sum", [], "float32"),
        ("tl.rng_rand", [ir.IntImm("int32", 1)], "uint32"),
        ("tl.rng_rand_float", [], "float32"),
        ("tl.rng_init", [], "void"),
    ],
)
def test_intrinsic_arity_is_not_hidden_by_type_arguments(name, args, ty):
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        call = ir.Call(name, args, ty_args=[ir.PrimType(ty)])
        call.validate()
        tirx.analysis.check_well_formed(tirx.Function([], [ir.Evaluate(call)], ir.NoneType()))
