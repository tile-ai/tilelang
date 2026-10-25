import numpy as np
import pytest
from tvm_tirx import ir, runtime, tirx

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


@pytest.mark.parametrize(
    "name,arity,effect",
    [
        ("tl.clamp", 3, ir.CallEffectKind.Pure),
        ("tl.region", None, ir.CallEffectKind.Pure),
        ("tl.ptx_cp_async", None, ir.CallEffectKind.Opaque),
        ("tl.ptx_stmatrix", None, ir.CallEffectKind.Opaque),
        ("tl.sync_warp", None, ir.CallEffectKind.Opaque),
        ("tl.rng_init", 4, ir.CallEffectKind.Opaque),
        ("tl.ds_read_tr16_b64", 1, ir.CallEffectKind.Pure),
        ("tl.tvm_mfma", 12, ir.CallEffectKind.Opaque),
        ("tl.cooperative_tensor_fill", 5, ir.CallEffectKind.Opaque),
        ("tl.simd.pset", None, ir.CallEffectKind.Opaque),
    ],
)
def test_native_intrinsic_contract(name, arity, effect):
    operation = ir.Op.get(name)
    assert ir.Op.get_attr_map("TCallEffectKind")[operation] == effect
    if arity is None:
        assert operation.var_args_info is not None
        assert operation.var_args_info.name == "args"
        arity = 3
    else:
        assert len(operation.args_info) == arity
        assert [argument.name for argument in operation.args_info] == [f"arg{index}" for index in range(arity)]
        assert operation.var_args_info is None
        with pytest.raises(TypeError, match="argument"):
            ir.Call(name, [0] * (arity + 1), ty="int32").validate()
    ir.Call(name, [0] * arity, ty="int32").validate()


def lower_intrinsic(name, arguments, dtype, backend):
    expression = ir.Call(name, arguments, ty=dtype)
    return ir.Op.get_attr_map(f"{backend}.FLowerIntrinsic")[expression.op](expression)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64", "float8_e5m2"])
def test_native_infinity(dtype):
    lowered = lower_intrinsic("tl.infinity", [0], dtype, "default")
    assert lowered.ty == ir.PrimType(dtype)
    assert np.isposinf(float(lowered))


@pytest.mark.parametrize("dtype", ["int32", "uint32", "float8_e4m3fn", "float4_e2m1fn"])
def test_infinity_rejects_nonrepresentable_types(dtype):
    with pytest.raises(TypeError, match="infinity"):
        lower_intrinsic("tl.infinity", [0], dtype, "default")


def test_pow_of_int_retains_typed_operand_and_location():
    value = ir.Var("value", "float32")
    lowered = lower_intrinsic("tl.pow_of_int", [value, -3], "float32", "cuda")
    assert lowered.op.name == "tirx.call_pure_extern"
    assert lowered.args[0].value == "tl::pow_of_int<-3>"
    assert lowered.args[1].same_as(value)
    assert lowered.ty == value.ty
    lowered.validate()
    with pytest.raises(ValueError, match="constant exponent"):
        lower_intrinsic("tl.pow_of_int", [value, ir.Var("exponent", "int32")], "float32", "cuda")


def test_round_integer_identity_and_float_helper():
    value = ir.Var("integer", "int32")
    assert lower_intrinsic("tl.round_ties_away_from_zero", [value], "int32", "cuda").same_as(value)
    value = ir.Var("floating", "float32")
    lowered = lower_intrinsic("tl.round_ties_away_from_zero", [value], "float32", "cuda")
    assert lowered.op.name == "tirx.call_pure_extern"
    assert lowered.args[0].value == "tl::RoundTiesAwayFromZero"
    assert lowered.args[1].same_as(value)
    lowered.validate()


@pytest.mark.parametrize("name,symbol", [("tl.any_of", "tl::Any"), ("tl.all_of", "tl::All")])
def test_logical_reductions_keep_pointer_type_and_read_effect(name, symbol):
    tensor = TensorView.create("values", [16], "bool", scope="local")
    expression = ir.Call(name, [tensor.pointer_at([2]), 8])
    assert expression.ty == ir.PrimType("bool")
    expression.validate()
    assert ir.Op.get_attr_map("TCallEffectKind")[expression.op] == ir.CallEffectKind.ReadState
    lowered = ir.Op.get_attr_map("cuda.FLowerIntrinsic")[expression.op](expression)
    assert lowered.args[0].value == symbol
    assert lowered.args[1].ty == ir.PtrType("bool", "local")
    lowered.validate()


def test_clamp_evaluates_each_argument_once():
    arguments = [ir.Call("tirx.call_extern", [ir.StringImm(name)], ty="float32") for name in ("value", "lower", "upper")]
    lowered = lower_intrinsic("tl.clamp", arguments, "float32", "c")
    calls = []
    stmt_functor.post_order_visit(
        lowered, lambda node: calls.append(node) if isinstance(node, ir.Call) and node.op.name == "tirx.call_extern" else None
    )
    assert len(calls) == 3
    for actual, expected in zip(calls, arguments):
        assert actual.same_as(expected)
    ir.assert_structural_equal(lowered, ir.load_json(ir.save_json(lowered)), map_free_vars=True)


def test_native_clamp_cpu_numerics():
    values = np.array([-np.inf, -3, -0.0, 0, 2, np.inf, np.nan, 1, 1], dtype="float32")
    lowers = np.array([-1, -1, -1, -1, -1, -1, -1, np.nan, -1], dtype="float32")
    uppers = np.array([1, 1, 1, 1, 1, 1, 1, 1, np.nan], dtype="float32")
    tensors = [TensorView.create(name, [len(values)], "float32") for name in ("values", "lowers", "uppers", "output")]
    index = ir.Var("index", "int32")
    expression = ir.Call("tl.clamp", [tensor.load(index) for tensor in tensors[:3]], ty="float32")
    body = ir.For(index, 0, len(values), 1, ir.ForKind.DEFAULT, tensors[3].store(index, expression))
    function = tirx.Function([tensor.tensor for tensor in tensors], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "clamp_native"
    )
    executable = tirx.compile(function, target="c")
    output = runtime.empty(values.shape, "float32")
    executable(runtime.tensor(values), runtime.tensor(lowers), runtime.tensor(uppers), output)
    expected = np.minimum(np.maximum(values, lowers), uppers)
    np.testing.assert_equal(output.numpy(), expected)
    assert np.signbit(output.numpy()[2])
