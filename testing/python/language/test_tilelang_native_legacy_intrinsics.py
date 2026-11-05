"""Retained public GPU/profile/q spellings construct real native operations."""

from fractions import Fraction
import inspect
import math

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

import tilelang.language as T
from tilelang.language.tir import op as facade
from tilelang.testing.packed import compile_packed_function


NATIVE_TARGETS = [
    "c",
    pytest.param("llvm", marks=pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM support is not built")),
]


def _compile_semantics(function, target):
    if target == "c":
        return compile_packed_function(function, tilelang_owned=False)
    return tirx.compile(function, target={"kind": "llvm", "mcpu": "generic"})


def _gpu_case(name):
    value = ir.Var("value", "float32")
    fragment = ir.Var("fragment", ir.PtrType("float16", "wmma.matrix_a"))
    pointer = ir.Var("pointer", ir.PtrType("float16"))
    accumulator = ir.Var("accumulator", ir.PtrType("float32", "wmma.accumulator"))
    stride = ir.Var("stride", "int32")
    mask = ir.const(0xFFFFFFFF, "uint32")
    cases = {
        "tvm_thread_invariant": ([value > 0], "tirx.gpu_thread_invariant", "bool", 1),
        "tvm_warp_shuffle": ([mask, value, 0, 32, 32], "tirx.gpu_warp_shuffle", "float32", 3),
        "tvm_warp_shuffle_up": ([mask, value, 1, 32, 32], "tirx.gpu_warp_shuffle_up", "float32", 3),
        "tvm_warp_shuffle_down": ([mask, value, 1, 32, 32], "tirx.gpu_warp_shuffle_down", "float32", 3),
        "tvm_warp_activemask": ([], "tirx.gpu_warp_activemask", "uint32", 3),
        "tvm_load_matrix_sync": ([fragment, 16, 16, 16, 0, pointer, stride, "row_major"], "tl.wmma_load_matrix_sync", "void", 3),
        "tvm_mma_sync": ([accumulator, 0, fragment, 0, fragment, 0, accumulator, 0], "tirx.gpu_mma_sync", "void", 3),
        "tvm_bmma_sync": ([accumulator, 0, fragment, 0, fragment, 0, accumulator, 0], "tirx.cuda.bmma_sync", "void", 3),
        "tvm_fill_fragment": ([accumulator, 16, 16, 16, 0, ir.const(0, "float32")], "tirx.gpu_fill_fragment", "void", 3),
        "tvm_store_matrix_sync": ([accumulator, 16, 16, 16, 0, pointer, stride, "row_major"], "tirx.gpu_store_matrix_sync", "void", 3),
    }
    return cases[name], [value, fragment, pointer, accumulator, stride]


GPU_NAMES = [
    "tvm_thread_invariant",
    "tvm_warp_shuffle",
    "tvm_warp_shuffle_up",
    "tvm_warp_shuffle_down",
    "tvm_warp_activemask",
    "tvm_load_matrix_sync",
    "tvm_mma_sync",
    "tvm_bmma_sync",
    "tvm_fill_fragment",
    "tvm_store_matrix_sync",
]


@pytest.mark.parametrize("name", GPU_NAMES)
@pytest.mark.parametrize("keywords", [False, True])
def test_public_gpu_calls_preserve_signature_type_and_native_validation(name, keywords):
    (args, target_name, dtype, effect), params = _gpu_case(name)
    operator = getattr(T, name)
    if keywords:
        parameter_names = inspect.signature(getattr(facade, name)).parameters
        call = operator(**dict(zip(parameter_names, args)))
    else:
        call = operator(*args)
    assert isinstance(call, ir.Call)
    assert call.op.name == target_name
    assert call.ty == ir.PrimType(dtype)
    assert int(call.op.get_attr("TCallEffectKind")) == effect
    assert len(call.ty_args) == 0
    call.validate()
    assert tirx.analysis.check_well_formed(tirx.Function(params, [ir.Evaluate(call)], ir.NoneType()))
    restored = ir.load_json(ir.save_json(call))
    restored.validate()
    assert tvm_ffi.structural_equal(call, restored, map_free_vars=True)
    if name in ("tvm_load_matrix_sync", "tvm_store_matrix_sync"):
        assert call.args[6].same_as(params[-1])


@pytest.mark.parametrize("name", GPU_NAMES)
def test_public_gpu_wrong_arity_is_not_silently_discarded(name):
    (args, _, _, _), _ = _gpu_case(name)
    wrong_args = args[:-1] if args else [0]
    with pytest.raises(TypeError):
        getattr(T, name)(*wrong_args)


def test_public_shuffle_rejects_non_integer_mask_in_native_contract():
    with pytest.raises((TypeError, ValueError)):
        call = T.tvm_warp_shuffle(ir.const(1, "float32"), ir.const(7, "int32"), 0, 32, 32)
        call.validate()


@pytest.mark.parametrize("dtype", ["bool", "int32", "int64", "float32"])
def test_thread_invariant_preserves_the_primitive_operand_type(dtype):
    value = ir.Var("value", dtype)
    call = T.tvm_thread_invariant(value)
    assert call.ty == value.ty
    assert call.args[0].same_as(value)
    call.validate()


def test_native_lowering_reaches_tilelang_cuda_emitter():
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})
    for name in GPU_NAMES:
        (args, _, _, _), params = _gpu_case(name)
        call = getattr(T, name)(*args)
        function = tirx.Function(params, [ir.Evaluate(call)], ir.PrimType("void")).with_attr("global_symbol", name)
        function = function.with_attr("calling_conv", 2).with_attr("target", target)
        lowered = tirx.transform.LowerIntrin()(ir.IRModule({name: function}))
        source = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(lowered, target).inspect_source("cuda")
        if name == "tvm_warp_activemask":
            assert "__activemask" in source
        elif name.startswith("tvm_warp_shuffle"):
            assert "shfl" in source
        elif name != "tvm_thread_invariant":
            assert "nvcuda::wmma::" in source
    # Raw pointer parameters exercise emitter dispatch, not a valid WMMA
    # fragment declaration or device execution.


@pytest.mark.parametrize("name", ["start_profile_intrinsic", "end_profile_intrinsic"])
def test_profile_markers_are_typed_and_pure(name):
    call = getattr(T, name)(id=17)
    assert call.op.name == "tl." + name
    assert call.ty == ir.PrimType("void")
    assert int(call.op.get_attr("TCallEffectKind")) == int(ir.CallEffectKind.Pure)
    call.validate()
    function = tirx.Function([], [ir.Evaluate(call)], ir.NoneType())
    assert tirx.analysis.check_well_formed(function)
    cleaned = tirx.transform.RemoveNoOp()(ir.IRModule({"main": function}))
    calls = []
    tvm_ffi.structural_walk(cleaned, (ir.Call, calls.append))
    assert not calls


@pytest.mark.parametrize("name", ["start_profile_intrinsic", "end_profile_intrinsic"])
def test_profile_llvm_ignores_id_without_evaluating_it(name):
    poison = tirx.call_extern("legacy_profile_id_must_not_execute", ty="int32")
    call = getattr(T, name)(poison)
    target = tvm_tirx.target.Target({"kind": "llvm", "mcpu": "generic"})
    function = tirx.Function([], [ir.Evaluate(call)], ir.NoneType()).with_attr("target", target)
    lowered = tirx.transform.LowerIntrin()(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(lowered)
    calls = []
    tvm_ffi.structural_walk(lowered, (ir.Call, calls.append))
    assert not calls
    # The ordinary C/CUDA emitters did not implement the Hexagon hook. The
    # LLVM-only rule must not silently generalize that backend capability.
    function = function.with_attr("target", tvm_tirx.target.Target("c"))
    lowered_c = tirx.transform.LowerIntrin()(ir.IRModule({"main": function}))
    calls = []
    tvm_ffi.structural_walk(lowered_c, (ir.Call, calls.append))
    assert any(getattr(node.op, "name", "") == "tl." + name for node in calls)


@pytest.mark.parametrize("name", ["start_profile_intrinsic", "end_profile_intrinsic"])
def test_profile_id_is_an_integer_expression(name):
    with pytest.raises((TypeError, ValueError)):
        getattr(T, name)(1.5).validate()


def _q_expected(kind, args, value):
    if kind == "q_multiply_shift":
        multiplier, q, shift = args
        exponent = shift - 1 if multiplier == 1 << 30 else shift - q
        multiplier = 1 if multiplier == 1 << 30 else multiplier
    else:
        multiplier, left, right, q, use_left, _use_right = args
        exponent = (left if use_left else 0) - right - q
    scale = Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**-exponent)
    return math.floor(value * multiplier * scale + Fraction(1, 2))


@pytest.mark.parametrize("target", NATIVE_TARGETS)
def test_quantized_public_operations_execute_rounding_and_final_narrowing(target):
    cases = [
        ("q_multiply_shift", [multiplier, 31, shift])
        for multiplier in [1 << 30, 1457846997, -1457846997, 0]
        for shift in [-2, -1, 0, 1, 2]
        if not (multiplier == 1 << 30 and shift == 1)
    ]
    cases += [("q_multiply_shift", [1 << 30, 23, 0]), ("q_multiply_shift", [1457846997, 23, 0])]
    cases += [
        ("q_multiply_shift_per_axis", [1457846997, left, right, 31, use_left, use_right])
        for left, right in [(0, 0), (2, 0), (0, 2)]
        for use_left in [False, True]
        for use_right in [False, True]
    ]
    value = ir.Var("value", "int32")
    output = ir.Var("output", tirx.TensorType((len(cases),), "int32"))
    stores = [ir.TensorStore(output, [ir.const(i, "int64")], getattr(T, kind)(value, *args)) for i, (kind, args) in enumerate(cases)]
    function = tirx.Function([value, output], stores, ir.NoneType()).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    executable = _compile_semantics(function, target)
    result = tvm_tirx.runtime.tensor(np.zeros(len(cases), dtype=np.int32))
    for x in [-32768, -4097, -17, -3, -1, 0, 1, 3, 17, 4097, 32767]:
        executable(x, result)
        assert result.numpy().tolist() == [_q_expected(kind, args, x) for kind, args in cases]


@pytest.mark.parametrize("target", NATIVE_TARGETS)
def test_quantized_non_overflowing_int64_product_narrows_to_signed_int32(target):
    x = ir.Var("x", "int32")
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    expression = T.q_multiply_shift(x, 1457846997, 23, 0)
    function = tirx.Function([x, output], [ir.TensorStore(output, [ir.const(0, "int64")], expression)], ir.NoneType())
    function = function.with_attr("global_symbol", "main")
    executable = _compile_semantics(function, target)
    result = tvm_tirx.runtime.tensor(np.zeros(1, dtype=np.int32))
    for value in [-2147483648, 2147483647]:
        expected = _q_expected("q_multiply_shift", [1457846997, 23, 0], value)
        expected = ((expected + (1 << 31)) % (1 << 32)) - (1 << 31)
        executable(value, result)
        assert int(result.numpy()[0]) == expected


@pytest.mark.parametrize("target", NATIVE_TARGETS)
def test_quantized_dynamic_shift_multiplier_and_left_flag_execute(target):
    x, shift, multiplier, left, right = [ir.Var(name, "int32") for name in ("x", "shift", "multiplier", "left", "right")]
    use_left = ir.Var("use_left", "bool")
    output = ir.Var("output", tirx.TensorType((2,), "int32"))
    expressions = [
        T.q_multiply_shift(x, 1457846997, 31, shift),
        T.q_multiply_shift_per_axis(x, multiplier, left, right, 31, use_left, False),
    ]
    stores = [ir.TensorStore(output, [ir.const(i, "int64")], expression) for i, expression in enumerate(expressions)]
    function = tirx.Function([x, shift, multiplier, left, right, use_left, output], stores, ir.NoneType())
    function = function.with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    executable = _compile_semantics(function, target)
    result = tvm_tirx.runtime.tensor(np.zeros(2, dtype=np.int32))
    for value, amount, mult, lshift, rshift, flag in [
        (-17, -2, 1457846997, 2, 0, False),
        (-3, 2, 1457846997, 2, 0, True),
        (3, 0, -1457846997, 0, 2, True),
        (32767, -1, 1457846997, 2, 2, True),
    ]:
        executable(value, amount, mult, lshift, rshift, flag, result)
        expected = [
            _q_expected("q_multiply_shift", [1457846997, 31, amount], value),
            _q_expected("q_multiply_shift_per_axis", [mult, lshift, rshift, 31, flag, False], value),
        ]
        assert result.numpy().tolist() == expected


def test_quantized_vector_lanes_and_right_flag_contract():
    x = ir.Var("x", "int32x4")

    def broadcast(value):
        return prim.Broadcast(ir.const(value, "int32"), 4)

    q = broadcast(31)
    multiplier = broadcast(1457846997)
    ordinary = T.q_multiply_shift(x, multiplier, q, broadcast(-1))
    per_axis = T.q_multiply_shift_per_axis(x, multiplier, broadcast(0), broadcast(2), q, ir.const(False), ir.const(False))
    for expression in (ordinary, per_axis):
        assert expression.ty == ir.PrimType("int32x4")
        assert tirx.analysis.check_well_formed(tirx.Function([x], [ir.Evaluate(expression)], ir.NoneType()))
        restored = ir.load_json(ir.save_json(expression))
        assert tvm_ffi.structural_equal(expression, restored, map_free_vars=True)
    poison_flag = tirx.call_extern("quantized_unused_right_flag", ty="bool")
    ignored = T.q_multiply_shift_per_axis(ir.Var("scalar", "int32"), 1457846997, 0, 2, 31, False, poison_flag)
    calls = []
    tvm_ffi.structural_walk(ignored, (ir.Call, calls.append))
    assert not any("quantized_unused_right_flag" in node.script() for node in calls)


def test_quantized_rejects_the_old_invalid_power_of_two_shift():
    with pytest.raises(ValueError, match="negative legacy rounding shift"):
        T.q_multiply_shift(ir.Var("value", "int32"), 1 << 30, 31, 1)


def test_quantized_rejects_nonconstant_generic_multiplier_and_wrong_shift_type():
    x = ir.Var("x", "int32")
    with pytest.raises(TypeError, match="constant"):
        T.q_multiply_shift(x, ir.Var("multiplier", "int32"), 31, 0)
    with pytest.raises(TypeError, match="int32"):
        T.q_multiply_shift(x, 1457846997, 31, ir.const(1, "int64"))


def test_public_intrinsics_retain_user_source_location():
    def source(value: T.int32):
        T.start_profile_intrinsic(0)
        T.evaluate(T.q_multiply_shift(value, 1457846997, 31, -1))
        T.evaluate(T.tvm_thread_invariant(value > 0))
        T.end_profile_intrinsic(0)

    function = T.prim_func(source)
    assert tirx.analysis.check_well_formed(function)
    evaluations = []
    tvm_ffi.structural_walk(function, (ir.Evaluate, evaluations.append))
    _, first_line = inspect.getsourcelines(source)
    for index, statement in enumerate(evaluations):
        assert isinstance(statement.value.loc, ir.SourceLoc)
        assert statement.value.loc.start_line == first_line + index + 1
    assert len(evaluations) == 4
