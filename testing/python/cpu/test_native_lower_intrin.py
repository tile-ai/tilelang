"""Check TileLang integer lowering through its own C emitter and packed ABI."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx.script import tirx as T

from testing.python.cpu.test_native_codegen_c import build_own_c


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("divisor", [None, 3, 8, -3])
def test_floor_divmod_integer_boundaries(dtype, divisor, tmp_path):
    info = np.iinfo(dtype)
    numerators = np.array([info.min, info.min + 1, -17, -9, -1, 0, 1, 9, 17, info.max], dtype=dtype)
    denominators = np.array([3, -7, 8, -8, 3, -3, 1, 7, -3, 7], dtype=dtype)
    n = len(numerators)

    def select_divisor(value):
        return value if divisor is None else tvm.ir.const(divisor, dtype)

    @T.function
    def divide(
        source: T.Tensor((n,), dtype),
        divisors: T.Tensor((n,), dtype),
        quotients: T.Tensor((n,), dtype),
        remainders: T.Tensor((n,), dtype),
    ):
        for i in range(n):
            quotients[i] = T.floordiv(source[i], select_divisor(divisors[i]))
            remainders[i] = T.floormod(source[i], select_divisor(divisors[i]))

    target = tvm.target.Target("c", host="c")
    original = divide.with_attr("target", target).with_attr("global_symbol", "main")
    lowered = tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(tvm.IRModule({"main": original}))
    (tmp_path / "tilelang_lowered.py").write_text(lowered.script())
    assert "floordiv" not in lowered.script() and "floormod" not in lowered.script()
    module, _ = build_own_c(lowered["main"], tmp_path)
    quotient = tvm.runtime.tensor(np.zeros_like(numerators))
    remainder = tvm.runtime.tensor(np.zeros_like(numerators))
    module(tvm.runtime.tensor(numerators), tvm.runtime.tensor(denominators), quotient, remainder)

    # Use Python integers, not a native-width reference that could share an
    # intermediate overflow. No denominator is zero or the signed min / -1.
    ds = [int(d) for d in denominators] if divisor is None else [divisor] * n
    expected_q = np.array([int(a) // b for a, b in zip(numerators, ds)], dtype=dtype)
    expected_r = np.array([int(a) % b for a, b in zip(numerators, ds)], dtype=dtype)
    np.testing.assert_array_equal(quotient.numpy(), expected_q)
    np.testing.assert_array_equal(remainder.numpy(), expected_r)


def test_lower_intrin_requires_explicit_target():
    function = tvm.tirx.Function([], [], ret_type=tvm.ir.PrimType("void"))
    with pytest.raises(ValueError, match="target attribute"):
        tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(tvm.IRModule({"main": function}))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_isfinite_default_lowering_preserves_nan_and_infinity(dtype, tmp_path):
    def isfinite(value):
        return tvm.ir.Call("tl.isfinite", [value])

    @T.function
    def finite(source: T.Tensor((5,), dtype), output: T.Tensor((5,), "bool")):
        for i in range(5):
            output[i] = isfinite(source[i])

    target = tvm.target.Target("c", host="c")
    function = finite.with_attr("target", target).with_attr("global_symbol", "main")
    lowered = tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(tvm.IRModule({"main": function}))
    assert "tl.isfinite" not in lowered.script()
    module, _ = build_own_c(lowered["main"], tmp_path)
    values = np.array([-np.inf, np.inf, np.nan, 0, -1.25], dtype=dtype)
    output = tvm.runtime.tensor(np.zeros(5, dtype="bool"))
    module(tvm.runtime.tensor(values), output)
    np.testing.assert_array_equal(output.numpy(), np.isfinite(values))


def test_explicit_isfinite_intrinsic_rejects_non_float_input():
    with pytest.raises(TypeError, match="requires a floating-point value"):
        tvm.ir.Call("tl.isfinite", [tvm.ir.Var("x", "int32")])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scalar_c_math_lowering(dtype, tmp_path):
    def constant(value):
        return tvm.ir.const(value, dtype)

    @T.function
    def math(
        source: T.Tensor((7,), dtype),
        absolute: T.Tensor((7,), dtype),
        exp2: T.Tensor((7,), dtype),
        remainder: T.Tensor((7,), dtype),
        multiply_add: T.Tensor((7,), dtype),
    ):
        for i in range(7):
            absolute[i] = T.abs(source[i])
            exp2[i] = T.exp2(source[i])
            remainder[i] = T.fmod(source[i], constant(2))
            multiply_add[i] = T.fma(source[i], constant(2), constant(0.5))

    target = tvm.target.Target("c", host="c")
    function = math.with_attr("target", target).with_attr("global_symbol", "main")
    lowered = tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(tvm.IRModule({"main": function}))
    module, source = build_own_c(lowered["main"], tmp_path)
    suffix = "f" if dtype == "float32" else ""
    for name in ("fabs", "exp2", "fmod", "fma"):
        assert name + suffix + "(" in source
    values = np.arange(-3, 4, dtype=dtype)
    outputs = [tvm.runtime.tensor(np.zeros_like(values)) for _ in range(4)]
    module(tvm.runtime.tensor(values), *outputs)
    references = [np.abs(values), np.exp2(values), np.fmod(values, 2), values * 2 + 0.5]
    for actual, reference in zip(outputs, references):
        np.testing.assert_array_equal(actual.numpy(), reference)


def test_c_intrinsic_rules_do_not_enable_implicit_fma():
    x, y, z = [tvm.ir.Var(name, "float32") for name in ("x", "y", "z")]
    function = tvm.tirx.Function([x, y, z], [tvm.ir.Evaluate(x * y + z)], ret_type=tvm.ir.PrimType("void"))
    function = function.with_attr("target", tvm.target.Target("c"))
    lowered = tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(tvm.IRModule({"main": function}))
    assert isinstance(lowered["main"].body.seq[0].value, tvm.ir.prim.Add)
