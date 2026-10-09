"""Target-independent legalization of documented integer per-axis shift flags."""

import pytest

import tilelang.language as T
import tilelang.cpu.language as CPU
from tilelang import tvm
from tvm import tirx


OPERATORS = [
    pytest.param(T.q_multiply_shift_per_axis, id="default-language"),
    pytest.param(CPU.q_multiply_shift_per_axis, id="cpu-language"),
]

FLAGS = [
    pytest.param(0, id="integer-false"),
    pytest.param(1, id="integer-true"),
    pytest.param(False, id="boolean-false"),
    pytest.param(True, id="boolean-true"),
    pytest.param(tirx.IntImm("int8", 0), id="int8-false"),
    pytest.param(tirx.IntImm("int8", 1), id="int8-true"),
    pytest.param(tirx.IntImm("int32", -1), id="negative-integer-true"),
    pytest.param(tirx.IntImm("int64", 1 << 32), id="wide-integer-true"),
    pytest.param(tirx.IntImm("uint64", 1 << 32), id="wide-unsigned-true"),
]


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("flag", FLAGS)
def test_per_axis_integer_flag_legalization(operator, flag):
    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    call = operator(x, y, 2, 7, 15, flag, 1)

    assert call.args[5].dtype == "bool"
    assert call.args[6].dtype == "bool"
    lowered = call.op.get_attr("default.FLegalize")(call)
    assert lowered.dtype == "int32"

    # Check the actual legalized arithmetic, including negative inputs and a
    # runtime multiplier, against an independent integer reference.
    enabled = bool(flag.value) if isinstance(flag, tirx.IntImm) else bool(flag)
    analyzer = tvm.arith.Analyzer()
    for x_value in (-1000, -31, -1, 0, 1, 31, 1000):
        for y_value in (1, 1033, 1 << 20):
            actual = analyzer.simplify(
                tirx.stmt_functor.substitute(
                    lowered,
                    {x: tirx.IntImm("int32", x_value), y: tirx.IntImm("int32", y_value)},
                )
            )
            shifted = x_value << 2 if enabled else x_value
            expected = (shifted * y_value + (1 << 21)) >> 22
            assert isinstance(actual, tirx.IntImm)
            assert actual.value == expected


@pytest.mark.parametrize("operator", OPERATORS)
def test_per_axis_boolean_flags_preserve_intrinsic(operator):
    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    actual = operator(x, y, 2, 7, 15, True, False)
    expected = tirx.q_multiply_shift_per_axis(x, y, 2, 7, 15, True, False)
    tvm.ir.assert_structural_equal(actual, expected)


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint64"])
def test_per_axis_symbolic_integer_flag_uses_nonzero(operator, dtype):
    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    flag = tirx.Var("flag", dtype)
    call = operator(x, y, 2, 7, 15, flag, 1)
    tvm.ir.assert_structural_equal(call.args[5], flag != tirx.IntImm(dtype, 0))
    lowered = call.op.get_attr("default.FLegalize")(call)
    values = [0, 1]
    if dtype.startswith("int"):
        values.append(-1)
    if dtype.endswith("64"):
        values.append(1 << 32)
    analyzer = tvm.arith.Analyzer()
    for value in values:
        actual = analyzer.simplify(
            tirx.stmt_functor.substitute(
                lowered,
                {x: tirx.IntImm("int32", -31), y: tirx.IntImm("int32", 1 << 20), flag: tirx.IntImm(dtype, value)},
            )
        )
        expected = -31 if value else -8
        assert isinstance(actual, tirx.IntImm)
        assert actual.value == expected
