import pytest

from tilelang._compat import ir, sym
from tvm_tirx.ir import prim


@pytest.mark.parametrize(
    "dtype,lower,upper,expected",
    [
        ("int8", 128, 255, (-128, -1)),
        ("int8", 127, 128, (-128, 127)),
        ("uint8", -3, -1, (253, 255)),
        ("uint8", -1, 0, (0, 255)),
        ("bool", 2, 5, (1, 1)),
        ("bool", -5, -2, (1, 1)),
        ("bool", -1, 1, (0, 1)),
    ],
)
def test_cast_bounds_enclose_wrapping_values(dtype, lower, upper, expected):
    analyzer = sym.Analyzer()
    source = ir.Var("source", "int32")
    analyzer.bind(source, ir.Range(lower, upper + 1))
    bound = analyzer.const_int_bound(prim.Cast(dtype, source))
    assert (bound.min_value, bound.max_value) == expected


@pytest.mark.parametrize(
    "dtype,value,expected",
    [("int8", 255, -1), ("uint8", -1, 255), ("bool", 2, 1), ("bool", -2, 1), ("bool", 0, 0)],
)
def test_modular_cast_constants_follow_conversion(dtype, value, expected):
    analyzer = sym.Analyzer()
    source = ir.Var("source", "int32")
    analyzer.bind(source, ir.const(value))
    bound = analyzer.modular_set(prim.Cast("int32", prim.Cast(dtype, source)))
    assert (bound.coeff, bound.base) == (0, expected)


def test_narrow_cast_modular_stride_wraps():
    analyzer = sym.Analyzer()
    source = ir.Var("source", "int32")
    bound = analyzer.modular_set(prim.Cast("uint8", source * 3 + 2))
    assert (bound.coeff, bound.base) == (1, 0)


def test_narrow_cast_interval_does_not_drop_wrapped_values():
    analyzer = sym.Analyzer()
    source = ir.Var("source", "int32")
    interval = analyzer.int_set(prim.Cast("uint8", source), {source: sym.IntervalSet(127, 257)})
    assert int(interval.min_value) == 0
    assert int(interval.max_value) == 255


def test_boolean_cast_interval_is_bounded_without_integer_type_limits():
    analyzer = sym.Analyzer()
    source = ir.Var("source", "int32")
    interval = analyzer.int_set(prim.Cast("bool", source), {source: sym.IntervalSet(-1, 1)})
    assert int(interval.min_value) == 0
    assert int(interval.max_value) == 1


@pytest.mark.parametrize("lower,upper,lossless", [(-128, 128, True), (0, 128, True), (0, 129, False), (-129, 0, False)])
def test_nested_integer_cast_elimination_requires_lossless_inner_cast(lower, upper, lossless):
    analyzer = sym.Analyzer()
    variable = ir.Var("variable", "int32")
    analyzer.bind(variable, ir.Range(lower, upper))
    expression = prim.Cast("int32", prim.Cast("int8", variable))
    simplified = analyzer.simplify(expression)
    assert simplified.same_as(variable) == lossless
