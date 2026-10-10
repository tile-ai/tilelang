import pytest

from tilelang import tvm
from tilelang.jit.adapter.utils import pythonic_expr


@pytest.mark.parametrize("style", ["legacy", "cxx"])
@pytest.mark.parametrize("op, symbol", [(tvm.tirx.Add, "+"), (tvm.tirx.Sub, "-"), (tvm.tirx.Mul, "*")])
def test_cast_groups_compound_operand(style, op, symbol):
    a = tvm.tirx.Var("a", "float32")
    b = tvm.tirx.Var("b", "float32")
    expr = tvm.tirx.Cast("int32", op(a, b))

    assert pythonic_expr(expr, {"int32": "int32_t"}, expression_style=style) == f"(int32_t)(a {symbol} b)"


@pytest.mark.parametrize("style", ["legacy", "python", "cxx"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_ignored_cast_preserves_operand_precedence(style, side):
    a = tvm.tirx.Var("a", "int32")
    b = tvm.tirx.Var("b", "int32")
    c = tvm.tirx.Var("c", "int64")
    cast = tvm.tirx.Cast("int64", a + b)
    expr = tvm.tirx.Mul(cast, c) if side == "left" else tvm.tirx.Mul(c, cast)
    expected = "(a + b) * c" if side == "left" else "c * (a + b)"

    assert pythonic_expr(expr, ignore_cast=True, expression_style=style) == expected


def test_ignored_nested_cast_preserves_nonassociative_operand():
    a = tvm.tirx.Var("a", "int32")
    b = tvm.tirx.Var("b", "int32")
    c = tvm.tirx.Var("c", "int32")
    inner = tvm.tirx.Cast("int64", b - c)
    outer = tvm.tirx.Cast("int32", inner)

    assert pythonic_expr(a - outer, ignore_cast=True) == "a - (b - c)"


def test_cast_preserves_atomic_operand_and_dtype_mapping():
    a = tvm.tirx.Var("a", "int32")
    expr = tvm.tirx.Cast("int64", a)

    assert pythonic_expr(expr, {"int64": "long long"}, var_name_map={a: "arg0"}) == "(long long)arg0"
    assert pythonic_expr(expr, expression_style="cxx") == "(int64_t)a"
    assert pythonic_expr(expr, ignore_cast=True) == "a"


if __name__ == "__main__":
    pytest.main([__file__])
