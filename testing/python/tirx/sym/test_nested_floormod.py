import pytest

from tilelang._compat import ir, sym
from tvm_tirx.ir import prim


@pytest.mark.parametrize("inner", [-12, -6, -2, 2, 6, 12])
@pytest.mark.parametrize("outer", [-2, 2])
def test_nested_floormod_with_divisible_moduli(inner, outer):
    variable = ir.Var("variable", "int64")
    analyzer = sym.Analyzer()
    nested = prim.floormod(prim.floormod(variable, inner), outer)
    expected = prim.floormod(variable, outer)
    assert analyzer.can_prove(nested == expected)
    for value in [-13, -7, -2, -1, 0, 1, 2, 7, 13]:
        assert (value % inner) % outer == value % outer


@pytest.mark.parametrize("inner,outer", [(-3, 2), (3, -2), (2, 3), (-2, -3)])
def test_nested_floormod_does_not_drop_nondivisible_modulus(inner, outer):
    variable = ir.Var("variable", "int64")
    analyzer = sym.Analyzer()
    nested = prim.floormod(prim.floormod(variable, inner), outer)
    assert not analyzer.can_prove(nested == prim.floormod(variable, outer))


def test_nested_floormod_keeps_outer_remainder_sign():
    variable = ir.Var("variable", "int64")
    analyzer = sym.Analyzer()
    assert not analyzer.can_prove(prim.floormod(prim.floormod(variable, 2), -2) == variable % 2)
    assert analyzer.can_prove(prim.floormod(prim.floormod(variable, -2), 2) == variable % 2)


@pytest.mark.parametrize("extent", [8, 32])
def test_floormod_of_three_term_mixed_radix_sum(extent):
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    replica = ir.Var("replica", "int32")
    analyzer = sym.Analyzer()
    analyzer.bind(row, ir.Range(0, extent))
    analyzer.bind(column, ir.Range(0, 32))
    analyzer.bind(replica, ir.Range(0, 16))
    expression = (row * 64 + replica * 4 + column // 8) % 128
    expected = row % 2 * 64 + replica * 4 + column // 8
    ir.assert_structural_equal(analyzer.simplify(expression), expected)


def test_floormod_does_not_split_terms_with_a_carry():
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    replica = ir.Var("replica", "int32")
    analyzer = sym.Analyzer()
    analyzer.bind(row, ir.Range(0, 8))
    analyzer.bind(column, ir.Range(0, 32))
    analyzer.bind(replica, ir.Range(0, 32))
    expression = (row * 64 + replica * 4 + column // 8) % 128
    assert not analyzer.can_prove(expression == row % 2 * 64 + replica * 4 + column // 8)
