"""Carver bound transfer keeps native dtype limits and context lifetimes."""

import pytest
from tvm_tirx import ir, sym

from tilelang.carver._constraints import const_bound_context


@pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "uint8", "uint32"])
def test_finite_bounds_limit_affine_accesses(dtype):
    variable = ir.Var("i", dtype)
    analyzer = sym.Analyzer()
    context = const_bound_context({variable: sym.ConstIntBound(2, 7)})
    analyzer.sync_constraint_context(context)
    actual = analyzer.const_int_bound(2 * variable + 1)
    assert (actual.min_value, actual.max_value) == (5, 15)
    assert all(isinstance(entry, sym.PredicateConstraint) for entry in context)


def test_unbounded_int32_does_not_materialize_an_int64_infinity_sentinel():
    variable = ir.Var("i", "int32")
    analyzer = sym.Analyzer()
    context = const_bound_context({variable: sym.ConstIntBound(0, sym.ConstIntBound.POS_INF)})
    analyzer.sync_constraint_context(context)
    actual = analyzer.const_int_bound(variable)
    assert (actual.min_value, actual.max_value) == (0, 2**31 - 1)
    for entry in context:
        assert entry.condition.ty == ir.PrimType("bool")
    analyzer.sync_constraint_context(sym.ConstraintContext())
    restored = analyzer.const_int_bound(variable)
    assert (restored.min_value, restored.max_value) == (-(2**31), 2**31 - 1)


def test_finite_bounds_outside_dtype_do_not_wrap():
    variable = ir.Var("i", "int8")
    analyzer = sym.Analyzer()
    analyzer.sync_constraint_context(const_bound_context({variable: sym.ConstIntBound(-1000, 1000)}))
    actual = analyzer.const_int_bound(variable)
    assert (actual.min_value, actual.max_value) == (-128, 127)


def test_replacement_context_releases_old_bounds_instead_of_intersecting():
    variable = ir.Var("i", "int32")
    analyzer = sym.Analyzer()
    analyzer.sync_constraint_context(const_bound_context({variable: sym.ConstIntBound(0, 3)}))
    analyzer.sync_constraint_context(const_bound_context({variable: sym.ConstIntBound(-2, 7)}))
    actual = analyzer.const_int_bound(variable)
    assert (actual.min_value, actual.max_value) == (-2, 7)


def test_same_name_axes_are_independent():
    first, second = ir.Var("i", "int64"), ir.Var("i", "int64")
    analyzer = sym.Analyzer()
    analyzer.sync_constraint_context(const_bound_context({first: sym.ConstIntBound(0, 3), second: sym.ConstIntBound(8, 11)}))
    actual = analyzer.const_int_bound(second - first)
    assert (actual.min_value, actual.max_value) == (5, 11)


def test_empty_inclusive_interval_remains_empty():
    variable = ir.Var("i", "int64")
    analyzer = sym.Analyzer()
    analyzer.sync_constraint_context(const_bound_context({variable: sym.ConstIntBound(0, -1)}))
    actual = analyzer.const_int_bound(variable)
    assert actual.max_value - actual.min_value + 1 == 0
