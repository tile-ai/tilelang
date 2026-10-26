import pytest

from tilelang._compat import ir, sym
from tvm_tirx.ir import prim


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "int32"])
@pytest.mark.parametrize("lanes", [2, 4, 8])
@pytest.mark.parametrize("operation", [prim.EQ, prim.NE, prim.LT, prim.LE, prim.GT, prim.GE])
def test_vector_comparison_bounds_are_scalar(dtype, lanes, operation):
    left = ir.Var("left", f"{dtype}x{lanes}")
    right = ir.Var("right", dtype)
    comparison = operation(left, prim.Broadcast(right, lanes))
    analyzer = sym.Analyzer()
    predicate = ir.Var("predicate", f"boolx{lanes}")
    analyzer.bind(predicate, comparison)
    for value in (comparison, predicate, prim.Cast(f"int32x{lanes}", comparison)):
        bounds = analyzer.int_set(value)
        assert bounds.min_value.ty.dtype.lanes == 1
        assert bounds.max_value.ty.dtype.lanes == 1
        assert int(bounds.min_value) == 0
        assert int(bounds.max_value) == 1


@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_vector_cast_keeps_lane_range(lanes):
    analyzer = sym.Analyzer()
    value = prim.Cast(f"int64x{lanes}", prim.Ramp(ir.const(5), 2, lanes))
    bounds = analyzer.int_set(value)
    assert int(bounds.min_value) == 5
    assert int(bounds.max_value) == 5 + 2 * (lanes - 1)


def test_unconstrained_vector_has_no_scalar_point_bound():
    bounds = sym.Analyzer().int_set(ir.Var("values", "int32x4"))
    assert bounds.is_everything()
