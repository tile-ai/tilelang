"""Allocation extents must contain every physical coordinate, including its origin."""

import itertools

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

import tilelang
import tilelang.language as T


CASES = [
    ((64, 64), lambda i, j: (0, i, (j // 8 ^ i % 8) * 8 + j % 8), (1, 64, 64)),
    ((8,), lambda i: ((i & 7) + 32,), (40,)),
    ((4, 64), lambda i, j: (j ^ (i % 4), i * 2), (64, 7)),
    ((2,), lambda i: ((i & 1) + 2147483646,), (2147483648,)),
    ((2,), lambda i: (i + 2147483646,), (2147483648,)),
    ((8, 16), lambda i, j: ((i | (j // 8)) * 8 + j % 8,), (64,)),
]


@pytest.mark.parametrize("shape,mapping,expected", CASES)
def test_extents_cover_finite_physical_domain(shape, mapping, expected):
    layout = tilelang.Layout(shape, mapping)
    extents = tuple(int(value) for value in layout.get_output_shape())
    points = [mapping(*indices) for indices in itertools.product(*(range(n) for n in shape))]
    assert extents == expected
    assert all(0 <= coordinate < extents[axis] for point in points for axis, coordinate in enumerate(point))
    # These examples attain their maximum. This is an independent finite-domain
    # oracle, not a comparison against the analyzer's own bound expression.
    assert extents == tuple(max(point[axis] for point in points) + 1 for axis in range(len(extents)))


@pytest.mark.parametrize("dtype,extent", [("uint8", 256), ("int8", 128), ("uint16", 65536)])
def test_capacity_increment_does_not_wrap_in_coordinate_type(dtype, extent):
    layout = tilelang.Layout((extent,), lambda i: (T.cast(i, dtype),))
    shape = layout.get_output_shape()
    assert tuple(int(value) for value in shape) == (extent,)


@pytest.mark.parametrize("origin", [32, 2147483646])
def test_region_bounds_keep_positive_origin_and_exclusive_end(origin):
    size = 2 if origin > 32 else 8
    layout = tilelang.Layout((size,), lambda i: ((i & (size - 1)) + origin,))
    bounds = layout.map_region_bounds([ir.Range(0, size)])
    assert len(bounds) == 1
    assert int(bounds[0].begin) == origin
    assert int(bounds[0].extent) == size


def test_swizzle_subregion_contains_every_mapped_point():
    layout = tilelang.Layout((64, 64), CASES[0][1])
    bounds = layout.map_region_bounds([ir.Range(4, 11), ir.Range(9, 18)])
    for i, j in itertools.product(range(4, 11), range(9, 18)):
        for axis, value in enumerate(CASES[0][1](i, j)):
            assert int(bounds[axis].begin) <= value
            assert value < int(bounds[axis].begin + bounds[axis].extent)


def test_unknown_coordinate_does_not_guess_a_capacity():
    layout = tilelang.Layout((8,), lambda i: (T.call_pure_extern("int32", "unknown_layout_index", i),))
    with pytest.raises(ValueError, match="Cannot infer a safe output extent"):
        layout.get_output_shape()


@pytest.mark.parametrize("origin", [0, -(2**31), -(2**31) + 1])
def test_region_span_retains_capacity_beyond_int32(origin):
    size = 2**31
    layout = tilelang.Layout((size,), lambda i: i + origin)
    (region,) = layout.map_region_bounds([ir.Range(0, ir.const(size, "int64"))])
    analyzer = sym.Analyzer()
    assert int(analyzer.simplify(region.begin)) == origin
    assert str(region.extent.ty) == "int64"
    assert int(analyzer.simplify(region.extent)) == size


def test_declared_int64_shape_preserves_linear_address_width():
    dimension = ir.const(65536, "int64")
    layout = tilelang.Layout((dimension, dimension), lambda i, j: (i, j))
    linear = layout.get_linearized_forward_index()
    assert str(linear.ty) == "int64"
    variables = list(layout.get_forward_vars())
    point = tvm_ffi.structural_map(
        linear,
        (ir.Var, lambda var: ir.const(65535, "int32") if any(var.same_as(v) for v in variables) else var),
        order="pre",
    )
    assert int(sym.Analyzer().simplify(point)) == 2**32 - 1


@pytest.mark.parametrize("lo,hi", [(-(2**62), 2**62 - 1), (-(2**63) + 1, 2**63 - 2)])
def test_unrepresentable_int64_region_span_is_rejected(lo, hi):
    assert hi - lo + 1 > 2**63 - 1
    layout = tilelang.Layout((2,), lambda i: tirx.Select(i == 0, ir.const(lo, "int64"), ir.const(hi, "int64")))
    with pytest.raises(ValueError, match="Layout region extent does not fit int64"):
        layout.map_region_bounds([ir.Range(0, 2)])


def test_largest_int64_region_span_remains_representable():
    lo, hi = -(2**62), 2**62 - 2
    layout = tilelang.Layout((2,), lambda i: tirx.Select(i == 0, ir.const(lo, "int64"), ir.const(hi, "int64")))
    (region,) = layout.map_region_bounds([ir.Range(0, 2)])
    assert int(sym.Analyzer().simplify(region.begin)) == lo
    assert str(region.extent.ty) == "int64"
    assert int(sym.Analyzer().simplify(region.extent)) == 2**63 - 1


@pytest.mark.parametrize("size", [1, 7, 2**31 - 1])
def test_symbolic_capacity_increment_does_not_wrap(size):
    n = ir.Var("n", "int32")
    layout = tilelang.Layout((n,), lambda i: i + 1)
    (capacity,) = layout.get_output_shape()
    specialized = tvm_ffi.structural_map(capacity, (ir.Var, lambda v: ir.const(size, "int32") if v.same_as(n) else v), order="pre")
    assert str(capacity.ty) == "int64"
    assert int(sym.Analyzer().simplify(specialized)) == size + 1


@pytest.mark.parametrize("origin", [-(2**31), -7, 0])
def test_symbolic_region_span_is_not_limited_to_endpoint_width(origin):
    begin = ir.Var("begin", "int32")
    layout = tilelang.Layout((2,), lambda i: tirx.Select(i == 0, begin, ir.const(0, "int32")))
    (region,) = layout.map_region_bounds([ir.Range(0, 2)])
    specialized = tvm_ffi.structural_map(
        region.extent, (ir.Var, lambda v: ir.const(origin, "int32") if v.same_as(begin) else v), order="pre"
    )
    assert str(region.extent.ty) == "int64"
    assert int(sym.Analyzer().simplify(specialized)) == 1 - origin
