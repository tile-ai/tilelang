"""A physical bounding box must include values on both sides of integer wrap."""

import pytest

import tilelang
from tvm_tirx import ir, sym
from tvm_tirx.ir import prim


@pytest.mark.parametrize("begin,extent", [(100, 200), (200, 200)])
def test_narrow_cast_region_contains_wrapped_coordinates(begin, extent):
    layout = tilelang.Layout([extent], lambda i: prim.Cast("uint8", i + begin))
    (region,) = layout.map_region_bounds([ir.Range(0, extent)])
    analyzer = sym.Analyzer()
    lower = int(analyzer.simplify(region.begin))
    upper = lower + int(analyzer.simplify(region.extent))
    coordinates = [(index + begin) % 256 for index in range(extent)]
    assert 0 in coordinates
    assert all(lower <= value < upper for value in coordinates)


def test_unsigned_add_region_contains_wrapped_zero_and_high_coordinate():
    layout = tilelang.Layout(
        [2],
        lambda i: prim.Cast("int32", prim.Add(prim.Cast("uint8", i), ir.const(255, "uint8"))),
    )
    (region,) = layout.map_region_bounds([ir.Range(0, 2)])
    analyzer = sym.Analyzer()
    lower = int(analyzer.simplify(region.begin))
    upper = lower + int(analyzer.simplify(region.extent))
    assert lower <= 0 < upper
    assert lower <= 255 < upper
