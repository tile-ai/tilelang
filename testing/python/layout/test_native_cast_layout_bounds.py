"""Native layout capacity includes every coordinate across a narrow cast."""

import pytest

import tilelang
from tvm_tirx import ir, sym
from tvm_tirx.ir import prim


@pytest.mark.parametrize("begin,extent,capacity", [(0, 100, 100), (100, 200, 256), (200, 200, 256)])
def test_unsigned_cast_layout_capacity(begin, extent, capacity):
    layout = tilelang.Layout([extent], lambda i: prim.Cast("uint8", i + begin))
    shape = sym.Analyzer().simplify(layout.get_output_shape()[0])
    assert isinstance(shape, ir.IntImm), shape
    assert int(shape.value) == capacity
    coordinates = [(index + begin) % 256 for index in range(extent)]
    assert all(0 <= coordinate < int(shape.value) for coordinate in coordinates)
