"""TileLang's physical layout capacity must cover wrapped coordinates."""

import pytest

import tilelang
from tvm_tirx import ir, sym
from tvm_tirx.ir import prim


@pytest.mark.parametrize("offset,expected", [(0, 2), (254, 256), (255, 256)])
def test_unsigned_add_layout_capacity(offset, expected):
    layout = tilelang.Layout(
        [2],
        lambda i: prim.Cast("int32", prim.Add(prim.Cast("uint8", i), ir.const(offset, "uint8"))),
    )
    shape = sym.Analyzer().simplify(layout.get_output_shape()[0])
    assert isinstance(shape, ir.IntImm), shape
    assert int(shape.value) == expected
    assert all(0 <= (index + offset) % 256 < int(shape.value) for index in range(2))
