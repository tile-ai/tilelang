"""Native inverse analysis preserves wide logical layout coordinates."""

import pytest
from tvm_tirx import ir

from tilelang import Layout


@pytest.mark.parametrize("rows,columns", [(3, 2), (4, 4), (8, 1)])
@pytest.mark.parametrize("unit_axis", [False, True])
def test_direct_widening_layout_inverse_round_trip(rows, columns, unit_axis):
    def forward(i, j):
        flat = ir.prim.Cast("int64", i) * columns + ir.prim.Cast("int64", j)
        return [flat, ir.IntImm("int64", 0)] if unit_axis else flat

    layout = Layout([ir.IntImm("int64", rows), ir.IntImm("int64", columns)], forward)
    before = ir.save_json(layout)
    for _ in range(2):
        inverse = layout.inverse()
        for row in range(rows):
            for column in range(columns):
                expected = [row * columns + column, 0] if unit_axis else [row * columns + column]
                physical = layout.map_forward_index([row, column])
                assert [int(value) for value in physical] == expected
                assert [int(value) for value in inverse.map_forward_index(physical)] == [row, column]
        assert ir.save_json(layout) == before
