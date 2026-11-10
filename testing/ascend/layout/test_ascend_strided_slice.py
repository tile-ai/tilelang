"""Strided slices preserve the physical buffer's base and row pitch."""

import pytest

from tilelang import tvm
from tilelang.layout import Layout, make_strided_slice
from tvm import tirx


@pytest.mark.parametrize(
    "strides,symbolic,first_offset,second_offset",
    [(None, False, 10, 19), ((16, 1), False, 18, 35), ((16, 1), True, 18, 35)],
    ids=["compact", "explicit-strides", "symbolic-base"],
)
def test_strided_slice_accumulates_base_offsets(strides, symbolic, first_offset, second_offset):
    base = tirx.Var("base", "int64") if symbolic else 128
    physical = tirx.decl_buffer((4, 8), "int32", strides=strides, elem_offset=base, scope="shared.dyn")
    # Logical (2, 1) maps to physical (1, 2).
    layout = Layout((8, 4), lambda i, j: (j, i))
    region = [tvm.ir.Range.from_min_extent(2, 4), tvm.ir.Range.from_min_extent(1, 2)]
    view = make_strided_slice(physical, layout, region)
    nested = make_strided_slice(
        view,
        Layout((2, 4), lambda i, j: (i, j)),
        [tvm.ir.Range.from_min_extent(1, 1), tvm.ir.Range.from_min_extent(1, 2)],
    )

    analyzer = tvm.arith.Analyzer()
    for result, shape, relative_offset in [(view, (2, 4), first_offset), (nested, (1, 2), second_offset)]:
        assert result.data.same_as(physical.data)
        assert result.dtype == physical.dtype
        assert list(result.shape) == list(shape)
        assert list(result.strides) == list(strides or (8, 1))
        assert analyzer.can_prove_equal(result.elem_offset, base + relative_offset)
