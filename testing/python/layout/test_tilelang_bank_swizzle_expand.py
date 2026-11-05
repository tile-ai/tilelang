import pytest

from tvm_tirx import ir, tirx

from tilelang.layout.swizzle import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
)


@pytest.mark.parametrize(
    "make_layout",
    [
        make_quarter_bank_swizzled_layout,
        make_half_bank_swizzled_layout,
        make_full_bank_swizzled_layout,
    ],
)
def test_bank_swizzle_layout_expand_leading_dims(make_layout):
    buf2d = ir.Var("A2", tirx.TensorType((8, 64), "float16", scope="shared"))
    buf3d = ir.Var("A3", tirx.TensorType((2, 8, 64), "float16", scope="shared"))
    buf4d = ir.Var("A4", tirx.TensorType((3, 2, 8, 64), "float16", scope="shared"))

    layout2d = make_layout(buf2d)
    assert make_layout(buf3d).is_equal(layout2d.expand([2]))
    assert make_layout(buf4d).is_equal(layout2d.expand([3, 2]))


if __name__ == "__main__":
    pytest.main([__file__])
