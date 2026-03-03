import pytest

import tilelang.testing
from tvm import tir

from tilelang.layout.swizzle import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
)

tilelang.testing.set_random_seed()


@pytest.mark.parametrize(
    "make_layout",
    [
        make_quarter_bank_swizzled_layout,
        make_half_bank_swizzled_layout,
        make_full_bank_swizzled_layout,
    ],
)
def test_bank_swizzle_layout_expand_leading_dims(make_layout):
    buf2d = tir.decl_buffer((8, 64), "float16", name="A2", scope="shared")
    buf3d = tir.decl_buffer((2, 8, 64), "float16", name="A3", scope="shared")
    buf4d = tir.decl_buffer((3, 2, 8, 64), "float16", name="A4", scope="shared")

    layout2d = make_layout(buf2d)
    assert make_layout(buf3d).is_equal(layout2d.expand([2]))
    assert make_layout(buf4d).is_equal(layout2d.expand([3, 2]))


if __name__ == "__main__":
    tilelang.testing.main()
