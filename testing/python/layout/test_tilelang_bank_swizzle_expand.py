import pytest

import tilelang.testing
from tvm import tirx

from tilelang.layout.swizzle import (
    make_swizzled_layout,
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
        make_swizzled_layout,
        make_full_bank_swizzled_layout,
    ],
)
def test_bank_swizzle_layout_expand_leading_dims(make_layout):
    buf2d = tirx.decl_buffer((8, 64), "float16", name="A2", scope="shared")
    buf3d = tirx.decl_buffer((2, 8, 64), "float16", name="A3", scope="shared")
    buf4d = tirx.decl_buffer((3, 2, 8, 64), "float16", name="A4", scope="shared")

    layout2d = make_layout(buf2d)
    assert make_layout(buf3d).is_equal(layout2d.expand([2]))
    assert make_layout(buf4d).is_equal(layout2d.expand([3, 2]))


@pytest.mark.parametrize("rows", [4, 8, 17, 20, 28])
@pytest.mark.parametrize(
    "cols,make_bank_layout",
    [(16, make_quarter_bank_swizzled_layout), (32, make_half_bank_swizzled_layout), (64, make_full_bank_swizzled_layout)],
)
def test_swizzled_layout_irregular_rows(rows, cols, make_bank_layout):
    buffer = tirx.decl_buffer((rows, cols), "float16", scope="shared")
    layout = make_swizzled_layout(buffer)
    shape = [int(dim) for dim in layout.get_output_shape()]
    addresses = set()
    for i in range(rows):
        for j in range(cols):
            address = tuple(int(index) for index in layout.map_forward_index([i, j]))
            assert all(0 <= index < extent for index, extent in zip(address, shape, strict=True))
            addresses.add(address)
    assert len(addresses) == rows * cols
    if rows in (4, 8):
        assert layout.is_equal(make_bank_layout(buffer))


if __name__ == "__main__":
    tilelang.testing.main()
