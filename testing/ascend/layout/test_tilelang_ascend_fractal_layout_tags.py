import pytest

import tilelang
import tilelang.testing
from tvm import tirx
from tilelang.layout import (
    make_ascend_l0c_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
    make_ascend_sf_layout,
    try_extract_fractal_layout,
)


@pytest.mark.parametrize("mn", [16, 32])
def test_small_uint16_sf_keeps_semantic_kind(mn):
    buf = tirx.decl_buffer((mn, 1), "uint16", name="SF", scope="shared")
    info = try_extract_fractal_layout(make_ascend_sf_layout(buf), buf)

    assert info.kind == 2
    assert info.c0_axis == 1
    assert info.row16_axis == 0


def test_ambiguous_small_shape_distinguishes_canonical_layouts():
    buf = tirx.decl_buffer((16, 1), "uint16", name="A", scope="shared")

    sf_info = try_extract_fractal_layout(make_ascend_sf_layout(buf), buf)
    major_info = try_extract_fractal_layout(make_ascend_major_k_layout(buf), buf)

    assert sf_info.kind == 2
    assert major_info.kind == 0


@pytest.mark.parametrize(
    "factory,shape,dtype,expected",
    [
        (make_ascend_major_k_layout, (16, 32), "float8_e4m3fn", (0, 1, 0)),
        (make_ascend_major_mn_layout, (32, 16), "float8_e4m3fn", (3, 0, 1)),
        (make_ascend_l0c_layout, (16, 16), "float32", (1, 1, 0)),
        (make_ascend_sf_layout, (16, 1), "uint16", (2, 1, 0)),
    ],
)
def test_canonical_layout_tags(factory, shape, dtype, expected):
    buf = tirx.decl_buffer(shape, dtype, name="A", scope="shared")
    info = try_extract_fractal_layout(factory(buf), buf)

    assert (info.kind, info.c0_axis, info.row16_axis) == expected


def test_expand_preserves_sf_tag():
    base_buf = tirx.decl_buffer((16, 1), "uint16", name="SF2", scope="shared")
    expanded_buf = tirx.decl_buffer((2, 16, 1), "uint16", name="SF3", scope="shared")

    layout = make_ascend_sf_layout(base_buf).expand([2])
    info = try_extract_fractal_layout(layout, expanded_buf)

    assert info.kind == 2
    assert list(layout.get_input_shape()) == [2, 16, 1]


if __name__ == "__main__":
    tilelang.testing.main()
