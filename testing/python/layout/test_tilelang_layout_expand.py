import pytest

import tilelang
import tilelang.testing
from tilelang.layout import Layout

tilelang.testing.set_random_seed()


def test_layout_expand_linear_scalar_output():
    base = Layout([16, 128], lambda j, k: j * 128 + k)
    expanded = base.expand([4])

    expected = Layout([4, 16, 128], lambda i, j, k: [i, j * 128 + k])
    assert expanded.is_equal(expected)

    assert list(expanded.get_input_shape()) == [4, 16, 128]
    assert list(expanded.get_output_shape()) == [4, 2048]


def test_layout_expand_multi_output_identity():
    base = Layout([16, 128], lambda j, k: [j, k])
    expanded = base.expand(4)

    expected = Layout([4, 16, 128], lambda i, j, k: [i, j, k])
    assert expanded.is_equal(expected)

    assert list(expanded.get_output_shape()) == [4, 16, 128]


def test_layout_expand_noop_empty():
    base = Layout([16, 128], lambda j, k: j * 128 + k)
    assert base.expand([]) is base
    assert base.expand(()) is base


def test_layout_expand_invalid_args():
    base = Layout([16, 128], lambda j, k: j * 128 + k)

    with pytest.raises(ValueError):
        _ = base.expand(0)
    with pytest.raises(ValueError):
        _ = base.expand([-1])
    with pytest.raises(TypeError):
        _ = base.expand("4")


if __name__ == "__main__":
    tilelang.testing.main()
