import pytest

import tilelang
import tilelang.testing
from tilelang.layout import Layout
from tilelang import _ffi_api

tilelang.testing.set_random_seed()


def test_layout_repeat_dim0_linear():
    atom = Layout([8, 64], lambda i, j: i * 64 + j)
    repeated = atom.repeat(dim=0, factor=2)

    expected = Layout([16, 64], lambda i, j: [i // 8, (i % 8) * 64 + j])
    assert repeated.is_equal(expected)

    assert list(repeated.get_input_shape()) == [16, 64]
    assert list(repeated.get_output_shape()) == [2, 512]


def test_layout_repeat_dim1_linear():
    atom = Layout([8, 64], lambda i, j: i * 64 + j)
    repeated = atom.repeat(dim=1, factor=3)

    expected = Layout([8, 192], lambda i, j: [j // 64, i * 64 + (j % 64)])
    assert repeated.is_equal(expected)

    assert list(repeated.get_input_shape()) == [8, 192]
    assert list(repeated.get_output_shape()) == [3, 512]


def test_layout_repeat_multi_output():
    atom = Layout([8, 64], lambda i, j: [i, j])
    repeated = atom.repeat(dim=0, factor=2)

    expected = Layout([16, 64], lambda i, j: [i // 8, i % 8, j])
    assert repeated.is_equal(expected)

    assert list(repeated.get_output_shape()) == [2, 8, 64]


def test_layout_repeat_factor_one_is_noop():
    atom = Layout([8, 64], lambda i, j: i * 64 + j)
    assert atom.repeat(dim=0, factor=1) is atom


def test_layout_repeat_invalid_args():
    atom = Layout([8, 64], lambda i, j: i * 64 + j)

    with pytest.raises(ValueError):
        _ = atom.repeat(dim=0, factor=0)
    with pytest.raises(ValueError):
        _ = atom.repeat(dim=0, factor=-1)
    with pytest.raises(ValueError):
        _ = atom.repeat(dim=2, factor=2)
    with pytest.raises(ValueError):
        _ = atom.repeat(dim=-3, factor=2)


def test_layout_repeat_invalid_args_cpp_raises_value_error():
    atom = Layout([8, 64], lambda i, j: i * 64 + j)

    with pytest.raises(ValueError):
        _ = _ffi_api.Layout_repeat(atom, 0, 0)
    with pytest.raises(ValueError):
        _ = _ffi_api.Layout_repeat(atom, 0, -1)
    with pytest.raises(ValueError):
        _ = _ffi_api.Layout_repeat(atom, 2, 2)
    with pytest.raises(ValueError):
        _ = _ffi_api.Layout_repeat(atom, -3, 2)


if __name__ == "__main__":
    tilelang.testing.main()
