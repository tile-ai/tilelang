"""Tests for validating inverses of TileLang layouts."""

import pytest

from tilelang.layout import Layout


def test_inverse_accepts_bijective_split():
    layout = Layout([128], lambda i: [i // 8, i % 8])

    inverse = layout.inverse()

    assert [int(dim) for dim in inverse.get_input_shape()] == [16, 8]
    assert [int(dim) for dim in inverse.get_output_shape()] == [128]


def test_inverse_rejects_unsupported_cyclic_shift():
    layout = Layout([128], lambda i: (i + 1) % 128)

    # Native IterMap now rejects the unsupported padding before producing an
    # inverse. It must still never return a map that changes coordinates.
    with pytest.raises(Exception, match="Cannot absorb left padding"):
        layout.inverse()


if __name__ == "__main__":
    pytest.main([__file__])
