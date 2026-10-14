import tilelang.testing
from tilelang import tvm
from tilelang.layout import Layout


def _assert_indices_equal(lhs, rhs):
    assert tvm.arith.Analyzer().can_prove_equal(lhs, rhs), f"{lhs} != {rhs}"


def test_linearized_forward_index_for_multi_dimensional_output():
    layout = Layout([2, 3, 4], lambda i, j, k: [i, j, k])
    linear = Layout([2, 3, 4], lambda i, j, k: i * 12 + j * 4 + k)

    _assert_indices_equal(
        layout.get_linearized_forward_index(),
        linear.get_linearized_forward_index(),
    )


def test_linearized_forward_index_for_split_output():
    layout = Layout([64], lambda i: [i // 32, i % 32])
    linear = Layout([64], lambda i: i)

    _assert_indices_equal(
        layout.get_linearized_forward_index(),
        linear.get_linearized_forward_index(),
    )


def test_linearized_forward_index_uses_output_shape():
    transposed = Layout([2, 32], lambda i, j: [j, i])
    expected = Layout([2, 32], lambda i, j: j * 2 + i)

    _assert_indices_equal(
        transposed.get_linearized_forward_index(),
        expected.get_linearized_forward_index(),
    )


if __name__ == "__main__":
    tilelang.testing.main()
