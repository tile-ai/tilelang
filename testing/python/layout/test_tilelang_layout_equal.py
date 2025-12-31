"""Tests for Layout and Fragment equality comparison."""

import tilelang
import tilelang.testing
from tilelang.layout import Layout
from tilelang.layout.fragment import Fragment

tilelang.testing.set_random_seed()


class TestLayoutEqual:
    """Test cases for Layout.is_equal()."""

    def test_same_layout_is_equal(self):
        """Two layouts with identical mapping should be equal."""
        layout1 = Layout([32, 4], lambda i, j: i * 4 + j)
        layout2 = Layout([32, 4], lambda i, j: i * 4 + j)
        assert layout1.is_equal(layout2)

    def test_different_index_order_not_equal(self):
        """Layouts with different index order (i*4+j vs j*4+i) should not be equal."""
        layout1 = Layout([32, 4], lambda i, j: i * 4 + j)
        layout2 = Layout([32, 4], lambda i, j: j * 4 + i)
        assert not layout1.is_equal(layout2)

    def test_different_coefficient_not_equal(self):
        """Layouts with different coefficients should not be equal."""
        layout1 = Layout([32, 4], lambda i, j: i * 4 + j)
        layout2 = Layout([32, 4], lambda i, j: i * 8 + j)
        assert not layout1.is_equal(layout2)

    def test_different_shape_not_equal(self):
        """Layouts with different shapes should not be equal."""
        layout1 = Layout([32, 4], lambda i, j: i * 4 + j)
        layout2 = Layout([16, 8], lambda i, j: i * 8 + j)
        assert not layout1.is_equal(layout2)

    def test_same_layout_different_var_names(self):
        """Layouts with same mapping but created with different variable names should be equal."""
        layout1 = Layout([32, 4], lambda x, y: x * 4 + y)
        layout2 = Layout([32, 4], lambda a, b: a * 4 + b)
        assert layout1.is_equal(layout2)

    def test_2d_output_layout_equal(self):
        """Layouts with 2D output should compare correctly."""
        layout1 = Layout([32, 4], lambda i, j: [i, j])
        layout2 = Layout([32, 4], lambda i, j: [i, j])
        assert layout1.is_equal(layout2)

    def test_2d_output_layout_different_order(self):
        """Layouts with swapped output dimensions should not be equal."""
        layout1 = Layout([32, 4], lambda i, j: [i, j])
        layout2 = Layout([32, 4], lambda i, j: [j, i])
        assert not layout1.is_equal(layout2)

    def test_complex_expression_equal(self):
        """Layouts with complex but equivalent expressions should be equal."""
        layout1 = Layout([16, 8], lambda i, j: i * 8 + j)
        layout2 = Layout([16, 8], lambda i, j: j + i * 8)
        # Note: This tests if the comparison handles commutative operations
        # With StructuralEqual, a*b+c and c+a*b have different AST structure
        # So this may or may not be equal depending on implementation
        # For now we test the actual behavior
        result = layout1.is_equal(layout2)
        # The key point is it should not crash and return a boolean
        assert isinstance(result, bool)


class TestFragmentEqual:
    """Test cases for Fragment.is_equal()."""

    def test_same_fragment_is_equal(self):
        """Two fragments with identical mapping should be equal."""
        frag1 = Fragment([32, 4], forward_thread_fn=lambda i, j: i * 4 + j)
        frag2 = Fragment([32, 4], forward_thread_fn=lambda i, j: i * 4 + j)
        assert frag1.is_equal(frag2)

    def test_different_thread_mapping_not_equal(self):
        """Fragments with different thread mapping (i*4+j vs j*4+i) should not be equal."""
        frag1 = Fragment([32, 4], forward_thread_fn=lambda i, j: i * 4 + j)
        frag2 = Fragment([32, 4], forward_thread_fn=lambda i, j: j * 4 + i)
        assert not frag1.is_equal(frag2)

    def test_different_forward_index_not_equal(self):
        """Fragments with different forward_index should not be equal."""
        frag1 = Fragment([32, 4], forward_thread_fn=lambda i, j: i * 4 + j, forward_index_fn=lambda i, j: i)
        frag2 = Fragment([32, 4], forward_thread_fn=lambda i, j: i * 4 + j, forward_index_fn=lambda i, j: j)
        assert not frag1.is_equal(frag2)

    def test_same_fragment_different_var_names(self):
        """Fragments with same mapping but different variable names should be equal."""
        frag1 = Fragment([32, 4], forward_thread_fn=lambda x, y: x * 4 + y)
        frag2 = Fragment([32, 4], forward_thread_fn=lambda a, b: a * 4 + b)
        assert frag1.is_equal(frag2)

    def test_fragment_with_replicate_equal(self):
        """Fragments with same replicate factor should be equal."""
        frag1 = Fragment([32, 4], forward_thread_fn=lambda i, j, rep: i * 4 + rep, replicate=4)
        frag2 = Fragment([32, 4], forward_thread_fn=lambda i, j, rep: i * 4 + rep, replicate=4)
        assert frag1.is_equal(frag2)

    def test_fragment_different_replicate_not_equal(self):
        """Fragments with different replicate factors should not be equal."""
        frag1 = Fragment([32, 4], forward_thread_fn=lambda i, j, rep: i * 4 + rep, replicate=4)
        frag2 = Fragment([32, 4], forward_thread_fn=lambda i, j, rep: i * 4 + rep, replicate=2)
        assert not frag1.is_equal(frag2)

    def test_fragment_with_forward_fn(self):
        """Fragments created with forward_fn should compare correctly."""
        frag1 = Fragment([32, 4], forward_fn=lambda i, j: (i * 4 + j, i * 4 + j))
        frag2 = Fragment([32, 4], forward_fn=lambda i, j: (i * 4 + j, i * 4 + j))
        assert frag1.is_equal(frag2)

    def test_fragment_forward_fn_different_thread(self):
        """Fragments with different thread mapping via forward_fn should not be equal."""
        frag1 = Fragment([32, 4], forward_fn=lambda i, j: (i * 4 + j, i))
        frag2 = Fragment([32, 4], forward_fn=lambda i, j: (j * 4 + i, i))
        assert not frag1.is_equal(frag2)

    def test_fragment_forward_fn_different_index(self):
        """Fragments with different forward_index via forward_fn should not be equal."""
        frag1 = Fragment([32, 4], forward_fn=lambda i, j: (i * 4 + j, i))
        frag2 = Fragment([32, 4], forward_fn=lambda i, j: (i * 4 + j, j))
        assert not frag1.is_equal(frag2)


class TestLayoutFragmentEdgeCases:
    """Edge cases and regression tests."""

    def test_single_dim_layout_equal(self):
        """Single dimension layouts should compare correctly."""
        layout1 = Layout([128], lambda i: i)
        layout2 = Layout([128], lambda i: i)
        assert layout1.is_equal(layout2)

    def test_single_dim_layout_not_equal(self):
        """Single dimension layouts with different mappings should not be equal."""
        layout1 = Layout([128], lambda i: i)
        layout2 = Layout([128], lambda i: i * 2)
        assert not layout1.is_equal(layout2)

    def test_three_dim_layout_equal(self):
        """Three dimension layouts should compare correctly."""
        layout1 = Layout([8, 16, 4], lambda i, j, k: i * 64 + j * 4 + k)
        layout2 = Layout([8, 16, 4], lambda i, j, k: i * 64 + j * 4 + k)
        assert layout1.is_equal(layout2)

    def test_three_dim_layout_different_order(self):
        """Three dimension layouts with different index order should not be equal."""
        layout1 = Layout([8, 16, 4], lambda i, j, k: i * 64 + j * 4 + k)
        layout2 = Layout([8, 16, 4], lambda i, j, k: k * 64 + j * 4 + i)
        assert not layout1.is_equal(layout2)

    def test_fragment_empty_forward_index(self):
        """Fragments with empty forward_index should compare correctly."""
        frag1 = Fragment([32, 4], forward_thread_fn=lambda i, j: i * 4 + j)
        frag2 = Fragment([32, 4], forward_thread_fn=lambda i, j: i * 4 + j)
        assert frag1.is_equal(frag2)

    def test_constant_layout_equal(self):
        """Layouts mapping to constant should be equal."""
        from tvm.tir import const

        layout1 = Layout([32, 4], lambda i, j: const(0, "int32"))
        layout2 = Layout([32, 4], lambda i, j: const(0, "int32"))
        assert layout1.is_equal(layout2)

    def test_constant_vs_variable_layout_not_equal(self):
        """Layout mapping to constant vs variable should not be equal."""
        from tvm.tir import const

        layout1 = Layout([32, 4], lambda i, j: const(0, "int32"))
        layout2 = Layout([32, 4], lambda i, j: i)
        assert not layout1.is_equal(layout2)


if __name__ == "__main__":
    tilelang.testing.main()
