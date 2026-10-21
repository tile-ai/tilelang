"""Unit tests for the WGMMA/UMMA K-panel stride decode.

``decode_k_panel_elems`` is the single source of truth for how far an atom offset
steps to reach the next K swizzle-atom panel. Reconstructing that step from the
operand's MN extent is wrong for an operand that is a slice of a wider buffer, so
the decode must either return the layout's own stride or refuse — never fall back
silently, which is the wrong-code path the decode exists to remove.
"""

import pytest

import tilelang.testing  # noqa: F401  (registers the test main)
from tilelang.layout import SwizzleMode, cute
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import decode_k_panel_elems

ATOM = 64  # elements per 128B swizzle atom at 16-bit dtypes
B128 = SwizzleMode.SWIZZLE_128B
NONE = SwizzleMode.NONE


def _k_mode(shape, stride):
    return cute.make_layout(shape, stride)


def test_canonical_multi_panel_returns_the_layout_stride():
    """(atom, panels):(1, panel_stride) -- the step is read off the layout."""
    assert decode_k_panel_elems(_k_mode([ATOM, 2], [1, 8192]), True, B128, ATOM, "WGMMA") == 8192
    assert decode_k_panel_elems(_k_mode([ATOM, 4], [1, 8192]), True, B128, ATOM, "WGMMA") == 8192
    # The stride is the buffer's, so a taller buffer gives a larger step for the
    # same operand shape -- exactly what a sliced operand needs.
    assert decode_k_panel_elems(_k_mode([ATOM, 2], [1, 16384]), True, B128, ATOM, "WGMMA") == 16384


def test_single_panel_and_mn_major_need_no_step():
    """None is only returned where ``ki // k_atom_size`` is always zero."""
    # Flat K mode: one atom wide, so there is no second panel to step to.
    assert decode_k_panel_elems(_k_mode(ATOM, 1), True, B128, ATOM, "WGMMA") is None
    # MN-major operands never reach the panel term.
    assert decode_k_panel_elems(_k_mode([ATOM, 2], [1, 8192]), False, B128, ATOM, "WGMMA") is None
    # Unswizzled layouts have no panels at all.
    assert decode_k_panel_elems(_k_mode(128, 1), True, NONE, 128, "WGMMA") is None


def test_panels_first_ordering_is_rejected_not_misread():
    """A (panels, atom) K mode must not have its trailing atom stride read as the step.

    Taking ``stride[-1]`` without checking that the leading sub-mode is the
    contiguous atom would return 1 here — a silently wrong panel step.
    """
    with pytest.raises(AssertionError, match=r"spans 2 swizzle-atom panels"):
        decode_k_panel_elems(_k_mode([2, ATOM], [8192, 1]), True, B128, ATOM, "WGMMA")


def test_multi_panel_without_a_constant_stride_asserts():
    """Refusing beats falling back to the extent-based reconstruction."""
    # Flat but two atoms wide: no panel sub-mode to read a stride from.
    with pytest.raises(AssertionError, match=r"spans 2 swizzle-atom panels"):
        decode_k_panel_elems(_k_mode(2 * ATOM, 1), True, B128, ATOM, "UMMA")
    # Three sub-modes: panel spacing is not expressible as one scalar step.
    with pytest.raises(AssertionError, match=r"spans 4 swizzle-atom panels"):
        decode_k_panel_elems(_k_mode([ATOM, 2, 2], [1, 8192, 16384]), True, B128, ATOM, "WGMMA")


def test_partial_atom_extent_still_counts_as_two_panels():
    """A K extent that is not a whole number of atoms must not read as single-panel."""
    with pytest.raises(AssertionError, match=r"spans 2 swizzle-atom panels"):
        decode_k_panel_elems(_k_mode(ATOM + 16, 1), True, B128, ATOM, "WGMMA")


if __name__ == "__main__":
    tilelang.testing.main()
