"""Tests for T.Kernel thread-extent validation."""

import pytest
import tilelang
import tilelang.testing
from tilelang.language.kernel import _normalize_threads

# ===========================================================================
# Validation tests (do not require GPU)
# ===========================================================================


@pytest.mark.parametrize("threads", [-1, 0, [-1], [128, 0], (128, 1, -2)])
def test_normalize_threads_rejects_non_positive(threads):
    """A non-positive extent must be rejected instead of launching an empty block."""
    with pytest.raises(ValueError, match="threads must be positive"):
        _normalize_threads(threads)


@pytest.mark.parametrize(
    "threads, expected",
    [
        (None, [128, 1, 1]),
        (256, [256, 1, 1]),
        ([32, 4], [32, 4, 1]),
        ((32, 2, 2), [32, 2, 2]),
    ],
)
def test_normalize_threads_accepts_positive(threads, expected):
    """Valid extents are still normalized to a 3-D thread block."""
    assert _normalize_threads(threads) == expected


if __name__ == "__main__":
    tilelang.testing.main()
