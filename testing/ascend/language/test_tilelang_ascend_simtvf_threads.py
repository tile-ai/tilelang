"""SimtVF thread-count upper bound validation."""

import pytest

import tilelang.ascend.language as T


def test_simtvf_rejects_more_than_2048_threads():
    with pytest.raises(ValueError, match="at most 2048"):
        T.SimtVF(threads=2049)
    with pytest.raises(ValueError, match="at most 2048"):
        T.SimtVF(threads=[64, 64, 1])
