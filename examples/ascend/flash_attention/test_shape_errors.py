"""Shape constraints should be rejected with actionable messages."""

import pytest

from example_gqa import gqa
from example_mha import mha


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: gqa(S1=65, G=2, S2=1024), r"q_len \(130\) must be a multiple of block_q \(128\)"),
        (lambda: gqa(S1=128, G=1, S2=1000), r"kv_len \(1000\) must be a multiple of block_kv \(128\)"),
        (lambda: mha(128, 1000), r"q_len \(1000\) must be a multiple of block_q \(128\)"),
        (lambda: gqa(S1=128, G=1, S2=128, num_blocks=0), r"num_blocks must be positive, got 0"),
        (lambda: gqa(S1=256, G=1, S2=128, num_blocks=3), r"must divide the 2 query tiles"),
    ],
)
def test_rejects_unsupported_shapes_with_message(call, match):
    with pytest.raises(AssertionError, match=match):
        call()
