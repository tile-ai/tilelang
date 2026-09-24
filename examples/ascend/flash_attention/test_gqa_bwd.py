"""Correctness test for the grouped-query FlashAttention backward kernels."""

import pytest

from example_gqa_bwd import run_correctness


TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("S1,G,S2", [(192, 2, 128), (320, 2, 640), (512, 2, 384)])
def test_gqa_bwd(S1, G, S2, target):
    # Three query tiles cover loops shorter than the four-slot input ring;
    # five and eight tiles exercise slot reuse and non-power-of-two tails.
    # G > 1 and multiple KV tiles cover shared-K/V and atomic dQ reductions.
    run_correctness(S1=S1, G=G, S2=S2, D=128, target=target)


if __name__ == "__main__":
    run_correctness(S1=512, G=2, S2=384, D=128, target="ascend")
