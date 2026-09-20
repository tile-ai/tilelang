"""pytest test for example_compress.py — two-stage compress + state-cache update.

Verifies the NPU kernel against the PyTorch reference for both the sparse
(few tokens complete a compression block) and dense (every token compresses)
regimes. Correctness is checked inside `run_compress_decode`
(verify=True asserts bit-correctness against the reference).
"""

import pytest

from example_compress import run_compress_decode


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
@pytest.mark.parametrize("all_seq_do_compress", [False, True])
def test_compress_decode(target, all_seq_do_compress):
    run_compress_decode(
        overlap_ratio=2,
        compress_ratio=4,
        dim=128,
        all_seq_do_compress=all_seq_do_compress,
        verify=True,
        target=target,
    )


if __name__ == "__main__":
    for flag in [False, True]:
        test_compress_decode("ascend", flag)
        print(f"PASS: test_compress_decode all_seq_do_compress={flag}")
