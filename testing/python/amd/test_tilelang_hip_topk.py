import sys
from pathlib import Path

import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "topk"
sys.path.insert(0, str(_EXAMPLE_DIR))

import example_topk  # noqa: E402


@tilelang.testing.requires_rocm
def test_topk():
    """Validate the generic Top-K kernel and its autotuned launch on gfx942."""
    torch.manual_seed(0)
    logits = torch.rand((64, 64), device="cuda", dtype=torch.float32)

    actual_values, actual_indices = example_topk.tl_topk(logits, 4, blk_m=64)
    expected_values, expected_indices = example_topk.ref_program(logits, 4)

    torch.testing.assert_close(actual_values, expected_values)
    torch.testing.assert_close(actual_indices, expected_indices)
