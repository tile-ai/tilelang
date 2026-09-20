"""pytest test for example_crosslevel_multibuffer.py — cross-level multi-buffer."""

import pytest
import torch
import tilelang

from example_crosslevel_multibuffer import crosslevel_multibuffer, ref_program, N


TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]


@pytest.mark.parametrize("target", TARGETS)
def test_crosslevel_multibuffer(target):
    device = torch.device("npu")
    torch.manual_seed(0)
    kernel = tilelang.compile(crosslevel_multibuffer(), target=target, out_idx=-1)
    a = torch.randn(N, dtype=torch.float32, device=device)
    out = kernel(a)
    torch.npu.synchronize()
    expected = ref_program(a)
    max_diff = (out - expected).abs().max().item()
    assert max_diff < 1e-2, f"max_diff={max_diff:.2e}"


if __name__ == "__main__":
    test_crosslevel_multibuffer()
    print("PASS: test_crosslevel_multibuffer")
