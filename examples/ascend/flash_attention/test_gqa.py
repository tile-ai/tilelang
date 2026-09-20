"""Correctness test for the grouped-query attention kernel."""

import pytest
import torch
import tilelang

from example_gqa import gqa, ref_program


TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]


@pytest.mark.parametrize("target", TARGETS)
def test_gqa(target):
    S1, G, S2, D = 8192, 32, 8192, 128
    kernel = tilelang.compile(
        gqa(S1, G, S2, D),
        target=target,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )

    device = torch.device("npu")
    torch.manual_seed(42)
    q = torch.randn(S1 * G, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(S2, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(S2, D, dtype=torch.bfloat16, device=device)

    output = kernel(q, k, v)
    torch.npu.synchronize()
    expected = ref_program(q, k, v, S1=S1, G=G)
    max_diff = (output.cpu().float() - expected.cpu().float()).abs().max().item()
    rel_diff = max_diff / expected.cpu().float().abs().max().item()
    assert rel_diff < 0.05, f"rel_diff={rel_diff:.4f}"


if __name__ == "__main__":
    test_gqa("ascend")
