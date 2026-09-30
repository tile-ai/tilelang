"""pytest test for example_mha.py — MHA Mix kernel (SimdVF softmax)."""

import pytest
import torch
import tilelang

from example_mha import mha, ref_program


TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]


@pytest.mark.parametrize("target", TARGETS)
def test_mha(target):
    D = 128
    SEQ_LEN = 4096

    kernel = tilelang.compile(
        mha(D, SEQ_LEN),
        target=target,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )

    device = torch.device("npu")
    torch.manual_seed(42)
    Q = torch.randn(SEQ_LEN, D, dtype=torch.bfloat16, device=device)
    K = torch.randn(SEQ_LEN, D, dtype=torch.bfloat16, device=device)
    V = torch.randn(SEQ_LEN, D, dtype=torch.bfloat16, device=device)

    O = kernel(Q, K, V)
    torch.npu.synchronize()

    expected = ref_program(Q, K, V)
    max_diff = (O.cpu().float() - expected.cpu()).abs().max().item()
    rel = max_diff / expected.cpu().abs().max().item()
    assert rel < 0.05, f"rel={rel:.4f}"


if __name__ == "__main__":
    test_mha("ascend")
    print("PASS: test_mha")
