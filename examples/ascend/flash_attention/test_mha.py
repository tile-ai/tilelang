"""pytest test for example_mha.py — MHA Mix kernel (SimdVF softmax)."""

import pytest
import torch
import tilelang

from example_mha import mha, ref_program


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_mha(dynamic):
    D = 128
    shapes = ((4096, 128), (4097, 128), (4097, 256)) if dynamic else ((4096, 4096),)

    kernel = tilelang.compile(
        mha(D, None if dynamic else 4096),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )

    device = torch.device("npu")
    torch.manual_seed(42)
    for q_len, kv_len in shapes:
        Q = torch.randn(q_len, D, dtype=torch.bfloat16, device=device)
        K = torch.randn(kv_len, D, dtype=torch.bfloat16, device=device)
        V = torch.randn(kv_len, D, dtype=torch.bfloat16, device=device)

        O = kernel(Q, K, V)
        torch.npu.synchronize()

        expected = ref_program(Q, K, V).cpu()
        max_diff = (O.cpu() - expected).abs().max().item()
        rel = max_diff / expected.abs().max().item()
        assert rel < 0.05, f"q_len={q_len}, kv_len={kv_len}, rel={rel:.4f}"


if __name__ == "__main__":
    test_mha(dynamic=False)
    test_mha(dynamic=True)
    print("PASS: test_mha")
