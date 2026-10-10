"""pytest tests for example_mhc_sinkhorn.py through the AscendC backend."""

import pytest
import torch
import tilelang

from example_mhc_sinkhorn import (
    mhc_sinkhorn_bwd,
    mhc_sinkhorn_fwd,
    ref_program_sinkhorn_bwd,
    ref_program_sinkhorn_fwd,
)


@pytest.mark.parametrize("target", ["ascend"])
def test_mhc_sinkhorn_fwd(target):
    num_tokens, hidden_size = 64, 4
    token_block_size, repeat, eps = 1, 10, 1e-6
    kernel = tilelang.compile(mhc_sinkhorn_fwd(hidden_size, token_block_size, repeat, eps), target=target, out_idx=-1)

    device = torch.device("npu")
    comb_res_mix = torch.randn((num_tokens, hidden_size, hidden_size), device=device, dtype=torch.float32)

    actual = kernel(comb_res_mix)
    torch.npu.synchronize()
    expected = ref_program_sinkhorn_fwd(comb_res_mix, repeat, eps)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("target", ["ascend"])
def test_mhc_sinkhorn_bwd(target):
    num_tokens, hidden_size = 128, 4
    token_block_size, repeat, eps = 32, 2, 1e-6
    kernel = tilelang.compile(mhc_sinkhorn_bwd(hidden_size, token_block_size, repeat, eps), target=target, out_idx=-1)

    device = torch.device("npu")
    grad_output = torch.randn((num_tokens, hidden_size, hidden_size), device=device, dtype=torch.float32)
    x = torch.randn((num_tokens, hidden_size, hidden_size), device=device, dtype=torch.float32)

    grad_input = kernel(grad_output, x)
    torch.npu.synchronize()
    expected = ref_program_sinkhorn_bwd(grad_output, x, repeat, eps)
    torch.testing.assert_close(grad_input, expected, rtol=1e-5, atol=2e-5)


if __name__ == "__main__":
    test_mhc_sinkhorn_fwd("ascend")
    print("PASS: test_mhc_sinkhorn_fwd")
    test_mhc_sinkhorn_bwd("ascend")
    print("PASS: test_mhc_sinkhorn_bwd")
