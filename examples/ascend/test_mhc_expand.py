"""pytest tests for example_mhc_expand.py through the AscendC backend."""

import pytest
import torch
import tilelang

from example_mhc_expand import mhc_expand_bwd, mhc_expand_fwd, ref_program_expand_bwd, ref_program_expand_fwd


@pytest.mark.parametrize("target", ["ascend"])
def test_mhc_expand_fwd(target):
    num_tokens, hidden, mhc_mult = 128, 1280, 4
    kernel = tilelang.compile(mhc_expand_fwd(hidden, mhc_mult), target=target, out_idx=-1)

    device = torch.device("npu")
    x = torch.randn((num_tokens, hidden), device=device, dtype=torch.bfloat16)

    actual = kernel(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, ref_program_expand_fwd(x, mhc_mult))


@pytest.mark.parametrize("target", ["ascend"])
def test_mhc_expand_bwd(target):
    num_tokens, hidden, mhc_mult = 128, 1280, 4
    kernel = tilelang.compile(mhc_expand_bwd(hidden, mhc_mult), target=target, out_idx=-1)

    device = torch.device("npu")
    o_grad = torch.randn((num_tokens, mhc_mult, hidden), device=device, dtype=torch.bfloat16)

    x_grad = kernel(o_grad)
    torch.npu.synchronize()
    torch.testing.assert_close(x_grad, ref_program_expand_bwd(o_grad), rtol=1e-5, atol=2e-5)


if __name__ == "__main__":
    test_mhc_expand_fwd("ascend")
    print("PASS: test_mhc_expand_fwd")
    test_mhc_expand_bwd("ascend")
    print("PASS: test_mhc_expand_bwd")
