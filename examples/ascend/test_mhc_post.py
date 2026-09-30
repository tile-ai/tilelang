"""pytest tests for example_mhc_post.py through the AscendC backend."""

import pytest
import torch
import tilelang

from example_mhc_post import mhc_post_fwd, ref_program


@pytest.mark.parametrize("target", ["ascend"])
def test_mhc_post_fwd(target):
    num_tokens, mhc, hidden = 8, 4, 1280
    kernel = tilelang.compile(mhc_post_fwd(mhc, hidden), target=target, out_idx=-1)

    device = torch.device("npu")
    a = torch.randn((num_tokens, mhc, mhc), device=device, dtype=torch.float32)
    b = torch.randn((num_tokens, mhc, hidden), device=device, dtype=torch.bfloat16)
    c = torch.randn((num_tokens, mhc), device=device, dtype=torch.float32)
    d = torch.randn((num_tokens, hidden), device=device, dtype=torch.bfloat16)

    actual = kernel(a, b, c, d)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, ref_program(a, b, c, d), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_mhc_post_fwd("ascend")
    print("PASS: test_mhc_post_fwd")
