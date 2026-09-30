"""Validate normalized values and reciprocal standard deviations."""

import pytest
import torch
import tilelang
import tilelang.testing
from example_rmsnorm import rms_norm_fwd, ref_program


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
@pytest.mark.parametrize("d", [4096, 5120, 7168])
def test_rmsnorm(d, target):
    batch, eps = 128, 1e-6  # Two rows per core exercise buffer reuse.
    x, weight = torch.randn(batch, d), torch.randn(d)
    kernel = tilelang.compile(rms_norm_fwd(batch, d), target=target, out_idx=[1, 3])
    y, rstd = kernel(x.flatten().npu(), weight.npu(), eps)
    torch.testing.assert_close(y.cpu().reshape(batch, d), ref_program(x, weight, eps), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rstd.cpu(), torch.rsqrt(x.square().mean(-1) + eps), rtol=1e-5, atol=1e-5)
