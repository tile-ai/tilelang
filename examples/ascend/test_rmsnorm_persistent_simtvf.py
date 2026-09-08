"""Pytest coverage for persistent-fragment RMSNorm SIMT_VF kernels."""

import pytest
import torch
import tilelang

from example_rmsnorm_persistent_simtvf import (
    rms_norm_persistent_fwd,
    rms_norm_persistent_gm_to_fragment_fwd,
)


BATCH = 4096
D = 7168
EPS = 1e-6


@pytest.mark.pto
@pytest.mark.parametrize(
    "kernel_builder",
    [
        pytest.param(rms_norm_persistent_fwd, id="ub-to-fragment"),
        pytest.param(rms_norm_persistent_gm_to_fragment_fwd, id="gm-to-fragment"),
    ],
)
def test_rmsnorm_persistent_simtvf_pto(kernel_builder):
    """Check both persistent-weight initialization paths over two iterations."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    x_cpu = torch.randn((BATCH, D), dtype=torch.float32, generator=generator)
    weight_cpu = torch.randn(D, dtype=torch.float32, generator=generator)

    device = torch.device("npu")
    x = x_cpu.to(device)
    weight = weight_cpu.to(device)

    kernel = tilelang.compile(
        kernel_builder(BATCH, D, "float32"),
        target="pto",
        out_idx=[1, 3],
    )
    y_flat, rstd = kernel(x.view(-1).contiguous(), weight, EPS)
    torch.npu.synchronize()

    x_float = x.float()
    expected_rstd = torch.rsqrt(x_float.pow(2).mean(dim=-1) + EPS)
    expected_y = x_float * expected_rstd.unsqueeze(-1) * weight.float().unsqueeze(0)

    torch.testing.assert_close(y_flat.view(BATCH, D), expected_y, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(rstd, expected_rstd, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
