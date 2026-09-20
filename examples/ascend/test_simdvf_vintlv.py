"""ASC interleave/deinterleave roundtrip coverage."""

import pytest
import torch
import tilelang
import tilelang.testing

from example_simdvf_vintlv import N, simulator_safe_randn, vintlv_kernel


@pytest.mark.parametrize("backend", ["asc", pytest.param("pto", marks=pytest.mark.pto)])
def test_simdvf_vintlv(backend):
    kernel = tilelang.compile(vintlv_kernel(backend), target=backend)
    device = torch.device("npu")
    x = simulator_safe_randn(N, dtype=torch.float32, device=device)
    y = simulator_safe_randn(N, dtype=torch.float32, device=device)
    x_bak = torch.empty(N, dtype=torch.float32, device=device)
    y_bak = torch.empty(N, dtype=torch.float32, device=device)
    kernel(x, y, x_bak, y_bak)
    torch.npu.synchronize()
    assert torch.equal(x_bak.cpu(), x.cpu())
    assert torch.equal(y_bak.cpu(), y.cpu())


if __name__ == "__main__":
    tilelang.testing.main()
