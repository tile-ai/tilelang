"""pytest test for example_simdvf_scalar_topk.py — SimdVF + scalar top-k on UB."""

import pytest
import torch
import tilelang
import tilelang.testing

from example_simdvf_scalar_topk import NUM_EXPERTS, NUM_TOPK, make_kernel, ref_program, simulator_safe_randn


@pytest.mark.parametrize("backend", ["asc", pytest.param("pto", marks=pytest.mark.pto)])
def test_simdvf_scalar_topk(backend):
    kernel = make_kernel(backend)
    device = torch.device("npu")
    n = 4096
    torch.manual_seed(42)
    logits = simulator_safe_randn((n, NUM_EXPERTS), dtype=torch.float32, device=device)
    out = kernel(logits)
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), ref_program(logits, NUM_TOPK).cpu())


if __name__ == "__main__":
    tilelang.testing.main()
