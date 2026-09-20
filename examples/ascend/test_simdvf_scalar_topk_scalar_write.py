"""pytest test for example_simdvf_scalar_topk_scalar_write.py — scalar GM write topk."""

import pytest
import torch
import tilelang
import tilelang.testing

from example_simdvf_scalar_topk_scalar_write import (
    NUM_EXPERTS,
    NUM_TOPK,
    make_kernel,
    ref_program,
    simulator_safe_randn,
)

# Include "(" so the assertion matches call sites, not declarations.
# ASC emits the AscendC template; PTO emits the PTODSL helper call.
BYPASS_CALLS = {
    "asc": "tl::write_gm_bypass_dcache(",
    "pto": "tl.write_gm_bypass_dcache(",
}


@pytest.mark.parametrize("backend", ["asc", pytest.param("pto", marks=pytest.mark.pto)])
def test_simdvf_scalar_topk_scalar_write(backend):
    kernel = make_kernel(backend)
    source = kernel.get_kernel_source()
    marker = BYPASS_CALLS[backend]
    assert source.count(marker) > 0, f"{backend} scalar GM write should emit {marker}"

    device = torch.device("npu")
    n = 4096
    torch.manual_seed(42)
    logits = simulator_safe_randn((n, NUM_EXPERTS), dtype=torch.float32, device=device)
    out = kernel(logits)
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), ref_program(logits, NUM_TOPK).cpu())


if __name__ == "__main__":
    tilelang.testing.main()
