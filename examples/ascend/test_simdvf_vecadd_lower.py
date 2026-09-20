"""pytest test for example_simdvf_vecadd_lower.py.

This variant exercises the SimdVF kernel with the Z3 auto-scheduler enabled
(the default), which requires AutoSchedule's MemoryAccessDetector to recognize
the simd vld/vsts buffer accesses (now emitted via tl.access_ptr).
"""

import pytest
import tilelang
import tilelang.testing
import torch

from example_simdvf_vecadd_lower import DEFAULT_N, ref_program, vector_add


def _run_vector_add(backend):
    n = DEFAULT_N
    kernel = tilelang.compile(vector_add(n, backend), target=backend, out_idx=-1)

    device = torch.device("npu")
    a = torch.randn(n, dtype=torch.float32, device="cpu").to(device)
    b = torch.randn(n, dtype=torch.float32, device="cpu").to(device)
    c = kernel(a, b)
    torch.npu.synchronize()

    assert torch.equal(c.cpu(), ref_program(a, b).cpu())


@pytest.mark.parametrize("backend", ["asc", pytest.param("pto", marks=pytest.mark.pto)])
def test_simdvf_vecadd_lower(backend):
    _run_vector_add(backend)


if __name__ == "__main__":
    tilelang.testing.main()
