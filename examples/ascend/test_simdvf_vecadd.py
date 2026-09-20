"""pytest test for example_simdvf_vecadd.py — high-level SIMD vector add with auto-schedule."""

import pytest
import torch
import tilelang

from example_simdvf_vecadd import ref_program, vector_add


TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]


@pytest.mark.parametrize("target", TARGETS)
def test_simdvf_vecadd(target):
    N = 2**30
    kernel = tilelang.compile(vector_add(N), target=target, out_idx=-1)
    device = torch.device("npu")
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()
    assert torch.equal(c, ref_program(a, b))


if __name__ == "__main__":
    test_simdvf_vecadd()
    print("PASS: test_simdvf_vecadd")
