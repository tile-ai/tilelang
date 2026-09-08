"""pytest test for example_simtvf_vector_add.py."""

import pytest
import torch
import tilelang

from example_simtvf_vector_add import ref_program, vector_add


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_simtvf_vector_add(target):
    N = 1024
    kernel = tilelang.compile(vector_add(N), target=target, out_idx=-1)
    device = torch.device("npu")
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()
    assert torch.equal(c, ref_program(a, b))


if __name__ == "__main__":
    test_simtvf_vector_add(target="ascend")
    print("PASS: test_simtvf_vector_add (ascend)")
    test_simtvf_vector_add(target="pto")
    print("PASS: test_simtvf_vector_add (pto)")
