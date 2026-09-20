"""pytest test for example_simtvf_vecadd_mutex.py — mutex-based get_buf/rls_buf sync."""

import pytest
import torch
import tilelang

from example_simtvf_vecadd_mutex import ref_program, vector_add


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_simtvf_vecadd_mutex(target):
    N = 2**30
    kernel = tilelang.compile(
        vector_add(N),
        out_idx=-1,
        target=target,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    device = torch.device("npu")
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()
    assert torch.equal(c, ref_program(a, b))


if __name__ == "__main__":
    test_simtvf_vecadd_mutex(target="ascend")
    print("PASS: test_simtvf_vecadd_mutex (ascend)")
