"""pytest test for example_simtvf_ubuf_multi.py — E = (A * B + C) * D."""

import pytest
import torch
import tilelang

from example_simtvf_ubuf_multi import ref_program, ubuf_multi


@pytest.mark.parametrize("target", ["ascend"])
def test_simtvf_ubuf_multi(target):
    N = 256
    kernel = tilelang.compile(ubuf_multi(N), target=target, out_idx=-1)
    device = torch.device("npu")
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    c = torch.randn(N, dtype=torch.float32, device=device)
    d = torch.randn(N, dtype=torch.float32, device=device)
    e = kernel(a, b, c, d)
    torch.npu.synchronize()
    assert torch.equal(e, ref_program(a, b, c, d))


if __name__ == "__main__":
    test_simtvf_ubuf_multi(target="ascend")
    print("PASS: test_simtvf_ubuf_multi (ascend)")
