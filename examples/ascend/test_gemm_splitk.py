"""Correctness and repeatability of the two split-K reduction modes."""

import pytest
import torch
import tilelang
import tilelang.testing
from example_gemm_splitk import gemm_splitk, ref_program


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("deterministic", [False, True], ids=["atomic", "ordered"])
def test_gemm_splitk(deterministic):
    m, k, n, split_k = 512, 4096, 512, 8
    kernel = tilelang.compile(gemm_splitk(m, k, n, split_k, deterministic), out_idx=-1)
    x = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    x_npu, w_npu = x.npu(), w.npu()
    result = kernel(x_npu, w_npu).cpu()
    torch.testing.assert_close(result, ref_program(x, w), rtol=1e-2, atol=1e-2)
    if deterministic:
        torch.testing.assert_close(kernel(x_npu, w_npu).cpu(), result, rtol=0, atol=0)
