"""Correctness tests for the auto-scheduled Ascend Split-K GEMM."""

import torch
import tilelang
import pytest

from example_gemm_splitk import gemm_splitk, ref_program


def _test_gemm_splitk(deterministic, target):
    M, K, N, split_k = 512, 8192, 512, 8
    device = torch.device("npu")
    torch.manual_seed(42)

    kernel = tilelang.compile(gemm_splitk(M, K, N, split_k, deterministic=deterministic), target=target, out_idx=-1)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    result = kernel(x, w)
    torch.npu.synchronize()

    torch.testing.assert_close(result, ref_program(x, w), rtol=1e-2, atol=1e-2)
    if deterministic:
        repeated = kernel(x, w)
        torch.npu.synchronize()
        assert torch.equal(result, repeated)


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_gemm_splitk(target):
    _test_gemm_splitk(deterministic=False, target=target)
    _test_gemm_splitk(deterministic=False, target=target)


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_gemm_splitk_deterministic(target):
    _test_gemm_splitk(deterministic=True, target=target)
    _test_gemm_splitk(deterministic=True, target=target)


if __name__ == "__main__":
    test_gemm_splitk(target="ascend")
    test_gemm_splitk_deterministic(target="ascend")
    print("PASS: test_gemm_splitk (ascend)")
