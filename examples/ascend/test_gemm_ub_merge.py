"""pytest test for example_gemm_ub_merge.py — UB merge with Cube + Vector."""

import pytest
import torch
import tilelang

from example_gemm_ub_merge import gemm_ub_merge, ref_program


def _run_gemm_ub_merge(target="ascend"):
    M, K, N = 256, 256, 128
    kernel = tilelang.compile(
        gemm_ub_merge(M, K, N),
        target=target,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    device = torch.device("npu")
    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(N, K, dtype=torch.float16, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()
    expected = ref_program(a, b)
    max_diff = (c - expected).abs().max().item()
    assert max_diff < 1e-2, f"max_diff={max_diff:.2e}"


def test_gemm_ub_merge():
    _run_gemm_ub_merge(target="ascend")


@pytest.mark.pto
def test_gemm_ub_merge_pto():
    _run_gemm_ub_merge(target="pto")


if __name__ == "__main__":
    test_gemm_ub_merge()
    print("PASS: test_gemm_ub_merge")
