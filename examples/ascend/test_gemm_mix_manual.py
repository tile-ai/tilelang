"""pytest test for example_gemm_mix_manual.py — manual Cube+Vector mix GEMM with flags.

Tests the manual flag pipelining and cross-core sync pattern at M=K=N=8192 with bfloat16.
"""

import torch
import tilelang
import pytest

from example_gemm_mix_manual import gemm, ref_program

TARGETS = ["ascend"]


def _test_gemm_mix_manual(target):
    M, K, N = 8192, 8192, 8192
    kernel = tilelang.compile(
        gemm(M, K, N),
        target=target,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    device = torch.device("npu")
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    c = kernel(x, w)
    torch.npu.synchronize()

    expected = ref_program(x, w)
    max_diff = torch.max(torch.abs(c - expected)).item()
    assert max_diff < 1e-2, f"Results mismatch! Max diff: {max_diff}"


@pytest.mark.parametrize("target", TARGETS)
def test_gemm_mix_manual(target):
    _test_gemm_mix_manual(target)


if __name__ == "__main__":
    test_gemm_mix_manual("ascend")
    print("PASS: test_gemm_mix_manual")
