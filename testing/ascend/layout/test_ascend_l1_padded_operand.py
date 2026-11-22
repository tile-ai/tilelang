"""Regression: an L1 GEMM operand padded in M or N (#4440).

The GM->L1 ND2NZ fill lays the tile out with the allocation row count as its
fractal pitch. Before this fix the L1->L0 descriptors used the GEMM region
extent instead, so a GEMM reading a narrower M/N sub-region of a padded L1
tile walked the wrong stride and produced silently wrong numbers.
"""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing


def _padded_l1_kernel(alloc_m, use_m, alloc_n, use_n, k, dtype="float16"):
    @T.prim_func
    def kernel(
        X: T.Tensor((alloc_m, k), dtype),
        W: T.Tensor((alloc_n, k), dtype),
        C: T.Tensor((use_m, use_n), "float32"),
    ):
        with T.Kernel(1):
            res = T.alloc_l0c((use_m, use_n), "float32")
            x_l1 = T.alloc_l1((alloc_m, k), dtype)
            w_l1 = T.alloc_l1((alloc_n, k), dtype)
            T.copy(X, x_l1)
            T.copy(W, w_l1)
            T.gemm(x_l1[0:use_m, 0:k], w_l1[0:use_n, 0:k], res, transpose_B=True, clear_accum=True)
            T.copy(res, C)

    return kernel


@pytest.mark.parametrize(
    "alloc_m,use_m,alloc_n,use_n",
    [
        (64, 64, 48, 48),  # control: no padding
        (128, 64, 48, 48),  # M padded
        (64, 64, 64, 48),  # N padded
        (128, 64, 64, 48),  # both padded
    ],
)
def test_l1_gemm_reads_padded_operand_subregion(alloc_m, use_m, alloc_n, use_n):
    torch.manual_seed(0)
    k = 64
    kernel = tilelang.compile(_padded_l1_kernel(alloc_m, use_m, alloc_n, use_n, k), target="ascend")
    x = (torch.randn(alloc_m, k, device="npu") * 0.25).half()
    w = (torch.randn(alloc_n, k, device="npu") * 0.25).half()
    c = torch.full((use_m, use_n), -999.0, device="npu", dtype=torch.float32)
    kernel(x, w, c)
    torch.npu.synchronize()
    ref = x[:use_m].float() @ w[:use_n].float().T
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
