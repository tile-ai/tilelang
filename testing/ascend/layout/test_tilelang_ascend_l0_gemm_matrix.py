import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing


def make_l0_gemm_kernel(M: int, K: int, N: int, dtype: str):
    @T.prim_func
    def gemm_kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            res = T.alloc_l0c((M, N), "float32")
            x_l1 = T.alloc_l1((M, K), dtype)
            w_l1 = T.alloc_l1((N, K), dtype)
            x_l0a = T.alloc_l0a((M, K), dtype)
            w_l0b = T.alloc_l0b((N, K), dtype)
            T.copy(X, x_l1)
            T.copy(W, w_l1)
            T.copy(x_l1, x_l0a)
            T.copy(w_l1, w_l0b)
            T.gemm(x_l0a, w_l0b, res, transpose_B=True, clear_accum=True)
            T.copy(res, C)

    return gemm_kernel


@pytest.mark.parametrize("dtype,atol,rtol", [("bfloat16", 1e-2, 1e-2), ("float8_e4m3fn", 1e-1, 5e-2)])
@pytest.mark.parametrize("m,k,n", [(1, 64, 64), (15, 15, 15), (32, 32, 32), (100, 100, 100)])
def test_l0_gemm_matrix(dtype, atol, rtol, m, k, n):
    kernel = tilelang.compile(make_l0_gemm_kernel(m, k, n, dtype), target="ascend", out_idx=-1)
    x = (torch.randn(m, k, device="npu") * 0.25).to(getattr(torch, dtype))
    w = (torch.randn(n, k, device="npu") * 0.25).to(getattr(torch, dtype))
    torch.testing.assert_close(kernel(x, w), x.float() @ w.float().T, atol=atol, rtol=rtol)
