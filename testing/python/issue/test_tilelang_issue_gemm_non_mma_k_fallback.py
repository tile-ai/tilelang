import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_gemm(m, n, k):
    @tilelang.jit
    def gemm(
        A: T.Tensor((m, k), T.float16),
        B: T.Tensor((k, n), T.float16),
        C: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((m, k), T.float16)
            B_shared = T.alloc_shared((k, n), T.float16)
            C_local = T.alloc_fragment((m, n), T.float16)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    return gemm


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("k", [16, 20, 28])
def test_gemm_falls_back_for_non_swizzle_stride(k):
    m = n = 64
    kernel = _make_gemm(m, n, k)

    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    kernel(a, b, c)

    torch.testing.assert_close(c, a @ b, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    tilelang.testing.main()
