"""Metal lowering of math intrinsics that TVM's default legalizer would
otherwise rewrite: rsqrt must reach Metal's native rsqrt(), not 1/sqrt()."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def rsqrt_kernel(N, threads=128):
    @T.prim_func
    def kernel(A: T.Tensor((N,), T.float32), B: T.Tensor((N,), T.float32)):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < N:
                    B[idx] = T.rsqrt(A[idx])

    return kernel


@tilelang.testing.requires_metal
def test_rsqrt_is_native():
    N = 1024
    kernel = tilelang.compile(rsqrt_kernel(N), target="metal")
    src = kernel.get_kernel_source()
    assert "rsqrt(" in src
    assert "1.000000e+00f / sqrt" not in src
    a = torch.rand(N, device="mps") + 0.5
    b = torch.empty(N, device="mps")
    kernel(a, b)
    torch.mps.synchronize()
    torch.testing.assert_close(b, torch.rsqrt(a), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    tilelang.testing.main()
