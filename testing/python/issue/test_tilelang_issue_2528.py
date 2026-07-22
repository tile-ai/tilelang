import torch
import tilelang
import tilelang.language as T


def make_kernel_2d(M, K, num_stages, dtype="float16"):
    """2-D shared buffer: shape [M, K], pipelined T.copy 触发 multi-version."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((M, K), dtype),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), dtype)  
            for _k in T.Pipelined(2, num_stages=num_stages):
                T.copy(A, A_shared)
                for i, j in T.Parallel(M, K):
                    B[i, j] = A_shared[i, j] * T.Cast(dtype, 2.0)

    return tilelang.compile(kernel)


def trial_2d(M, K, num_stages):
    kernel = make_kernel_2d(M, K, num_stages)
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.empty_like(a)
    kernel(a, b)
    torch.testing.assert_close(b, a * 2.0, rtol=1e-3, atol=1e-3)
    print(f"PASS  2D shape=[{M},{K}] num_stages={num_stages}")


if __name__ == "__main__":
    trial_2d(M=8, K=4, num_stages=2)

    trial_2d(M=16, K=4, num_stages=2)

    trial_2d(M=8, K=4, num_stages=1)

    @T.prim_func
    def kernel_3d(
        A: T.Tensor((2, 4, 4), "float16"),
        B: T.Tensor((2, 4, 4), "float16"),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((2, 4, 4), "float16")
            for _ in T.Pipelined(2, num_stages=2):
                T.copy(A, S)
                for i, j, k in T.Parallel(2, 4, 4):
                    B[i, j, k] = S[i, j, k] * T.Cast("float16", 2.0)

    k3 = tilelang.compile(kernel_3d)
    a3 = torch.randn(2, 4, 4, dtype=torch.float16, device="cuda")
    b3 = torch.empty_like(a3)
    k3(a3, b3)
    torch.testing.assert_close(b3, a3 * 2.0, rtol=1e-3, atol=1e-3)
    print("PASS  3D shape=[2,4,4] num_stages=2")
