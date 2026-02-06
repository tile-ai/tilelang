import tilelang
import tilelang.language as T
import torch
import tilelang.testing


@tilelang.jit(
    out_idx=[1],
)
def tilelang_ternary(M, N, block_M, block_N, dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = A[by * block_M + i, bx * block_N + j] if (by * block_M + i) < (M // 2) else 0

    return main


def run_tilelang_ternary(M=128, N=128, block_M=32, block_N=32, dtype=T.float16):
    kernel = tilelang_ternary(M, N, block_M, block_N, dtype)
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    ref_b = torch.zeros_like(b)
    for i in range(M):
        for j in range(N):
            if i < M // 2:
                ref_b[i, j] = a[i, j]
            else:
                ref_b[i, j] = 0

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


def test_tilelang_ternary():
    run_tilelang_ternary(M=128, N=128, block_M=32, block_N=32)


if __name__ == "__main__":
    tilelang.testing.main()
