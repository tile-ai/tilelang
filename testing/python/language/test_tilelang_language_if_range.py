import tilelang
import tilelang.language as T
import torch
import tilelang.testing


@tilelang.jit(
    out_idx=[1],
)
def tilelang_if_range(M, N, block_M, block_N, dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                row_idx = by * block_M + i
                col_idx = bx * block_N + j
                # Test condition: ca < i < cb where ca=16, cb=96
                if 16 < row_idx < 96:
                    B[row_idx, col_idx] = A[row_idx, col_idx] * 2.0
                else:
                    B[row_idx, col_idx] = A[row_idx, col_idx] * 0.5

    return main


def run_tilelang_if_range(M=128, N=128, block_M=32, block_N=32, dtype=T.float16):
    kernel = tilelang_if_range(M, N, block_M, block_N, dtype)
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)

    # Reference computation
    ref_b = torch.zeros_like(a)
    for i in range(M):
        for j in range(N):
            # ca < i < cb where ca=16, cb=96
            if 16 < i < 96:
                ref_b[i, j] = a[i, j] * 2.0
            else:
                ref_b[i, j] = a[i, j] * 0.5

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


def test_tilelang_if_range():
    run_tilelang_if_range(M=128, N=128, block_M=32, block_N=32)


if __name__ == "__main__":
    tilelang.testing.main()
