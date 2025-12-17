import tilelang
import tilelang.testing
import tilelang.language as T
import torch


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def tilelang_composable_copy(M, N, block_M, block_N, dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M * N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_local = T.alloc_fragment([block_M, block_N], dtype)
            B_local = T.alloc_fragment([block_M * block_N], dtype)
            T.copy(A[by * block_M, bx * block_N], A_local)
            for i, j in T.Parallel(block_M, block_N):
                B_local[i * block_N + j] = A_local[i, j]
            for i in T.Parallel(block_M * block_N):
                B[by * block_M * N + bx * block_N + i // block_N * N + i % block_N] = B_local[i]

    return main


def run_tilelang_composable_copy(M=1024, N=1024, block_M=128, block_N=128, dtype=T.float16):
    program = tilelang_composable_copy(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b.flatten(), a.flatten(), rtol=1e-2, atol=1e-2)


def test_tilelang_copy():
    run_tilelang_composable_copy(M=1024, N=1024, block_M=128, block_N=128)
    run_tilelang_composable_copy(M=1024, N=576, block_M=32, block_N=576)
    run_tilelang_composable_copy(M=1024, N=576, block_M=32, block_N=576, dtype=T.float32)


if __name__ == "__main__":
    tilelang.testing.main()
