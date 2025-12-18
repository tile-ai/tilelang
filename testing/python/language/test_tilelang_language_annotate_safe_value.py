import tilelang
import tilelang.language as T
import tilelang.testing
import torch


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def tilelang_copy(M, N, block_M, block_N, dtype=T.float16, pad_value=0):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.annotate_safe_value({A: pad_value})
            for i, j in T.Parallel(block_M, block_N):
                A_shared[i, j] = A[by * block_M + i - 10, bx * block_N + j]

            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = A_shared[i, j]

    return main


def run_tilelang_copy(M=1024, N=1024, block_M=128, block_N=128, dtype=T.float16, pad_value=0):
    program = tilelang_copy(M, N, block_M, block_N, dtype, pad_value=pad_value)
    kernel = tilelang.compile(
        program, out_idx=[1], target="cuda", pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    ref_b = torch.zeros_like(a)
    for i in range(M):
        if i >= 10:
            ref_b[i, :] = a[i - 10, :]
        else:
            ref_b[i, :] = pad_value
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


def test_tilelang_copy():
    run_tilelang_copy(M=1024, N=1024, block_M=128, block_N=128, pad_value=10)


if __name__ == "__main__":
    tilelang.testing.main()
