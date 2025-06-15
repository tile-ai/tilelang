# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
import torch


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def tilelang_copy(M, N, block_M, block_N, dtype="float16"):

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = A[by * block_M + i, bx * block_N + j]

    return main


def run_tilelang_copy(M=1024, N=1024, block_M=128, block_N=128, dtype="float16"):
    program = tilelang_copy(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="cuda",
        pass_configs={
            "tl.disable_warp_specialized": True,
            "tl.disable_tma_lower": True
        })
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy():
    run_tilelang_copy(M=1024, N=1024, block_M=128, block_N=128)
    run_tilelang_copy(M=1024, N=576, block_M=32, block_N=576)
    run_tilelang_copy(M=1024, N=576, block_M=32, block_N=576, dtype="float")


if __name__ == "__main__":
    tilelang.testing.main()
