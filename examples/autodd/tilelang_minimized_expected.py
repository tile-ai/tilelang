"""
This is the expected output after running AutoDD on tilelang_buggy.py.
AutoDD automatically simplified the 200+ line buggy program to ~30 lines
while preserving the ability to reproduce the error.

The minimized code clearly shows the root cause of the bug:
- A_shared has shape (block_M, block_K)
- B_shared has shape (block_M, block_N) - should be (block_K, block_N)
- This causes a dimension mismatch in T.gemm()
"""

import tilelang.language as T


class MatmulConfig:
    def __init__(self, *args, **kwargs):
        self.M = 1
        self.N = 1
        self.K = 1
        self.block_M = 2
        self.block_N = 1
        self.block_K = 1


def buggy_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32, *args, **kwargs):
    @T.prim_func
    def matmul_kernel():
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)  # Bug: should be (block_K, block_N)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.gemm(A_shared, B_shared, C_local)


def run_kernel(config, *args, **kwargs):
    M, N, K = (config.M, config.N, config.K)
    block_M, block_N, block_K = (config.block_M, config.block_N, config.block_K)
    buggy_matmul(M, N, K, block_M, block_N, block_K)


def main(*args, **kwargs):
    config = MatmulConfig()
    try:
        run_kernel(config)
    except Exception as e:
        print(f"{e}")


main()
