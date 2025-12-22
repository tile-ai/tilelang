import torch
from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import tilelang.language as T
from tilelang.utils import map_torch_type


@tl.jit
def tensor_null_test(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32, with_bias=False):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
        Bias: T.Tensor((N), accum_dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            if with_bias:
                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] += Bias[bx * block_N + j]

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_test(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    a = torch.randn(M, K, device="cuda", dtype=map_torch_type(dtype))
    b = torch.randn(N, K, device="cuda", dtype=map_torch_type(dtype))
    c = torch.zeros(M, N, device="cuda", dtype=map_torch_type(accum_dtype))
    kernel = tensor_null_test(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, with_bias=False)
    kernel(a, b, c, None)


def test_nullptr():
    run_test(1024, 1024, 1024, 128, 128, 32)


if __name__ == "__main__":
    tilelang.testing.main()
