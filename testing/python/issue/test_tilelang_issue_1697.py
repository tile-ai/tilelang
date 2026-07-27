import tilelang.language as T
import tilelang.testing
import tilelang
import torch


def matmu_jit_kernel(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_jit_kernel(M, N, K, block_M, block_N, block_K):
    program = matmu_jit_kernel(M, N, K, block_M, block_N, block_K)

    matmul_kernel = tilelang.compile(program, out_idx=-1, execution_backend="tvm_ffi")

    A = torch.randn(M, K, dtype=torch.float16).cuda()
    B = torch.randn(K, N, dtype=torch.float16).cuda()

    C = matmul_kernel(A, B)

    tilelang.testing.torch_assert_close(C, torch.matmul(A, B), atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.05)


def test_gemm_jit_kernel_zero_dim():
    run_gemm_jit_kernel(512, 1024, 0, 128, 256, 32)


if __name__ == "__main__":
    tilelang.testing.main()
