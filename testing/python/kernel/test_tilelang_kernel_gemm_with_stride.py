import tilelang.testing
import tilelang
import tilelang.language as T
import torch


def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K * 2), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N * 2), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Clear local accumulation
            T.clear(C_local)
            T.clear(B_shared)
            T.clear(A_shared)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                # Copy tile of A
                # T.copy(A[by * block_M, ko * block_K], A_shared)
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k + block_K] = A[by * block_M + i, ko * block_K + k]

                # Copy tile of B
                # T.copy(B[ko * block_K, bx * block_N], B_shared)
                for i, k in T.Parallel(block_K, block_N):
                    B_shared[i, k] = B[ko * block_K + i, bx * block_N + k]

                # Perform a tile-level GEMM on the shared buffers
                # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
                T.gemm(A_shared[:, block_K:], B_shared[0:block_K, 0:block_N], C_local)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_with_stride_ss(M: int, N: int, K: int, block_M: int, block_N: int, block_K: int):
    # 1. Define the kernel (matmul) and compile/lower it into an executable module
    func = matmul(M, N, K, block_M, block_N, block_K)

    # 2. Compile the kernel into a torch function
    # out_idx specifies the index of the output buffer in the argument list
    # if out_idx is specified, the tensor will be created during runtime
    # target currently can be "cuda" or "hip" or "cpu".
    jit_kernel = tilelang.compile(
        func,
        out_idx=[2],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    # Create random input tensors on the GPU
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Run the kernel through the Profiler
    c = jit_kernel(a, b)

    print(c)
    # Reference multiplication using PyTorch
    ref_c = a @ b

    # Validate correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 5)
def test_tilelang_kernel_gemm_with_stride():
    run_gemm_with_stride_ss(128, 128, 64, 32, 32, 32)


if __name__ == "__main__":
    tilelang.testing.main()
