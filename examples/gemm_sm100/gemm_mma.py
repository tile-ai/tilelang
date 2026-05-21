import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((N, K), dtype)
    C = T.empty((M, N), dtype)

    # Initialize Kernel Context
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_N, block_K), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # Clear local accumulation
        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
            # Copy tile of A
            # This is a sugar syntax for parallelized copy
            # for i, k in T.Parallel(M, block_K):
            #     A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
            T.copy(A[by * block_M, ko * block_K], A_shared)

            # Copy tile of B
            T.copy(B[bx * block_N, ko * block_K], B_shared)

            # Perform a tile-level GEMM on the shared buffers
            # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)

        # Copy result back to global memory
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


M = 128
N = 128
K = 32
block_M = 128
block_N = 128
block_K = 32

a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(N, K, device="cuda", dtype=torch.float16)
c = matmul(a, b, block_M, block_N, block_K)
print(matmul.get_kernel_source(a, b, block_M, block_N, block_K))

ref_c = a @ b.T
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("All checks passed. ✅")

tl_latency = do_bench(lambda: matmul(a, b, block_M, block_N, block_K), backend="cupti")
torch_latency = do_bench(lambda: a @ b.T, backend="cupti")
print(f"Tilelang latency: {tl_latency} ms")
print(f"Flops: {2 * M * N * K / (tl_latency / 1e3) / 1e12} TFLOPS")
print(f"Torch latency: {torch_latency} ms")
print(f"Flops: {2 * M * N * K / (torch_latency / 1e3) / 1e12} TFLOPS")
