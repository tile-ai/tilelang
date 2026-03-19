import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
import argparse


@tilelang.jit
def matmul_non_persistent(
    A,
    B,
    block_M: int = 128,
    block_N: int = 256,
    block_K: int = 64,
    threads: int = 256,
    num_stages: int = 3,
    dtype: T.dtype = T.float16,
    accum_dtype=T.float32,
):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), dtype)

        T.use_swizzle(10)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[bx * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, by * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C_shared)
        T.copy(C_shared, C[bx * block_M, by * block_N])

    return C


@tilelang.jit
def matmul_persistent(
    A,
    B,
    block_M: int = 128,
    block_N: int = 256,
    block_K: int = 64,
    threads: int = 256,
    num_stages: int = 3,
    dtype: T.dtype = T.float16,
    accum_dtype=T.float32,
):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty([M, N], dtype)

    sm_num = driver.get_num_sms()

    with T.Kernel(sm_num, threads=threads) as (block_id):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), dtype)

        for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)], sm_num, block_id):
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[bx * block_M, by * block_N])

    return C


def ref_program(A, B):
    return A @ B


def main(M=4096, N=4096, K=4096):
    import torch

    total_flops = 2 * M * N * K

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    threads = 256
    num_stages = 3

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    persistent_kernel = matmul_persistent.compile(
        a, b, block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K, threads=threads, num_stages=num_stages
    )
    persistent_profiler = persistent_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    persistent_profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("Persistent GEMM: All check passed.")
    persistent_latency = persistent_profiler.do_bench(warmup=500)
    print(f"Persistent GEMM Latency: {persistent_latency} ms")
    print(f"Persistent GEMM TFlops: {total_flops / persistent_latency * 1e-9} TFlops")

    non_persistent_kernel = matmul_non_persistent.compile(
        a, b, block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K, threads=threads, num_stages=num_stages
    )
    non_persistent_profiler = non_persistent_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    non_persistent_profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("Non-Persistent GEMM: All check passed.")
    non_persistent_latency = non_persistent_profiler.do_bench(warmup=500)
    print(f"Non-Persistent GEMM Latency: {non_persistent_latency} ms")
    print(f"Non-Persistent GEMM TFlops: {total_flops / non_persistent_latency * 1e-9} TFlops")

    print(f"Persistent GEMM Speedup: {non_persistent_latency / persistent_latency}")


def run_regression_perf(M=4096, N=4096, K=4096):
    import torch

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    threads = 256
    num_stages = 3

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    persistent_kernel = matmul_persistent.compile(
        a, b, block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K, threads=threads, num_stages=num_stages
    )
    persistent_profiler = persistent_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    return persistent_profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192, help="M dimension")
    parser.add_argument("--N", type=int, default=8192, help="N dimension")
    parser.add_argument("--K", type=int, default=8192, help="K dimension")
    args = parser.parse_args()
    M, N, K = args.M, args.N, args.K
    main(M, N, K)
