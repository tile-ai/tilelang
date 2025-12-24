import tilelang
import tilelang.language as T

tilelang.disable_cache()


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
@tilelang.jit(out_idx=[2])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    num_stages = 2
    mbarrier_list = [128, 128] * num_stages

    @T.prim_func
    def main(
        A: T.Tensor[(M, K), dtype],
        B: T.Tensor[(K, N), dtype],
        C: T.Tensor[(M, N), dtype],
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            B_shared = T.alloc_shared((num_stages, block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # create mbarrier for tma
            T.create_list_of_mbarrier(mbarrier_list)

            with T.ws(0):
                T.clear(C_local)

            for ko in range(T.ceildiv(K, block_K)):
                with T.ws(1):
                    T.mbarrier_wait_parity(mbarrier=ko % num_stages + num_stages, parity=((ko // num_stages) % num_stages) ^ 1)
                    T.copy(A[by * block_M : (by + 1) * block_M, ko * block_K : (ko + 1) * block_K], A_shared[ko % num_stages, :, :])
                    T.copy(B[ko * block_K : (ko + 1) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbarrier=ko % num_stages)
                with T.ws(0):
                    T.mbarrier_wait_parity(mbarrier=ko % num_stages, parity=(ko // num_stages) % num_stages)
                    T.gemm(A_shared[ko % num_stages, :, :], B_shared[ko % num_stages, :, :], C_local)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)

            with T.ws(0):
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def main(M=16384, N=16384, K=16384):
    tilelang.disable_cache()
    block_M = 128
    block_N = 128
    block_K = 64
    jit_kernel = matmul(M, N, K, block_M, block_N, block_K)

    print(jit_kernel.get_kernel_source())

    import torch

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Run the kernel through the Profiler
    c = jit_kernel(a, b)

    # Reference multiplication using PyTorch
    ref_c = a @ b

    # Validate correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    # 4. Retrieve and inspect the generated CUDA source (optional)
    # cuda_source = jit_kernel.get_kernel_source()
    # print("Generated CUDA kernel:\n", cuda_source)

    # 5.Profile latency with kernel
    profiler = jit_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    latency = profiler.do_bench()

    print(f"Latency: {latency} ms")


def run_regression_perf(M=16384, N=16384, K=16384):
    tilelang.disable_cache()
    block_M = 128
    block_N = 128
    block_K = 64
    jit_kernel = matmul(M, N, K, block_M, block_N, block_K)
    profiler = jit_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main()
