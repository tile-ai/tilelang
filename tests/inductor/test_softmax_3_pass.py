import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench


M = 8192
N = 65536
dtype = T.float32
block_M = 2
block_N = 16384


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def softmax():
    @T.prim_func
    def softmax_kernel(
        input: T.Tensor((M, N), dtype),
        output: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(M // block_M, threads=256) as (bx,):
            rmax_shared = T.alloc_reducer((block_M,), dtype, "max", replication="all")
            rsum_shared = T.alloc_reducer((block_M,), dtype, "sum", replication="all")
            T.fill(rmax_shared, -T.infinity(dtype))
            T.fill(rsum_shared, T.float32(0.0))

            input_shared_dyn = T.alloc_fragment((block_M, block_N), dtype)

            for i in range(N // block_N):
                T.copy(input[bx * block_M : bx * block_M + block_M, i * block_N : (i + 1) * block_N], input_shared_dyn)
                for ii, jj in T.Parallel(block_M, block_N):
                    rmax_shared[ii] = T.max(rmax_shared[ii], input_shared_dyn[ii, jj])

            T.finalize_reducer(rmax_shared)

            for i in range(N // block_N):
                T.copy(input[bx * block_M : bx * block_M + block_M, i * block_N : (i + 1) * block_N], input_shared_dyn)
                for ii, jj in T.Parallel(block_M, block_N):
                    rsum_shared[ii] = rsum_shared[ii] + T.exp(input_shared_dyn[ii, jj] - rmax_shared[ii])

            T.finalize_reducer(rsum_shared)

            for i in range(N // block_N):
                T.copy(input[bx * block_M : bx * block_M + block_M, i * block_N : (i + 1) * block_N], input_shared_dyn)
                for ii, jj in T.Parallel(block_M, block_N):
                    output[bx * block_M + ii, i * block_N + jj] = T.exp(input_shared_dyn[ii, jj] - rmax_shared[ii]) / rsum_shared[ii]

    return softmax_kernel


@torch.compile()
def fn_torch(a):
    return torch.softmax(a, dim=1)


def main():
    kernel = softmax()
    a = torch.randn((M, N), device="cuda")
    c = torch.empty((M, N), device="cuda")
    kernel(a, c)

    ref_c = fn_torch(a)
    torch.testing.assert_close(c, ref_c, rtol=1e-4, atol=1e-4)

    tilelang_time = do_bench(lambda: kernel(a, c), backend="cupti")
    torch_time = do_bench(lambda: fn_torch(a), backend="cupti")
    print(f"TileLang kernel time: {tilelang_time} ms")
    print(f"Torch kernel time: {torch_time} ms")
    print(f"Speedup: {torch_time / tilelang_time}x")


if __name__ == "__main__":
    main()
