import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench


M = 8192
N = 65536
dtype = T.float32
block_M = 2
block_N = 16384

scale = 1.44269504  # log2(e)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def softmax():
    @T.prim_func
    def softmax_kernel(
        input: T.Tensor((M, N), dtype),
        output: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(M // block_M, threads=256) as (bx,):
            input_frag = T.alloc_fragment((block_M, block_N), dtype)
            exp_frag = T.alloc_fragment((block_M, block_N), dtype)
            output_frag = T.alloc_fragment((block_M, block_N), dtype)
            lse = T.alloc_fragment((block_M,), dtype)  # log-sum-exp per row
            local_max = T.alloc_fragment((block_M,), dtype)  # local max per tile
            local_sum = T.alloc_fragment((block_M,), dtype)  # local sum(exp) per tile

            T.fill(lse, -T.infinity(dtype))

            # Pass 1: Online softmax — fused max + sum in single pass
            for i in range(N // block_N):
                T.copy(input[bx * block_M : bx * block_M + block_M, i * block_N : (i + 1) * block_N], input_frag)

                T.reduce_max(input_frag, local_max, dim=1, clear=True)

                for ii, jj in T.Parallel(block_M, block_N):
                    exp_frag[ii, jj] = T.exp2(input_frag[ii, jj] * scale - local_max[ii] * scale)

                T.reduce_sum(exp_frag, local_sum, dim=1, clear=True)

                for ii in T.Parallel(block_M):
                    lse[ii] = local_max[ii] * scale + T.log2(T.exp2(lse[ii] - local_max[ii] * scale) + local_sum[ii])

            # Pass 2: Normalize — exp(x * scale - lse) = exp(x - max) / sum
            for i in range(N // block_N):
                T.copy(input[bx * block_M : bx * block_M + block_M, i * block_N : (i + 1) * block_N], input_frag)

                for ii, jj in T.Parallel(block_M, block_N):
                    output_frag[ii, jj] = T.exp2(input_frag[ii, jj] * scale - lse[ii])

                T.copy(output_frag, output[bx * block_M : bx * block_M + block_M, i * block_N : (i + 1) * block_N])

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
