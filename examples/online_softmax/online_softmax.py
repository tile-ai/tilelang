import torch
import tilelang as tl
import tilelang.language as T
from tilelang.profiler import do_bench
from typing import Callable


@tl.jit(out_idx=[1])
def softmax_kernel(
    M,
    N,
    dtype: T.dtype = T.float16,
) -> "Callable":
    BN = min(tl.next_power_of_2(N), 8192)
    NN = tl.cdiv(N, BN)

    accum_dtype = T.float32

    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        X: T.Tensor([M, N], dtype),
        Y: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(M, threads=128) as (i_m):
            x = T.alloc_fragment([BN], dtype)
            y = T.alloc_fragment([BN], dtype)
            lse = T.alloc_fragment([1], accum_dtype)
            max_x = T.alloc_fragment([1], dtype)
            exp_x = T.alloc_fragment([BN], accum_dtype)
            sum_exp_x = T.alloc_fragment([1], accum_dtype)
            T.fill(lse, -T.infinity(accum_dtype))

            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)

                T.reduce_max(x, max_x, dim=0, clear=True)

                for j in T.Parallel(BN):
                    exp_x[j] = T.exp2(x[j] * scale - max_x[0] * scale)

                T.reduce_sum(exp_x, sum_exp_x, dim=0, clear=True)

                lse[0] = max_x[0] * scale + T.log2(T.exp2(lse[0] - max_x[0] * scale) + sum_exp_x[0])

            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)

                for j in T.Parallel(BN):
                    y[j] = T.exp2(x[j] * scale - lse[0])

                T.copy(y, Y[i_m, i_n * BN : (i_n + 1) * BN])

    return main


M = 8192
N = 8192
kernel = softmax_kernel(M, N)
dtype = torch.float16
X = torch.randn(M, N, dtype=dtype, device="cuda")
Y = kernel(X)
Y_ref = X.softmax(dim=1)

torch.testing.assert_close(Y, Y_ref, rtol=1e-2, atol=1e-2)

t1 = do_bench(lambda: X.softmax(dim=1), warmup=25, rep=100)
t2 = do_bench(lambda: kernel(X), warmup=25, rep=100)
print(f"torch latency: {t1:.3f} ms")
print(f"TileLang latency: {t2:.3f} ms")
print(f"Speedup: {t1 / t2:.3f}x")
