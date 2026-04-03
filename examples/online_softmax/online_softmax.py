import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit
def softmax_kernel(X, BLOCK_M=1, BLOCK_N=8192, dtype: T.dtype = T.float16):
    X: T.Tensor([M, N], dtype)
    Y = T.empty([M, N], dtype)

    accum_dtype = T.float32

    scale = 1.44269504  # log2(e)

    with T.Kernel(T.ceildiv(M, BLOCK_M), threads=128) as (i_m):
        x = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
        y = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
        lse = T.alloc_fragment([BLOCK_M], accum_dtype)
        max_x = T.alloc_fragment([BLOCK_M], dtype)
        exp_x = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
        sum_exp_x = T.alloc_fragment([BLOCK_M], accum_dtype)
        T.fill(lse, -T.infinity(accum_dtype))

        for i_n in T.Pipelined(T.ceildiv(N, BLOCK_N)):
            T.copy(X[i_m * BLOCK_M, i_n * BLOCK_N], x)
            T.reduce_max(x, max_x, dim=1, clear=True)

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                exp_x[i, j] = T.exp2(x[i, j] * scale - max_x[i] * scale)

            T.reduce_sum(exp_x, sum_exp_x, dim=1, clear=True)

            for i in T.Parallel(BLOCK_M):
                lse[i] = max_x[i] * scale + T.log2(T.exp2(lse[i] - max_x[i] * scale) + sum_exp_x[i])

        for i_n in T.Pipelined(T.ceildiv(N, BLOCK_N)):
            T.copy(X[i_m * BLOCK_M, i_n * BLOCK_N], x)

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                y[i, j] = T.exp2(x[i, j] * scale - lse[i])

            T.copy(y, Y[i_m * BLOCK_M, i_n * BLOCK_N])

    return Y


M = 8192
N = 8192
dtype = torch.float16
X = torch.randn(M, N, dtype=dtype, device="cuda")
Y = softmax_kernel(X)
Y_ref = X.softmax(dim=1)

torch.testing.assert_close(Y, Y_ref, rtol=1e-2, atol=1e-2)

t1 = do_bench(lambda: X.softmax(dim=1), warmup=25, rep=100)
t2 = do_bench(lambda: softmax_kernel(X), warmup=25, rep=100)
print(f"torch latency: {t1:.3f} ms")
print(f"TileLang latency: {t2:.3f} ms")
print(f"Speedup: {t1 / t2:.3f}x")
