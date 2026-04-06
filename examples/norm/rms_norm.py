import torch
import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
def rms_norm_splitk(A, blk_m, blk_k):
    M, N = T.const("M, N")
    dtype = T.float

    A: T.Tensor((M, N), dtype)
    B = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
        A_shared = T.alloc_shared((blk_m, blk_k), dtype)
        A_local = T.alloc_fragment((blk_m, blk_k), dtype)
        A_powsum = T.alloc_fragment((blk_m,), dtype)

        num_k_step = T.ceildiv(N, blk_k)
        T.clear(A_local)
        for k in T.Serial(num_k_step):
            T.copy(A[bx * blk_m, k * blk_k], A_shared)
            for i, j in T.Parallel(blk_m, blk_k):
                A_local[i, j] += A_shared[i, j] * A_shared[i, j]
        T.reduce_sum(A_local, A_powsum, dim=1)
        for i in T.Parallel(blk_m):
            A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)

        for k in T.Serial(num_k_step):
            # reverse, better cache hit rate
            T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
            for i, j in T.Parallel(blk_m, blk_k):
                A_shared[i, j] *= A_powsum[i]
            T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return B


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
def rms_norm(A, blk_m):
    M, N = T.const("M, N")
    dtype = T.float

    A: T.Tensor((M, N), dtype)
    B = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
        A_local = T.alloc_fragment((blk_m, N), dtype)
        A_pow_local = T.alloc_fragment((blk_m, N), dtype)
        A_powsum = T.alloc_fragment((blk_m,), dtype)

        T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_local)
        for i, j in T.Parallel(blk_m, N):
            A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
        T.reduce_sum(A_pow_local, A_powsum, dim=1)
        for i in T.Parallel(blk_m):
            A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)
        for i, j in T.Parallel(blk_m, N):
            A_local[i, j] *= A_powsum[i]
        T.copy(A_local, B[bx * blk_m : (bx + 1) * blk_m, :])

    return B


def ref_program(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12)


def test_rms_norm(M=1024, N=1024, blk_m=1):
    kernel = rms_norm.compile(M=M, N=N, blk_m=blk_m)
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    M, N, blk_m, blk_k = 8192, 8192, 1, 512
    kernel = rms_norm.compile(M=M, N=N, blk_m=blk_m)
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = profiler.do_bench(warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))
