import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(
    A,
    B,
    C,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    split_k: int = 4,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
    out_dtype: T.dtype = T.float32,
):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[N, K], dtype]
    C: T.Tensor[[M, N], out_dtype]
    splitK = K // split_k

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
            T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
            T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C_shared)

        T.atomic_add(C[by * block_M, bx * block_N], C_shared)


def main():
    M = 1024
    N = 1024
    K = 1024

    import torch

    torch.random.manual_seed(42)
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()
    c = torch.zeros(M, N).cuda().float()
    matmul(a, b, c)

    ref_c = a @ b

    torch.testing.assert_close(c, ref_c.to(c.dtype), rtol=1e-2, atol=1e-2)


def run_regression_perf():
    M = 4096
    N = 4096
    K = 4096

    import torch

    torch.random.manual_seed(42)
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()
    c = torch.zeros(M, N).cuda().float()
    from tilelang.profiler import do_bench

    kernel = matmul.compile(a, b, c)

    def run_kernel_only():
        kernel(a, b, c)

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    main()
