import torch
import tilelang
import tilelang.language as T


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    @T.prim_func
    def gemm_fp8(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_fp8


def test_gemm_fp8(M, N, K, dtype):
    torch_dtype = T.dtype(dtype).as_torch()

    kernel = matmul(M, N, K, 128, 128, 64, dtype)

    a = torch.randn(M, K, dtype=torch.float16, device="cuda").to(dtype=torch_dtype)
    b = torch.randn(N, K, dtype=torch.float16, device="cuda").to(dtype=torch_dtype)

    c = kernel(a, b)

    ref_c = (a.half() @ b.half().T).to(dtype=torch_dtype)

    print(c)
    print(ref_c)

    diff = calc_diff(c, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3


def main():
    test_gemm_fp8(1024, 1024, 1024, T.float8_e4m3fn)
    test_gemm_fp8(1024, 1024, 1024, T.float8_e5m2)


def run_regression_perf():
    M, N, K = 1024, 1024, 1024
    dtype = "float8_e4m3"
    kernel_e4m3 = matmul(M, N, K, 128, 128, 64, dtype)
    profiler_e4m3 = kernel_e4m3.get_profiler(tilelang.TensorSupplyType.Integer)
    latency_e4m3 = profiler_e4m3.do_bench(warmup=25, backend="cupti")
    dtype = "float8_e5m2"
    kernel_e5m2 = matmul(M, N, K, 128, 128, 64, dtype)
    profiler_e5m2 = kernel_e5m2.get_profiler(tilelang.TensorSupplyType.Integer)
    latency_e5m2 = profiler_e5m2.do_bench(warmup=25, backend="cupti")
    return (latency_e4m3 + latency_e5m2) / 2


if __name__ == "__main__":
    main()
