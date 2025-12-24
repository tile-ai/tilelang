import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    # for fp8 gemm, do one promote after 4 wgmma inst, i.e. block_K = 128.
    # if block_K < 128, promote after 128/block_K iters.
    # if block_K > 128, promote after every iter.
    update_interval = 128 // block_K if block_K < 128 else 1

    @T.prim_func
    def gemm_fp8_2xAcc(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Promote to enable 2xAcc
                if (k + 1) % update_interval == 0:
                    for i, j in T.Parallel(block_M, block_N):
                        C_local_accum[i, j] += C_local[i, j]
                    T.clear(C_local)
            # Tail processing
            if K_iters % update_interval != 0:
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j]
            # TMA store
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm_fp8_2xAcc


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def test_gemm_fp8(M, N, K, dtype):
    torch_dtype = T.dtype(dtype).as_torch()

    kernel = matmul(M, N, K, 128, 128, 64, dtype)

    a = torch.rand(M, K, dtype=torch.float16, device="cuda")
    a = (100 * (2 * a - 1)).to(dtype=torch_dtype)
    b = torch.rand(N, K, dtype=torch.float16, device="cuda")
    b = (100 * (2 * b - 1)).to(dtype=torch_dtype)

    c = kernel(a, b)

    ref_c = a.float() @ b.float().T

    diff = calc_diff(c, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3


def main():
    test_gemm_fp8(1024, 1024, 8192, T.float8_e4m3fn)
    test_gemm_fp8(1024, 1024, 8192, T.float8_e5m2)


def run_regression_perf():
    M, N, K = 1024, 1024, 8192
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
