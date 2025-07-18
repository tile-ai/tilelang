import torch
import tilelang
import tilelang.testing
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, threads, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullCol)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_threads_test(threads, M=1024, N=192, K=1024, block_M=64, block_N=192, block_K=32):
    func = matmul(M, N, K, block_M, block_N, block_K, threads)
    jit_kernel = tilelang.compile(func, out_idx=-1, target="cuda")

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    ref_c = a @ b
    c = jit_kernel(a, b)

    tilelang.testing.torch_assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_gemm_threads_2wgs():
    run_gemm_threads_test(128 * 2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_gemm_threads_4wgs():
    run_gemm_threads_test(128 * 4)


if __name__ == "__main__":
    tilelang.testing.main()
