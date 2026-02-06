import tilelang.testing
import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    out_idx=-1,  # create the output tensor during runtime
    verbose=True,
)
def matmul_kernel_jit(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A=False,
    trans_B=True,
    in_dtype=T.float16,
    out_dtype=T.float32,
    accum_dtype=T.float32,
    num_stages=2,
    threads=128,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def test_par_compile():
    configs = [
        (1024, 1024, 1024, 128, 128, 32),
        (2048, 2048, 2048, 256, 256, 64),
        (4096, 4096, 4096, 64, 64, 128),
    ]
    kernels = matmul_kernel_jit.par_compile(configs)
    for (M, N, K, _, _, _), kernel in zip(configs, kernels):
        A = torch.randn(M, K, dtype=torch.float16).cuda()
        B = torch.randn(N, K, dtype=torch.float16).cuda()
        ref = (A @ B.T).float()
        C = kernel(A, B)
        tilelang.testing.torch_assert_close(C, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
