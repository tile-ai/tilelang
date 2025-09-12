import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import torch


@tilelang.jit(execution_backend='torch')
def matmul(M, N, K, block_M, block_N, block_K, dtype="float32", accum_dtype="float"):

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope='shared')
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope='shared')
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                for i, j, k in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, k] * B_shared[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return gemm


def assert_gemm(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float32",
    accum_dtype="float",
):
    jit_kernel = matmul(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)

    torch_dtype = getattr(torch, dtype)
    a = torch.randn(M, N, dtype=torch_dtype, device='mps')
    b = torch.randn(N, K, dtype=torch_dtype, device='mps')
    c = torch.zeros(K, M, dtype=torch_dtype, device='mps')

    jit_kernel(a, b, c)

    assert torch.allclose(a @ b, c)

    assert jit_kernel.kernel_source is not None


def test_gemm_codegen():
    assert_gemm(1024, 1024, 1024, 16, 16, 16)


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
