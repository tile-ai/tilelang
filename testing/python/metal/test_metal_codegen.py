import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import torch


@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float32, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                for i, j in T.Parallel(block_M, block_N):
                    for k in T.Serial(block_K):
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
    dtype=T.float32,
    accum_dtype=T.float32,
    atol=1e-8,
):
    jit_kernel = matmul(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)

    torch_dtype = dtype.as_torch()
    a, b = None, None
    if "int" in dtype:
        a = torch.randint(100, (M, K), dtype=torch_dtype, device="mps")
        b = torch.randint(100, (K, N), dtype=torch_dtype, device="mps")
    else:
        a = torch.randn(M, K, dtype=torch_dtype, device="mps")
        b = torch.randn(K, N, dtype=torch_dtype, device="mps")
    c = torch.zeros(M, N, dtype=torch_dtype, device="mps")

    jit_kernel(a, b, c)

    assert torch.allclose(a @ b, c, atol=atol)

    assert jit_kernel.kernel_source is not None


@tilelang.jit
def local_var_conditional_store(n=128):
    @T.prim_func
    def kernel(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n, threads=128) as i:
            x = T.alloc_var(T.float32)
            x = A[i]
            if A[i] > 0:
                x = x + 1.0
            else:
                x = x - 1.0
            B[i] = x

    return kernel


@tilelang.testing.requires_metal
def test_gemm_float32():
    assert_gemm(1024, 1024, 1024, 16, 16, 16)


@tilelang.testing.requires_metal
def test_gemm_float16():
    assert_gemm(1024, 1024, 1024, 16, 16, 16, dtype=T.float16, atol=1)


@tilelang.testing.requires_metal
def test_gemm_int32():
    assert_gemm(1024, 1024, 1024, 16, 16, 16, dtype=T.int32, atol=1)


@tilelang.testing.requires_metal
def test_gemm_bfloat16():
    assert_gemm(256, 256, 256, 16, 16, 16, dtype=T.bfloat16, accum_dtype=T.float32, atol=2)


@tilelang.testing.requires_metal
def test_local_var_conditional_store():
    kernel = local_var_conditional_store()
    a = torch.randn(128, dtype=torch.float32, device="mps")
    b = torch.empty_like(a)

    kernel(a, b)

    torch.testing.assert_close(b, torch.where(a > 0, a + 1.0, a - 1.0))


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
