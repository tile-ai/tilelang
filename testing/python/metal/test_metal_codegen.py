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


@tilelang.jit(target="metal")
def matmul_with_t_gemm(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float32,
    accum_dtype=T.float32,
    transpose_B=False,
    num_stages=0,
    threads=128,
):
    B_shape = (N, K) if transpose_B else (K, N)
    B_shared_shape = (block_N, block_K) if transpose_B else (block_K, block_N)

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor(B_shape, dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared(B_shared_shape, dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                if transpose_B:
                    T.copy(B[bx * block_N, ko * block_K], B_shared, coalesced_width=2)
                else:
                    T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                T.gemm(A_shared, B_shared, C_local, transpose_B=transpose_B)

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


def assert_t_gemm(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float32,
    accum_dtype=T.float32,
    atol=1e-8,
    transpose_B=False,
    num_stages=0,
    threads=128,
):
    jit_kernel = matmul_with_t_gemm(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
        transpose_B=transpose_B,
        num_stages=num_stages,
        threads=threads,
    )

    torch_dtype = dtype.as_torch()
    b_shape = (N, K) if transpose_B else (K, N)
    if "int" in dtype:
        a = torch.randint(100, (M, K), dtype=torch_dtype, device="mps")
        b = torch.randint(100, b_shape, dtype=torch_dtype, device="mps")
    else:
        a = torch.randn(M, K, dtype=torch_dtype, device="mps")
        b = torch.randn(b_shape, dtype=torch_dtype, device="mps")
    c = torch.zeros(M, N, dtype=torch_dtype, device="mps")

    jit_kernel(a, b, c)

    b_ref = b.T if transpose_B else b
    assert torch.allclose(a @ b_ref, c, atol=atol)
    assert jit_kernel.kernel_source is not None
    assert "threadIdx.x) == 0" not in jit_kernel.kernel_source


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
def test_t_gemm_float32():
    assert_t_gemm(128, 128, 128, 16, 16, 16)


@tilelang.testing.requires_metal
def test_t_gemm_float16_accum_float32():
    assert_t_gemm(64, 64, 64, 16, 16, 16, dtype=T.float16, accum_dtype=T.float32, atol=1)


@tilelang.testing.requires_metal
def test_t_gemm_transpose_b_float32():
    assert_t_gemm(128, 128, 128, 16, 16, 16, transpose_B=True)


@tilelang.testing.requires_metal
def test_t_gemm_pipelined_float32():
    assert_t_gemm(64, 64, 64, 16, 16, 16, num_stages=2)


@tilelang.testing.requires_metal
def test_t_gemm_single_thread_float32():
    assert_t_gemm(16, 16, 16, 16, 16, 16, threads=1)


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
