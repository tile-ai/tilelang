import torch
import tilelang.testing
import tilelang.language as T
import pytest


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True})
def vectorize_test(N, M, stride_A, stride_B):
    @T.prim_func
    def main(
        A: T.StridedTensor[(N, M), (1, stride_A), T.float32],  # noqa: F821
        B: T.StridedTensor[(N, M), (1, stride_B), T.float32],  # noqa: F821
    ):
        with T.Kernel(M // 128, threads=128) as (bx):
            tx = T.get_thread_binding(0)
            col = bx * 128 + tx

            for row in T.vectorized(N):
                B[row, col] = A[row, col]

    return main


def run_vectorize(N, M, stride_A, stride_B):
    assert N % 128 == 0 and M % 128 == 0
    assert stride_A >= N and stride_B >= N

    jit_kernel = vectorize_test(N, M, stride_A, stride_B)

    base_a = torch.randn(stride_A, M, device="cuda", dtype=torch.float32)
    base_b = torch.zeros(stride_B, M, device="cuda", dtype=torch.float32)
    a = torch.as_strided(base_a, size=(N, M), stride=(1, stride_A))
    b = torch.as_strided(base_b, size=(N, M), stride=(1, stride_B))

    jit_kernel(a, b)

    torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)

    code = jit_kernel.get_kernel_source()

    vectorize_size = 1
    while vectorize_size <= 2 and stride_A % (vectorize_size * 2) == 0 and stride_B % (vectorize_size * 2) == 0:
        vectorize_size *= 2

    if vectorize_size == 4:
        assert "float4" in code
    elif vectorize_size == 2:
        assert "float2" in code


def test_vectorize():
    N, M = 512, 256

    run_vectorize(N, M, N, N)
    run_vectorize(N, M, N + 2, N + 4)
    run_vectorize(N, M, N + 4, N + 8)
    run_vectorize(N, M, N + 8, N + 16)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True})
def vectorize_test_invariant_index(N, M, K):
    @T.prim_func
    def main(
        A: T.Tensor[(N, M), T.float32],  # noqa: F821
        B: T.Tensor[(N, M), T.float32],  # noqa: F821
        C: T.Tensor[(N, M // K), T.float32],  # noqa: F821
    ):
        with T.Kernel(N // 128, threads=128) as (bx):
            tx = T.get_thread_binding(0)
            row = bx * 128 + tx

            for col in T.vectorized(M):
                B[row, col] = A[row, col] * C[row, col // K]

    return main


def run_vectorize_invariant_index(N, M, K):
    assert N % 128 == 0 and M % K == 0

    jit_kernel = vectorize_test_invariant_index(N, M, K)

    a = torch.randn(N, M, device="cuda", dtype=torch.float32)
    b = torch.zeros(N, M, device="cuda", dtype=torch.float32)
    c = torch.randn(N, M // K, device="cuda", dtype=torch.float32)

    jit_kernel(a, b, c)

    indices = torch.arange(a.size(1)) // K
    ret = a * c[:, indices]
    torch.testing.assert_close(b, ret, atol=1e-8, rtol=1e-8)

    code = jit_kernel.get_kernel_source()

    vectorize_size = 1
    while vectorize_size <= 2 and K % (vectorize_size * 2) == 0:
        vectorize_size *= 2

    if vectorize_size == 4:
        assert "float4" in code
    elif vectorize_size == 2:
        assert "float2" in code


def test_vectorize_invariant_index():
    N, M = 512, 256

    run_vectorize_invariant_index(N, M, 2)
    run_vectorize_invariant_index(N, M, 4)
    run_vectorize_invariant_index(N, M * 3, 6)
    run_vectorize_invariant_index(N, M, 8)
    run_vectorize_invariant_index(N, M * 3, 12)
    run_vectorize_invariant_index(N, M * 7, 14)


@tilelang.jit
def vectorize_test_all_dtypes(dtype, vec_num):
    @T.prim_func
    def main(A: T.Tensor[(64,), dtype]):
        with T.Kernel(1, threads=256):
            for i in T.vectorized(vec_num):
                A[i] = T.cast(i + 1, dtype)

    return main


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e8m0fnu,
    ],
)
@pytest.mark.parametrize("vec_num", [1, 2, 4, 8])
def test_vectorize_all_dtypes(dtype, vec_num):
    x = torch.empty((64,), dtype=dtype, device="cuda")
    kernel = vectorize_test_all_dtypes(dtype, vec_num)
    kernel(x)


if __name__ == "__main__":
    tilelang.testing.main()
