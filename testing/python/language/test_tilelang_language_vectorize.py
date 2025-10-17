import torch
import tilelang.testing
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True})
def vectorize_test(N, M, stride_A, stride_B):
    assert N % 128 == 0 and M % 128 == 0

    @T.prim_func
    def main(
            A: T.StridedTensor[(N, M), (1, stride_A), "float32"],  # noqa: F821
            B: T.StridedTensor[(N, M), (1, stride_B), "float32"],  # noqa: F821
    ):
        with T.Kernel(M // 128, threads=128) as (bx):
            tx = T.get_thread_binding(0)
            col = bx * 128 + tx

            for row in T.vectorized(N):
                B[row, col] = A[row, col]

    return main


def run_vectorize(N, M, stride_A, stride_B):
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
    while vectorize_size <= 2 and \
          stride_A % (vectorize_size * 2) == 0 and \
          stride_B % (vectorize_size * 2) == 0:
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


if __name__ == "__main__":
    tilelang.testing.main()
