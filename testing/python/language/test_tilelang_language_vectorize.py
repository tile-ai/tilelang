import torch
import tilelang.testing
import tilelang.language as T

from tilelang.cuda.intrinsics import make_mma_swizzle_layout
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
    N, M = 128, 128

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
    N, M = 128, 128

    run_vectorize_invariant_index(N, M, 2)
    run_vectorize_invariant_index(N, M, 4)
    run_vectorize_invariant_index(N, M * 3, 6)
    run_vectorize_invariant_index(N, M * 7, 14)


@tilelang.jit
def vectorize_invariant_store_accumulate(M, K):
    @T.prim_func
    def main(
        A: T.Tensor[(M, K), T.float16],  # noqa: F821
        Init: T.Tensor[(M,), T.float16],  # noqa: F821
        B: T.Tensor[(M,), T.float16],  # noqa: F821
    ):
        with T.Kernel(M // 128, threads=128) as bx:
            tx = T.get_thread_binding(0)
            row = bx * 128 + tx
            B[row] = Init[row]
            # B[row] does not depend on the vectorized loop variable k, so
            # vectorizing this loop turns the accumulate into a broadcast
            # store: every lane reads the same old B[row], and the last
            # lane's store wins, dropping all addends but A[row, K-1].
            for k in T.vectorized(K):
                B[row] = B[row] + A[row, k]

    return main


@tilelang.testing.requires_cuda
def test_vectorize_invariant_store_accumulate():
    """A loop-invariant store address must keep the loop scalar.

    Regression test for the VectorizePlanner bucketing in
    src/transform/loop_vectorize.cc: a loop-invariant store computes
    vector_size=1 in ComputeBufferVectorSize, but the entry was moved to the
    local/fragment bucket which the simple-case strategy ignores, so the loop
    was vectorized into a broadcast store. On CUDA this compiles and silently
    produces wrong results (last lane wins); on the `c` target it fails to
    compile (`vec_type` has no `.s0/.s1` accessors).
    """
    M, K = 128, 8
    kernel = vectorize_invariant_store_accumulate(M, K)

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    init = torch.randn(M, device="cuda", dtype=torch.float16)
    b = init.clone()
    kernel(a, init, b)

    expected = init + a.sum(dim=1)
    torch.testing.assert_close(b, expected, atol=1e-2, rtol=1e-2)


@tilelang.jit
def vectorize_test_all_dtypes(dtype, vec_num):
    @T.prim_func
    def main(A: T.Tensor[(64,), dtype]):
        with T.Kernel(1, threads=256):
            for i in T.vectorized(vec_num):
                A[i] = T.cast(i + 1, dtype)

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.uint64,
        torch.int8,
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


@tilelang.jit
def vectorize_broadcast_int8(vec_num):
    with T.Kernel(1, threads=128):
        a = T.alloc_local((64,), "int8")
        b = T.alloc_var("int8")

        for i in T.vectorized(vec_num):
            a[i] = b


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("vec_num", [4, 32])
def test_vectorize_broadcast_int8(vec_num):
    """Test broadcasting a non-constant int8 value to a vectorized store."""
    vectorize_broadcast_int8.compile(vec_num=vec_num)


@tilelang.jit
def vectorize_test_call_infinity():
    A = T.empty((4,), dtype=T.float32)
    with T.Kernel(1, threads=128):
        for i in T.vectorized(4):
            A[i] = T.infinity(T.float32)
    return A


def test_vectorize_call_infinity():
    kernel = vectorize_test_call_infinity.compile()
    assert "float4" in kernel.get_kernel_source()


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_VECTORIZE_PLANNER_VERBOSE: True, tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: False}
)
def vectorize_test_call_bitwise_logical():
    A = T.empty((128, 32), dtype=T.float32)
    with T.Kernel(1, threads=128):
        A_shared = T.alloc_shared((128, 32), dtype=T.float32)
        T.annotate_layout({A_shared: make_mma_swizzle_layout(A_shared)})
        for i, j in T.Parallel(128, 32):
            A_shared[i, j] = A[i, j]
    return A


def test_vectorize_call_bitwise_logical():
    kernel = vectorize_test_call_bitwise_logical.compile()
    print(kernel.get_kernel_source())
    assert "float4" in kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
