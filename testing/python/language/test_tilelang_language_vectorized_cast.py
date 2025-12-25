import pytest
import torch
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def vectorized_cast_kernel(M: int, dtype_A: str, dtype_B: str):
    assert M % 256 == 0

    @T.prim_func
    def main(
        A: T.Tensor[(M,), dtype_A],  # noqa: F821
        B: T.Tensor[(M,), dtype_B],  # noqa: F821
    ):
        with T.Kernel(1, threads=128):
            T.copy(A, B)

    return main


@tilelang.jit
def parallel_vectorized_cast_kernel(M: int, dtype_A: str, dtype_B: str):
    assert M % 256 == 0

    @T.prim_func
    def main(
        A: T.Tensor[(M,), dtype_A],  # noqa: F821
        B: T.Tensor[(M,), dtype_B],  # noqa: F821
    ):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_fragment((M,), dtype_A)
            B_local = T.alloc_fragment((M,), dtype_B)

            T.copy(A, A_local)
            for i in T.Parallel(M):
                B_local[i] = A_local[i]
            T.copy(B_local, B)

    return main


def run_vectorized_cast(src_dtype: T.dtype, dst_dtype: T.dtype, check_str: str, lanes: int = 2):
    """Run the vectorized cast kernel and check the correctness.
    Args:
        src_dtype: The source data type.
        dst_dtype: The destination data type.
        check_str: Used to ensure vectorized cast is used.
        lanes: The number of lanes of the source and destination data types.
    """

    M = 128 * lanes
    kernel = vectorized_cast_kernel(M, src_dtype, dst_dtype)
    kernel_parallel = parallel_vectorized_cast_kernel(M, src_dtype, dst_dtype)

    code = kernel.get_kernel_source()
    code_parallel = kernel_parallel.get_kernel_source()
    print(code)
    assert check_str in code and check_str in code_parallel, f"Cast {src_dtype} to {dst_dtype} with {lanes=} is not vectorized!"

    if src_dtype == T.float4_e2m1fn or dst_dtype == T.float4_e2m1fn:
        return

    A_float = torch.randn(M, dtype=torch.float32, device="cuda")
    A = A_float.to(src_dtype.as_torch())

    A = A_float.to(src_dtype.as_torch())
    B = torch.zeros(M, dtype=dst_dtype.as_torch(), device="cuda")
    C = torch.zeros(M, dtype=dst_dtype.as_torch(), device="cuda")

    kernel(A, B)
    kernel_parallel(A, C)

    torch.testing.assert_close(A.to(dst_dtype.as_torch()), B)
    torch.testing.assert_close(A.to(dst_dtype.as_torch()), C)


@pytest.mark.parametrize(
    "src_dtype, dst_dtype, check_str, lanes",
    [
        (T.float32, T.float16, "__float22half2_rn", 2),
        (T.float32, T.float16, "__float22half2_rn", 4),
        (T.float16, T.float32, "__half22float2", 2),
        (T.float16, T.float32, "__half22float2", 4),
        (T.float32, T.float8_e4m3fn, "__nv_cvt_float2_to_fp8x2", 2),
        (T.float32, T.float8_e4m3fn, "__nv_cvt_float2_to_fp8x2", 4),
        (T.float32, T.float8_e5m2, "__nv_cvt_float2_to_fp8x2", 2),
        (T.float32, T.float8_e5m2, "__nv_cvt_float2_to_fp8x2", 4),
        (T.float32, T.bfloat16, "__float22bfloat162_rn", 2),
        (T.float32, T.bfloat16, "__float22bfloat162_rn", 4),
        (T.bfloat16, T.float32, "__bfloat1622float2", 2),
        (T.bfloat16, T.float32, "__bfloat1622float2", 4),
    ],
)
def test_vectorized_cast(src_dtype, dst_dtype, check_str, lanes):
    run_vectorized_cast(src_dtype, dst_dtype, check_str, lanes)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize(
    "src_dtype, dst_dtype, check_str, lanes",
    [
        (T.float8_e4m3fn, T.float32, "__tl_cvt_fp8x2_to_float2", 2),
        (T.float8_e4m3fn, T.float32, "__tl_cvt_fp8x2_to_float2", 4),
        (T.float8_e5m2, T.float32, "__tl_cvt_fp8x2_to_float2", 2),
        (T.float8_e5m2, T.float32, "__tl_cvt_fp8x2_to_float2", 4),
    ],
)
def test_vectorized_cast_fp8(src_dtype, dst_dtype, check_str, lanes):
    run_vectorized_cast(src_dtype, dst_dtype, check_str, lanes)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize(
    "src_dtype, dst_dtype, check_str, lanes",
    [
        # FP4 <-> Half
        (T.float4_e2m1fn, T.float16, "__tl_cvt_fp4x2_to_half2", 2),
        (T.float16, T.float4_e2m1fn, "__tl_cvt_half2_to_fp4x2", 2),
        # FP4 <-> Float
        (T.float4_e2m1fn, T.float32, "__tl_cvt_fp4x2_to_float2", 2),
        (T.float32, T.float4_e2m1fn, "__tl_cvt_float2_to_fp4x2", 2),
        # FP4 <-> Double
        (T.float4_e2m1fn, T.float64, "__tl_cvt_fp4x2_to_double2", 2),
        (T.float64, T.float4_e2m1fn, "__tl_cvt_double2_to_fp4x2", 2),
        # FP4 <-> BFloat16
        (T.float4_e2m1fn, T.bfloat16, "__tl_cvt_fp4x2_to_bfloat162", 2),
        (T.bfloat16, T.float4_e2m1fn, "__tl_cvt_bfloat162_to_fp4x2", 2),
    ],
)
def test_vectorized_cast_fp4(src_dtype, dst_dtype, check_str, lanes):
    run_vectorized_cast(src_dtype, dst_dtype, check_str, lanes)


if __name__ == "__main__":
    tilelang.testing.main()
