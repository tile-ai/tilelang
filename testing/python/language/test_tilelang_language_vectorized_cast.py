import torch
import tilelang.testing
import tilelang.language as T

str2dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


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


def run_vectorized_cast(src_dtype_str: str, dst_dtype_str: str, check_str: str, lanes: int = 2):
    """Run the vectorized cast kernel and check the correctness.
    Args:
        src_dtype_str: The source data type string.
        dst_dtype_str: The destination data type string.
        check_str: Used to ensure vectorized cast is used.
        lanes: The number of lanes of the source and destination data types.
    """

    M = 128 * lanes
    kernel = vectorized_cast_kernel(M, src_dtype_str, dst_dtype_str)
    kernel_parallel = parallel_vectorized_cast_kernel(M, src_dtype_str, dst_dtype_str)

    A = torch.randn(M, dtype=str2dtype[src_dtype_str]).cuda()
    B = torch.zeros(M, dtype=str2dtype[dst_dtype_str]).cuda()
    C = torch.zeros(M, dtype=str2dtype[dst_dtype_str]).cuda()

    kernel(A, B)
    kernel_parallel(A, C)

    torch.testing.assert_close(A.to(str2dtype[dst_dtype_str]), B)
    torch.testing.assert_close(A.to(str2dtype[dst_dtype_str]), C)

    code = kernel.get_kernel_source()
    code_parallel = kernel_parallel.get_kernel_source()

    assert check_str in code and check_str in code_parallel, \
        f"Cast {src_dtype_str} to {dst_dtype_str} with {lanes=} is not vectorized!"


def test_vectorized_cast():
    # fp32 -> fp16
    run_vectorized_cast("float32", "float16", "__float22half2_rn", 2)
    run_vectorized_cast("float32", "float16", "__float22half2_rn", 4)

    # fp16 -> fp32
    run_vectorized_cast("float16", "float32", "__half22float2", 2)
    run_vectorized_cast("float16", "float32", "__half22float2", 4)

    # fp32 -> fp8_e4m3
    run_vectorized_cast("float32", "float8_e4m3", "__nv_cvt_float2_to_fp8x2", 2)
    run_vectorized_cast("float32", "float8_e4m3", "__nv_cvt_float2_to_fp8x2", 4)

    # fp32 -> fp8_e5m2
    run_vectorized_cast("float32", "float8_e5m2", "__nv_cvt_float2_to_fp8x2", 2)
    run_vectorized_cast("float32", "float8_e5m2", "__nv_cvt_float2_to_fp8x2", 4)

    # fp32 -> bf16
    run_vectorized_cast("float32", "bfloat16", "__float22bfloat162_rn", 2)
    run_vectorized_cast("float32", "bfloat16", "__float22bfloat162_rn", 4)

    # bf16 -> fp32
    run_vectorized_cast("bfloat16", "float32", "__bfloat1622float2", 2)
    run_vectorized_cast("bfloat16", "float32", "__bfloat1622float2", 4)


if __name__ == "__main__":
    tilelang.testing.main()
