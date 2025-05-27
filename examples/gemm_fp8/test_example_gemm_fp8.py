import tilelang.testing
import example_tilelang_gemm_fp8_2xAcc
import example_tilelang_gemm_fp8_intrinsic
import example_tilelang_gemm_fp8


def test_example_tilelang_gemm_fp8_2xAcc():
    example_tilelang_gemm_fp8_2xAcc.main()


def test_example_tilelang_gemm_fp8_intrinsic():
    example_tilelang_gemm_fp8_intrinsic.main()


def test_example_tilelang_gemm_fp8():
    example_tilelang_gemm_fp8.main()


if __name__ == "__main__":
    tilelang.testing.main()
