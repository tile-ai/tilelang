import tilelang.testing
import example_tilelang_gemm_fp8
import example_tilelang_gemm_fp8_2xAcc
import example_tilelang_gemm_fp8_intrinsic


def regression_example_tilelang_gemm_fp8_2xAcc():
    tilelang.testing.process_func(example_tilelang_gemm_fp8_2xAcc.run_regression_perf)


def regression_example_tilelang_gemm_fp8_intrinsic():
    tilelang.testing.process_func(example_tilelang_gemm_fp8_intrinsic.run_regression_perf)


def regression_example_tilelang_gemm_fp8():
    tilelang.testing.process_func(example_tilelang_gemm_fp8.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
