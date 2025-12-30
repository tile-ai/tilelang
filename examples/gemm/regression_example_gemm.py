import tilelang.testing
import example_gemm
import example_gemm_autotune
import example_gemm_intrinsics
import example_gemm_schedule


def regression_example_gemm_autotune():
    tilelang.testing.process_func(example_gemm_autotune.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_gemm_intrinsics():
    tilelang.testing.process_func(example_gemm_intrinsics.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_gemm_schedule():
    tilelang.testing.process_func(example_gemm_schedule.run_regression_perf)


def regression_example_gemm():
    tilelang.testing.process_func(example_gemm.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
