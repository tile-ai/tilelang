import tilelang.testing
import example_warp_specialize_gemm_barrierpipe_stage2
import example_warp_specialize_gemm_copy_0_gemm_1
import example_warp_specialize_gemm_copy_1_gemm_0
import example_warp_specialize_gemm_softpipe_stage2


def regression_example_warp_specialize_gemm_barrierpipe_stage2():
    tilelang.testing.process_func(example_warp_specialize_gemm_barrierpipe_stage2.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_warp_specialize_gemm_copy_0_gemm_1():
    tilelang.testing.process_func(example_warp_specialize_gemm_copy_0_gemm_1.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_warp_specialize_gemm_copy_1_gemm_0():
    tilelang.testing.process_func(example_warp_specialize_gemm_copy_1_gemm_0.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_warp_specialize_gemm_softpipe_stage2():
    tilelang.testing.process_func(example_warp_specialize_gemm_softpipe_stage2.run_regression_perf, M=1024, N=1024, K=1024)


if __name__ == "__main__":
    tilelang.testing.regression()
