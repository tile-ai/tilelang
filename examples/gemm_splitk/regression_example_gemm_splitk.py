import tilelang.testing
import example_tilelang_gemm_splitk
import example_tilelang_gemm_splitk_vectorize_atomicadd


def regression_example_tilelang_gemm_splitk():
    tilelang.testing.process_func(example_tilelang_gemm_splitk.run_regression_perf)


def regression_example_tilelang_gemm_splitk_vectorize_atomicadd():
    tilelang.testing.process_func(example_tilelang_gemm_splitk_vectorize_atomicadd.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
