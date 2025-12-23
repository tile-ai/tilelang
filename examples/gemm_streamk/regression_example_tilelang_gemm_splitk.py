import tilelang.testing
import example_tilelang_gemm_streamk


def regression_example_tilelang_gemm_streamk():
    tilelang.testing.process_func(example_tilelang_gemm_streamk.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
