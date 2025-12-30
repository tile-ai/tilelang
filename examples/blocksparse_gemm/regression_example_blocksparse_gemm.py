import tilelang.testing
import example_blocksparse_gemm


def regression_example_blocksparse_gemm():
    tilelang.testing.process_func(example_blocksparse_gemm.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
