import tilelang.testing
import example_gemv


def regression_example_gemv():
    tilelang.testing.process_func(example_gemv.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
