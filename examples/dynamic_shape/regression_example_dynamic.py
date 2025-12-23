import tilelang.testing
import example_dynamic


def regression_example_dynamic():
    tilelang.testing.process_func(example_dynamic.run_regression_perf, M=1024, N=1024, K=1024)


if __name__ == "__main__":
    tilelang.testing.regression()
