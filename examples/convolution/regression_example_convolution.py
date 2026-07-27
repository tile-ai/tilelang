import tilelang.testing
import example_convolution
import example_convolution_autotune


def regression_example_convolution():
    tilelang.testing.process_func(example_convolution.run_regression_perf)


def regression_example_convolution_autotune():
    tilelang.testing.process_func(example_convolution_autotune.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
