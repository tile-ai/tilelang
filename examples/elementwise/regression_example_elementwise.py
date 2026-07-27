import tilelang.testing
import example_elementwise_add


def regression_example_elementwise_add():
    tilelang.testing.process_func(example_elementwise_add.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
