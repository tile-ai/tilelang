import tilelang.testing
import example_dynamic


def regression_example_dynamic():
    tilelang.testing.process_func(example_dynamic.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
