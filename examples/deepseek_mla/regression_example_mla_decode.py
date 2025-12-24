import tilelang.testing
import example_mla_decode


def regression_example_mla_decode():
    tilelang.testing.process_func(example_mla_decode.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
