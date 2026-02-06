import tilelang.testing
import example_topk


def regression_example_topk():
    tilelang.testing.process_func(example_topk.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
