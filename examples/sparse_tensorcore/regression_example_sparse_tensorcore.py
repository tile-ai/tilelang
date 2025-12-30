import tilelang.testing
import tilelang
import tilelang_example_sparse_tensorcore


def regression_example_sparse_tensorcore():
    tilelang.testing.process_func(tilelang_example_sparse_tensorcore.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
