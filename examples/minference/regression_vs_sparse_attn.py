import tilelang.testing
import example_vertical_slash_sparse_attn


def regression_example_vertical_slash_sparse_attn():
    tilelang.testing.process_func(example_vertical_slash_sparse_attn.run_regression_perf, argv=[])


if __name__ == "__main__":
    tilelang.testing.regression()
