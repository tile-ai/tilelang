import tilelang.testing
import example_linear_attn_bwd
import example_linear_attn_fwd


def regression_example_linear_attn_fwd():
    tilelang.testing.process_func(example_linear_attn_fwd.run_regression_perf)


def regression_example_linear_attn_bwd():
    tilelang.testing.process_func(example_linear_attn_bwd.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
