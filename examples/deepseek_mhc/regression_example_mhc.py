import tilelang.testing
import example_mhc_post
import example_mhc_pre


def regression_example_mhc_post():
    tilelang.testing.process_func(example_mhc_post.run_regression_perf)


def regression_example_mhc_pre():
    tilelang.testing.process_func(example_mhc_pre.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
