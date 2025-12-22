import tilelang.testing
import example_tilelang_nsa_fwd
import example_tilelang_nsa_decode


def regression_example_tilelang_nsa_fwd():
    tilelang.testing.process_func(example_tilelang_nsa_fwd.run_regression_perf)


def regression_example_tilelang_nsa_fwd_decode():
    tilelang.testing.process_func(example_tilelang_nsa_decode.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
