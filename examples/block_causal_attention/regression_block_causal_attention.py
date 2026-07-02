import tilelang.testing

import block_causal_attention


def regression_block_causal_attention():
    tilelang.testing.process_func(block_causal_attention.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
