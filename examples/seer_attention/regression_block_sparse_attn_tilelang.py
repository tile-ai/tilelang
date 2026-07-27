import tilelang.testing
import block_sparse_attn_tilelang


def regression_block_sparse_attn_tilelang():
    tilelang.testing.process_func(block_sparse_attn_tilelang.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
