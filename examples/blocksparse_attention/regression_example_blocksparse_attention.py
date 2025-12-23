import tilelang.testing
import example_tilelang_block_sparse_attn
import example_tilelang_sparse_gqa_decode_varlen_indice
import example_tilelang_sparse_gqa_decode_varlen_mask


def regression_example_tilelang_block_sparse_attn():
    tilelang.testing.process_func(example_tilelang_block_sparse_attn.run_regression_perf)


def regression_example_tilelang_sparse_gqa_decode_varlen_indice():
    tilelang.testing.process_func(example_tilelang_sparse_gqa_decode_varlen_indice.run_regression_perf, batch=1, max_cache_seqlen=2048)


def regression_example_tilelang_sparse_gqa_decode_varlen_mask():
    tilelang.testing.process_func(example_tilelang_sparse_gqa_decode_varlen_mask.run_regression_perf, batch=1, max_cache_seqlen=2048)


if __name__ == "__main__":
    tilelang.testing.regression()
