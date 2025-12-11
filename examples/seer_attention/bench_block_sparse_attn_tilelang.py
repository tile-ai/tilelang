import tilelang.testing.benchmark
import block_sparse_attn_tilelang


def bench_block_sparse_attn_tilelang():
    tilelang.testing.benchmark.process_func(block_sparse_attn_tilelang.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
