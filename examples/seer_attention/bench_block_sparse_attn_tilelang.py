import tilelang.tools.bench
import block_sparse_attn_tilelang


def bench_block_sparse_attn_tilelang():
    tilelang.tools.bench.process_func(block_sparse_attn_tilelang.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
