import tilelang.testing.benchmark
import example_vertical_slash_sparse_attn


def bench_example_vertical_slash_sparse_attn():
    tilelang.testing.benchmark.process_func(example_vertical_slash_sparse_attn.benchmark, argv=[])


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
