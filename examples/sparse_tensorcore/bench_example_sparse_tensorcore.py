import tilelang.testing.benchmark
import tilelang
import tilelang_example_sparse_tensorcore


def bench_example_sparse_tensorcore():
    tilelang.testing.benchmark.process_func(tilelang_example_sparse_tensorcore.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
