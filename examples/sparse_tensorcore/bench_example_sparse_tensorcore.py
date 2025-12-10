import tilelang.tools.bench
import tilelang
import tilelang_example_sparse_tensorcore


def bench_example_sparse_tensorcore():
    tilelang.tools.bench.process_func(tilelang_example_sparse_tensorcore.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
