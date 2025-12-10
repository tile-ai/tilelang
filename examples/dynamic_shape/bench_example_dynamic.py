import tilelang.tools.bench
import example_dynamic


def bench_example_dynamic():
    tilelang.tools.bench.process_func(example_dynamic.benchmark, M=1024, N=1024, K=1024)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
