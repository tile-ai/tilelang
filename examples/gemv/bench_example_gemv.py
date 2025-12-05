import tilelang.tools.bench
import example_gemv


def bench_example_gemv():
    tilelang.tools.bench.process_func(example_gemv.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
