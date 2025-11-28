import tilelang.tools.bench
import example_elementwise_add


def bench_example_elementwise_add():
    tilelang.tools.bench.process_func(example_elementwise_add.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
