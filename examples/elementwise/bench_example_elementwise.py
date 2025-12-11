import tilelang.testing.benchmark
import example_elementwise_add


def bench_example_elementwise_add():
    tilelang.testing.benchmark.process_func(example_elementwise_add.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
