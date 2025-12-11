import tilelang.testing.benchmark
import example_mla_decode


def bench_example_mla_decode():
    tilelang.testing.benchmark.process_func(example_mla_decode.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
