import tilelang.testing.benchmark
import example_convolution
import example_convolution_autotune


def bench_example_convolution():
    tilelang.testing.benchmark.process_func(example_convolution.benchmark)


def bench_example_convolution_autotune():
    tilelang.testing.benchmark.process_func(example_convolution_autotune.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
