import tilelang.tools.bench
import example_convolution
import example_convolution_autotune


def bench_example_convolution():
    tilelang.tools.bench.process_func(example_convolution.main)


def bench_example_convolution_autotune():
    tilelang.tools.bench.process_func(example_convolution_autotune.main)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
