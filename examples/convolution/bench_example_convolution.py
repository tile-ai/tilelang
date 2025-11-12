import tilelang.tools.bench
import tilelang.testing
import example_convolution
import example_convolution_autotune


def bench_example_convolution():
    tilelang.tools.bench.process_func(example_convolution.main)


def bench_example_convolution_autotune():
    tilelang.tools.bench.process_func(example_convolution_autotune.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
