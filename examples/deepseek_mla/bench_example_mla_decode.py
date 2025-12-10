import tilelang.tools.bench
import example_mla_decode


def bench_example_mla_decode():
    tilelang.tools.bench.process_func(example_mla_decode.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
