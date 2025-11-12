import tilelang.tools.bench
import example_mla_decode


def bench_example_mla_decode():
    tilelang.tools.bench.process_func(example_mla_decode.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
