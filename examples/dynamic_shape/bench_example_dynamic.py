import tilelang.tools.bench
import example_dynamic


def bench_example_dynamic():
    tilelang.tools.bench.process_func(example_dynamic.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
