import tilelang.tools.bench
import example_gemv


def bench_example_gemv():
    tilelang.tools.bench.process_func(example_gemv.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
