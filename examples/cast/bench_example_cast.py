import tilelang.tools.bench
import example_group_per_split_token_cast_to_fp8
import example_per_token_cast_to_fp8


def bench_example_group_per_split_token_cast_to_fp8():
    tilelang.tools.bench.process_func(example_group_per_split_token_cast_to_fp8.main)


def bench_example_per_token_cast_to_fp8():
    tilelang.tools.bench.process_func(example_per_token_cast_to_fp8.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
