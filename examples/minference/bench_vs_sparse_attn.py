import tilelang.tools.bench
import example_vertical_slash_sparse_attn


def bench_example_vertical_slash_sparse_attn():
    tilelang.tools.bench.process_func(example_vertical_slash_sparse_attn.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
