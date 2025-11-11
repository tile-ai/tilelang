import tilelang.tools.bench
import example_blocksparse_gemm


def bench_example_blocksparse_gemm():
    tilelang.tools.bench.process_func(example_blocksparse_gemm.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
