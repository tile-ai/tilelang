import tilelang.tools.bench
import example_blocksparse_gemm


def bench_example_blocksparse_gemm():
    tilelang.tools.bench.process_func(example_blocksparse_gemm.main)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
