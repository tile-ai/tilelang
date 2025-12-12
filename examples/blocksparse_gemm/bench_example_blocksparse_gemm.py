import tilelang.testing.benchmark
import example_blocksparse_gemm


def bench_example_blocksparse_gemm():
    tilelang.testing.benchmark.process_func(example_blocksparse_gemm.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
