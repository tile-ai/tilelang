import tilelang.tools.bench
import example_tilelang_gemm_streamk


def bench_example_tilelang_gemm_streamk():
    tilelang.tools.bench.process_func(example_tilelang_gemm_streamk.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
