import tilelang.tools.bench
import example_gemm
import example_gemm_autotune
import example_gemm_intrinsics
import example_gemm_schedule


def bench_example_gemm_autotune():
    tilelang.tools.bench.process_func(example_gemm_autotune.main)


def bench_example_gemm_intrinsics():
    tilelang.tools.bench.process_func(example_gemm_intrinsics.main)


def bench_example_gemm_schedule():
    tilelang.tools.bench.process_func(example_gemm_schedule.main)


def bench_example_gemm():
    tilelang.tools.bench.process_func(example_gemm.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
