import tilelang.tools.bench
import example_gemm
import example_gemm_autotune
import example_gemm_intrinsics
import example_gemm_schedule


def bench_example_gemm_autotune():
    tilelang.tools.bench.process_func(example_gemm_autotune.benchmark, M=1024, N=1024, K=1024)


def bench_example_gemm_intrinsics():
    tilelang.tools.bench.process_func(example_gemm_intrinsics.benchmark, M=1024, N=1024, K=1024)


def bench_example_gemm_schedule():
    tilelang.tools.bench.process_func(example_gemm_schedule.benchmark)


def bench_example_gemm():
    tilelang.tools.bench.process_func(example_gemm.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
