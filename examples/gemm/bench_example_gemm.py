import tilelang.tools.bench
import example_gemm
import example_gemm_autotune
import example_gemm_intrinsics
import example_gemm_schedule


def bench_example_gemm_autotune():
    tilelang.tools.bench.process_func(example_gemm_autotune.main,M=1024, N=1024, K=1024, with_roller=True)


def bench_example_gemm_intrinsics():
    tilelang.tools.bench.process_func(example_gemm_intrinsics.main,M=1024, N=1024, K=1024)


def bench_example_gemm_schedule():
    tilelang.tools.bench.process_func(example_gemm_schedule.main)


def bench_example_gemm():
    tilelang.tools.bench.process_func(example_gemm.main)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
