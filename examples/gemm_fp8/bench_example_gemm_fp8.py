import tilelang.tools.bench
import example_tilelang_gemm_fp8
import example_tilelang_gemm_fp8_2xAcc
import example_tilelang_gemm_fp8_intrinsic


def bench_example_tilelang_gemm_fp8_2xAcc():
    tilelang.tools.bench.process_func(example_tilelang_gemm_fp8_2xAcc.main)


def bench_example_tilelang_gemm_fp8_intrinsic():
    tilelang.tools.bench.process_func(example_tilelang_gemm_fp8_intrinsic.main)


def bench_example_tilelang_gemm_fp8():
    tilelang.tools.bench.process_func(example_tilelang_gemm_fp8.main)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
