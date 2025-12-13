import tilelang.testing.benchmark
import example_dequant_gemm_bf16_fp4_hopper
import example_dequant_gemm_bf16_mxfp4_hopper
import example_dequant_gemm_bf16_mxfp4_hopper_tma
import example_dequant_gemm_fp4_hopper
import example_dequant_gemm_w4a8
import example_dequant_gemv_fp16xint4
import example_dequant_groupedgemm_bf16_mxfp4_hopper


def bench_example_dequant_gemv_fp16xint4():
    tilelang.testing.benchmark.process_func(example_dequant_gemv_fp16xint4.benchmark)


def bench_example_dequant_gemm_fp4_hopper():
    tilelang.testing.benchmark.process_func(example_dequant_gemm_fp4_hopper.benchmark)


def bench_example_dequant_gemm_bf16_fp4_hopper():
    tilelang.testing.benchmark.process_func(example_dequant_gemm_bf16_fp4_hopper.benchmark)


def bench_example_dequant_gemm_bf16_mxfp4_hopper():
    tilelang.testing.benchmark.process_func(example_dequant_gemm_bf16_mxfp4_hopper.benchmark)


def bench_example_dequant_gemm_bf16_mxfp4_hopper_tma():
    tilelang.testing.benchmark.process_func(example_dequant_gemm_bf16_mxfp4_hopper_tma.benchmark)


def bench_example_dequant_groupedgemm_bf16_mxfp4_hopper():
    tilelang.testing.benchmark.process_func(example_dequant_groupedgemm_bf16_mxfp4_hopper.benchmark)


def bench_example_dequant_gemm_w4a8():
    tilelang.testing.benchmark.process_func(example_dequant_gemm_w4a8.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
