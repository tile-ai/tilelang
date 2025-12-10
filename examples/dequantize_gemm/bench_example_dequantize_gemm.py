import tilelang.tools.bench
import example_dequant_gemm_bf16_fp4_hopper
import example_dequant_gemm_bf16_mxfp4_hopper
import example_dequant_gemm_bf16_mxfp4_hopper_tma
import example_dequant_gemm_fp4_hopper
import example_dequant_gemm_w4a8
import example_dequant_gemv_fp16xint4
import example_dequant_groupedgemm_bf16_mxfp4_hopper


def bench_example_dequant_gemv_fp16xint4():
    tilelang.tools.bench.process_func(example_dequant_gemv_fp16xint4.benchmark)


def bench_example_dequant_gemm_fp4_hopper():
    tilelang.tools.bench.process_func(example_dequant_gemm_fp4_hopper.benchmark)


def bench_example_dequant_gemm_bf16_fp4_hopper():
    tilelang.tools.bench.process_func(example_dequant_gemm_bf16_fp4_hopper.benchmark)


def bench_example_dequant_gemm_bf16_mxfp4_hopper():
    tilelang.tools.bench.process_func(example_dequant_gemm_bf16_mxfp4_hopper.benchmark)


def bench_example_dequant_gemm_bf16_mxfp4_hopper_tma():
    tilelang.tools.bench.process_func(example_dequant_gemm_bf16_mxfp4_hopper_tma.benchmark)


def bench_example_dequant_groupedgemm_bf16_mxfp4_hopper():
    tilelang.tools.bench.process_func(example_dequant_groupedgemm_bf16_mxfp4_hopper.benchmark)


def bench_example_dequant_gemm_w4a8():
    tilelang.tools.bench.process_func(example_dequant_gemm_w4a8.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
