import tilelang.tools.bench
import example_dequant_gemm_bf16_mxfp4_hopper
import example_dequant_gemm_bf16_mxfp4_hopper_tma
import example_dequant_gemm_fp4_hopper
import example_dequant_gemm_w4a8
import example_dequant_gemv_fp16xint4
import example_dequant_groupedgemm_bf16_mxfp4_hopper

@tilelang.testing.requires_cuda
def bench_example_dequant_gemv_fp16xint4():
    tilelang.tools.bench.process_func(example_dequant_gemv_fp16xint4.main)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def bench_example_dequant_gemm_fp4_hopper():
    tilelang.tools.bench.process_func(example_dequant_gemm_fp4_hopper.main)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def bench_example_dequant_gemm_bf16_mxfp4_hopper():
    tilelang.tools.bench.process_func(example_dequant_gemm_bf16_mxfp4_hopper.main)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def bench_example_dequant_gemm_bf16_mxfp4_hopper_tma():
    tilelang.tools.bench.process_func(example_dequant_gemm_bf16_mxfp4_hopper_tma.main)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def bench_example_dequant_groupedgemm_bf16_mxfp4_hopper():
    tilelang.tools.bench.process_func(example_dequant_groupedgemm_bf16_mxfp4_hopper.main)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def bench_example_dequant_gemm_w4a8():
    tilelang.tools.bench.process_func(example_dequant_gemm_w4a8.main)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()
