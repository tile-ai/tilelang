import tilelang.tools.bench
import example_warp_specialize_gemm_barrierpipe_stage2
import example_warp_specialize_gemm_copy_0_gemm_1
import example_warp_specialize_gemm_copy_1_gemm_0
import example_warp_specialize_gemm_softpipe_stage2

@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def bench_example_warp_specialize_gemm_barrierpipe_stage2():
    tilelang.tools.bench.process_func(example_warp_specialize_gemm_barrierpipe_stage2.main)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def bench_example_warp_specialize_gemm_copy_0_gemm_1():
    tilelang.tools.bench.process_func(example_warp_specialize_gemm_copy_0_gemm_1.main)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def bench_example_warp_specialize_gemm_copy_1_gemm_0():
    tilelang.tools.bench.process_func(example_warp_specialize_gemm_copy_1_gemm_0.main)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def bench_example_warp_specialize_gemm_softpipe_stage2():
    tilelang.tools.bench.process_func(example_warp_specialize_gemm_softpipe_stage2.main)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()
