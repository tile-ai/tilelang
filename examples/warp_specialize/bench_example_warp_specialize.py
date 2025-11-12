import tilelang.tools.bench
import example_warp_specialize_gemm_barrierpipe_stage2
import example_warp_specialize_gemm_copy_0_gemm_1
import example_warp_specialize_gemm_copy_1_gemm_0
import example_warp_specialize_gemm_softpipe_stage2


def bench_example_warp_specialize_gemm_barrierpipe_stage2():
    tilelang.tools.bench.process_func(example_warp_specialize_gemm_barrierpipe_stage2.main)


def bench_example_warp_specialize_gemm_copy_0_gemm_1():
    tilelang.tools.bench.process_func(example_warp_specialize_gemm_copy_0_gemm_1.main)


def bench_example_warp_specialize_gemm_copy_1_gemm_0():
    tilelang.tools.bench.process_func(example_warp_specialize_gemm_copy_1_gemm_0.main)


def bench_example_warp_specialize_gemm_softpipe_stage2():
    tilelang.tools.bench.process_func(example_warp_specialize_gemm_softpipe_stage2.main)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
