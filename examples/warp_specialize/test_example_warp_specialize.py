# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing

import example_warp_specialize_flashmla
import example_warp_specialize_gemm_barrierpipe_stage2
import example_warp_specialize_gemm_copy_0_gemm_1
import example_warp_specialize_gemm_copy_1_gemm_0
import example_warp_specialize_gemm_softpipe_stage2


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_warp_specialize_flashmla():
    example_warp_specialize_flashmla.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_warp_specialize_gemm_barrierpipe_stage2():
    example_warp_specialize_gemm_barrierpipe_stage2.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_warp_specialize_gemm_copy_0_gemm_1():
    example_warp_specialize_gemm_copy_0_gemm_1.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_warp_specialize_gemm_copy_1_gemm_0():
    example_warp_specialize_gemm_copy_1_gemm_0.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_warp_specialize_gemm_softpipe_stage2():
    example_warp_specialize_gemm_softpipe_stage2.main()


if __name__ == "__main__":
    tilelang.testing.main()
