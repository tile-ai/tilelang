# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing

import example_dequant_gemv_fp16xint4
import example_dequant_gemm_fp4_hopper


@tilelang.testing.requires_cuda
def test_example_dequant_gemv_fp16xint4():
    example_dequant_gemv_fp16xint4.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_dequant_gemm_fp4_hopper():
    example_dequant_gemm_fp4_hopper.main()


if __name__ == "__main__":
    tilelang.testing.main()
