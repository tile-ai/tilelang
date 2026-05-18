import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

import example_dequant_gemv_fp16xint4
import example_dequant_gemm_fp4_hopper
import example_dequant_gemm_bf16_mxfp4_hopper
import example_dequant_gemm_w4a8


@tilelang.testing.requires_cuda
def test_example_dequant_gemv_fp16xint4():
    example_dequant_gemv_fp16xint4.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_dequant_gemm_fp4_hopper():
    example_dequant_gemm_fp4_hopper.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_dequant_gemm_bf16_mxfp4_hopper():
    example_dequant_gemm_bf16_mxfp4_hopper.main()


def test_example_dequant_gemm_bf16_mxfp4_hopper_ws_bias_lower():
    with tvm.target.Target("cuda -arch=sm_90"):
        func = example_dequant_gemm_bf16_mxfp4_hopper.matmul.jit_impl.get_tir(
            256,
            256,
            256,
            T.bfloat16,
            T.bfloat16,
            T.float32,
            num_bits=4,
            scale_size=32,
            block_M=256,
            block_N=128,
            block_K=128,
            num_stages=2,
            threads=256,
            split=1,
            fast_dequant=True,
            with_bias=True,
        )
        tilelang.lower(func, target="cuda -arch=sm_90", enable_device_compile=False)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_dequant_gemm_w4a8():
    example_dequant_gemm_w4a8.main()


if __name__ == "__main__":
    tilelang.testing.main()
