import tilelang.testing

import sm120_nvfp4_blockscaled_gemm


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_sm120_nvfp4_blockscaled_gemm():
    for out_dtype in ("bfloat16", "float32"):
        sm120_nvfp4_blockscaled_gemm.main(
            ["--m", "256", "--n", "384", "--k", "512", "--out-dtype", out_dtype, "--verify", "--n-warmup", "1", "--n-repeat", "1"]
        )


if __name__ == "__main__":
    tilelang.testing.main()
