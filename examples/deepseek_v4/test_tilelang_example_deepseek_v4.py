import tilelang.testing

import act_quant
import fp8_fp4_gemm_1d1d_sm100
import sparse_attn_fwd_sm90


@tilelang.testing.requires_cuda
def test_example_act_quant():
    act_quant.test_fp8_act_quant(M=64, N=256, block_size=128)
    act_quant.test_fp4_act_quant(M=64, N=256, block_size=32)
    act_quant.test_round_trip_error()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(10, 0)
def test_example_fp8_fp4_gemm_1d1d():
    fp8_fp4_gemm_1d1d_sm100.check_correctness(M=256, N=8192, K=512)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_sparse_attn_fwd_sm90():
    sparse_attn_fwd_sm90.test_correctness()


if __name__ == "__main__":
    tilelang.testing.main()
