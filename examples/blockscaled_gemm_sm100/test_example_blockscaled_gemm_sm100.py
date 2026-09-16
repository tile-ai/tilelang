import torch

import tilelang.language as T
import tilelang.testing

import gemm_mxfp8_blockscaled_1d1d as ex


def _quantized_inputs(M, N, K, sf_granularity_k, transpose_B):
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_nt = torch.randn(N, K, device="cuda", dtype=torch.float16)
    a, sfa, _ = ex.quantize_fp8_with_packed_ue8m0(x, gran_k=sf_granularity_k)
    b_nt, sfb, _ = ex.quantize_fp8_with_packed_ue8m0(w_nt, gran_k=sf_granularity_k)
    b = b_nt if transpose_B else b_nt.T.contiguous()
    return a, b, sfa, sfb


def _check(kernel, M, N, K, block_M, block_N, block_K, num_stages, transpose_B):
    sf_granularity_k = 128
    a, b, sfa, sfb = _quantized_inputs(M, N, K, sf_granularity_k, transpose_B)
    c = kernel(
        a,
        b,
        sfa,
        sfb,
        block_M,
        block_N,
        block_K,
        T.float8_e4m3fn,
        T.bfloat16,
        T.float,
        num_stages,
        sf_granularity_k,
        transpose_B,
    )
    ref = ex.blockscaled_gemm_ref(a, b, sfa, sfb, sf_granularity_k, transpose_B=transpose_B)
    torch.testing.assert_close(c.float(), ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_example_mxfp8_blockscaled_gemm_1cta():
    # The 1-CTA kernel has no `use_2cta` short-circuit in instruction
    # selection, so it exercises the scope/target driven block-scaled dispatch.
    _check(ex.mxfp8_blockscaled_gemm, 512, 512, 512, 128, 128, 128, 4, transpose_B=False)
    _check(ex.mxfp8_blockscaled_gemm, 512, 512, 512, 128, 128, 128, 4, transpose_B=True)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_example_mxfp8_blockscaled_gemm_2cta():
    _check(ex.mxfp8_blockscaled_gemm_2cta, 256, 512, 512, 128, 256, 128, 4, transpose_B=False)
    _check(ex.mxfp8_blockscaled_gemm_2cta, 256, 512, 512, 128, 256, 128, 4, transpose_B=True)


if __name__ == "__main__":
    tilelang.testing.main()
