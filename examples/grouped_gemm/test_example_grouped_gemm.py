import tilelang.testing

import example_grouped_gemm_bwd
import example_grouped_gemm_fwd
import example_grouped_gemm_fwd_ptr


def _run_grouped_gemm_fwd_small():
    """Run the portable tensor-based grouped GEMM forward validation."""
    example_grouped_gemm_fwd.run_tilelang_grouped_gemm(
        [5, 9, 13],
        K=64,
        M=96,
        block_M=64,
        block_N=64,
        block_K=32,
        trans_b=False,
        num_stages=2,
        threads=256,
        profile=False,
    )


def _run_grouped_gemm_bwd_small():
    """Run the portable grouped GEMM forward and weight-gradient validation."""
    example_grouped_gemm_bwd.run_tilelang_grouped_gemm(
        [5, 9, 13],
        K=64,
        M=96,
        block_M=64,
        block_N=64,
        block_K=32,
        trans_b=False,
        num_stages=2,
        threads=256,
        profile=False,
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_example_grouped_gemm_fwd_small():
    _run_grouped_gemm_fwd_small()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_example_grouped_gemm_fwd_ptr_small():
    example_grouped_gemm_fwd_ptr.run_tilelang_grouped_gemm_ptr(
        [5, 9, 13],
        K=64,
        N=96,
        block_M=64,
        block_N=64,
        block_K=32,
        num_stages=1,
        threads=256,
        profile=False,
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_example_grouped_gemm_bwd_small():
    _run_grouped_gemm_bwd_small()


@tilelang.testing.requires_rocm
def test_example_grouped_gemm_rocm():
    """Run the portable grouped GEMM validations on ROCm."""
    _run_grouped_gemm_fwd_small()
    _run_grouped_gemm_bwd_small()


if __name__ == "__main__":
    tilelang.testing.main()
