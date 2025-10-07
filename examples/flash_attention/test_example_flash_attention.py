import tilelang.testing

import example_gqa_bwd
import example_gqa_bwd_wgmma_pipelined
import example_mha_bwd
import example_mha_bwd_bhsd
import example_mha_fwd_bhsd_wgmma_pipelined
import example_gqa_fwd_bshd
import example_mha_fwd_bshd
import example_gqa_fwd_bshd_wgmma_pipelined
import example_mha_fwd_bshd_wgmma_pipelined
import example_mha_fwd_varlen
import example_mha_bwd_wgmma_pipelined
import example_mha_fwd_bhsd


@tilelang.testing.requires_cuda
def test_example_gqa_bwd():
    example_gqa_bwd.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_gqa_bwd_wgmma_pipelined():
    example_gqa_bwd_wgmma_pipelined.main()


@tilelang.testing.requires_cuda
def test_example_mha_bwd():
    example_mha_bwd.main()


@tilelang.testing.requires_cuda
def test_example_mha_bwd_bhsd():
    example_mha_bwd_bhsd.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_mha_bwd_wgmma_pipelined():
    example_mha_bwd_wgmma_pipelined.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_gqa_fwd_bshd_wgmma_pipelined():
    example_gqa_fwd_bshd_wgmma_pipelined.main()


@tilelang.testing.requires_cuda
def test_example_gqa_fwd_bshd():
    example_gqa_fwd_bshd.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_mha_fwd_bhsd_wgmma_pipelined():
    example_mha_fwd_bhsd_wgmma_pipelined.main()


@tilelang.testing.requires_cuda
def test_example_mha_fwd_bhsd():
    example_mha_fwd_bhsd.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_mha_fwd_bshd_wgmma_pipelined():
    example_mha_fwd_bshd_wgmma_pipelined.main()


@tilelang.testing.requires_cuda
def test_example_mha_fwd_bshd():
    example_mha_fwd_bshd.main()


@tilelang.testing.requires_cuda
def test_example_mha_fwd_varlen():
    example_mha_fwd_varlen.main()


if __name__ == "__main__":
    tilelang.testing.main()
