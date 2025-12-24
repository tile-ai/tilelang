import tilelang.testing

import example_mha_sink_fwd_bhsd
import example_mha_sink_fwd_bhsd_wgmma_pipelined
import example_gqa_sink_fwd_bhsd_wgmma_pipelined
import example_mha_sink_bwd_bhsd
import example_gqa_sink_bwd_bhsd


@tilelang.testing.requires_cuda
def test_example_mha_sink_fwd_bhsd_full_attn():
    example_mha_sink_fwd_bhsd.main()


@tilelang.testing.requires_cuda
def test_example_mha_sink_fwd_bhsd_sliding_window():
    example_mha_sink_fwd_bhsd.main(window_size=128)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_mha_sink_fwd_bhsd_wgmma_pipelined_full_attn():
    example_mha_sink_fwd_bhsd_wgmma_pipelined.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_mha_sink_fwd_bhsd_wgmma_pipelined_sliding_window():
    example_mha_sink_fwd_bhsd_wgmma_pipelined.main(window_size=128)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_gqa_sink_fwd_bhsd_wgmma_pipelined_full_attn():
    example_gqa_sink_fwd_bhsd_wgmma_pipelined.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_gqa_sink_fwd_bhsd_wgmma_pipelined_sliding_window():
    example_gqa_sink_fwd_bhsd_wgmma_pipelined.main(window_size=128)


@tilelang.testing.requires_cuda
def test_example_mha_sink_bwd_bhsd():
    example_mha_sink_bwd_bhsd.main()


@tilelang.testing.requires_cuda
def test_example_mha_sink_bwd_bhsd_sliding_window():
    example_mha_sink_bwd_bhsd.main(window_size=128)


@tilelang.testing.requires_cuda
def test_example_gqa_sink_bwd_bhsd():
    example_gqa_sink_bwd_bhsd.main()


@tilelang.testing.requires_cuda
def test_example_gqa_sink_bwd_bhsd_sliding_window():
    example_gqa_sink_bwd_bhsd.main(window_size=128)


if __name__ == "__main__":
    tilelang.testing.main()
