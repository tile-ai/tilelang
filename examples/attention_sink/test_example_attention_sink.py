import tilelang.testing

import example_mha_sink_fwd_bshd


@tilelang.testing.requires_cuda
def test_example_mha_sink_fwd_bshd_full_attn():
    example_mha_sink_fwd_bshd.main()


@tilelang.testing.requires_cuda
def test_example_mha_sink_fwd_bshd_sliding_window():
    example_mha_sink_fwd_bshd.main(window_size=128)


if __name__ == "__main__":
    tilelang.testing.main()
