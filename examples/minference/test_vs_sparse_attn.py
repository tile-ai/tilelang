import tilelang.testing

import example_vertical_slash_sparse_attn


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_vs_sparse_attn():
    # The example copies Q with T.tma_copy, which needs TMA (sm_90+).
    example_vertical_slash_sparse_attn.main()


if __name__ == "__main__":
    tilelang.testing.main()
