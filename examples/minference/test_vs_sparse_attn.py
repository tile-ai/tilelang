import tilelang.testing

import example_vertical_slash_sparse_attn


@tilelang.testing.requires_cuda
def test_vs_sparse_attn():
    example_vertical_slash_sparse_attn.main(argv=[])


if __name__ == "__main__":
    tilelang.testing.main()
